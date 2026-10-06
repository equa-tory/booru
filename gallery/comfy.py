"""AI edit: ComfyUI img2img driven by an Ollama-written prompt.

One image goes in, a request typed by the user ("make her hair blue") is turned
into Danbooru-style tags by an Ollama model, and ComfyUI re-draws the image with
the WAI-Illustrious checkpoint at a partial denoise (so it is an EDIT of the input,
not a new picture — denoise 1.0 throws the input away). Everything here is plain
HTTP (httpx, trust_env=False because this machine exports HTTP(S)_PROXY) and the
ComfyUI API workflow is built in code (`build_img2img`); the same JSON is shipped
in docs/comfyui/booru_img2img_api.json so it can be opened in ComfyUI.

Settings live in the server-side pref `comfy` (see gallery/prefs.py).
"""
import hashlib
import io
import json
import re
import time

import httpx

from . import prefs

DEFAULTS = {
    'enabled': True,
    'comfy_url': 'http://127.0.0.1:8188',
    'ollama_url': 'http://127.0.0.1:11434',
    'ollama_model': 'qwen2.5:7b-instruct-q4_K_M',
    'checkpoint': 'models/waiIllustriousSDXL_v150_fp8.safetensors',
    'steps': 20,        # the Tesla P4 needs ~9 s per step at 1 MP (28 steps = 4 min)
    'cfg': 5.5,
    'sampler': 'euler_ancestral',
    'scheduler': 'normal',
    'denoise': 0.6,
    'megapixels': 0.8,
    'max_temp': 72,          # wait for the GPU to cool below this (°C) before sampling; 0 = off
    'free_after': True,      # ask ComfyUI to unload its models once the picture is done
    'quality': 'masterpiece, best quality, amazing quality, very aesthetic, absurdres',
    'negative': ('bad quality, worst quality, worst detail, sketch, censor, lowres, bad anatomy, '
                 'bad hands, jpeg artifacts, watermark, text, signature'),
}
DENOISE_MIN, DENOISE_MAX = 0.2, 0.85
MAX_UPLOAD_MP = 4.0          # bigger sources are shrunk before upload; ComfyUI rescales to `megapixels` anyway
MAX_TAGS = 100

RATING_TAGS = {'general', 'sensitive', 'questionable', 'explicit',
               'rating:general', 'rating:sensitive', 'rating:questionable', 'rating:explicit'}
QUALITY_WORDS = {'masterpiece', 'best quality', 'amazing quality', 'very aesthetic', 'absurdres', 'highres',
                 'high quality', 'good quality', 'great quality', 'newest', 'recent', 'ultra-detailed',
                 'score_9', 'score_8_up', 'score_7_up', '8k', 'hdr', 'worst quality', 'low quality',
                 'bad quality', 'normal quality'}


class ComfyError(Exception):
    pass


# ── settings ────────────────────────────────────────────────────
def config():
    cfg = dict(DEFAULTS)
    saved = prefs.get('comfy', {})
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in DEFAULTS and v not in (None, ''):
                cfg[k] = v
            elif k in ('quality', 'negative') and v == '':
                cfg[k] = ''          # an empty quality prefix / negative is a valid choice
    return clean_config(cfg)


def clean_config(cfg):
    """Coerce a settings dict to the right types and ranges (used for both saved
    prefs and values posted from the settings form)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in DEFAULTS})

    def num(key, cast, lo, hi):
        try:
            out[key] = min(hi, max(lo, cast(out[key])))
        except (TypeError, ValueError):
            out[key] = DEFAULTS[key]
    num('steps', int, 1, 150)
    num('cfg', float, 1.0, 30.0)
    num('denoise', float, DENOISE_MIN, DENOISE_MAX)
    num('megapixels', float, 0.25, 2.5)
    num('max_temp', int, 0, 110)
    out['enabled'] = bool(out['enabled'])
    out['free_after'] = bool(out['free_after'])
    for k in ('comfy_url', 'ollama_url'):
        out[k] = str(out[k]).strip().rstrip('/') or DEFAULTS[k]
    for k in ('ollama_model', 'checkpoint', 'sampler', 'scheduler'):
        out[k] = str(out[k]).strip() or DEFAULTS[k]
    for k in ('quality', 'negative'):
        out[k] = str(out[k]).strip()
    return out


def save_config(patch):
    cfg = config()
    cfg.update({k: v for k, v in patch.items() if k in DEFAULTS})
    cfg = clean_config(cfg)
    prefs.set('comfy', cfg)
    return cfg


# ── prompts ─────────────────────────────────────────────────────
def escape_tag(tag):
    """Danbooru name -> prompt text: `nakano_miku_(go-toubun)` would otherwise be read
    as a weight group, so spaces replace underscores and parentheses are escaped."""
    t = str(tag).strip().replace('_', ' ')
    t = re.sub(r'\s+', ' ', t)
    t = t.replace('\\(', '(').replace('\\)', ')')
    return t.replace('(', '\\(').replace(')', '\\)')


def clean_tags(tags):
    """Normalise a tag list coming from the tagger or the LLM: lowercase, no
    underscores, no rating/quality tags (those are added separately), no duplicates."""
    out, seen = [], set()
    for t in tags or []:
        t = re.sub(r'\s+', ' ', str(t).strip().lower().replace('_', ' ').replace('\\(', '(').replace('\\)', ')'))
        t = t.strip(' ,.;')
        if not t or t in RATING_TAGS or t in QUALITY_WORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:MAX_TAGS]


def join_prompt(*parts):
    """Comma-join prompt fragments; tags are escaped, ready-made strings kept as they are."""
    bits = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, str):
            p = p.strip(' ,')
            if p:
                bits.append(p)
        else:
            bits.extend(escape_tag(t) for t in p if str(t).strip())
    return ', '.join(bits)


SYSTEM_PROMPT = """You convert a picture-editing request into prompt tags for an anime image model (Illustrious / Danbooru tags).
You get the tags that describe the CURRENT picture and the user's request (any language). The picture will be re-drawn from itself, so the new tag list must describe the WHOLE picture after the edit.
Rules:
- Return the complete new tag list: keep every current tag that still applies, remove tags the request contradicts, add Danbooru tags for the change.
- Tags are lowercase Danbooru tags with spaces (e.g. "long hair", "blue eyes", "white dress"). No sentences, no quality tags (masterpiece, best quality...), no rating tags.
- "negative" lists only things to avoid that relate to the request (e.g. the old hair colour). It may be empty.
- "denoise" is how much of the picture may change: 0.40-0.50 small details and accessories (glasses, a ribbon), 0.55-0.65 colours of hair, eyes or clothes, clothes, expression, objects, 0.65-0.75 pose, framing or big changes. Never above 0.8.
- "note" is one short sentence saying what you changed.
Answer with JSON only: {"positive": [tags], "negative": [tags], "denoise": number, "note": string}"""

PROMPT_SCHEMA = {
    'type': 'object',
    'properties': {
        'positive': {'type': 'array', 'items': {'type': 'string'}},
        'negative': {'type': 'array', 'items': {'type': 'string'}},
        'denoise': {'type': 'number'},
        'note': {'type': 'string'},
    },
    'required': ['positive', 'negative', 'denoise', 'note'],
}


def _client(timeout=10.0):
    return httpx.Client(timeout=timeout, trust_env=False)


def parse_llm_prompt(text, default_denoise):
    """The LLM's JSON answer -> {'positive','negative','denoise','note'}; raises ValueError
    when nothing usable is in it."""
    text = (text or '').strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r'\{.*\}', text, re.S)      # a model that wrapped the JSON in prose / a code fence
        if not m:
            raise ValueError('no JSON in the model answer')
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError('unexpected JSON')

    def as_tags(v):
        if isinstance(v, str):
            v = v.split(',')
        return clean_tags(v if isinstance(v, list) else [])
    pos = as_tags(data.get('positive'))
    if not pos:
        raise ValueError('the model returned no tags')
    try:
        den = float(data.get('denoise'))
    except (TypeError, ValueError):
        den = default_denoise
    return {
        'positive': pos,
        'negative': as_tags(data.get('negative')),
        'denoise': min(DENOISE_MAX, max(DENOISE_MIN, den)),
        'note': str(data.get('note') or '')[:300],
    }


def to_prompt(request_text, current_tags, cfg=None):
    """Ask Ollama to turn the user's request + the picture's current tags into the new
    tag list. Never raises: if Ollama is down or answers garbage, falls back to the
    current tags plus the request text itself (`note` says so). The model stays loaded
    for 2 minutes (a preview followed by "generate" loads it once); `generate` unloads
    it before ComfyUI starts if it sits in VRAM (here Ollama runs on the CPU)."""
    cfg = cfg or config()
    current = clean_tags(current_tags)
    user = ('Current tags: ' + (', '.join(current) or '(none)') + '\n'
            'Request: ' + request_text.strip())
    body = {
        'model': cfg['ollama_model'], 'stream': False, 'think': False, 'keep_alive': '2m',
        'format': PROMPT_SCHEMA,
        'options': {'temperature': 0.2, 'num_ctx': 4096},
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}],
    }
    try:
        with _client(180.0) as c:
            r = c.post(cfg['ollama_url'] + '/api/chat', json=body)
        if r.status_code != 200:
            raise ComfyError(f'Ollama HTTP {r.status_code}: {r.text[:200]}')
        out = parse_llm_prompt((r.json().get('message') or {}).get('content', ''), cfg['denoise'])
        out['source'] = 'llm'
        return out
    except Exception as e:
        return {'positive': clean_tags(current + [request_text]), 'negative': [], 'denoise': cfg['denoise'],
                'note': f'Ollama unavailable ({str(e)[:120]}) — used your text as is', 'source': 'fallback'}


# ── the ComfyUI API workflow ────────────────────────────────────
def build_img2img(image_name, positive, negative, seed, cfg=None, denoise=None, loras=(), temp_node=None):
    """ComfyUI API-format workflow (dict) for an img2img edit:

        load checkpoint -> (LoRAs) -> encode prompts
        load image -> scale to ~1 MP -> VAE encode -> [wait for GPU temp] -> KSampler (denoise) -> VAE decode -> preview

    `positive`/`negative` are final prompt strings. `loras` = [(name, strength)].
    `temp_node` = True adds the WaitForGPUTemperature passthrough (None = follow cfg['max_temp']).
    The output node is PreviewImage so nothing is written to ComfyUI's output folder."""
    cfg = cfg or config()
    den = cfg['denoise'] if denoise is None else denoise
    den = min(DENOISE_MAX, max(DENOISE_MIN, float(den)))
    wf = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': cfg['checkpoint']}},
        '2': {'class_type': 'LoadImage', 'inputs': {'image': image_name}},
        '3': {'class_type': 'ImageScaleToTotalPixels',
              'inputs': {'image': ['2', 0], 'upscale_method': 'lanczos',
                         'megapixels': float(cfg['megapixels']), 'resolution_steps': 8}},
        '4': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['3', 0], 'vae': ['1', 2]}},
    }
    model, clip = ['1', 0], ['1', 1]
    for i, (name, strength) in enumerate(loras):
        nid = str(20 + i)
        wf[nid] = {'class_type': 'LoraLoader',
                   'inputs': {'model': model, 'clip': clip, 'lora_name': name,
                              'strength_model': float(strength), 'strength_clip': float(strength)}}
        model, clip = [nid, 0], [nid, 1]
    wf['5'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': positive, 'clip': clip}}
    wf['6'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': negative, 'clip': clip}}
    latent = ['4', 0]
    if temp_node is None:
        temp_node = int(cfg['max_temp']) > 0
    if temp_node and int(cfg['max_temp']) > 0:
        wf['7'] = {'class_type': 'WaitForGPUTemperature',
                   'inputs': {'passthrough': latent, 'max_temp_c': int(cfg['max_temp']), 'gpu_index': 0,
                              'poll_interval_sec': 5.0, 'max_wait_sec': 600}}
        latent = ['7', 0]
    wf['8'] = {'class_type': 'KSampler',
               'inputs': {'model': model, 'seed': int(seed), 'steps': int(cfg['steps']), 'cfg': float(cfg['cfg']),
                          'sampler_name': cfg['sampler'], 'scheduler': cfg['scheduler'],
                          'positive': ['5', 0], 'negative': ['6', 0], 'latent_image': latent,
                          'denoise': den}}
    wf['9'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['8', 0], 'vae': ['1', 2]}}
    wf['10'] = {'class_type': 'PreviewImage', 'inputs': {'images': ['9', 0]}}
    return wf


EXAMPLE_PROMPT = ('masterpiece, best quality, amazing quality, very aesthetic, absurdres, 1girl, solo, long hair, '
                  'blue hair, glasses, school uniform')


def example_workflow():
    """The workflow with the shipped defaults and an example prompt; written to
    docs/comfyui/booru_img2img_api.json (a test keeps the file in sync). Open it in ComfyUI
    (drag the file in, API format), pick your own image in "Load Image" and run it."""
    return build_img2img('example.png', EXAMPLE_PROMPT, DEFAULTS['negative'], 123456789, dict(DEFAULTS), 0.6, temp_node=True)


def workflow_problems(wf):
    """Static sanity check of an API workflow: every link points at an existing node
    and a plausible output slot. Returns a list of problems (empty = fine)."""
    problems = []
    for nid, node in wf.items():
        for key, val in node['inputs'].items():
            if isinstance(val, list):
                if len(val) != 2 or str(val[0]) not in wf:
                    problems.append(f'node {nid} input {key}: bad link {val}')
    return problems


# ── ComfyUI HTTP ────────────────────────────────────────────────
def _comfy_get(cfg, path, timeout=5.0):
    try:
        with _client(timeout) as c:
            r = c.get(cfg['comfy_url'] + path)
    except httpx.HTTPError as e:
        raise ComfyError(f'ComfyUI not reachable at {cfg["comfy_url"]} ({e.__class__.__name__})')
    if r.status_code != 200:
        raise ComfyError(f'ComfyUI {path}: HTTP {r.status_code}')
    return r


def status(cfg=None):
    """Live state for the settings panel. Short timeouts: never hangs the page."""
    cfg = cfg or config()
    out = {'config': cfg, 'comfy': {'ok': False}, 'ollama': {'ok': False}}
    try:
        stats = _comfy_get(cfg, '/system_stats').json()
        dev = (stats.get('devices') or [{}])[0]
        info = {'ok': True, 'version': stats.get('system', {}).get('comfyui_version', ''),
                'gpu': dev.get('name', ''), 'vram_free': dev.get('vram_free', 0), 'vram_total': dev.get('vram_total', 0)}
        ks = _comfy_get(cfg, '/object_info/KSampler').json()['KSampler']['input']['required']
        info['samplers'] = ks['sampler_name'][0]
        info['schedulers'] = ks['scheduler'][0]
        ck = _comfy_get(cfg, '/object_info/CheckpointLoaderSimple').json()['CheckpointLoaderSimple']['input']['required']
        info['checkpoints'] = ck['ckpt_name'][0]
        info['temp_node'] = bool(_comfy_get(cfg, '/object_info/WaitForGPUTemperature').json())
        out['comfy'] = info
    except (ComfyError, KeyError, ValueError, IndexError) as e:
        out['comfy'] = {'ok': False, 'error': str(e)}
    try:
        with _client(4.0) as c:
            ver = c.get(cfg['ollama_url'] + '/api/version').json().get('version', '')
            tags = c.get(cfg['ollama_url'] + '/api/tags').json().get('models', [])
        out['ollama'] = {'ok': True, 'version': ver, 'models': sorted(m['name'] for m in tags)}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        out['ollama'] = {'ok': False, 'error': f'Ollama not reachable at {cfg["ollama_url"]} ({e.__class__.__name__})'}
    return out


def _png_bytes(path):
    """The source image as PNG bytes (shrunk when huge, flattened to RGB)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if getattr(im, 'is_animated', False):
            im.seek(0)
        if im.mode not in ('RGB', 'L'):
            bg = Image.new('RGB', im.size, (255, 255, 255))
            rgba = im.convert('RGBA')
            bg.paste(rgba, mask=rgba.split()[3])
            im = bg
        else:
            im = im.convert('RGB')
        w, h = im.size
        if w * h > MAX_UPLOAD_MP * 1_000_000:
            k = (MAX_UPLOAD_MP * 1_000_000 / (w * h)) ** 0.5
            im = im.resize((max(8, int(w * k)), max(8, int(h * k))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'PNG')
        return buf.getvalue()


def upload_image(path, cfg=None):
    """Upload the source image to ComfyUI's input folder (subfolder `booru`); returns
    the name to put in LoadImage."""
    cfg = cfg or config()
    data = _png_bytes(path)
    name = f'src_{hashlib.md5(data).hexdigest()[:16]}.png'      # same picture re-edited -> same file, no pile-up
    try:
        with _client(60.0) as c:
            r = c.post(cfg['comfy_url'] + '/upload/image',
                       files={'image': (name, data, 'image/png')},
                       data={'subfolder': 'booru', 'type': 'input', 'overwrite': 'true'})
    except httpx.HTTPError as e:
        raise ComfyError(f'upload to ComfyUI failed ({e.__class__.__name__})')
    if r.status_code != 200:
        raise ComfyError(f'upload to ComfyUI failed: HTTP {r.status_code} {r.text[:200]}')
    j = r.json()
    sub = j.get('subfolder') or ''
    return f'{sub}/{j["name"]}' if sub else j['name']


def queue_prompt(wf, cfg=None):
    cfg = cfg or config()
    try:
        with _client(30.0) as c:
            r = c.post(cfg['comfy_url'] + '/prompt', json={'prompt': wf, 'client_id': 'booru'})
    except httpx.HTTPError as e:
        raise ComfyError(f'ComfyUI not reachable ({e.__class__.__name__})')
    if r.status_code != 200:
        detail = r.text[:600]
        try:
            j = r.json()
            errs = [f'{v.get("class_type")}: ' + '; '.join(x.get('message', '') + ' ' + x.get('details', '')
                                                           for x in v.get('errors', []))
                    for v in (j.get('node_errors') or {}).values()]
            detail = (j.get('error', {}).get('message', '') + ' ' + ' | '.join(errs)).strip() or detail
        except ValueError:
            pass
        raise ComfyError(f'ComfyUI rejected the workflow: {detail}')
    return r.json()['prompt_id']


def cancel_prompt(prompt_id, cfg=None):
    """Best effort: drop it from the queue and interrupt it if it is running."""
    cfg = cfg or config()
    try:
        with _client(5.0) as c:
            c.post(cfg['comfy_url'] + '/queue', json={'delete': [prompt_id]})
            c.post(cfg['comfy_url'] + '/interrupt', json={'prompt_id': prompt_id})
    except httpx.HTTPError:
        pass


def wait_for_result(prompt_id, cfg=None, check=None, progress=None, timeout=900, poll=1.5):
    """Poll /history until the prompt finished; returns the output image descriptor
    {'filename','subfolder','type'}. `check()` is called every poll (it raises to
    cancel: ComfyUI is interrupted first). `progress(text)` gets queue position / state."""
    cfg = cfg or config()
    t0 = time.time()
    last = ''
    while True:
        if check:
            try:
                check()
            except BaseException:
                cancel_prompt(prompt_id, cfg)
                raise
        if time.time() - t0 > timeout:
            cancel_prompt(prompt_id, cfg)
            raise ComfyError(f'ComfyUI did not finish within {timeout // 60} minutes')
        hist = _comfy_get(cfg, f'/history/{prompt_id}', 10.0).json()
        item = hist.get(prompt_id)
        if item:
            st = item.get('status') or {}
            if st.get('status_str') == 'error':
                msg = ''
                for kind, d in st.get('messages', []):
                    if kind == 'execution_error':
                        msg = f'{d.get("node_type", "")}: {d.get("exception_message", "")}'.strip()
                raise ComfyError('ComfyUI failed: ' + (msg[:400] or 'unknown error'))
            for out in (item.get('outputs') or {}).values():
                if out.get('images'):
                    return out['images'][0]
            if st.get('completed'):
                raise ComfyError('ComfyUI finished without producing an image')
        if progress:
            text = 'generating…'
            try:
                q = _comfy_get(cfg, '/queue', 5.0).json()
                ids = [p[1] for p in q.get('queue_pending', [])]
                if prompt_id in ids:
                    text = f'waiting in the ComfyUI queue (#{ids.index(prompt_id) + 1})'
            except ComfyError:
                pass
            if text != last:
                progress(text)
                last = text
        time.sleep(poll)


def fetch_output(desc, cfg=None):
    cfg = cfg or config()
    try:
        with _client(60.0) as c:
            r = c.get(cfg['comfy_url'] + '/view', params={
                'filename': desc['filename'], 'subfolder': desc.get('subfolder', ''), 'type': desc.get('type', 'temp')})
    except httpx.HTTPError as e:
        raise ComfyError(f'could not download the result ({e.__class__.__name__})')
    if r.status_code != 200 or not r.content:
        raise ComfyError(f'could not download the result: HTTP {r.status_code}')
    return r.content


def free_comfy(cfg=None, timeout=3.0):
    """Ask ComfyUI to unload its models and free VRAM (best effort)."""
    cfg = cfg or config()
    try:
        with _client(timeout) as c:
            c.post(cfg['comfy_url'] + '/free', json={'unload_models': True, 'free_memory': True})
        return True
    except httpx.HTTPError:
        return False


def unload_ollama(cfg=None):
    """Unload every Ollama model that occupies VRAM (the 8 GB card must hold the
    SDXL checkpoint). Models running on the CPU are left alone. Best effort."""
    cfg = cfg or config()
    try:
        with _client(10.0) as c:
            for m in c.get(cfg['ollama_url'] + '/api/ps').json().get('models', []):
                if m.get('size_vram', 0) > 0:
                    c.post(cfg['ollama_url'] + '/api/generate', json={'model': m['name'], 'keep_alive': 0})
    except (httpx.HTTPError, ValueError):
        pass


def generate(src_path, positive, negative, seed, denoise, cfg=None, check=None, progress=None):
    """Upload `src_path`, run the img2img workflow, return (png_bytes, workflow_used)."""
    cfg = cfg or config()
    stat = status_light(cfg)
    if progress:
        progress('uploading to ComfyUI…')
    unload_ollama(cfg)
    name = upload_image(src_path, cfg)
    wf = build_img2img(name, positive, negative, seed, cfg, denoise, temp_node=stat['temp_node'])
    problems = workflow_problems(wf)
    if problems:
        raise ComfyError('internal workflow error: ' + problems[0])
    pid = queue_prompt(wf, cfg)
    if progress:
        progress('generating…')
    desc = wait_for_result(pid, cfg, check=check, progress=progress)
    return fetch_output(desc, cfg), wf


def status_light(cfg):
    """Just what `generate` needs: is ComfyUI up, does it have the temperature node."""
    _comfy_get(cfg, '/system_stats')
    try:
        return {'temp_node': bool(_comfy_get(cfg, '/object_info/WaitForGPUTemperature').json())}
    except ComfyError:
        return {'temp_node': False}
