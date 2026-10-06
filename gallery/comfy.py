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
import os
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
    'denoise': 0.65,
    'megapixels': 0.8,
    'max_temp': 72,          # wait for the GPU to cool below this (°C) before sampling; 0 = off
    'free_after': True,      # ask ComfyUI to unload its models once the picture is done
    'quality': 'masterpiece, best quality, amazing quality, very aesthetic, absurdres',
    'negative': ('bad quality, worst quality, worst detail, sketch, censor, lowres, bad anatomy, '
                 'bad hands, jpeg artifacts, watermark, text, signature'),
    'lora_bases': 'Illustrious, NoobAI',   # LoRA base models that fit the checkpoint ('' = all)
}
DENOISE_MIN, DENOISE_MAX = 0.2, 0.9
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
            elif k in ('quality', 'negative', 'lora_bases') and v == '':
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
    for k in ('quality', 'negative', 'lora_bases'):
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


ADD_WEIGHT = 1.3        # the new tags are put first and weighted, or the ~40 tags describing the old picture drown them
MAX_NEGATIVE = 20

# The model only has to return a DIFF (tags to add / tags to remove); the code builds the
# final prompt. Small models asked to rewrite the whole list just echoed the old picture
# (kept "school uniform" next to "astronaut suit", put removals only in the negative).
# No tool calling is needed: Ollama enforces the JSON schema (`format`) for ANY model.
SYSTEM_PROMPT = """You edit pictures for an anime image model that understands Danbooru tags.
You get the tags of the CURRENT picture and the user's request (any language). Answer with what changes:
- "add": Danbooru tags for what the request wants to see (lowercase, spaces instead of underscores). Be concrete: an outfit gets its main tag plus its typical parts.
- "remove": tags copied EXACTLY from the current list that contradict the request or that the user wants gone. Replacing an outfit removes ALL tags of the old clothes (shirt, skirt, uniform, sweater, cardigan, socks, shoes...). Changing a colour removes the old colour tag.
- "denoise": how much the picture must change: 0.45 small item or accessory, 0.6 colours (hair, eyes, clothes), 0.7 removing something, 0.75 outfit / costume, 0.8 pose, background or composition. Use 0.55 to 0.85.
- "note": one short sentence describing the edit.
Examples:
Current: 1girl, long hair, brown hair, school uniform, skirt, smile | Request: make her hair blue
{"add": ["blue hair"], "remove": ["brown hair"], "denoise": 0.6, "note": "brown hair -> blue hair"}
Current: 1girl, school uniform, white shirt, pleated skirt, black socks, loafers, cardigan, standing | Request: put her in a bikini
{"add": ["bikini", "swimsuit", "navel", "barefoot"], "remove": ["school uniform", "white shirt", "pleated skirt", "black socks", "loafers", "cardigan"], "denoise": 0.75, "note": "school uniform -> bikini"}
Current: 1girl, glasses, red eyes, hat, smile | Request: remove the glasses
{"add": [], "remove": ["glasses"], "denoise": 0.7, "note": "glasses removed"}
Current: 1girl, sitting, indoors, window | Request: add a cat on her lap
{"add": ["cat", "cat on lap", "animal on lap"], "remove": [], "denoise": 0.6, "note": "added a cat on her lap"}
Answer with JSON only."""

PROMPT_SCHEMA = {
    'type': 'object',
    'properties': {
        'add': {'type': 'array', 'items': {'type': 'string'}},
        'remove': {'type': 'array', 'items': {'type': 'string'}},
        'denoise': {'type': 'number'},
        'note': {'type': 'string'},
    },
    'required': ['add', 'remove', 'denoise', 'note'],
}


def _client(timeout=10.0):
    return httpx.Client(timeout=timeout, trust_env=False)


def _norm(t):
    return re.sub(r'[\s_]+', ' ', str(t).strip().lower().replace('\\(', '(').replace('\\)', ')')).strip(' ,.;')


def parse_llm_prompt(text, default_denoise):
    """The LLM's JSON answer -> {'add','remove','denoise','note'}; raises ValueError when
    nothing usable is in it (no JSON, or neither additions nor removals)."""
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
    add, remove = as_tags(data.get('add')), as_tags(data.get('remove'))
    if not add and not remove:
        raise ValueError('the model returned no change')
    try:
        den = float(data.get('denoise'))
    except (TypeError, ValueError):
        den = default_denoise
    return {'add': add, 'remove': remove,
            'denoise': min(DENOISE_MAX, max(DENOISE_MIN, den)),
            'note': str(data.get('note') or '')[:300]}


def apply_diff(current, add, remove):
    """Build the final tag lists: new tags first (weighted when shown via `prompt_texts`),
    then every current tag that was not removed; removed tags go to the negative.
    A tag both added and removed counts as added. Matching ignores case/underscores."""
    cur = clean_tags(current)
    add = [t for t in clean_tags(add)]
    add_n = {_norm(t) for t in add}
    rem_n = {_norm(t) for t in clean_tags(remove)} - add_n
    keep = [t for t in cur if _norm(t) not in rem_n and _norm(t) not in add_n]
    neg = [t for t in clean_tags(remove) if _norm(t) not in add_n][:MAX_NEGATIVE]
    return add, keep, neg


def prompt_texts(add, keep, neg):
    """(positive text, negative text) as shown in the box / sent to ComfyUI (without the
    quality prefix and default negative, which are added on run)."""
    pos = ', '.join([f'({escape_tag(t)}:{ADD_WEIGHT})' for t in add] + [escape_tag(t) for t in keep])
    return pos, ', '.join(escape_tag(t) for t in neg)


def to_prompt(request_text, current_tags, cfg=None):
    """Ask Ollama what the request changes (add/remove tags) and build the prompt from the
    picture's current tags. Never raises: if Ollama is down or answers garbage, the request
    text itself becomes the weighted first part of the prompt (`note` says so).
    Returns {'positive','negative' (final texts), 'add','remove','denoise','note','source'}.
    The model stays loaded for 2 minutes (preview then generate loads it once); `generate`
    unloads it before ComfyUI starts if it sits in VRAM (here Ollama runs on the CPU)."""
    cfg = cfg or config()
    current = clean_tags(current_tags)
    user = ('Current: ' + (', '.join(current) or '(none)') + ' | Request: ' + request_text.strip())
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
        d = parse_llm_prompt((r.json().get('message') or {}).get('content', ''), cfg['denoise'])
        source, note = 'llm', d['note']
    except Exception as e:
        d = {'add': clean_tags([request_text]), 'remove': [], 'denoise': max(cfg['denoise'], 0.65)}
        source, note = 'fallback', f'Ollama gave no usable answer ({str(e)[:120]}) — your text was used as is'
    add, keep, neg = apply_diff(current, d['add'], d['remove'])
    pos_text, neg_text = prompt_texts(add, keep, neg)
    return {'positive': pos_text, 'negative': neg_text, 'add': add, 'remove': neg,
            'denoise': d['denoise'], 'note': note, 'source': source}


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


def generate(src_path, positive, negative, seed, denoise, cfg=None, check=None, progress=None, loras=()):
    """Upload `src_path`, run the img2img workflow (with `loras` = [(name, strength)]),
    return (png_bytes, workflow_used)."""
    cfg = cfg or config()
    stat = status_light(cfg)
    if progress:
        progress('uploading to ComfyUI…')
    unload_ollama(cfg)
    name = upload_image(src_path, cfg)
    wf = build_img2img(name, positive, negative, seed, cfg, denoise, loras=loras, temp_node=stat['temp_node'])
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


# ── LoRAs (catalog from ComfyUI-Lora-Manager, suggestions, trigger-word groups) ────────
MAX_LORAS = 6
CATALOG_TTL = 600
_catalog_cache = {'t': 0.0, 'key': None, 'items': None}
_LORA_SYNTAX = re.compile(r'<lora:[^>]*>', re.I)
GENERIC_TOKENS = {'illustrious', 'illustriousxl', 'illu', 'illus', 'il', 'ill', 'ilxl', 'illxl', 'xl', 'sdxl', 'lora',
                  'nochekaiser', 'character', 'characters', 'concept', 'style', 'v1', 'v2', 'v3', 'v4', 'v5', 'the',
                  'and', 'of', 'a', 'an', 'in', 'with', 'for', 'to', 'by', 'from', 'epoch', 'version', 'model', 'anime',
                  'safetensors', 'pony', 'noobai', 'her', 'his', 'make', 'add', 'put', 'remove', 'on', 'it', 'is'}


def clean_trigger_group(text):
    """One trigger-word group -> clean comma list ('' when nothing is left).
    A1111 `<lora:x:1>` syntax is dropped (meaningless in ComfyUI)."""
    words = [re.sub(r'\s+', ' ', w).strip() for w in _LORA_SYNTAX.sub('', str(text or '')).split(',')]
    return ', '.join(w for w in words if w)


def _tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+', str(text).lower().replace('_', ' ')) if t not in GENERIC_TOKENS and not t.isdigit()]


def _lora_item(it):
    folder = (it.get('folder') or '').strip('/')
    fname = it.get('file_name') or ''
    path = it.get('file_path') or ''
    ext = os.path.splitext(path)[1] if path else '.safetensors'
    name = (folder + '/' if folder else '') + fname + (ext or '.safetensors')
    groups = [g for g in (clean_trigger_group(x) for x in ((it.get('civitai') or {}).get('trainedWords') or [])) if g]
    tags = [str(t).lower() for t in (it.get('tags') or [])]
    kind = 'character' if ('character' in folder.lower().split('/') or 'character' in tags) else 'other'
    # preview_url is '/api/lm/previews?path=<absolute file>'; keep the path of image previews
    # (videos are skipped) for the booru proxy, the browser can't reach ComfyUI's 127.0.0.1
    from urllib.parse import parse_qs, urlparse
    pv = it.get('preview_url') or ''
    pv = parse_qs(urlparse(pv).query).get('path', [''])[0] if '?' in pv else pv
    preview = pv if re.search(r'\.(jpe?g|png|webp|gif)$', pv, re.I) else ''
    return {'name': name, 'title': it.get('model_name') or fname, 'file': fname, 'folder': folder,
            'base': it.get('base_model') or '', 'tags': tags, 'groups': groups, 'kind': kind, 'preview': preview}


def lora_catalog(cfg=None, force=False):
    """All LoRAs usable with the checkpoint (base model in cfg['lora_bases']), from the
    LoRA Manager API (100 per page), cached 10 minutes per process. Without LoRA Manager,
    ComfyUI's plain /models/loras names are used (no tags, no trigger words)."""
    cfg = cfg or config()
    key = (cfg['comfy_url'], cfg['lora_bases'])
    now = time.time()
    if not force and _catalog_cache['items'] is not None and _catalog_cache['key'] == key and now - _catalog_cache['t'] < CATALOG_TTL:
        return _catalog_cache['items']
    items = []
    try:
        with _client(20.0) as c:
            page = 1
            while page <= 100:
                r = c.get(cfg['comfy_url'] + '/api/lm/loras/list', params={'page': page, 'page_size': 100})
                if r.status_code != 200:
                    raise ComfyError(f'LoRA Manager HTTP {r.status_code}')
                d = r.json()
                items += [_lora_item(it) for it in d.get('items', [])]
                if page >= int(d.get('total_pages') or 1):
                    break
                page += 1
    except (httpx.HTTPError, ValueError, ComfyError):
        items = []
        try:
            names = _comfy_get(cfg, '/models/loras', 10.0).json()
            items = [{'name': n, 'title': os.path.splitext(n.rsplit('/', 1)[-1])[0], 'file': os.path.splitext(n.rsplit('/', 1)[-1])[0],
                      'folder': n.rsplit('/', 1)[0] if '/' in n else '', 'base': '', 'tags': [], 'groups': [],
                      'kind': 'character' if '/character' in n.lower() else 'other', 'preview': ''} for n in names]
        except (ComfyError, ValueError):
            items = []
    bases = {b.strip().lower() for b in cfg['lora_bases'].split(',') if b.strip()}
    if bases:
        items = [i for i in items if not i['base'] or i['base'].lower() in bases]
    items.sort(key=lambda i: i['title'].lower())
    _catalog_cache.update(t=now, key=key, items=items)
    return items


def search_loras(q, catalog, limit=12):
    toks = _tokens(q) or [w for w in re.findall(r'[a-z0-9]+', q.lower())]
    if not toks:
        return catalog[:limit]
    out = []
    for it in catalog:
        hay = ' '.join([it['title'], it['file'], it['folder'], ' '.join(it['tags'])]).lower().replace('_', ' ')
        if all(t in hay for t in toks):
            title_hit = all(t in it['title'].lower() for t in toks)
            out.append((0 if title_hit else 1, it['title'].lower(), it))
    return [x[2] for x in sorted(out, key=lambda x: x[:2])[:limit]]


def default_groups(item):
    """Indexes of the trigger groups pre-selected for a LoRA: a character's first group
    (its identity; the others are usually outfits), else all of up to 2 groups, else the first."""
    n = len(item['groups'])
    if not n:
        return []
    if item['kind'] == 'character' or n > 2:
        return [0]
    return list(range(n))


def _char_name(tag):
    """'nakano_miku_(go-toubun_no_hanayome)' -> 'nakano miku'."""
    return re.sub(r'\(.*?\)', '', str(tag).replace('_', ' ')).strip()


def suggest_loras(char_tags, terms, catalog, limit=8):
    """Deterministic LoRA suggestions: characters whose full name matches the picture's
    character tags, then concept/clothing LoRAs whose tags or trigger words match the
    request / added tags. Returns items + score, reason, default_groups."""
    scored = {}

    def bump(it, score, reason):
        cur = scored.setdefault(it['name'], {'item': it, 'score': 0, 'reasons': []})
        cur['score'] += score
        if reason not in cur['reasons']:
            cur['reasons'].append(reason)

    for ct in char_tags or []:
        name = _char_name(ct)
        toks = {t for t in re.findall(r'[a-z0-9]+', name.lower()) if not t.isdigit()}
        if not toks:
            continue
        for it in catalog:
            first = it['groups'][0] if it['groups'] else ''
            hay = set(re.findall(r'[a-z0-9]+', ' '.join([it['title'], it['file'], ' '.join(it['tags']), first]).lower().replace('_', ' ')))
            if toks <= hay:
                bump(it, 100 + len(toks) + (5 if it['kind'] == 'character' else 0), 'character: ' + name)
    phrases = []
    for t in terms or []:
        t = re.sub(r'\s+', ' ', str(t).lower().replace('_', ' ')).strip()
        if t and t not in GENERIC_TOKENS and t not in phrases:
            phrases.append(t)
    for it in catalog:
        if it['kind'] == 'character':
            continue
        # key trigger words = the first 3 of each group (the rest are often generic: 1girl, solo, school uniform...)
        key_words = {w.strip().lower().replace('_', ' ') for g in it['groups'] for w in g.split(',')[:3]}
        tagset = {t.replace('_', ' ') for t in it['tags']}
        title_toks = set(_tokens(it['title'] + ' ' + it['file']))
        for ph in phrases:
            if ph in tagset or ph in key_words:
                bump(it, 10, ph)
            elif len(ph) > 3 and set(_tokens(ph)) and set(_tokens(ph)) <= title_toks:
                bump(it, 6, ph)
    out = []
    for v in sorted(scored.values(), key=lambda v: (-v['score'], v['item']['title'].lower())):
        if v['score'] < 10:
            continue
        it = dict(v['item'])
        it.update(score=v['score'], reason=', '.join(v['reasons'][:4]), default_groups=default_groups(it))
        out.append(it)
        if len(out) >= limit:
            break
    return out


def trigger_text(words):
    """Selected trigger words -> prompt text: parentheses escaped, underscores kept."""
    parts = []
    for g in words or []:
        for w in clean_trigger_group(g).split(','):
            w = w.strip()
            if w and w not in parts:
                parts.append(w)
    return ', '.join(w.replace('\\(', '(').replace('\\)', ')').replace('(', '\\(').replace(')', '\\)') for w in parts)


def drop_terms(text, remove_text):
    """Comma text minus the items that also appear in `remove_text` (case/underscore/escape
    insensitive): trigger words the user selected must never sit in the negative too."""
    gone = {_norm(w) for w in str(remove_text or '').split(',') if w.strip()}
    return ', '.join(w.strip() for w in str(text or '').split(',') if w.strip() and _norm(w) not in gone)


def resolve_loras(requested, catalog):
    """Browser LoRA list -> [{'name','title','strength','words'}]; names must exist in
    the catalog (raises ComfyError otherwise)."""
    if not requested:
        return []
    if not isinstance(requested, list):
        raise ComfyError('bad LoRA list')
    if len(requested) > MAX_LORAS:
        raise ComfyError(f'at most {MAX_LORAS} LoRAs')
    by_name = {i['name']: i for i in catalog}
    out, seen = [], set()
    for r in requested:
        name = str((r or {}).get('name') or '')
        if name in seen:
            continue
        it = by_name.get(name)
        if not it:
            raise ComfyError(f'unknown LoRA: {name[:120]}')
        try:
            strength = min(2.0, max(0.0, float(r.get('strength', 1.0))))
        except (TypeError, ValueError):
            strength = 1.0
        words = [clean_trigger_group(w) for w in (r.get('words') or []) if isinstance(w, str) and clean_trigger_group(w)][:20]
        out.append({'name': name, 'title': it['title'], 'strength': round(strength, 2), 'words': words})
        seen.add(name)
    return out


def fetch_lora_preview(path, cfg=None):
    """(bytes, content_type) of a LoRA Manager preview image, or None."""
    cfg = cfg or config()
    if not re.search(r'\.(jpe?g|png|webp|gif)$', path, re.I):
        return None
    try:
        with _client(10.0) as c:
            r = c.get(cfg['comfy_url'] + '/api/lm/previews', params={'path': path})
    except httpx.HTTPError:
        return None
    ctype = r.headers.get('content-type', '')
    if r.status_code != 200 or not ctype.startswith('image/'):
        return None
    return r.content, ctype
