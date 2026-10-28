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
    # "⚡ fast": a distillation LoRA (DMD2) on top of the same checkpoint — few steps at cfg 1
    # (no negative pass), ~5x quicker on the P4. Used when a job asks for it.
    'fast_lora': 'speed/dmd2_sdxl_4step_lora_fp16.safetensors',
    'fast_strength': 1.0,
    'fast_steps': 8,
    'fast_sampler': 'lcm',
    'fast_scheduler': 'sgm_uniform',
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
            elif k in ('quality', 'negative', 'lora_bases', 'fast_lora') and v == '':
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
    num('fast_strength', float, 0.0, 2.0)
    num('fast_steps', int, 1, 30)
    out['enabled'] = bool(out['enabled'])
    out['free_after'] = bool(out['free_after'])
    for k in ('comfy_url', 'ollama_url'):
        out[k] = str(out[k]).strip().rstrip('/') or DEFAULTS[k]
    for k in ('ollama_model', 'checkpoint', 'sampler', 'scheduler', 'fast_sampler', 'fast_scheduler'):
        out[k] = str(out[k]).strip() or DEFAULTS[k]
    for k in ('quality', 'negative', 'lora_bases', 'fast_lora'):
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


# ── ✨ create: a description becomes a full prompt (+ LoRA trigger groups) ─────────
# Kept byte-identical between calls and placed first: Ollama then reuses the cached prefix.
PLAN_SYSTEM_PROMPT = """You write prompts for an anime image model that understands Danbooru tags.
The user describes a picture (any language). Answer with:
- "characters": named characters as Danbooru names, lowercase, spaces instead of underscores, WITHOUT the series in brackets (e.g. "hayasaka ai", "nakano miku"). Empty when nobody is named.
- "tags": Danbooru tags for everything else in the picture, most important first: count (1girl, 2girls, 1boy, solo), hair, eyes, outfit and its parts, expression, pose, action, objects, background, lighting, framing (upper body, full body, cowboy shot, close-up). 15 to 35 tags. Lowercase, spaces instead of underscores. Never quality tags.
- "negative": things the user explicitly does NOT want (e.g. "no glasses" -> "glasses"). Usually empty.
- "orientation": "portrait", "landscape" or "square" (people standing or a single character: portrait; scenery or several people side by side: landscape).
- "importance": for the tags that matter most, {"tag": <a tag from "tags">, "level": 2-4}: 4 = the core of the request (what the user asked for), 3 = clearly asked, 2 = useful detail. Tags you only added to fill the picture are left out (level 1).
- "note": one short sentence describing the picture.
When a source picture is given, its tags describe it now; "tags" must describe the FINAL picture (keep what the user does not change).
When the request turns a person into a named character (or another person), DROP that person's old hair, eye and clothing tags: the character brings its own look; keep the pose, background and the rest.
Examples:
Request: hayasaka ai in a maid outfit smiling in a cafe
{"characters": ["hayasaka ai"], "tags": ["1girl", "solo", "maid", "maid headdress", "maid apron", "black dress", "white apron", "frills", "smile", "looking at viewer", "standing", "holding tray", "cafe", "indoors", "table", "window", "sunlight", "cowboy shot"], "negative": [], "orientation": "portrait", "importance": [{"tag": "maid", "level": 4}, {"tag": "cafe", "level": 3}, {"tag": "smile", "level": 3}, {"tag": "maid headdress", "level": 2}], "note": "Hayasaka Ai as a smiling maid in a cafe"}
Request: two girls on a beach at sunset, no hats
{"characters": [], "tags": ["2girls", "beach", "sunset", "ocean", "sand", "orange sky", "cloud", "swimsuit", "bikini", "barefoot", "smile", "holding hands", "wind", "long hair", "from side", "wide shot"], "negative": ["hat"], "orientation": "landscape", "importance": [{"tag": "2girls", "level": 4}, {"tag": "beach", "level": 4}, {"tag": "sunset", "level": 4}, {"tag": "holding hands", "level": 3}], "note": "two girls holding hands on a beach at sunset"}
Answer with JSON only."""

PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'characters': {'type': 'array', 'items': {'type': 'string'}},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'negative': {'type': 'array', 'items': {'type': 'string'}},
        'orientation': {'type': 'string', 'enum': ['portrait', 'landscape', 'square']},
        'importance': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'tag': {'type': 'string'}, 'level': {'type': 'integer'}}, 'required': ['tag', 'level']}},
        'note': {'type': 'string'},
    },
    'required': ['characters', 'tags', 'negative', 'orientation', 'importance', 'note'],
}

GROUPS_SYSTEM_PROMPT = """You choose trigger words for LoRAs (add-ons of an anime image model).
Each LoRA has numbered trigger-word groups. For a character LoRA, one group is usually the character itself (name, hair, eyes) and the others are outfits or variants.
Pick, for every LoRA, the group numbers that fit the user's description:
- a character LoRA: the group with the character's look, plus the ONE outfit/variant group the description asks for (none when no outfit is described or nothing fits);
- any other LoRA: the groups whose words match the description (a style or concept LoRA with a single group: that group).
Answer {"picks": [{"lora": <LoRA number>, "groups": [<group numbers>]}]} with JSON only."""

GROUPS_SCHEMA = {
    'type': 'object',
    'properties': {'picks': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {'lora': {'type': 'integer'}, 'groups': {'type': 'array', 'items': {'type': 'integer'}}},
        'required': ['lora', 'groups']}}},
    'required': ['picks'],
}


def _ollama_json(cfg, system, user, schema, timeout=180.0):
    """One schema-constrained Ollama chat call -> parsed dict (raises on any problem)."""
    body = {
        'model': cfg['ollama_model'], 'stream': False, 'think': False, 'keep_alive': '2m',
        'format': schema, 'options': {'temperature': 0.2, 'num_ctx': 4096},
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
    }
    with _client(timeout) as c:
        r = c.post(cfg['ollama_url'] + '/api/chat', json=body)
    if r.status_code != 200:
        raise ComfyError(f'Ollama HTTP {r.status_code}: {r.text[:200]}')
    text = ((r.json().get('message') or {}).get('content') or '').strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            raise ValueError('no JSON in the model answer')
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError('unexpected JSON')
    return data


def parse_plan(data):
    """The planner's JSON -> {'characters','tags','negative','orientation','note'} (cleaned);
    raises ValueError when it holds no tags and no characters."""
    def as_list(v):
        if isinstance(v, str):
            v = v.split(',')
        return clean_tags(v if isinstance(v, list) else [])
    chars = [re.sub(r'\s*\(.*?\)\s*', ' ', c).strip() for c in as_list(data.get('characters'))]
    chars = [c for c in dict.fromkeys(chars) if c][:4]
    tags = [t for t in as_list(data.get('tags')) if t not in chars]
    if not tags and not chars:
        raise ValueError('the model returned no tags')
    orient = data.get('orientation') if data.get('orientation') in SIZE_RATIOS else 'portrait'
    levels = {}
    for it in data.get('importance') or []:
        try:
            t, lv = clean_tags([it.get('tag')]), int(it.get('level'))
        except (AttributeError, TypeError, ValueError):
            continue
        if t and t[0] in tags:
            levels[t[0]] = min(4, max(1, lv))
    return {'characters': chars, 'tags': tags, 'negative': as_list(data.get('negative'))[:MAX_NEGATIVE],
            'orientation': orient, 'levels': levels, 'note': str(data.get('note') or '')[:300]}


FORCE_WEIGHTS = {1: 1.2, 2: 1.4}
_FORCED_RE = re.compile(r'(\({1,2})([^()]+?)(\){1,2})')


def forced_terms(text):
    """Bracketed parts of a description must end up in the prompt: "girl (blue hair)" ->
    [('blue hair', 1.2)], "((red eyes, glasses))" -> [('red eyes', 1.4), ('glasses', 1.4)].
    A part that already carries an explicit weight (`(x:1.5)`) keeps it."""
    out, seen = [], set()
    for m in _FORCED_RE.finditer(str(text or '')):
        depth = min(len(m.group(1)), len(m.group(3)), 2)
        inner, weight = m.group(2), FORCE_WEIGHTS[depth]
        wm = re.match(r'^(.*):\s*([0-9]*\.?[0-9]+)\s*$', inner)
        if wm:
            inner, weight = wm.group(1), min(2.0, max(0.1, float(wm.group(2))))
        for part in inner.split(','):
            t = clean_tags([part])
            if t and t[0] not in seen:
                seen.add(t[0])
                out.append((t[0], weight))
    return out


def strip_forced(text):
    """The description with the bracket marks removed (words kept), for the LLM / fallback."""
    return re.sub(r'\s+', ' ', re.sub(r':\s*[0-9]*\.?[0-9]+\s*\)', ')', str(text or '')).replace('(', ' ').replace(')', ' ')).strip()


LEVEL_WEIGHTS = {2: 1.1, 3: 1.2, 4: 1.3}      # importance 1-4 -> prompt weight (1 = plain)


def weighted_tags(tags, levels):
    """Tags as prompt text, most important first: level 2-4 become (tag:1.1-1.3)."""
    order = sorted(tags, key=lambda t: -levels.get(t, 1))
    return ', '.join(f'({escape_tag(t)}:{LEVEL_WEIGHTS[levels[t]]:g})' if levels.get(t, 1) > 1 else escape_tag(t) for t in order)


_COLOR_SLOT = re.compile(r'^(?:(?:light|dark|very)\s+)?(aqua|black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|'
                         r'silver|white|yellow|multicolored|two-tone|streaked|gradient)\s+(hair|eyes)$')   # colours only


def compose_positive(quality, triggers, pos_text):
    """The final positive: quality, then the user's WEIGHTED tags ((x:1.2)... — what the
    description asked for), then the LoRA trigger words, then the rest of the prompt.
    Trigger words that contradict a described hair / eye colour are dropped ("pink hair" from
    a character's trigger group when the user asked for blue hair). Trigger words also
    present in the prompt are not repeated."""
    items = [x.strip() for x in re.split(r',(?![^()]*\))', str(pos_text or '')) if x.strip()]
    first = [x for x in items if re.match(r'^\(.+:\s*[0-9.]+\)$', x)]
    rest = [x for x in items if x not in first]
    plain = lambda x: _norm(re.sub(r'^\((.+?):\s*[0-9.]+\)$', r'\1', x).replace('\\(', '(').replace('\\)', ')'))
    slots = {}
    for x in items:
        m = _COLOR_SLOT.match(plain(x))
        if m:
            slots.setdefault(m.group(2), set()).add(m.group(1))
    have = {plain(x) for x in items}
    trig = []
    for w in str(triggers or '').split(','):
        w = w.strip()
        if not w or _norm(w) in have:
            continue
        m = _COLOR_SLOT.match(_norm(w))
        if m and m.group(2) in slots and m.group(1) not in slots[m.group(2)]:
            continue                     # the description chose another colour
        trig.append(w)
    return join_prompt(quality, ', '.join(first), ', '.join(trig), ', '.join(rest))


def forced_text(forced):
    return ', '.join(f'({escape_tag(t)}:{w:g})' for t, w in forced)


def plan_prompt(description, cfg=None, current_tags=None):
    """Description (+ the source picture's tags) -> plan dict + 'source' ('llm'/'fallback').
    Never raises: without a usable answer the description's comma parts become the tags."""
    cfg = cfg or config()
    forced = [t for t, _w in forced_terms(description)]
    description = strip_forced(description)
    user = 'Request: ' + description
    if forced:
        user += '\nMust be in "tags": ' + ', '.join(forced)
    if current_tags:
        user = 'Source picture tags: ' + ', '.join(clean_tags(current_tags)[:60]) + '\n' + user
    try:
        plan = parse_plan(_ollama_json(cfg, PLAN_SYSTEM_PROMPT, user, PLAN_SCHEMA))
        plan['source'] = 'llm'
    except Exception as e:
        plan = {'characters': [], 'tags': clean_tags(description.split(',') + forced), 'negative': [],
                'orientation': 'portrait', 'source': 'fallback',
                'note': f'Ollama gave no usable answer ({str(e)[:120]}) — your text was used as is'}
    return plan


def groups_question(description, items):
    """The user message for GROUPS_SYSTEM_PROMPT: numbered LoRAs with numbered groups."""
    lines = ['Description: ' + description.strip(), '']
    for i, it in enumerate(items, 1):
        lines.append(f'LoRA {i}: {it["title"]}' + (' (character)' if it['kind'] == 'character' else ''))
        for g, words in enumerate(it['groups'], 1):
            lines.append(f'  group {g}: {words[:300]}')
    return '\n'.join(lines)


def parse_group_picks(data, items):
    """{'picks': [{'lora': n, 'groups': [..]}]} (1-based) -> {lora name: [0-based group idx]};
    out-of-range numbers are dropped, a LoRA with nothing valid is left out (caller falls back)."""
    out = {}
    for p in (data or {}).get('picks') or []:
        try:
            li = int(p.get('lora')) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if not 0 <= li < len(items):
            continue
        n = len(items[li]['groups'])
        idx = []
        for g in p.get('groups') or []:
            try:
                g = int(g) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= g < n and g not in idx:
                idx.append(g)
        if idx:
            out[items[li]['name']] = sorted(idx)
    return out


def pick_groups(description, items, cfg=None, terms=()):
    """Which trigger groups fit the description, per LoRA. First `match_groups` (the
    planned tags `terms` / the description against each group's distinctive words); only
    LoRAs where that finds nothing go to the LLM, in ONE call; then `default_groups`.
    Returns ({name: [idx]}, {name: 'match'|'llm'|'default'})."""
    cfg = cfg or config()
    out, how, ask = {}, {}, []
    for it in items:
        idx, score = match_groups(it, terms, description)
        if score:
            out[it['name']], how[it['name']] = idx, 'match'
        elif len(it['groups']) > 1:
            ask.append(it)
    picks = {}
    if ask:
        try:
            picks = parse_group_picks(_ollama_json(cfg, GROUPS_SYSTEM_PROMPT, groups_question(description, ask), GROUPS_SCHEMA), ask)
        except Exception as e:
            print(f'create: group pick failed: {e}')
    for it in items:
        if it['name'] in out:
            continue
        got = picks.get(it['name'])
        how[it['name']] = 'llm' if got else 'default'
        got = got or default_groups(it)
        if it['kind'] == 'character' and it['groups'] and 0 not in got:
            got = [0] + got                    # the identity group (as in default_groups) is never dropped
        out[it['name']] = got
    return out, how


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
    return _sample_chain(wf, ['4', 0], positive, negative, seed, cfg, den, loras, temp_node)


def build_txt2img(positive, negative, seed, cfg=None, width=832, height=1216, batch=1, loras=(), temp_node=None):
    """ComfyUI API workflow for a NEW picture (the ✨ create page): empty latent of
    width×height (multiples of 8) -> the same LoRA chain / prompts / sampler as img2img at
    denoise 1.0. `batch` pictures share one sampling run."""
    cfg = cfg or config()
    wf = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': cfg['checkpoint']}},
        '4': {'class_type': 'EmptyLatentImage',
              'inputs': {'width': int(width) // 8 * 8, 'height': int(height) // 8 * 8, 'batch_size': int(batch)}},
    }
    return _sample_chain(wf, ['4', 0], positive, negative, seed, cfg, 1.0, loras, temp_node)


def build_inpaint(image_name, mask_name, positive, negative, seed, cfg=None, denoise=0.75, loras=(), temp_node=None):
    """Redraw only the painted area of a (cropped) picture: VAE-encode it, SetLatentNoiseMask
    with the soft mask + DifferentialDiffusion (soft edges blend), the same LoRAs / prompts /
    sampler. The caller pastes the result back into the full picture through the mask."""
    cfg = cfg or config()
    wf = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': cfg['checkpoint']}},
        '2': {'class_type': 'LoadImage', 'inputs': {'image': image_name}},
        '3': {'class_type': 'LoadImageMask', 'inputs': {'image': mask_name, 'channel': 'red'}},
        '4': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['2', 0], 'vae': ['1', 2]}},
        '11': {'class_type': 'SetLatentNoiseMask', 'inputs': {'samples': ['4', 0], 'mask': ['3', 0]}},
    }
    den = min(1.0, max(DENOISE_MIN, float(denoise)))
    return _sample_chain(wf, ['11', 0], positive, negative, seed, cfg, den, loras, temp_node, diff=True)


def _sample_chain(wf, latent, positive, negative, seed, cfg, den, loras, temp_node, diff=False):
    """Shared tail of the workflows: LoRAs -> [DifferentialDiffusion] -> prompts -> [GPU temp
    wait] -> KSampler -> VAE decode -> PreviewImage (node ids 5-10, 12, LoRAs 20+)."""
    model, clip = ['1', 0], ['1', 1]
    for i, (name, strength) in enumerate(loras):
        nid = str(20 + i)
        wf[nid] = {'class_type': 'LoraLoader',
                   'inputs': {'model': model, 'clip': clip, 'lora_name': name,
                              'strength_model': float(strength), 'strength_clip': float(strength)}}
        model, clip = [nid, 0], [nid, 1]
    if diff:
        wf['12'] = {'class_type': 'DifferentialDiffusion', 'inputs': {'model': model}}
        model = ['12', 0]
    wf['5'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': positive, 'clip': clip}}
    wf['6'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': negative, 'clip': clip}}
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


class ProgressWatcher:
    """Reads ComfyUI's websocket (stdlib only: no websocket package in the venv) and calls
    `on_step(value, max)` for the sampler steps of OUR prompt. ComfyUI sends progress only
    to the client id that queued the prompt, so `queue_prompt` must use `self.client_id`.
    Best effort: any error just ends the thread (the job itself never depends on it)."""

    def __init__(self, cfg, on_step):
        import threading
        import uuid
        self.cfg, self.on_step = cfg, on_step
        self.client_id = 'booru-' + uuid.uuid4().hex[:12]
        self.prompt_id = None
        self._stop = threading.Event()
        self._sock = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout=3.0):
        self._thread.start()
        self._ready.wait(timeout)              # connected before the prompt is queued, or no progress
        return self

    def stop(self):
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except OSError:
            pass

    def _run(self):
        import base64
        import socket
        import struct
        from urllib.parse import urlparse
        try:
            u = urlparse(self.cfg['comfy_url'])
            s = socket.create_connection((u.hostname, u.port or 80), timeout=5)
            self._sock = s
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall((f'GET /ws?clientId={self.client_id} HTTP/1.1\r\nHost: {u.hostname}:{u.port or 80}\r\n'
                       f'Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n'
                       f'Sec-WebSocket-Version: 13\r\n\r\n').encode())
            buf = b''
            while b'\r\n\r\n' not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    return
                buf += chunk
            head, buf = buf.split(b'\r\n\r\n', 1)
            if b' 101 ' not in head.split(b'\r\n', 1)[0]:
                return
            self._ready.set()
            s.settimeout(1.0)

            def read(n):
                nonlocal buf
                while len(buf) < n:
                    if self._stop.is_set():
                        raise EOFError
                    try:
                        chunk = s.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise EOFError
                    buf += chunk
                out, buf = buf[:n], buf[n:]
                return out
            while not self._stop.is_set():
                b1, b2 = read(2)
                op, n = b1 & 0x0F, b2 & 0x7F
                if n == 126:
                    n = struct.unpack('>H', read(2))[0]
                elif n == 127:
                    n = struct.unpack('>Q', read(8))[0]
                mask = read(4) if b2 & 0x80 else None
                data = read(n)
                if mask:
                    data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
                if op == 8:
                    return
                if op == 9:                    # ping -> masked pong
                    m = os.urandom(4)
                    s.sendall(bytes([0x8A, 0x80 | len(data)]) + m + bytes(c ^ m[i % 4] for i, c in enumerate(data)))
                    continue
                if op != 1:                    # binary = latent previews, ignored
                    continue
                try:
                    msg = json.loads(data)
                except ValueError:
                    continue
                d = msg.get('data') or {}
                if msg.get('type') == 'progress' and (self.prompt_id is None or d.get('prompt_id') in (None, self.prompt_id)):
                    try:
                        self.on_step(int(d.get('value', 0)), int(d.get('max', 0)))
                    except Exception:
                        pass
        except (OSError, EOFError, ValueError, struct.error):
            pass
        finally:
            self._ready.set()
            try:
                if self._sock:
                    self._sock.close()
            except OSError:
                pass


def queue_prompt(wf, cfg=None, client_id='booru'):
    cfg = cfg or config()
    try:
        with _client(30.0) as c:
            r = c.post(cfg['comfy_url'] + '/prompt', json={'prompt': wf, 'client_id': client_id})
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


def generate(src_path, positive, negative, seed, denoise, cfg=None, check=None, progress=None, loras=(), step=None):
    """Upload `src_path`, run the img2img workflow (with `loras` = [(name, strength)]),
    return (png_bytes, workflow_used). `step(value, max)` gets live sampler progress."""
    cfg = cfg or config()
    stat = status_light(cfg)
    if progress:
        progress('uploading to ComfyUI…')
    unload_ollama(cfg)
    name = upload_image(src_path, cfg)
    wf = build_img2img(name, positive, negative, seed, cfg, denoise, loras=loras, temp_node=stat['temp_node'])
    return _run_workflow(wf, cfg, check, progress, step), wf


def generate_txt2img(positive, negative, seed, width, height, cfg=None, check=None, progress=None, loras=(), step=None):
    """Run the txt2img workflow (a NEW picture); returns (png_bytes, workflow_used)."""
    cfg = cfg or config()
    stat = status_light(cfg)
    unload_ollama(cfg)
    wf = build_txt2img(positive, negative, seed, cfg, width, height, loras=loras, temp_node=stat['temp_node'])
    return _run_workflow(wf, cfg, check, progress, step), wf


def generate_inpaint(img_path, mask_path, positive, negative, seed, denoise, cfg=None, check=None, progress=None,
                     loras=(), step=None):
    """Upload a crop + its mask, redraw the masked part; returns (png_bytes of the crop, workflow)."""
    cfg = cfg or config()
    stat = status_light(cfg)
    if progress:
        progress('uploading to ComfyUI…')
    unload_ollama(cfg)
    name, mname = upload_image(img_path, cfg), upload_image(mask_path, cfg)
    wf = build_inpaint(name, mname, positive, negative, seed, cfg, denoise, loras=loras, temp_node=stat['temp_node'])
    return _run_workflow(wf, cfg, check, progress, step), wf


def inpaint_crop(src, mask, megapixels):
    """Where to redraw a painted area: (box, (w, h)) — the mask's bounding box plus context
    (half its size, at least 64 px) clamped to the picture, and the size to render it at
    (about `megapixels`, multiples of 8). src/mask: PIL images of the same size."""
    x0, y0, x1, y1 = mask.getbbox()
    W, H = src.size
    m = max(64, max(x1 - x0, y1 - y0) // 2)
    box = (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))
    bw, bh = box[2] - box[0], box[3] - box[1]
    k = (megapixels * 1_000_000 / (bw * bh)) ** 0.5
    k = min(k, 4.0)                                   # a tiny painted spot: upscale at most 4×
    return box, (max(64, int(bw * k) // 8 * 8), max(64, int(bh * k) // 8 * 8))


def _run_workflow(wf, cfg, check, progress, step=None):
    problems = workflow_problems(wf)
    if problems:
        raise ComfyError('internal workflow error: ' + problems[0])
    watcher = ProgressWatcher(cfg, step).start() if step else None
    try:
        pid = queue_prompt(wf, cfg, watcher.client_id if watcher else 'booru')
        if watcher:
            watcher.prompt_id = pid
        if progress:
            progress('generating…')
        desc = wait_for_result(pid, cfg, check=check, progress=progress)
    finally:
        if watcher:
            watcher.stop()
    return fetch_output(desc, cfg)


def fast_setup(cfg, loras):
    """"⚡ fast" job: (cfg with the fast steps/cfg 1/sampler, LoRA list + the speed LoRA).
    Raises ComfyError when no speed LoRA is configured."""
    if not cfg.get('fast_lora'):
        raise ComfyError('no speed LoRA set (settings → AI edit → fast LoRA)')
    fast = dict(cfg, steps=cfg['fast_steps'], cfg=1.0, sampler=cfg['fast_sampler'], scheduler=cfg['fast_scheduler'])
    return fast, list(loras) + [(cfg['fast_lora'], cfg['fast_strength'])]


SIZE_RATIOS = {'portrait': 2 / 3, 'square': 1.0, 'landscape': 3 / 2}
SIZE_MP = {'draft': 0.4, 'normal': None, 'large': 1.0}      # None = cfg['megapixels']


def size_for(orientation, quality, cfg):
    """(width, height), multiples of 64, for an orientation and a size preset."""
    ratio = SIZE_RATIOS.get(orientation, SIZE_RATIOS['portrait'])
    mp = SIZE_MP.get(quality) or float(cfg['megapixels'])
    w = (mp * 1_000_000 * ratio) ** 0.5
    h = w / ratio
    return max(512, int(round(w / 64)) * 64), max(512, int(round(h / 64)) * 64)


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


def _key_words(item):
    """Key trigger words = the first 3 of each group (the rest are often generic: 1girl, solo, school uniform...)."""
    return {w.strip().lower().replace('_', ' ') for g in item['groups'] for w in g.split(',')[:3]}


_common_cache = {'id': None, 'words': frozenset()}


def _common_key_words(catalog):
    """Key words found in more than 3% of the LoRAs (1girl, solo, looking at viewer...): they
    say nothing about a LoRA. A planned prompt is full of them, and matching them suggested
    hundreds of unrelated LoRAs."""
    if _common_cache['id'] != id(catalog):
        from collections import Counter
        df = Counter(w for it in catalog for w in _key_words(it))
        limit = max(5, len(catalog) * 0.03)
        _common_cache.update(id=id(catalog), words=frozenset(w for w, n in df.items() if n > limit))
    return _common_cache['words']


def _group_words(g):
    return {_norm(w) for w in g.split(',') if w.strip()}


def match_groups(item, terms, text=''):
    """Deterministic trigger-group choice: score each group by its DISTINCTIVE words (left
    out: words in more than half of the LoRA's groups, e.g. blonde hair / blue eyes of a
    character) found among `terms` (planned tags) or inside `text` (the description).
    Returns (indexes, best score): a character's identity group 0 plus its best other
    group; any other LoRA: every group with a hit. Score 0 = nothing matched."""
    groups = [_group_words(g) for g in item['groups']]
    n = len(groups)
    if not n:
        return [], 0
    from collections import Counter
    seen = Counter(w for g in groups for w in g)
    shared = {w for w, c in seen.items() if n >= 2 and c > n / 2}
    termset = {_norm(t) for t in terms or []}
    text = ' ' + re.sub(r'[^a-z0-9]+', ' ', str(text).lower()) + ' '
    scores = []
    for g in groups:
        s = 0
        for w in g - shared:
            if w in termset or (len(w) > 3 and ' ' + w + ' ' in text):
                s += 1
        scores.append(s)
    if item['kind'] == 'character':              # score = the best outfit/variant group (group 0 is the identity)
        if n == 1:
            return [0], 0
        top = max(range(1, n), key=lambda i: (scores[i], -i))
        return ([0, top] if scores[top] else [0]), scores[top]
    best = max(scores)
    return [i for i, s in enumerate(scores) if s], best


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
    common = _common_key_words(catalog)
    for t in terms or []:
        t = re.sub(r'\s+', ' ', str(t).lower().replace('_', ' ')).strip()
        if t and t not in GENERIC_TOKENS and t not in common and t not in phrases:
            phrases.append(t)
    for it in catalog:
        if it['kind'] == 'character':
            continue
        key_words = _key_words(it)
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
