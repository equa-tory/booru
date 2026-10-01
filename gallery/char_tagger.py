"""PixAI tagger v0.9 (EVA02, 448px, 13,461 tags of which 3,720 characters, trained
on Danbooru through Jan 2025) via the DeepGHS ONNX export.

Two uses:
- run_character_tagger(): a second-opinion CHARACTERS-ONLY pass next to WD14
  (per-post button, settings bulk re-tag, optional auto-run). Only ever adds tags.
- run_pixai_general(): general + character tags, for when PixAI is selected as
  the main AI tagger. It has no rating tags (explicit/questionable...) and a
  different vocabulary than WD14 — the settings panel warns about both.

~318M params / 0.62 TFLOPs per image: about 0.3 s on a Tesla P4. FP32 only — the
P4 has no fast FP16/int8, so the 1.27 GB FP32 export is also the right one.
Files/download/delete are handled by gallery/ai_models.py.
"""
import csv
import threading

from django.conf import settings

from . import ai_models

MODEL_ID = 'pixai'                 # value stored in Post.char_model
SIZE = 448
EXPECTED_BYTES = ai_models.MODELS['pixai']['bytes']
CHAR_THRESHOLD = getattr(settings, 'AI_CHAR_MODEL_THRESHOLD', 0.85)   # the repo's thresholds.csv value
GENERAL_THRESHOLD = getattr(settings, 'AI_PIXAI_GENERAL_THRESHOLD', 0.30)
MAX_CHARACTERS = 12
MAX_GENERAL = 40

_tags_cache = None
_tags_lock = threading.Lock()


# ── files / download (thin wrappers over ai_models) ─────────────
def _cached(filename):
    return ai_models.cached_file('pixai', filename)


def model_ready():
    return ai_models.is_ready('pixai')


def model_bytes():
    return ai_models.size_bytes('pixai')


def download_model(on_progress=None):
    ai_models.download('pixai', on_progress)


# ── inference ───────────────────────────────────────────────────
def _tags():
    """(names, is_character mask) from selected_tags.csv. Order matters: it is
    the model's output order (the file is NOT sorted by category)."""
    global _tags_cache
    with _tags_lock:
        if _tags_cache is None:
            import numpy as np
            with open(_cached('selected_tags.csv'), encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            _tags_cache = ([r['name'] for r in rows],
                           np.array([r['category'] == '4' for r in rows], dtype=bool))
        return _tags_cache


def _preprocess(img):
    """preprocess.json: resize to 448x448 (bilinear, aspect NOT preserved),
    RGB -> CHW float in [0,1], normalize mean=std=0.5 (-> [-1, 1])."""
    import numpy as np
    from PIL import Image
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, RGB
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None, ...].copy()          # 1x3xHxW


def _infer(file_path, thumb_path=''):
    """(probabilities over all 13,461 tags, names, is_char mask)."""
    import numpy as np
    from . import ai_runtime
    from .utils import load_image_for_tagging
    if not model_ready():
        raise RuntimeError('PixAI model not downloaded yet — Settings → AI models')
    img = load_image_for_tagging(file_path, thumb_path)
    model = ai_runtime.get_model(MODEL_ID, _cached('model.onnx'))
    names, is_char = _tags()
    # The export has 3 outputs: embedding (1024), logits and prediction (sigmoid scores, 13461).
    out = np.asarray(model.run(['prediction'], {model.get_inputs()[0].name: _preprocess(img)})[0][0], dtype=np.float32)
    if out.min() < 0.0 or out.max() > 1.0:                   # logits -> probabilities
        out = 1.0 / (1.0 + np.exp(-out))
    return out, names, is_char


def run_character_tagger(file_path, thumb_path=''):
    """[(name, probability), ...] for characters >= CHAR_THRESHOLD, most
    confident first (at most MAX_CHARACTERS)."""
    import numpy as np
    out, names, is_char = _infer(file_path, thumb_path)
    idx = np.nonzero((out >= CHAR_THRESHOLD) & is_char)[0]
    idx = idx[np.argsort(-out[idx])][:MAX_CHARACTERS]
    return [(names[i], float(out[i])) for i in idx]


def run_pixai_general(file_path, thumb_path=''):
    """Same dict shape as the WD14 tagger: {'general': [...], 'character': [...]}.
    General tags >= GENERAL_THRESHOLD, most confident first (cap MAX_GENERAL)."""
    import numpy as np
    out, names, is_char = _infer(file_path, thumb_path)
    g = np.nonzero((out >= GENERAL_THRESHOLD) & ~is_char)[0]
    g = g[np.argsort(-out[g])][:MAX_GENERAL]
    c = np.nonzero((out >= CHAR_THRESHOLD) & is_char)[0]
    c = c[np.argsort(-out[c])][:MAX_CHARACTERS]
    return {'general': [names[i] for i in g], 'character': [names[i] for i in c], 'model': MODEL_ID}
