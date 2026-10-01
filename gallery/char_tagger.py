"""Second-opinion CHARACTER tagger: PixAI tagger v0.9 (EVA02, 448px, 13,461 tags
of which 3,720 characters, trained on Danbooru through Jan 2025) via the DeepGHS
ONNX export. WD14 stays the main tagger; this one only ever adds character tags.

~318M params / 0.62 TFLOPs per image: about 0.3 s on a Tesla P4, so it is run on
demand (per-post button, settings bulk run) rather than on every scan. FP32 only
— the P4 has no fast FP16/int8, so the 1.27 GB FP32 export is also the right one.
"""
import csv
import glob
import os
import threading

from django.conf import settings

REPO = 'deepghs/pixai-tagger-v0.9-onnx'
MODEL_ID = 'pixai'                 # value stored in Post.char_model
SIZE = 448
EXPECTED_BYTES = 1_270_000_000     # model.onnx, for the download progress bar
CHAR_THRESHOLD = getattr(settings, 'AI_CHAR_MODEL_THRESHOLD', 0.85)   # the repo's thresholds.csv value
MAX_CHARACTERS = 12

_tags_cache = None
_tags_lock = threading.Lock()


# ── files / download ────────────────────────────────────────────
def _hub_dir():
    from huggingface_hub import constants
    return os.path.join(constants.HF_HUB_CACHE, 'models--' + REPO.replace('/', '--'))


def _cached(filename):
    """Local path of a cached repo file, or None. No network access."""
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(REPO, filename)
    return p if isinstance(p, str) and os.path.exists(p) else None


def model_ready():
    return bool(_cached('model.onnx') and _cached('selected_tags.csv'))


def model_bytes():
    p = _cached('model.onnx')
    try:
        return os.path.getsize(p) if p else 0
    except OSError:
        return 0


def download_progress_bytes():
    """Bytes of an in-flight download (the .incomplete blob), 0 if none."""
    total = 0
    for f in glob.glob(os.path.join(_hub_dir(), 'blobs', '*.incomplete')):
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return total


def download_model(on_progress=None):
    """Download the model files into the Hugging Face cache (resumable). While
    it runs, on_progress(done_bytes, total_bytes) is called about every 2 s."""
    from huggingface_hub import constants, snapshot_download
    constants.HF_HUB_DOWNLOAD_TIMEOUT = 30    # a stalled connection must error out, not hang for ever
    constants.HF_HUB_DISABLE_XET = True       # the Xet transfer path stalls/restarts on this network; plain HTTP is reliable
    err = []

    def _dl():
        # snapshot_download resumes from the .incomplete blob, so just retry
        for attempt in range(1, 6):
            try:
                snapshot_download(REPO, allow_patterns=['model.onnx', 'selected_tags.csv', 'preprocess.json', 'thresholds.csv'])
                err.clear()
                return
            except BaseException as e:    # noqa: BLE001
                err[:] = [e]
                import time as _t
                _t.sleep(3 * attempt)

    t = threading.Thread(target=_dl, name='booru-pixai-download', daemon=True)
    t.start()
    while t.is_alive():
        t.join(2)
        if on_progress:
            on_progress(download_progress_bytes(), EXPECTED_BYTES)
    if err:
        raise err[0]
    if not model_ready():
        raise RuntimeError('download finished but the model files are not in the cache')


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


def run_character_tagger(file_path, thumb_path=''):
    """[(name, probability), ...] for characters >= CHAR_THRESHOLD, most
    confident first (at most MAX_CHARACTERS)."""
    import numpy as np
    from . import ai_runtime
    from .utils import load_image_for_tagging
    if not model_ready():
        raise RuntimeError('character model not downloaded yet')
    img = load_image_for_tagging(file_path, thumb_path)
    model = ai_runtime.get_model(MODEL_ID, _cached('model.onnx'))
    names, is_char = _tags()
    # The export has 3 outputs: embedding (1024), logits and prediction (sigmoid scores, 13461).
    out = np.asarray(model.run(['prediction'], {model.get_inputs()[0].name: _preprocess(img)})[0][0], dtype=np.float32)
    if out.min() < 0.0 or out.max() > 1.0:                   # logits -> probabilities
        out = 1.0 / (1.0 + np.exp(-out))
    idx = np.nonzero((out >= CHAR_THRESHOLD) & is_char)[0]
    idx = idx[np.argsort(-out[idx])][:MAX_CHARACTERS]
    return [(names[i], float(out[i])) for i in idx]
