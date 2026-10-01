"""Registry of the AI models the gallery can use: where each one lives, whether
it is on disk, and download / delete.

- wd14   the default tagger (SmilingWolf/wd-vit-tagger-v3, 362 MB)
- pixai  PixAI tagger v0.9 ONNX (DeepGHS export, 1.27 GB) — characters add-on / alternative main tagger
- custom "my model": a clone of wd14 that can learn extra tags (gallery/custom_model.py)

Hugging Face models live in the normal HF cache; "my model" lives in
settings.AI_MODELS_DIR/custom (default: <project>/ai_models, untracked).
"""
import os
import shutil
import threading
import time

from django.conf import settings

MODELS = {
    'wd14': {
        'name': 'Default tagger (WD14 ViT v3)', 'repo': 'SmilingWolf/wd-vit-tagger-v3',
        'files': ['model.onnx', 'selected_tags.csv'], 'bytes': 378_536_310,
    },
    'pixai': {
        'name': 'PixAI tagger v0.9', 'repo': 'deepghs/pixai-tagger-v0.9-onnx',
        'files': ['model.onnx', 'selected_tags.csv', 'preprocess.json', 'thresholds.csv'], 'bytes': 1_271_365_854,
    },
    'custom': {'name': 'My model', 'repo': None, 'files': ['model.onnx'], 'bytes': 0},
}
KEYS = tuple(MODELS)
HF_KEYS = ('wd14', 'pixai')


def models_dir():
    d = str(getattr(settings, 'AI_MODELS_DIR', os.path.join(settings.BASE_DIR, 'ai_models')))
    os.makedirs(d, exist_ok=True)
    return d


def custom_dir():
    return os.path.join(models_dir(), 'custom')


def _hub_dir(key):
    from huggingface_hub import constants
    return os.path.join(constants.HF_HUB_CACHE, 'models--' + MODELS[key]['repo'].replace('/', '--'))


def cached_file(key, filename):
    """Local path of a model file, or None. No network access."""
    if key == 'custom':
        p = os.path.join(custom_dir(), filename)
        return p if os.path.exists(p) else None
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(MODELS[key]['repo'], filename)
    return p if isinstance(p, str) and os.path.exists(p) else None


def is_ready(key):
    if key == 'custom':
        return bool(cached_file('custom', 'model.onnx') and cached_file('custom', 'meta.json'))
    return all(cached_file(key, f) for f in MODELS[key]['files'][:2])


def size_bytes(key):
    p = cached_file(key, 'model.onnx')
    try:
        return os.path.getsize(p) if p else 0
    except OSError:
        return 0


def status(key):
    ready = is_ready(key)
    m = MODELS[key]
    return {'key': key, 'name': m['name'], 'ready': ready,
            'mb': round(size_bytes(key) / 1048576) if ready else 0,
            'expected_mb': round(m['bytes'] / 1048576) if m['bytes'] else 0}


# ── download ────────────────────────────────────────────────────
def download_progress_bytes(key):
    """Bytes of an in-flight download (the .incomplete blobs), 0 if none."""
    import glob
    total = 0
    for f in glob.glob(os.path.join(_hub_dir(key), 'blobs', '*.incomplete')):
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return total


def download(key, on_progress=None):
    """Download a Hugging Face model into the HF cache (resumable). While it
    runs, on_progress(done_bytes, total_bytes) is called about every 2 s."""
    if key not in HF_KEYS:
        raise ValueError(f'{key} is not a downloadable model')
    from huggingface_hub import constants, snapshot_download
    constants.HF_HUB_DOWNLOAD_TIMEOUT = 30    # a stalled connection must error out, not hang for ever
    constants.HF_HUB_DISABLE_XET = True       # the Xet transfer path stalls/restarts on this network; plain HTTP is reliable
    m = MODELS[key]
    err = []

    def _dl():
        # snapshot_download resumes from the .incomplete blob, so just retry
        for attempt in range(1, 6):
            try:
                snapshot_download(m['repo'], allow_patterns=m['files'])
                err.clear()
                return
            except BaseException as e:    # noqa: BLE001
                err[:] = [e]
                time.sleep(3 * attempt)

    t = threading.Thread(target=_dl, name=f'booru-download-{key}', daemon=True)
    t.start()
    while t.is_alive():
        t.join(2)
        if on_progress:
            on_progress(download_progress_bytes(key), m['bytes'])
    if err:
        raise err[0]
    if not is_ready(key):
        raise RuntimeError('download finished but the model files are not in the cache')


# ── delete ──────────────────────────────────────────────────────
def delete(key):
    """Remove a model from disk. Returns the bytes freed."""
    path = custom_dir() if key == 'custom' else _hub_dir(key)
    freed = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                freed += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)
    return freed
