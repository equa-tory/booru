"""ONNX Runtime sessions for the AI taggers.

- Uses the CUDA provider when onnxruntime-gpu + a working CUDA/cuDNN stack are
  present; otherwise (or if CUDA fails at run time) transparently runs on CPU.
- Models are loaded lazily and UNLOADED after AI_IDLE_UNLOAD_SECONDS of disuse,
  so an idle gallery does not keep the GPU's VRAM (it is shared with e.g.
  ComfyUI). Each gunicorn worker is its own process with its own sessions.
- A CUDA failure (cuDNN mismatch, out of memory...) switches this process to
  CPU for CUDA_RETRY_SECONDS, then CUDA is tried again.
"""
import gc
import logging
import threading
import time

from django.conf import settings

log = logging.getLogger('booru.ai')

IDLE_UNLOAD_SECONDS = getattr(settings, 'AI_IDLE_UNLOAD_SECONDS', 120)
CUDA_RETRY_SECONDS = 600

_lock = threading.RLock()
_models = {}              # key -> ManagedModel
_cuda_off_until = 0.0     # monotonic time; CUDA is skipped until then
_cuda_off_reason = ''
_preloaded = False
_reaper_started = False


def _now():
    return time.monotonic()


def cuda_enabled():
    return _now() >= _cuda_off_until


def _disable_cuda(reason):
    global _cuda_off_until, _cuda_off_reason
    _cuda_off_until = _now() + CUDA_RETRY_SECONDS
    _cuda_off_reason = str(reason)[:200]
    log.warning('AI runtime: CUDA disabled for %ss, using CPU (%s)', CUDA_RETRY_SECONDS, _cuda_off_reason)


def _make_session(path, use_cuda):
    """Create an InferenceSession; returns (session, provider_name)."""
    global _preloaded
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = []
    if use_cuda and 'CUDAExecutionProvider' in avail:
        if not _preloaded and hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls()      # pick up the pip-installed nvidia-* CUDA/cuDNN libs
            except Exception as e:      # noqa: BLE001
                log.info('preload_dlls: %s', e)
            _preloaded = True
        providers.append(('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'HEURISTIC',
            'arena_extend_strategy': 'kSameAsRequested',   # don't over-reserve VRAM
        }))
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions()
    so.log_severity_level = 3           # silence the harmless "Memcpy nodes" warning
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    return sess, sess.get_providers()[0].replace('ExecutionProvider', '')


class ManagedModel:
    """Quacks like an onnxruntime InferenceSession (get_inputs / run) but loads
    lazily, falls back to CPU, and can be unloaded when idle."""

    def __init__(self, key, path):
        self.key, self.path = key, path
        self._sess = None
        self.provider = ''
        self.last_used = _now()
        self._input_name = None
        self._run_lock = threading.Lock()    # one inference at a time per model per process

    # -- lifecycle
    def _ensure(self, force_cpu=False):
        if self._sess is None:
            self._sess, self.provider = _make_session(self.path, cuda_enabled() and not force_cpu)
            self._input_name = self._sess.get_inputs()[0].name
            _start_reaper()
        return self._sess

    def unload(self):
        with _lock:
            self._sess = None
        gc.collect()

    # -- onnxruntime-like API
    def get_inputs(self):
        return self._ensure().get_inputs()

    def run(self, output_names, feed):
        with self._run_lock:
            sess = self._ensure()
            self.last_used = _now()
            try:
                return sess.run(output_names, feed)
            except Exception as e:           # noqa: BLE001
                if self.provider != 'CUDA':
                    raise
                # CUDA failed mid-run (cuDNN mismatch, OOM, ...): drop to CPU and retry once
                _disable_cuda(e)
                self._sess = None
                gc.collect()
                sess = self._ensure(force_cpu=True)
                return sess.run(output_names, feed)


def get_model(key, path):
    """The managed model for `key` (created on first use; path changes reload it)."""
    with _lock:
        m = _models.get(key)
        if m is None or m.path != path:
            m = _models[key] = ManagedModel(key, path)
        return m


def unload_idle(ttl=None):
    """Unload models unused for `ttl` seconds. Returns how many were unloaded."""
    ttl = IDLE_UNLOAD_SECONDS if ttl is None else ttl
    n = 0
    with _lock:
        for m in list(_models.values()):
            if m._sess is not None and _now() - m.last_used >= ttl and m._run_lock.acquire(blocking=False):
                try:
                    m._sess = None
                    n += 1
                finally:
                    m._run_lock.release()
    if n:
        gc.collect()
        log.info('AI runtime: unloaded %d idle model(s)', n)
    return n


def _reaper_loop():
    while True:
        time.sleep(30)
        try:
            unload_idle()
        except Exception as e:               # noqa: BLE001
            log.warning('AI reaper: %s', e)


def _start_reaper():
    global _reaper_started
    with _lock:
        if _reaper_started:
            return
        _reaper_started = True
    threading.Thread(target=_reaper_loop, name='booru-ai-reaper', daemon=True).start()


def runtime_info():
    """Describe the ONNX runtime for the settings panel (does not load any model)."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        ver = ort.__version__
    except Exception as e:                   # noqa: BLE001
        return {'ok': False, 'error': str(e)}
    with _lock:
        loaded = {k: m.provider for k, m in _models.items() if m._sess is not None}
    return {
        'ok': True, 'onnxruntime': ver,
        'cuda_available': 'CUDAExecutionProvider' in avail,
        'cuda_enabled': cuda_enabled(),
        'cuda_off_reason': _cuda_off_reason if not cuda_enabled() else '',
        'loaded_in_this_worker': loaded,
        'idle_unload_seconds': IDLE_UNLOAD_SECONDS,
    }
