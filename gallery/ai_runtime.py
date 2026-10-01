"""ONNX Runtime sessions for the AI taggers.

- Uses the CUDA provider when onnxruntime-gpu + a working CUDA/cuDNN stack are
  present; otherwise (or if CUDA fails at run time) transparently runs on CPU.
- Models are loaded lazily and UNLOADED after AI_IDLE_UNLOAD_SECONDS of disuse,
  so an idle gallery does not keep the GPU's VRAM (it is shared with e.g.
  ComfyUI). Each gunicorn worker is its own process with its own sessions.
- A CUDA failure (cuDNN mismatch, out of memory...) switches this process to
  CPU for CUDA_RETRY_SECONDS, then CUDA is tried again.
- "Free VRAM now": request_unload_all() drops a signal file that EVERY worker's
  reaper notices within a few seconds and answers by force-unloading all models.
"""
import gc
import logging
import os
import subprocess
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
_force_until = 0.0        # monotonic; until then every model counts as idle (keeps freeing reloads)
_seen_signal = 0.0        # mtime of the unload signal this process has already acted on
REAPER_TICK = 3           # seconds between checks (one os.stat)
FORCE_WINDOW = 60         # seconds a forced unload keeps freeing models that get reloaded


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


def _signal_path():
    from . import ai_models
    return os.path.join(ai_models.models_dir(), '.unload')


def _signal_mtime():
    try:
        return os.stat(_signal_path()).st_mtime
    except OSError:
        return 0.0


def request_unload_all():
    """Ask every worker process to unload all models (cross-process)."""
    p = _signal_path()
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(time.time()))
    os.replace(tmp, p)


def force_unload():
    """Drop every loaded model in THIS process right now, even one that is
    mid-inference (the running call keeps its own reference until it returns).
    Returns how many were loaded."""
    global _force_until
    n = 0
    with _lock:
        _force_until = _now() + FORCE_WINDOW
        for m in _models.values():
            if m._sess is not None:
                m._sess = None
                n += 1
    gc.collect()
    if n:
        log.info('AI runtime: force-unloaded %d model(s)', n)
    return n


def forget(key):
    """Unload and unregister a model (used before deleting its files)."""
    with _lock:
        m = _models.pop(key, None)
    if m is not None:
        m._sess = None
        gc.collect()


def gpu_info():
    """{'name','used_mb','total_mb'} from nvidia-smi, or None when unavailable."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        name, used, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(',')]
        return {'name': name, 'used_mb': int(used), 'total_mb': int(total)}
    except Exception:                        # noqa: BLE001
        return None


_last_idle_check = 0.0


def _reaper_tick():
    """One reaper step: act on a new unload signal, keep freeing models during the
    force window, otherwise do the normal idle sweep (every 30 s)."""
    global _seen_signal, _last_idle_check
    sig = _signal_mtime()
    if sig > _seen_signal:
        _seen_signal = sig
        force_unload()
    elif _now() < _force_until:
        unload_idle(ttl=0)
    elif _now() - _last_idle_check >= 30:
        _last_idle_check = _now()
        unload_idle()


def _reaper_loop():
    global _seen_signal
    _seen_signal = max(_seen_signal, _signal_mtime())      # ignore signals sent before this process loaded anything
    while True:
        time.sleep(REAPER_TICK)
        try:
            _reaper_tick()
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
