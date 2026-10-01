"""Database backups: scheduled + on-demand snapshots of db.sqlite3, and restore.

Only the SQLite DB is backed up — media files on disk are the source of truth
and are never touched. Snapshots use sqlite3's online-backup API in a single
step, so they are consistent while gunicorn workers keep reading/writing (WAL).
Settings live under the 'backup' key of prefs.json; the default folder comes
from settings.BACKUP_DIR (set it in booru/local_settings.py).
"""
import contextlib
import json
import os
import shutil
import sqlite3
import threading
import time
from urllib.parse import quote

from django.conf import settings

try:
    import fcntl          # cross-process lock between gunicorn workers (Linux/macOS)
except ImportError:       # Windows dev: fall back to the in-process lock only
    fcntl = None

AUTO_PREFIX = 'booru-'                     # booru-YYYYmmdd-HHMMSS.sqlite3 — the only files rotation deletes
SNAPSHOT_NAME = 'pre-restore.sqlite3'      # safety copy of the live DB taken before every restore
EXTS = ('.sqlite3', '.sqlite', '.db')
DEFAULTS = {'enabled': True, 'max_backups': 1, 'interval_hours': 48}

_proc_lock = threading.Lock()


class _Busy(RuntimeError):
    pass


def db_path():
    return str(settings.DATABASES['default']['NAME'])


def _prefs_path():
    return os.path.join(settings.BASE_DIR, 'prefs.json')


def _read_prefs():
    try:
        with open(_prefs_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def default_dir():
    return str(getattr(settings, 'BACKUP_DIR', os.path.join(settings.BASE_DIR, 'backups')))


# ── config ──────────────────────────────────────────────────────
def get_config():
    saved = _read_prefs().get('backup')
    cfg = dict(DEFAULTS, path=default_dir())
    if isinstance(saved, dict):
        cfg.update({k: saved[k] for k in ('enabled', 'max_backups', 'interval_hours', 'path') if k in saved})
    return cfg


def save_config(data):
    """Validate + persist. Raises ValueError with a user-facing message."""
    try:
        max_backups = int(data.get('max_backups'))
        interval = float(data.get('interval_hours'))
    except (TypeError, ValueError):
        raise ValueError('max backups and hours must be numbers')
    if not 1 <= max_backups <= 100:
        raise ValueError('max backups must be between 1 and 100')
    if not 1 <= interval <= 24 * 365:
        raise ValueError('interval must be between 1 and 8760 hours')
    path = os.path.expanduser(str(data.get('path') or '').strip())
    if not os.path.isabs(path):
        raise ValueError('folder must be an absolute path')
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.write-test')
        with open(probe, 'w'):
            pass
        os.remove(probe)
    except OSError as e:
        raise ValueError(f'cannot write to {path}: {e.strerror or e}')
    cfg = {'enabled': bool(data.get('enabled')), 'max_backups': max_backups,
           'interval_hours': int(interval) if interval == int(interval) else interval,
           'path': os.path.normpath(path)}
    prefs = _read_prefs()
    prefs['backup'] = cfg
    tmp = _prefs_path() + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(prefs, f)
    os.replace(tmp, _prefs_path())
    return cfg


# ── listing ─────────────────────────────────────────────────────
def list_backups(d=None):
    """Restorable files in the backup folder, newest first."""
    d = d or get_config()['path']
    out = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        if not name.lower().endswith(EXTS):
            continue
        p = os.path.join(d, name)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if os.path.isfile(p):
            out.append({'name': name, 'size': st.st_size, 'mtime': st.st_mtime,
                        'auto': name.startswith(AUTO_PREFIX)})
    out.sort(key=lambda b: b['mtime'], reverse=True)
    return out


def _last_auto(d):
    autos = [b for b in list_backups(d) if b['auto']]
    return autos[0]['mtime'] if autos else None


def info():
    cfg = get_config()
    d = cfg['path']
    last = _last_auto(d)
    nxt = None
    if cfg['enabled']:
        nxt = (last + cfg['interval_hours'] * 3600) if last else time.time()
    try:
        free = shutil.disk_usage(d).free if os.path.isdir(d) else None
    except OSError:
        free = None
    try:
        db_size = os.path.getsize(db_path())
    except OSError:
        db_size = None
    return {'config': cfg, 'backups': list_backups(d), 'last': last, 'next': nxt,
            'dir_exists': os.path.isdir(d), 'free': free, 'db_size': db_size}


# ── locking ─────────────────────────────────────────────────────
@contextlib.contextmanager
def _lock(d):
    """Non-blocking exclusive lock shared by every worker process. Yields True
    if we got it."""
    if not _proc_lock.acquire(blocking=False):
        yield False
        return
    fh = None
    try:
        got = True
        if fcntl:
            try:
                os.makedirs(d, exist_ok=True)
                fh = open(os.path.join(d, '.booru-backup.lock'), 'w')
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                got = False
        yield got
    finally:
        if fh:
            with contextlib.suppress(OSError):
                fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        _proc_lock.release()


# ── backup ──────────────────────────────────────────────────────
def _snapshot(src_con, dest_path):
    """Copy the (live) DB behind `src_con` into a standalone file at dest_path,
    atomically (write to .partial, then rename)."""
    tmp = dest_path + '.partial'
    if os.path.exists(tmp):
        os.remove(tmp)
    dst = sqlite3.connect(tmp)
    try:
        src_con.backup(dst)                       # one step: consistent, never restarts mid-copy
        dst.execute('PRAGMA journal_mode=DELETE')  # standalone file, no -wal/-shm companions
    except BaseException:
        dst.close()
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    dst.close()
    os.replace(tmp, dest_path)


def _prune(d, keep):
    autos = sorted((b for b in list_backups(d) if b['auto']), key=lambda b: b['name'], reverse=True)
    for b in autos[max(1, keep):]:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(d, b['name']))


def run_backup(task=None):
    """Take a snapshot into the backup folder, then rotate. Returns the file name."""
    cfg = get_config()
    d = cfg['path']
    os.makedirs(d, exist_ok=True)
    with _lock(d) as got:
        if not got:
            raise _Busy('another backup/restore is already running')
        need = os.path.getsize(db_path())
        if shutil.disk_usage(d).free < need * 1.1:
            raise RuntimeError(f'not enough free space in {d}')
        name = f'{AUTO_PREFIX}{time.strftime("%Y%m%d-%H%M%S")}.sqlite3'
        if task is not None:
            task.message = f'writing {name}…'; task.save(update_fields=['message'])
        live = sqlite3.connect(db_path(), timeout=30)
        try:
            _snapshot(live, os.path.join(d, name))
        finally:
            live.close()
        _prune(d, cfg['max_backups'])
    if task is not None:
        task.message = f'saved {name}'; task.save(update_fields=['message'])
    return name


# ── restore ─────────────────────────────────────────────────────
def validate_backup(path):
    """Raise ValueError unless `path` is an intact booru database."""
    try:
        con = sqlite3.connect(f'file:{quote(path)}?mode=ro', uri=True)
        try:
            if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('file is corrupt (integrity check failed)')
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
    except sqlite3.DatabaseError as e:
        raise ValueError(f'not a valid SQLite database: {e}')
    if not {'gallery_post', 'gallery_photo', 'gallery_tag'} <= tables:
        raise ValueError('not a booru database (gallery tables missing)')


def restore_from(path, task=None):
    """Replace the live DB with the snapshot at `path`.

    - validates first; takes a safety copy of the current DB (pre-restore.sqlite3)
    - overwrites the live file in place via the backup API, so the other gunicorn
      workers' open connections simply see the new content
    - carries the current login sessions across so you stay logged in
    - runs `migrate`, so an older snapshot is brought up to the current schema
    - re-creates this task's own row, which the restored DB would not contain
    """
    from django.core.management import call_command
    from gallery.models import Task

    validate_backup(path)
    d = get_config()['path']
    os.makedirs(d, exist_ok=True)
    keep = None
    if task is not None:
        keep = dict(pk=task.pk, kind=task.kind, total=task.total, done=task.done,
                    message='restoring database…', started_at=task.started_at)
    with _lock(d) as got:
        if not got:
            raise _Busy('another backup/restore is already running')
        live = sqlite3.connect(db_path(), timeout=60)
        try:
            try:
                sessions = live.execute(
                    'SELECT session_key, session_data, expire_date FROM django_session').fetchall()
            except sqlite3.DatabaseError:
                sessions = []
            if os.path.abspath(path) != os.path.abspath(os.path.join(d, SNAPSHOT_NAME)):
                _snapshot(live, os.path.join(d, SNAPSHOT_NAME))
            src = sqlite3.connect(f'file:{quote(path)}?mode=ro', uri=True)
            try:
                src.backup(live)                 # no Django writes between here and the commit: it holds the write lock
            finally:
                src.close()
            if sessions:
                try:
                    live.executemany(
                        'INSERT OR REPLACE INTO django_session (session_key, session_data, expire_date) VALUES (?,?,?)',
                        sessions)
                    live.commit()
                except sqlite3.DatabaseError:
                    pass
        finally:
            live.close()
    try:
        call_command('migrate', interactive=False, verbosity=0)
    except Exception as e:
        print(f'restore: migrate failed: {e}')
    if keep:
        started = keep.pop('started_at')
        Task.objects.update_or_create(pk=keep.pop('pk'), defaults=dict(keep, status='running'))
        Task.objects.filter(pk=task.pk).update(started_at=started)


# ── scheduler ───────────────────────────────────────────────────
_scheduler_started = False


def _due(cfg):
    if not cfg['enabled']:
        return False
    last = _last_auto(cfg['path'])
    return last is None or time.time() - last >= cfg['interval_hours'] * 3600


def _scheduled_run():
    from django.utils import timezone
    from gallery.models import Task
    if not _due(get_config()):
        return
    task = Task.objects.create(kind='backup', message='scheduled backup…')
    try:
        run_backup(task)                       # takes the cross-worker lock
        status, err = 'done', ''
    except _Busy:
        task.delete()                          # another worker is already doing it
        return
    except Exception as e:
        status, err = 'error', str(e)[:2000]
    task.refresh_from_db()
    task.status, task.error, task.finished_at = status, err, timezone.now()
    task.save()


def _scheduler_loop():
    from django.db import connection
    import random
    time.sleep(90 + random.uniform(0, 30))    # let the app finish booting; jitter so the 4 workers don't all fire at once
    while True:
        try:
            _scheduled_run()
        except Exception as e:
            print(f'backup scheduler: {e}')
        finally:
            connection.close()
        time.sleep(600)


def start_scheduler():
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    threading.Thread(target=_scheduler_loop, name='booru-backup', daemon=True).start()
