"""Server-side preferences (quick links, main AI model, backup config, key bindings,
recent posts...). Stored in the database (`Setting` rows holding JSON), so writes are
transactional across the gunicorn workers and every database backup contains them.

The old `prefs.json` is imported once (then renamed to prefs.json.imported).
"""
import json
import os

from django.conf import settings
from django.db import transaction

from .models import Setting

_legacy_checked = False


def legacy_path():
    return os.path.join(settings.BASE_DIR, 'prefs.json')


def _import_legacy():
    """First use: copy the keys of the old prefs.json into the table (never overwrites)."""
    global _legacy_checked
    if _legacy_checked:
        return
    _legacy_checked = True
    path = legacy_path()
    if not os.path.isfile(path) or Setting.objects.exists():
        return
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return
    if isinstance(data, dict):
        for k, v in data.items():
            Setting.objects.get_or_create(key=str(k)[:100], defaults={'value': json.dumps(v)})
    try:
        os.replace(path, path + '.imported')
    except OSError:
        pass


def _load(raw, default=None):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def get(key, default=None):
    _import_legacy()
    row = Setting.objects.filter(key=key).first()
    return default if row is None else _load(row.value, default)


def all():
    _import_legacy()
    return {r.key: _load(r.value) for r in Setting.objects.all()}


def set(key, value):          # noqa: A001 - mirrors dict-style naming
    _import_legacy()
    Setting.objects.update_or_create(key=key, defaults={'value': json.dumps(value)})


def update(key, fn, default=None):
    """Atomic read-modify-write: value = fn(current). The write lock is taken up front
    (SQLite IMMEDIATE transactions), so concurrent updates queue instead of racing."""
    _import_legacy()
    with transaction.atomic():
        row = Setting.objects.filter(key=key).first()
        new = fn(default if row is None else _load(row.value, default))
        Setting.objects.update_or_create(key=key, defaults={'value': json.dumps(new)})
        return new
