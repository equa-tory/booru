from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-local-booru-secret-key-change-in-prod'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [    'django.contrib.contenttypes',    'django.contrib.sessions',    'django.contrib.staticfiles',    'gallery',]

MIDDLEWARE = [    'django.middleware.security.SecurityMiddleware',    'gallery.middleware.CacheHeadersMiddleware',    'django.middleware.common.CommonMiddleware',    'django.contrib.sessions.middleware.SessionMiddleware',    'gallery.middleware.LoginRequiredMiddleware',    'django.middleware.csrf.CsrfViewMiddleware',    'django.middleware.clickjacking.XFrameOptionsMiddleware',]

ROOT_URLCONF = 'booru.urls'

# Unique cookie names: other local Django apps use the default `sessionid` /
# `csrftoken`, and cookies are shared across ports on the same host, so they
# kept logging each other out.
SESSION_COOKIE_NAME = 'booru_sessionid'
CSRF_COOKIE_NAME = 'booru_csrftoken'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.request',
        'gallery.context.quick_links',
    ]},
}]

WSGI_APPLICATION = 'booru.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        # file-backed test DB (not the default in-memory one) so concurrency tests across threads behave like production
        'TEST': {'NAME': BASE_DIR / 'test_db.sqlite3'},
        # WAL lets reads and writes happen concurrently (the gunicorn gevent
        # workers otherwise serialize on a single write lock, which is the main
        # reason adds feel slower as the library grows). busy_timeout makes a
        # worker wait for the lock instead of erroring out immediately.
        'OPTIONS': {
            'init_command': (
                'PRAGMA journal_mode=WAL;'
                'PRAGMA synchronous=NORMAL;'
                'PRAGMA busy_timeout=5000;'
            ),
            'transaction_mode': 'IMMEDIATE',
        },
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'  # сюда collectstatic

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Simple gallery password (override in booru/local_settings.py)
GALLERY_PASSWORD = 'booru'

# Machine-specific overrides (MEDIA_ROOT, GALLERY_PASSWORD, SECRET_KEY...).
# booru/local_settings.py is untracked; see local_settings.py.example.
try:
    from .local_settings import *  # noqa: F401,F403
except ImportError:
    pass
