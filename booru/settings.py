from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-local-booru-secret-key-change-in-prod'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [    'django.contrib.contenttypes',    'django.contrib.sessions',    'django.contrib.staticfiles',    'gallery',]

MIDDLEWARE = [    'django.middleware.security.SecurityMiddleware',    'django.middleware.common.CommonMiddleware',    'django.contrib.sessions.middleware.SessionMiddleware',    'gallery.middleware.LoginRequiredMiddleware',    'django.middleware.csrf.CsrfViewMiddleware',    'django.middleware.clickjacking.XFrameOptionsMiddleware',]

ROOT_URLCONF = 'booru.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.request',
    ]},
}]

WSGI_APPLICATION = 'booru.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'  # сюда collectstatic

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
#MEDIA_ROOT = '/@/Media_SRV/Photo'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Simple gallery password (change this!)
GALLERY_PASSWORD = 'booru'
