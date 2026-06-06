#!/bin/bash

venv/bin/python3 manage.py makemigrations && venv/bin/python3 manage.py migrate

sudo systemctl daemon-reload

sudo systemctl restart booru

venv/bin/gunicorn \
    --workers 4 \
    --worker-class gevent \
    --worker-connections 100 \
    --bind 0.0.0.0:3000 \
    --timeout 900 \
    booru.wsgi:application
