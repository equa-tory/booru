#!/usr/bin/env bash
# One-shot installer: venv + dependencies + local settings + database + systemd service.
# Safe to re-run (keeps your booru/local_settings.py and database; updates deps,
# migrates, re-installs the unit and restarts it).
#
#   ./install.sh                  interactive
#   ./install.sh --yes            no questions (use env vars below / defaults)
#   ./install.sh --no-service     everything except the systemd unit
#   ./install.sh --gpu            also switch the AI taggers to the NVIDIA GPU (onnxruntime-gpu + CUDA 12 libs, ~3 GB)
#   ./install.sh --print-unit     just print the rendered unit and exit
#
# Env overrides: MEDIA_ROOT  GALLERY_PASSWORD  BACKUP_DIR  PORT
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

YES=0; SERVICE=1; PRINT_UNIT=0; GPU=0
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --no-service) SERVICE=0 ;;
    --gpu) GPU=1 ;;
    --print-unit) PRINT_UNIT=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

SVC_USER="${SUDO_USER:-${USER:-$(id -un)}}"
PORT="${PORT:-3002}"
say()  { printf '\033[1;35m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ask()  { # ask VAR "prompt" default  — keeps an env-provided value, prompts only when interactive
  local var="$1" prompt="$2" def="$3" cur="${!1:-}" reply
  if [[ -n "$cur" ]]; then return; fi
  if [[ $YES -eq 1 || ! -t 0 ]]; then printf -v "$var" '%s' "$def"; return; fi
  read -r -p "$prompt [$def]: " reply || true
  printf -v "$var" '%s' "${reply:-$def}"
}
pip_install() {  # the machine's configured pip index (e.g. a local mirror) may be down/empty — retry against PyPI
  venv/bin/pip install --quiet "$@" && return 0
  warn "pip install failed with the configured index — retrying against pypi.org"
  venv/bin/pip install --quiet --index-url https://pypi.org/simple "$@"
}
as_root() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

render_unit() {
  sed -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@USER@|$SVC_USER|g" -e "s|@PORT@|$PORT|g" "$APP_DIR/booru.service"
}
if [[ $PRINT_UNIT -eq 1 ]]; then render_unit; exit 0; fi

# ── prerequisites ───────────────────────────────────────────────
say "checking prerequisites"
command -v python3 >/dev/null || die "python3 not found"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || die "Python 3.12+ required (Django 6); found $(python3 --version)"
python3 -c 'import venv, ensurepip' 2>/dev/null \
  || die "python3-venv missing — on Debian/Ubuntu: sudo apt install python3-venv"
for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null || warn "$bin not found — video thumbnails / audio detection need it (sudo apt install ffmpeg)"
done

# ── virtualenv + dependencies ───────────────────────────────────
if [[ ! -x venv/bin/python ]]; then say "creating virtualenv"; python3 -m venv venv; fi
say "installing dependencies (requirements.txt)"
venv/bin/pip install --quiet --upgrade pip || warn "could not upgrade pip (offline?) — continuing"
pip_install -r requirements.txt

if [[ $GPU -eq 1 ]]; then
  command -v nvidia-smi >/dev/null || warn "nvidia-smi not found — is the NVIDIA driver installed? continuing anyway"
  say "installing GPU runtime (replaces CPU onnxruntime; large download)"
  venv/bin/pip uninstall -y --quiet onnxruntime >/dev/null 2>&1 || true
  pip_install -r requirements-gpu.txt
  venv/bin/python - <<'PY' || warn "CUDA provider not available — the taggers will keep using the CPU"
import onnxruntime as ort
ort.preload_dlls()
assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()
print('   onnxruntime', ort.__version__, '- CUDA provider available')
PY
fi

# ── local settings (untracked: password, media + backup folders) ──
LOCAL=booru/local_settings.py
if [[ -f $LOCAL ]]; then
  say "keeping existing $LOCAL"
else
  say "creating $LOCAL"
  MEDIA_ROOT="${MEDIA_ROOT:-}"; GALLERY_PASSWORD="${GALLERY_PASSWORD:-}"; BACKUP_DIR="${BACKUP_DIR:-}"
  ask MEDIA_ROOT "media folder (photos live here, inbox/ inside it)" "$APP_DIR/media"
  ask BACKUP_DIR "database backup folder" "$APP_DIR/backups"
  GENERATED_PW=0
  if [[ -z $GALLERY_PASSWORD ]]; then
    if [[ $YES -eq 0 && -t 0 ]]; then read -r -s -p "gallery password (empty = generate one): " GALLERY_PASSWORD; echo; fi
    if [[ -z $GALLERY_PASSWORD ]]; then GALLERY_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"; GENERATED_PW=1; fi
  fi
  export MEDIA_ROOT GALLERY_PASSWORD BACKUP_DIR
  python3 - "$LOCAL" <<'PY'
import os, secrets, sys
with open(sys.argv[1], 'w') as f:
    f.write("# Machine-specific overrides. Untracked (see .gitignore) — never commit this file.\n")
    f.write(f"MEDIA_ROOT = {os.environ['MEDIA_ROOT']!r}\n")
    f.write(f"GALLERY_PASSWORD = {os.environ['GALLERY_PASSWORD']!r}\n")
    f.write(f"BACKUP_DIR = {os.environ['BACKUP_DIR']!r}\n")
    f.write(f"SECRET_KEY = {secrets.token_urlsafe(50)!r}\n")
os.chmod(sys.argv[1], 0o600)
PY
  [[ $GENERATED_PW -eq 1 ]] && say "generated gallery password: $GALLERY_PASSWORD   (stored in $LOCAL)"
fi

# ── folders + database ──────────────────────────────────────────
read -r MEDIA_DIR BACKUP_PATH < <(venv/bin/python - <<'PY'
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'booru.settings')
from django.conf import settings
print(settings.MEDIA_ROOT, getattr(settings, 'BACKUP_DIR', ''))
PY
)
say "preparing folders ($MEDIA_DIR, $BACKUP_PATH)"
mkdir -p "$MEDIA_DIR/inbox" "$MEDIA_DIR/thumbs" ${BACKUP_PATH:+"$BACKUP_PATH"}
say "migrating database"
venv/bin/python manage.py migrate --noinput

# ── systemd service ─────────────────────────────────────────────
if [[ $SERVICE -eq 1 ]]; then
  if ! command -v systemctl >/dev/null; then
    warn "systemd not found — skipping the service. Run manually: venv/bin/python manage.py runserver 0.0.0.0:$PORT"
  else
    say "installing systemd unit (user=$SVC_USER, port=$PORT)"
    render_unit | as_root tee /etc/systemd/system/booru.service >/dev/null
    as_root systemctl daemon-reload
    as_root systemctl enable booru >/dev/null 2>&1
    as_root systemctl restart booru
    sleep 2
    if systemctl is-active --quiet booru; then say "booru is running"; else warn "service did not start — see: journalctl -u booru -n 50"; fi
  fi
fi

HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
say "done → http://${HOST:-localhost}:$PORT/"
[[ $SERVICE -eq 1 ]] && echo "   logs: journalctl -u booru -f    restart: sudo systemctl restart booru"
echo "   put photos in $MEDIA_DIR/inbox/ and click “scan inbox”"
