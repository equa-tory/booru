#!/usr/bin/env bash
# One-shot installer: venv + dependencies + local settings + database + systemd service.
# Safe to re-run (keeps your booru/local_settings.py and database; updates deps,
# migrates, re-installs the unit and restarts it).
#
#   ./install.sh                  interactive
#   ./install.sh --yes            no questions (use env vars below / defaults)
#   ./install.sh --no-service     everything except the systemd unit
#   ./install.sh --gpu            also switch the AI taggers to the NVIDIA GPU (onnxruntime-gpu + CUDA 12 libs, ~3 GB)
#   ./install.sh --proxy URL      send pip through a proxy, e.g. http://127.0.0.1:1081 (xray)
#   ./install.sh --print-unit     just print the rendered unit and exit
#
# Env overrides: MEDIA_ROOT  GALLERY_PASSWORD  BACKUP_DIR  PORT
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

YES=0; SERVICE=1; PRINT_UNIT=0; GPU=0
shift_proxy=0
for a in "$@"; do
  if [[ $shift_proxy -eq 2 ]]; then
    export HTTPS_PROXY="$a" HTTP_PROXY="$a" https_proxy="$a" http_proxy="$a"; shift_proxy=0; continue
  fi
  case "$a" in
    --yes|-y) YES=1 ;;
    --no-service) SERVICE=0 ;;
    --gpu) GPU=1 ;;
    --proxy) shift_proxy=2 ;;
    --proxy=*) export HTTPS_PROXY="${a#--proxy=}" HTTP_PROXY="${a#--proxy=}" https_proxy="${a#--proxy=}" http_proxy="${a#--proxy=}" ;;
    --print-unit) PRINT_UNIT=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# A pull-through pip mirror (e.g. Nexus) first downloads big wheels (CUDA libs are GBs)
# from upstream before it serves them — pip's 15 s default read timeout gives up
# too early, and the mirror then has the file cached for the next try.
export PIP_TIMEOUT="${PIP_TIMEOUT:-900}" PIP_RETRIES="${PIP_RETRIES:-5}"

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
pip_run() {  # pip_run <install|download> args… — the configured index (e.g. a local mirror) first, then PyPI itself
  local cmd="$1"; shift
  venv/bin/pip "$cmd" --quiet "$@" && return 0
  warn "pip $cmd failed with the configured index — retrying against pypi.org (use --proxy URL if it is blocked)"
  venv/bin/pip "$cmd" --quiet --index-url https://pypi.org/simple "$@"
}
pip_install() { pip_run install "$@"; }
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
# Keep the GPU build if it is already installed: requirements.txt pins the CPU
# `onnxruntime`, which shares its package directory with `onnxruntime-gpu`.
if venv/bin/pip show onnxruntime-gpu >/dev/null 2>&1; then GPU=1; fi
say "installing dependencies (requirements.txt)"
venv/bin/pip install --quiet --upgrade pip || warn "could not upgrade pip (offline?) — continuing"
if [[ $GPU -eq 1 ]]; then
  REQ="$(mktemp)"; grep -v -i '^onnxruntime==' requirements.txt > "$REQ"
  pip_install -r "$REQ"; rm -f "$REQ"
else
  pip_install -r requirements.txt
fi

if [[ $GPU -eq 1 ]]; then
  command -v nvidia-smi >/dev/null || warn "nvidia-smi not found — is the NVIDIA driver installed? continuing anyway"
  if venv/bin/python -c "import importlib.metadata as m,sys; sys.exit(0 if m.version('onnxruntime-gpu')=='1.26.0' and m.version('nvidia-cudnn-cu12')=='9.10.2.21' else 1)" 2>/dev/null; then
    say "GPU runtime already installed"
  else
    # Download everything FIRST (several GB; a flaky mirror/proxy must not leave the
    # venv without any onnxruntime), then swap offline in a few seconds.
    WH="$(mktemp -d)"; trap 'rm -rf "$WH"' EXIT
    say "downloading GPU runtime (several GB — a pull-through mirror may need a few minutes per file)"
    pip_run download -r requirements-gpu.txt -d "$WH"
    say "switching onnxruntime → onnxruntime-gpu"
    venv/bin/pip uninstall -y --quiet onnxruntime >/dev/null 2>&1 || true
    venv/bin/pip install --quiet --no-index --find-links "$WH" -r requirements-gpu.txt
    rm -rf "$WH"
  fi
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
