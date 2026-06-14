#!/usr/bin/env python3
"""
watch_downloads.py — infinite watcher for a downloads folder.

What it does (polling loop, no extra dependencies):
  • Watches DOWNLOADS_DIR for NEW files only. Files already present when the
    script starts are remembered and ignored (so it never re-imports your whole
    downloads history on (re)start).
  • Ignores files that are still downloading: browser part-files
    (.crdownload/.part/.tmp/...) are skipped, and a file is only acted on once
    its size has stayed the same across a couple of polls.
  • Single media file (mp4, png, jpg, jpeg, webm, gif, …) → MOVED into the
    gallery inbox root.
  • .zip archive → extracted ONCE into  inbox/_/<zipname>/  (a multi-image
    post), then the original .zip is deleted. If that folder already exists it
    is treated as already-extracted (no second extraction).
  • Everything else is left alone.

Run it:
    python watch_downloads.py --downloads ~/Downloads --inbox /path/to/media/inbox
or set env vars BOORU_DOWNLOADS and BOORU_INBOX and just run `python watch_downloads.py`.

Tip: run it under systemd or `nohup python watch_downloads.py &` so it stays alive.
"""

import os
import sys
import time
import shutil
import zipfile
import argparse

# file types we treat as a single media post
MEDIA_EXTS = {'.mp4', '.webm', '.mov', '.mkv', '.avi',
              '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.avif'}
ARCHIVE_EXTS = {'.zip'}
# partial-download markers — never touch these
PARTIAL_EXTS = {'.crdownload', '.part', '.tmp', '.download', '.opdownload',
                '.partial', '.!qb', '.aria2'}

POLL_SECONDS = 3        # how often to scan the downloads folder
STABLE_POLLS = 2        # size must be unchanged this many polls before acting


def log(msg):
    print(f'[watch] {time.strftime("%H:%M:%S")} {msg}', flush=True)


def unique_path(dest_dir, filename):
    """Return a non-colliding path inside dest_dir for `filename`."""
    base = os.path.basename(filename)
    dest = os.path.join(dest_dir, base)
    if not os.path.exists(dest):
        return dest
    stem, ext = os.path.splitext(base)
    i = 1
    while os.path.exists(os.path.join(dest_dir, f'{stem}_{i}{ext}')):
        i += 1
    return os.path.join(dest_dir, f'{stem}_{i}{ext}')


def safe_extract_zip(zip_path, target_dir):
    """Extract a zip into target_dir, guarding against path traversal."""
    os.makedirs(target_dir, exist_ok=True)
    target_abs = os.path.abspath(target_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # skip directory entries and anything that escapes the target
            dest = os.path.abspath(os.path.join(target_dir, member))
            if not dest.startswith(target_abs + os.sep) and dest != target_abs:
                log(f'  ! skipping unsafe path in zip: {member}')
                continue
            if member.endswith('/'):
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, 'wb') as out:
                shutil.copyfileobj(src, out)


def handle_zip(zip_path, inbox):
    """Extract zip into inbox/_/<name>/ exactly once, then delete the zip."""
    name = os.path.splitext(os.path.basename(zip_path))[0]
    multi_root = os.path.join(inbox, '_')
    target = os.path.join(multi_root, name)
    # avoid a second extraction if the folder already exists with content
    if os.path.isdir(target) and os.listdir(target):
        log(f'  zip target already exists, not extracting again: {target}')
    else:
        try:
            safe_extract_zip(zip_path, target)
            log(f'  extracted → {target}')
        except zipfile.BadZipFile:
            log(f'  ! bad/incomplete zip, leaving it: {zip_path}')
            return False
        except Exception as e:
            log(f'  ! extract failed ({e}), leaving zip: {zip_path}')
            return False
    # delete the original archive
    try:
        os.remove(zip_path)
        log(f'  deleted archive: {os.path.basename(zip_path)}')
    except OSError as e:
        log(f'  ! could not delete zip: {e}')
    return True


def handle_media(path, inbox):
    """Move a single media file into the inbox root."""
    dest = unique_path(inbox, os.path.basename(path))
    try:
        shutil.move(path, dest)
        log(f'  moved → {os.path.basename(dest)}')
        return True
    except Exception as e:
        log(f'  ! move failed ({e}): {path}')
        return False


def is_partial(name):
    lower = name.lower()
    if any(lower.endswith(ext) for ext in PARTIAL_EXTS):
        return True
    if lower.startswith('.') or lower.endswith('~'):
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description='Watch a downloads folder and feed the booru inbox.')
    ap.add_argument('--downloads', default=os.environ.get('BOORU_DOWNLOADS', ''),
                    help='folder to watch (or set BOORU_DOWNLOADS)')
    ap.add_argument('--inbox', default=os.environ.get('BOORU_INBOX', ''),
                    help='gallery inbox root (or set BOORU_INBOX)')
    ap.add_argument('--interval', type=float, default=POLL_SECONDS)
    args = ap.parse_args()

    downloads = os.path.expanduser(args.downloads)
    inbox = os.path.expanduser(args.inbox)
    if not downloads or not os.path.isdir(downloads):
        sys.exit(f'downloads folder not found: {downloads!r} (use --downloads or BOORU_DOWNLOADS)')
    if not inbox or not os.path.isdir(inbox):
        sys.exit(f'inbox folder not found: {inbox!r} (use --inbox or BOORU_INBOX)')
    os.makedirs(os.path.join(inbox, '_'), exist_ok=True)

    # remember everything already present → those are "old", ignore them
    seen = set(os.listdir(downloads))
    # pending = name -> (last_size, stable_count)
    pending = {}
    log(f'watching {downloads}')
    log(f'inbox    {inbox}')
    log(f'ignoring {len(seen)} pre-existing item(s)')

    while True:
        try:
            entries = os.listdir(downloads)
        except OSError as e:
            log(f'! cannot list downloads ({e})'); time.sleep(args.interval); continue

        current = set(entries)
        # forget pending items that vanished
        for gone in [n for n in pending if n not in current]:
            pending.pop(gone, None)

        for name in entries:
            if name in seen:
                continue
            path = os.path.join(downloads, name)
            if not os.path.isfile(path):       # ignore folders / sockets
                continue
            if is_partial(name):               # still downloading
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in MEDIA_EXTS and ext not in ARCHIVE_EXTS:
                seen.add(name)                 # not interesting → stop checking
                continue

            # stability check: only act once the size has settled
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            last_size, stable = pending.get(name, (-1, 0))
            if size == last_size and size > 0:
                stable += 1
            else:
                stable = 0
            pending[name] = (size, stable)
            if stable < STABLE_POLLS:
                continue                       # wait until size is stable

            # act
            log(f'new file: {name} ({size} bytes)')
            ok = handle_zip(path, inbox) if ext in ARCHIVE_EXTS else handle_media(path, inbox)
            seen.add(name)
            pending.pop(name, None)

        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('stopped')
