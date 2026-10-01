# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
A self-hosted, single-user "booru"-style photo/video/PDF gallery built on Django 6 + SQLite, with an htmx-driven frontend. No cloud, one shared password (`GALLERY_PASSWORD`; set it in the untracked `booru/local_settings.py`, which overrides `booru/settings.py` — see `local_settings.py.example`. The repo is public, so never put real secrets in `settings.py`). Files on disk are the source of truth; the DB only stores paths + metadata and never moves/renames originals except via explicit "organize"/"merge" actions.

## Commands
- Dev server: `python manage.py runserver`
- Migrate / make migrations: `python manage.py migrate` / `python manage.py makemigrations`
- Fresh install / re-install: `./install.sh [--yes] [--no-service]` — venv, `pip install -r requirements.txt`, generates the untracked `booru/local_settings.py` (password, media + backup folders, SECRET_KEY), migrates, renders `booru.service` (template with `@APP_DIR@/@USER@/@PORT@`) into `/etc/systemd/system/` and restarts it (default port 3002). Idempotent.
- Production (Linux/systemd): `./start.sh` — uses `venv/bin/python3`, runs makemigrations+migrate, `sudo`-restarts the `booru` systemd unit, then runs gunicorn (4 gevent workers, port 3001, 900s timeout). Windows dev uses runserver.
- Dependencies: `pip install -r requirements.txt` (pinned; includes gunicorn/gevent/onnxruntime/PyMuPDF). No linter is configured.
- Tests: `python manage.py test` (single test: `python manage.py test gallery.tests.ScanTests.test_scan_prunes_deleted_files_and_empty_posts`). `gallery/tests.py` covers scan, task cancel/dismiss, tag recount, auth cookies and backup/restore; scan tests run against a temp `MEDIA_ROOT`, backup tests against a temp SQLite file, never the real data.
- Downloads watcher (optional sidecar): `python watch_downloads.py --downloads <dir> --inbox <media/inbox>` — auto-moves new downloads/zips into the inbox.

## Adding media
Paths like `media/inbox/` below are relative to `settings.MEDIA_ROOT`, which may point outside the repo (overridden in the untracked `booru/local_settings.py`; the `settings.py` default is `./media`).
Two paths, both funnel through `gallery/utils.py::scan_inbox`:
1. Drop files into `media/inbox/` (subfolders OK), then click "scan inbox" (POST `/api/scan-bg/`).
2. Browser upload (POST `/api/upload/`) writes into `media/inbox/` then ingests.
Convention: `media/inbox/_/<folder>/` = one multi-image Post; loose files in `media/inbox/` = one Post each. Optional per-month tidy folders `inbox/YYYY-MM/DD/`.

## Architecture
- Single Django app `gallery`; project package `booru`.
- Models (`gallery/models.py`): `Post` (a gallery item) has many `Photo` (individual files, incl. video/PDF, `is_video` flag, `phash` for dedupe). `Tag` (M2M to Post) has a `category` (general/character/artist/meta/ai) and a denormalized `count` kept current via `Tag.update_count()`. `Folder` is either manual (explicit M2M posts) or "smart" (stores a query string re-run through `_build_post_qs`), and nests into a tree via `parent` (self-FK, `on_delete=CASCADE`) — organizational; opening a folder shows only its own directly-assigned posts unless `include_subfolders` is set (the "⊞" toggle on a folder row), which folds in every descendant folder's posts too (`Folder.descendant_ids()`, applied in `_build_post_qs`). `Task` is a DB-backed row tracking background jobs so progress survives page reloads and is visible to every gunicorn worker.
- `Photo.rel_path`/`rel_thumb_path` are stored relative to `settings.MEDIA_ROOT` (so relocating the whole media folder only means updating `MEDIA_ROOT`); the `file_path`/`thumb_path` properties resolve them to absolute paths and transparently tolerate legacy absolute values from before this became relative. `rebase_photo_paths()` in `utils.py` (wired to the "rebase paths" button) bulk-converts any remaining absolute rows.
- Ingestion/thumbnailing lives in `gallery/utils.py`: `ingest_photo`, `create_post_from_files`, `make_thumb`/`make_video_thumb` (ffmpeg)/`make_pdf_thumb` (PyMuPDF→pdftoppm fallback), `compute_phash`/`phash_distance`. Thumbnails are keyed by md5 of the source path and written to `media/thumbs/`.
- Views (`gallery/views.py`) are the whole controller layer — page renders + a large JSON API (URLs in `gallery/urls.py`). Frontend is server-rendered templates (`templates/gallery/`) + `static/js/htmx.min.js`; there is no JS build step. Infinite scroll pulls JSON from `/api/posts/`.

## Search DSL (in `gallery/views.py`)
The gallery query is built by `_build_post_qs`; token parsing is `_parse_tag_tokens` + `_term_to_q`. Supported search-box syntax:
- `a b` = AND, `( a ~ b )` = OR group (braces and spaces are significant), `-tag` = NOT
- `tag~` = fuzzy (Levenshtein), `ta*1` = glob wildcard, `file:name` / `folder:name` = path substring
Reuse these helpers rather than writing new query logic. `random` sort uses a seeded deterministic shuffle (`_apply_seeded_order`) so gallery/scroll/prev-next stay consistent — the seed lives in the URL.

## Background tasks
Heavy operations (scan, merge, ai_tag, dupes) run via `_start_task(kind, fn)` which spawns a daemon thread and records progress in a `Task` row. Each thread MUST `connection.close()` when done (already handled in the runner). The frontend polls `/api/tasks/`. There are both synchronous (`scan`, `merge_posts`, `ai_tag_all`) and background (`scan_bg`, `merge_bg`, `ai_tag_all_bg`) variants of the big operations — the `_bg` ones are the ones wired to the UI.
Tasks are cancellable: the stop button sets `Task.cancel_requested` (`/api/tasks/<id>/cancel/`) and every work fn calls `check_cancel(task)` (`gallery/utils.py`, raises `TaskCancelled`) between units of work — new long loops must do the same, and must never be cancelled mid-post. Finished cards dismiss individually (`/api/tasks/<id>/dismiss/`); `running` rows with no progress for 30 min are swept to `error` (the thread died with its worker). Bulk code should batch tag counts (`add_tags_to_post(..., recount=False)` + one `recount_tags()`), not call `Tag.update_count()` per tag/file.

## AI tagging
`run_ai_tagger` runs the WD14 ONNX tagger (`SmilingWolf/wd-vit-tagger-v3`, lazily downloaded + cached in `_get_wd14_model`, CUDA→CPU providers). For videos/PDFs it tags the generated thumbnail instead of the original. Always go through `apply_ai_tags(post)` (views.py), which wraps `run_ai_tagger` → `{'general': [...], 'character': [...]}`: general/rating tags (>= `AI_GENERAL_THRESHOLD` 0.35, first 40 in model order — unchanged) land in category `ai`; characters (>= `AI_CHARACTER_THRESHOLD` 0.85, top 12 by confidence, kept OUTSIDE the 40 cap — they sit last in the model's tag order and used to be cut off) land in category `character` (an existing `ai`-category character tag is promoted; user-set categories are never touched). It sets `Post.ai_tagged` and `Post.char_tagged`; `ai_tagged=True, char_tagged=False` = tagged by the old tagger, i.e. the candidates for a future characters-only re-tag (not built yet). Thresholds can be overridden in `local_settings.py`. NOTE: the venv has CPU-only `onnxruntime`, so the tagger runs on CPU despite the CUDA provider being requested (the P4 needs a CUDA-12 `onnxruntime-gpu` build; CUDA 13 builds dropped Pascal; FP16/int8 models gain nothing on it).
- Separately, `sync_sound_tag`/`has_audio_stream` (ffprobe-based, in `gallery/utils.py`) add/remove a `sound` tag on video posts; driven per-post during ingest and in bulk via the `sound_tag_all_bg` background task.

## Middleware & caching (`gallery/middleware.py`)
- Auth uses unique names (`booru_sessionid`, `booru_csrftoken`, session key `booru_authed`) because other local Django apps share the default cookie names across ports and logged each other out — don't revert to the defaults.
- `LoginRequiredMiddleware`: session password gate; `sw.js` and `/login|/logout` are exempt.
- `CacheHeadersMiddleware`: `/static/` cached a year (immutable), `/media/` a day; thumbnails cache-bust via a `?v=<mtime>` param added in `Photo.thumb_url`.

## Database backups (`gallery/backup.py`)
Only `db.sqlite3` is backed up (media files are never touched), via sqlite's online-backup API in one step (consistent under WAL, no stray `-wal/-shm`). Settings (enabled, max backups, interval hours, folder) live under the `backup` key of `prefs.json`; the default folder is `settings.BACKUP_DIR`. UI: the "⚙ settings" modal (`templates/gallery/_settings_modal.html`), which also hosts the maintenance buttons (tag sound, keys, re-tidy all, rebase paths). A scheduler thread started from `GalleryConfig.ready()` (gunicorn/runserver only) checks every 10 min; a flock'd lock file in the backup folder keeps the 4 workers from duplicating a run. Rotation only deletes `booru-*.sqlite3`. Restore (from the folder listing or an upload) validates the file, saves `pre-restore.sqlite3`, overwrites the live DB in place, carries `django_session` rows across, runs `migrate`, and re-creates its own Task row (the restored DB doesn't contain it).

## Settings panel debug overlays
"⚙ settings" → debug overlays toggles badges on every grid card (AI ✓/✗, characters-checked ✓/✗, folder name/none, post id). They are CSS-only: each card always carries a `.dbg` block (server template `_photo_grid.html` AND the JS `makeCard` in `index.html` — keep both in sync) and body classes `dbg-ai/ch/folder/id` (localStorage `dbgOverlay`, applied by `applyDbg()` in `base.html`) reveal them. `posts_json` supplies `ai`, `chars`, `folders`; the grid query prefetches `folders` for this — keep it.

## Thumbnails
`_regen_photo_thumb(photo, pct)` (views.py) is the single implementation behind the per-photo regen button and the bulk "pic thumbs" action (`/api/bulk-regen-thumb/`, covers that are pictures/gif/pdf); videos in a bulk selection go through `bulk_video_thumb` (needs the frame %).

## Duplicate detection
`duplicates` view groups posts by cover-image perceptual hash (`phash_distance`); videos only compare with videos (tighter threshold), GIF vs still is avoided, and `Post.not_dupes` (symmetrical M2M) pairs are skipped.

## Gotchas
- `README.md` is stale (describes no-login, CLIP/transformers tagging, swapping `run_ai_tagger`); trust this file and the code instead.
- User prefs (recent items, etc.) are stored in a plain `prefs.json` at `BASE_DIR` (`_prefs_path()` in `views.py`, `/api/prefs/`, `/api/pref/set/`), not in the DB, and it is untracked.
- `booru-main/` in the repo root is an untracked stray copy of the project; ignore it and don't edit files there.
- SQLite is tuned for concurrency in `settings.py` (WAL, `busy_timeout`, `transaction_mode=IMMEDIATE`) because gunicorn gevent workers otherwise serialize on the write lock.
- Gallery grid relies on `prefetch_related('tags','images')`; `Post.cover`/`image_count`/`has_video` read the prefetched cache to avoid N+1 queries — preserve the prefetch when touching those code paths.
- The hardcoded UNC path prefix in `duplicates`/`post_detail` (`\\192.168.1.50\@\Media_SRV\Photo\`) is the owner's file-server path for "open in explorer" links.
- `views.py` contains commented-out dead blocks and a legacy WD14-swap note; ignore them.
