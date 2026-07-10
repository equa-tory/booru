# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
A self-hosted, single-user "booru"-style photo/video/PDF gallery built on Django 6 + SQLite, with an htmx-driven frontend. No cloud, one shared password (`GALLERY_PASSWORD` in `booru/settings.py`). Files on disk are the source of truth; the DB only stores paths + metadata and never moves/renames originals except via explicit "organize"/"merge" actions.

## Commands
- Dev server: `python manage.py runserver`
- Migrate / make migrations: `python manage.py migrate` / `python manage.py makemigrations`
- Production (Linux/systemd): `./start.sh` — runs migrate then gunicorn (gevent workers, port 3001). Windows dev uses runserver.
- Tests: `python manage.py test` (note: `gallery/tests.py` is currently an empty stub — there is no real suite yet).
- Downloads watcher (optional sidecar): `python watch_downloads.py --downloads <dir> --inbox <media/inbox>` — auto-moves new downloads/zips into the inbox.

## Adding media
Two paths, both funnel through `gallery/utils.py::scan_inbox`:
1. Drop files into `media/inbox/` (subfolders OK), then click "scan inbox" (POST `/api/scan-bg/`).
2. Browser upload (POST `/api/upload/`) writes into `media/inbox/` then ingests.
Convention: `media/inbox/_/<folder>/` = one multi-image Post; loose files in `media/inbox/` = one Post each. Optional per-month tidy folders `inbox/YYYY-MM/DD/`.

## Architecture
- Single Django app `gallery`; project package `booru`.
- Models (`gallery/models.py`): `Post` (a gallery item) has many `Photo` (individual files, incl. video/PDF, `is_video` flag, `phash` for dedupe). `Tag` (M2M to Post) has a `category` (general/character/artist/meta/ai) and a denormalized `count` kept current via `Tag.update_count()`. `Folder` is either manual (explicit M2M posts) or "smart" (stores a query string re-run through `_build_post_qs`). `Task` is a DB-backed row tracking background jobs so progress survives page reloads and is visible to every gunicorn worker.
- Ingestion/thumbnailing lives in `gallery/utils.py`: `ingest_photo`, `create_post_from_files`, `make_thumb`/`make_video_thumb` (ffmpeg)/`make_pdf_thumb` (PyMuPDF→pdftoppm fallback), `compute_phash`/`phash_distance`. Thumbnails are keyed by md5 of the source path and written to `media/thumbs/`.
- Views (`gallery/views.py`) are the whole controller layer — page renders + a large JSON API (URLs in `gallery/urls.py`). Frontend is server-rendered templates (`templates/gallery/`) + `static/js/htmx.min.js`; there is no JS build step. Infinite scroll pulls JSON from `/api/posts/`.

## Search DSL (in `gallery/views.py`)
The gallery query is built by `_build_post_qs`; token parsing is `_parse_tag_tokens` + `_term_to_q`. Supported search-box syntax:
- `a b` = AND, `( a ~ b )` = OR group (braces and spaces are significant), `-tag` = NOT
- `tag~` = fuzzy (Levenshtein), `ta*1` = glob wildcard, `file:name` / `folder:name` = path substring
Reuse these helpers rather than writing new query logic. `random` sort uses a seeded deterministic shuffle (`_apply_seeded_order`) so gallery/scroll/prev-next stay consistent — the seed lives in the URL.

## Background tasks
Heavy operations (scan, merge, ai_tag, dupes) run via `_start_task(kind, fn)` which spawns a daemon thread and records progress in a `Task` row. Each thread MUST `connection.close()` when done (already handled in the runner). The frontend polls `/api/tasks/`. There are both synchronous (`scan`, `merge_posts`, `ai_tag_all`) and background (`scan_bg`, `merge_bg`, `ai_tag_all_bg`) variants of the big operations — the `_bg` ones are the ones wired to the UI.

## AI tagging
`run_ai_tagger` runs the WD14 ONNX tagger (`SmilingWolf/wd-vit-tagger-v3`, lazily downloaded + cached in `_get_wd14_model`, CUDA→CPU providers). For videos/PDFs it tags the generated thumbnail instead of the original. Tags applied land in the `ai` category and set `Post.ai_tagged=True`.

## Middleware & caching (`gallery/middleware.py`)
- `LoginRequiredMiddleware`: session password gate; `sw.js` and `/login|/logout` are exempt.
- `CacheHeadersMiddleware`: `/static/` cached a year (immutable), `/media/` a day; thumbnails cache-bust via a `?v=<mtime>` param added in `Photo.thumb_url`.

## Duplicate detection
`duplicates` view groups posts by cover-image perceptual hash (`phash_distance`); videos only compare with videos (tighter threshold), GIF vs still is avoided, and `Post.not_dupes` (symmetrical M2M) pairs are skipped.

## Gotchas
- SQLite is tuned for concurrency in `settings.py` (WAL, `busy_timeout`, `transaction_mode=IMMEDIATE`) because gunicorn gevent workers otherwise serialize on the write lock.
- Gallery grid relies on `prefetch_related('tags','images')`; `Post.cover`/`image_count`/`has_video` read the prefetched cache to avoid N+1 queries — preserve the prefetch when touching those code paths.
- The hardcoded UNC path prefix in `duplicates`/`post_detail` (`\\192.168.1.50\@\Media_SRV\Photo\`) is the owner's file-server path for "open in explorer" links.
- `views.py` contains commented-out dead blocks and a legacy WD14-swap note; ignore them.
