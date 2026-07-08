import os
import re
import json
import random
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.views.decorators.http import require_POST
from django.db.models import Q, Case, When, IntegerField, F
from django.db.models.functions import Mod
from django.conf import settings

from .models import Post, Photo, Tag, Task, Folder
from .utils import (scan_inbox, create_post_from_files, ingest_photo,
                    add_tags_to_post, delete_post, phash_distance, make_thumb,
                    make_video_thumb, retag_all_videos)


# ── Search syntax helpers ──────────────────────────────────────
# Supported in the search box (tokens are split on whitespace):
#   tag1 tag2        AND   — posts having both
#   ( a ~ b )        OR    — posts having at least one (braces + spaces matter)
#   -tag1            NOT   — posts without the tag
#   night~           FUZZY — Levenshtein-close tag names (night/fight/bright…)
#   ta*1             GLOB  — tags starting "ta" and ending "1" (* = anything)

def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy_tag_names(base):
    """Tag names within a small edit distance of `base` (for the `tag~` form)."""
    base = base.lower()
    budget = 2 if len(base) <= 4 else 3
    out = []
    for name in Tag.objects.filter(count__gt=0).values_list('name', flat=True):
        if abs(len(name) - len(base)) > budget:
            continue
        if _levenshtein(base, name.lower()) <= budget:
            out.append(name)
    return out


def _term_to_q(term):
    """Translate one search term into a Q over Post.tags.
    Returns (Q, multi) — multi=True means the term may match several tag
    names, so it has OR semantics (a post matches if ANY of its tags fit)."""
    term = term.strip()
    if not term:
        return None, False
    if term.startswith('file:') and len(term) > 5:   # search by file name
        return Q(images__file_path__icontains=term[5:]), True
    if term.startswith('folder:') and len(term) > 7:  # search by folder name
        return Q(images__file_path__icontains=term[7:]), True
    if '*' in term:                                   # wildcard glob
        pattern = '^' + re.escape(term).replace(r'\*', '.*') + '$'
        return Q(tags__name__iregex=pattern), True
    if term.endswith('~') and len(term) > 1:          # fuzzy
        names = _fuzzy_tag_names(term[:-1])
        return (Q(tags__name__in=names) if names else Q(pk__in=[])), True
    return Q(tags__name=term), False                  # plain exact


def _parse_tag_tokens(tokens):
    """Parse search tokens into (and_clauses, or_clauses, not_clauses)."""
    ands, ors, nots = [], [], []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == '(':                                # OR group: ( a ~ b )
            group = []
            i += 1
            while i < n and tokens[i] != ')':
                if tokens[i] != '~':
                    group.append(tokens[i])
                i += 1
            i += 1                                    # skip ')'
            q, has = Q(), False
            for g in group:
                sub, _ = _term_to_q(g)
                if sub is not None:
                    q |= sub
                    has = True
            if has:
                ors.append(q)
            continue
        if tok in ('~', ')'):                         # stray separators
            i += 1
            continue
        if tok.startswith('-') and len(tok) > 1:      # NOT
            sub, _ = _term_to_q(tok[1:])
            if sub is not None:
                nots.append(sub)
        else:
            sub, multi = _term_to_q(tok)
            if sub is not None:
                (ors if multi else ands).append(sub)
        i += 1
    return ands, ors, nots


def _apply_seeded_order(posts, seed):
    """Deterministic shuffle so 'random' stays stable across the gallery,
    infinite scroll and prev/next navigation (all share the URL's seed).

    Order by (id * a) % P with a large, seed-derived multiplier `a`. The
    multiplier must be large so that id*a wraps the modulus and actually
    permutes the order (a small `a` would leave rows in plain id order). This
    is computed entirely in SQL — far cheaper than a CASE/WHEN over every post,
    which made paging back to the gallery slow once the library grew large."""
    P = 2_000_003
    a = (seed * 2_654_435_761 + 12_345) % P or 1
    return posts.annotate(_rnd=Mod(F('id') * a, P)).order_by('_rnd', 'id')


def _ordered_by_ids(id_list):
    """Posts limited to id_list, preserving the given order (for 'similar')."""
    posts = Post.objects.prefetch_related('tags', 'images').filter(id__in=id_list)
    order = Case(*[When(id=pk, then=pos) for pos, pk in enumerate(id_list)],
                 output_field=IntegerField())
    return posts.order_by(order)


# ── Helpers ────────────────────────────────────────────────────

def _build_post_qs(request):
    q_tags     = request.GET.getlist('tag')
    min_rating = request.GET.get('min_rating', '')
    exact_rating = request.GET.get('rating', '')  # exact rating filter
    fav_only   = request.GET.get('fav', '')
    multi_only  = request.GET.get('multi_only', '')
    single_only = request.GET.get('single_only', '')
    folder_id   = request.GET.get('folder', '')   # manual folder filter (smart folders redirect via their saved query instead)
    sort_by     = request.GET.get('sort', 'new')   # new | old | rating | fav | random

    # Explicit id list (used by "find similar") — show exactly these posts in
    # the given order and skip every other filter.
    explicit_ids = request.GET.get('ids', '')
    if explicit_ids:
        id_list = [int(x) for x in explicit_ids.split(',') if x.strip().isdigit()]
        return _ordered_by_ids(id_list), q_tags, 'ids', '', ''

    posts = Post.objects.prefetch_related('tags', 'images').all()

    if q_tags:
        ands, ors, nots = _parse_tag_tokens(q_tags)
        for q in ands:
            posts = posts.filter(q)
        for q in ors:
            posts = posts.filter(q)
        for q in nots:
            posts = posts.exclude(q)
        if ands or ors or nots:
            posts = posts.distinct()

    if min_rating.isdigit():
        posts = posts.filter(rating__gte=int(min_rating))

    if exact_rating.isdigit():
        posts = posts.filter(rating=int(exact_rating))

    if fav_only == '1':
        posts = posts.filter(fav=True)

    if folder_id.isdigit():
        posts = posts.filter(folders__id=int(folder_id)).distinct()

    if multi_only == '1':
        from django.db.models import Count as _Count
        posts = posts.annotate(_img_count=_Count('images')).filter(_img_count__gt=1)

    if single_only == '1':
        from django.db.models import Count as _Count2
        if multi_only == '1':
            pass  # conflicting filters
        else:
            posts = posts.annotate(_img_count2=_Count2('images')).filter(_img_count2__lte=1)

    if sort_by == 'old':
        posts = posts.order_by('added_at')
    elif sort_by == 'rating':
        posts = posts.order_by('-rating', '-added_at')
    elif sort_by == 'fav':
        posts = posts.order_by('-fav', '-added_at')
    elif sort_by == 'rated_time':
        # most recently rated first; unrated posts fall to the bottom
        posts = posts.filter(rated_at__isnull=False).order_by('-rated_at')
    elif sort_by == 'faved_time':
        # most recently favorited first; only favorited posts
        posts = posts.filter(fav=True, faved_at__isnull=False).order_by('-faved_at')
    elif sort_by == 'random':
        seed = request.GET.get('seed', '')
        if seed.isdigit():
            posts = _apply_seeded_order(posts, int(seed))
        else:
            posts = posts.order_by('?')
    else:  # new (default)
        posts = posts.order_by('-added_at')

    return posts, q_tags, sort_by, multi_only, single_only


# ── Pages ──────────────────────────────────────────────────────

def index(request):
    # Random sort needs a stable seed in the URL so the gallery, infinite
    # scroll and prev/next all walk the SAME shuffle. Add one if missing.
    if (request.GET.get('sort') == 'random' and not request.GET.get('seed')
            and not request.GET.get('ids')
            and not request.headers.get('HX-Request')):
        p = request.GET.copy()
        p['seed'] = str(random.randint(1, 2_000_000_000))
        return redirect(f'{request.path}?{p.urlencode()}')

    posts, q_tags, sort_by, multi_only, single_only = _build_post_qs(request)

    # Full nav query string (tags + sort + filters + seed/ids, minus paging) so
    # links into a post carry the exact browsing context for prev/next.
    np = request.GET.copy()
    np.pop('page', None)
    np.pop('scroll', None)
    nav_qs = np.urlencode()

    paginator = Paginator(posts, 40)
    page_obj  = paginator.get_page(request.GET.get('page', 1))

    sort_tags = request.GET.get('sort_tags', 'count')  # 'count' or 'name'

    # custom category order: meta, char, art, gen, ai
    from django.db.models import Case, When, IntegerField, Value
    cat_order = Case(
        When(category='meta', then=Value(0)),
        When(category='character', then=Value(1)),
        When(category='artist', then=Value(2)),
        When(category='general', then=Value(3)),
        When(category='ai', then=Value(4)),
        default=Value(9), output_field=IntegerField(),
    )

    if q_tags:
        from django.db.models import Count
        post_ids     = posts.values_list('id', flat=True)
        base_qs_tags = (Tag.objects
            .filter(posts__id__in=post_ids)
            .annotate(filtered_count=Count('posts', distinct=True))
            .filter(filtered_count__gt=0)
            .annotate(cat_rank=cat_order))
        if sort_tags == 'name':
            popular_tags = base_qs_tags.order_by('-fav', 'cat_rank', 'name')
        else:
            popular_tags = base_qs_tags.order_by('-fav', 'cat_rank', '-filtered_count')
    else:
        base = Tag.objects.filter(count__gt=0).annotate(cat_rank=cat_order)
        if sort_tags == 'name':
            popular_tags = base.order_by('-fav', 'cat_rank', 'name')
        else:
            popular_tags = base.order_by('-fav', 'cat_rank', '-count')

    # Only render the top N tags in the sidebar. The full list lives behind the
    # search box / tag sheet / "edit tags" page. Rendering thousands of <a>
    # tags into every gallery page made paging back to the gallery slow on the
    # phone (the markup is parsed even though the sidebar is hidden on mobile).
    tag_total    = popular_tags.count()
    # fast mode (cookie set from the more menu) renders far fewer tags so the
    # gallery page is lighter to parse on a phone.
    tag_cap = 40 if request.COOKIES.get('fastMode') == '1' else 300
    popular_tags = list(popular_tags[:tag_cap])
    is_htmx = request.headers.get('HX-Request')
    scroll_mode = request.GET.get('scroll', '0') == '1'
    if is_htmx:
        return render(request, 'gallery/_photo_grid.html', {
            'page_obj': page_obj, 'q_tags': q_tags, 'nav_qs': nav_qs,
            'scroll_mode': scroll_mode, 'request': request,
        })


    # Build base query string (everything except page) for pagination links
    p = request.GET.copy()
    p.pop('page', None)
    base_qs = ('&' + p.urlencode()) if p else ''

    return render(request, 'gallery/index.html', {
        'page_obj': page_obj,
        'popular_tags': popular_tags,
        'tag_total': tag_total,
        'tag_cap': tag_cap,
        'q_tags': q_tags,
        'nav_qs': nav_qs,
        'min_rating': request.GET.get('min_rating', ''),
        'exact_rating': request.GET.get('rating', ''),
        'fav_only':   request.GET.get('fav', ''),
        'filtering_active': bool(q_tags),
        'scroll_mode': scroll_mode,
        'base_qs': base_qs,
        'sort_tags': sort_tags,
        'sort_by': sort_by,
        'multi_only': request.GET.get('multi_only',''),
        'single_only': request.GET.get('single_only',''),
        'sort_options': [('new','newest'),('old','oldest'),('rating','rating'),('fav','fav first'),('rated_time','recently rated'),('faved_time','recently liked'),('random','random')],
        'folders': Folder.objects.all(),
        'active_folder': request.GET.get('folder', ''),
        'current_query': p.urlencode(),  # current filters, minus page — used by "save as smart folder"
    })


def posts_json(request):
    """JSON API for infinite scroll — returns page of posts as JSON."""
    posts, q_tags, sort_by, multi_only, single_only = _build_post_qs(request)
    paginator = Paginator(posts, 40)
    page_obj  = paginator.get_page(request.GET.get('page', 1))

    # Full nav context (minus paging) so each card links back into the same
    # sorted/filtered list for correct prev/next.
    np = request.GET.copy()
    np.pop('page', None)
    np.pop('scroll', None)
    nav_qs = np.urlencode()
    suffix = ('?' + nav_qs) if nav_qs else ''

    result = []
    for post in page_obj:
        cover = post.cover
        if not cover:
            continue
        result.append({
            'id':         post.pk,
            'rating':     post.rating,
            'fav':        post.fav,
            'tag_count':  len(post.tags.all()),
            'img_count':  post.image_count,
            'thumb_url':  cover.thumb_url,
            'is_video':   cover.is_video,
            'has_video':  post.has_video,
            'has_gif':    post.has_gif,
            'url':        f'/post/{post.pk}/{suffix}',
        })

    # legacy tag-only query string (kept for back-compat)
    tag_qs = '?' + '&'.join(f'tag={t}' for t in q_tags) if q_tags else ''

    return JsonResponse({
        'posts':    result,
        'page':     page_obj.number,
        'has_next': page_obj.has_next(),
        'total':    paginator.count,
        'tag_qs':   tag_qs,
    })


def post_detail(request, pk):
    post   = get_object_or_404(Post, pk=pk)
    q_tags = request.GET.getlist('tag')
    images = list(post.images.order_by('order', 'id'))

    # Store referrer for back button — prefer HTTP_REFERER that points to index
    referer  = request.META.get('HTTP_REFERER', '')
    back_url = ''
    if referer and '/post/' not in referer:
        # came from index — use it
        back_url = referer
    elif 'back_url' in request.session:
        back_url = request.session['back_url']
    # save back_url in session for post-to-post navigation
    if back_url:
        request.session['back_url'] = back_url

    # Full search query string (tags + filters + sort) for neighbor navigation
    search_params = []
    for key in ('tag', 'sort', 'min_rating', 'rating', 'fav',
                'multi_only', 'single_only', 'folder', 'seed', 'ids'):
        for val in request.GET.getlist(key):
            search_params.append(f'{key}={val}')
    search_qs = '&'.join(search_params)

    return render(request, 'gallery/detail.html', {
        'post': post, 'images': images, 'q_tags': q_tags,
        'back_url': back_url,
        'search_qs': search_qs,
    })


def duplicates(request):
    # Compare at POST level using cover image phash.
    # - Only different posts are ever compared (within-post images are never
    #   treated as duplicates of each other).
    # - GIFs are never compared against non-GIF images (animated vs still).
    # - not_dupes relationships between posts are respected.
    posts = list(Post.objects.prefetch_related('not_dupes', 'images').all())

    def _net_path(file_path):
        r"""Build the same \\server\share path the detail page shows."""
        rel = os.path.relpath(file_path, settings.MEDIA_ROOT).replace(os.sep, '/')
        prefix = '\\\\192.168.1.50\\@\\Media_SRV\\Photo\\'
        return prefix + rel.replace('/', '\\')

    # Build a list of post_id -> cover info
    from .utils import compute_phash
    post_data = []
    for post in posts:
        cover = post.cover
        if not cover:
            continue
        # lazy backfill: covers (incl. videos) missing a pHash get one from
        # their thumbnail so they can take part in duplicate detection.
        if not cover.phash:
            base = cover.thumb_path or cover.file_path
            if base and os.path.exists(base):
                ph = compute_phash(base)
                if ph:
                    cover.phash = ph
                    cover.save(update_fields=['phash'])
        if not cover.phash:
            continue
        src = cover.thumb_path or cover.file_path
        rel = os.path.relpath(src, settings.MEDIA_ROOT)
        post_data.append({
            'post_id':   post.pk,
            'post':      post,
            'phash':     cover.phash,
            'is_gif':    cover.is_gif,
            'is_video':  cover.is_video,
            'thumb_url': '/media/' + rel.replace(os.sep, '/'),
            'title':     post.title,
            'img_count': post.image_count,
            'file_size': cover.file_size,
            'file_path': _net_path(cover.file_path),
        })

    # Build ignored pairs set for O(1) lookup
    ignored_pairs = set()
    for post in posts:
        for other in post.not_dupes.all():
            pair = tuple(sorted([post.pk, other.pk]))
            ignored_pairs.add(pair)

    def _comparable(a, b):
        # video only ever compares with video; image/gif cross-compare freely.
        if a['is_video'] or b['is_video']:
            return a['is_video'] and b['is_video']
        return True

    groups = []
    used   = set()
    for i, p in enumerate(post_data):
        if p['post_id'] in used: continue
        group = [p]
        for j, q in enumerate(post_data):
            if i == j or q['post_id'] in used: continue
            # never compare two photos from the same post
            if p['post_id'] == q['post_id']: continue
            if not _comparable(p, q): continue
            pair = tuple(sorted([p['post_id'], q['post_id']]))
            if pair in ignored_pairs: continue
            # Video thumbnails (often dark/title frames) collide far too easily,
            # so require a near-exact match for video-vs-video; images/gifs keep
            # the looser perceptual threshold.
            both_video = p['is_video'] and q['is_video']
            threshold = 2 if both_video else 8
            if phash_distance(p['phash'], q['phash']) <= threshold:
                group.append(q)
                used.add(q['post_id'])
        if len(group) > 1:
            used.add(p['post_id'])
            groups.append(group)

    return render(request, 'gallery/duplicates.html', {'groups': groups})


# ── Scan / upload ──────────────────────────────────────────────

# ── Background tasks ────────────────────────────────────────────
import threading
import traceback
from django.utils import timezone


def _start_task(kind, fn, total=0, message=''):
    """Create a Task row and run `fn(task)` in a background thread so the work
    survives the page being closed and its progress is visible to every worker
    (state lives in the DB)."""
    task = Task.objects.create(kind=kind, total=total, message=message)
    tid = task.id

    def runner():
        from django.db import connection
        try:
            t = Task.objects.get(pk=tid)
            fn(t)
            t.refresh_from_db()
            if t.status == 'running':
                t.status = 'done'
                t.finished_at = timezone.now()
                t.save()
        except Exception as e:
            traceback.print_exc()
            try:
                t = Task.objects.get(pk=tid)
                t.status = 'error'
                t.error = str(e)[:2000]
                t.finished_at = timezone.now()
                t.save()
            except Exception:
                pass
        finally:
            connection.close()

    threading.Thread(target=runner, daemon=True).start()
    return task


def tasks_list(request):
    """Active tasks + recently finished ones (last few minutes), with elapsed
    time. The frontend polls this to show progress / notifications."""
    cutoff = timezone.now() - timezone.timedelta(minutes=5)
    qs = Task.objects.filter(Q(status='running') | Q(finished_at__gte=cutoff))[:20]
    out = []
    for t in qs:
        out.append({
            'id': t.id, 'kind': t.kind, 'status': t.status,
            'done': t.done, 'total': t.total, 'message': t.message,
            'error': t.error, 'elapsed': t.elapsed,
            'finished': bool(t.finished_at),
        })
    return JsonResponse({'tasks': out})


@require_POST
def tasks_clear(request):
    """Dismiss finished/errored tasks from the list."""
    Task.objects.exclude(status='running').delete()
    return JsonResponse({'ok': True})


# ── Background variants of the heavy operations ─────────────────
def _do_scan(task):
    removed = 0
    for photo in Photo.objects.all():
        if not os.path.exists(photo.file_path):
            if photo.thumb_path and os.path.exists(photo.thumb_path):
                try: os.remove(photo.thumb_path)
                except OSError: pass
            photo.delete(); removed += 1
    empty = Post.objects.filter(images__isnull=True)
    removed += empty.count(); empty.delete()
    for tag in Tag.objects.all(): tag.update_count()
    Tag.objects.filter(count=0).delete()

    task.message = 'scanning inbox…'; task.save(update_fields=['message'])
    new_posts, extend_posts = scan_inbox()
    task.total = len(new_posts) + len(extend_posts); task.save(update_fields=['total'])
    added = 0
    for i, (title, paths) in enumerate(new_posts):
        create_post_from_files(paths, title=title)
        added += len(paths)
        task.done = i + 1; task.message = f'added {added} file(s)'; task.save(update_fields=['done', 'message'])
    base = len(new_posts)
    for j, (post, paths) in enumerate(extend_posts):
        start_order = post.images.count()
        for k, path in enumerate(sorted(paths)):
            ingest_photo(path, post, order=start_order + k)
        added += len(paths)
        task.done = base + j + 1; task.save(update_fields=['done'])
    # fix placeholder video/pdf thumbs
    for photo in Photo.objects.filter(is_video=True):
        if not photo.thumb_path or not os.path.exists(photo.thumb_path) or os.path.getsize(photo.thumb_path) < 5000:
            thumb = make_video_thumb(photo.file_path)
            if thumb: photo.thumb_path = thumb; photo.save(update_fields=['thumb_path'])
    task.message = f'added {added}, removed {removed} (total {Post.objects.count()})'
    task.save(update_fields=['message'])


@require_POST
def scan_bg(request):
    return JsonResponse({'task_id': _start_task('scan', _do_scan, message='scan starting…').id})


def _merge_one_group(group):
    from .utils import move_post_to_folder
    group = [g for g in group if g]
    if len(group) < 2:
        return
    target_id, source_ids = group[0], group[1:]
    try:
        target = Post.objects.get(pk=target_id)
    except Post.DoesNotExist:
        return
    max_order = target.images.count()
    for src_id in source_ids:
        try:
            src = Post.objects.get(pk=src_id)
        except Post.DoesNotExist:
            continue
        try:
            n = src.images.count()
            for i, photo in enumerate(src.images.order_by('order', 'id')):
                photo.post = target; photo.order = max_order + i
                photo.save(update_fields=['post', 'order'])
            max_order += n
            for tag in src.tags.all(): target.tags.add(tag)
            src.delete()
        except Exception as e:
            print(f'merge error src {src_id}: {e}')
    for tag in target.tags.all(): tag.update_count()
    cover = target.cover
    if cover:
        folder_base = os.path.splitext(os.path.basename(cover.file_path))[0]
        if not target.title:
            target.title = folder_base; target.save(update_fields=['title'])
        try: move_post_to_folder(target, folder_base)
        except Exception as e: print(f'merge folder move error: {e}')


@require_POST
def merge_bg(request):
    """Merge many groups in the background. body: {groups: [[target, src,…], …]}"""
    data = json.loads(request.body)
    groups = [g for g in data.get('groups', []) if len(g) >= 2]
    if not groups:
        return JsonResponse({'error': 'no groups'}, status=400)

    def work(task):
        import time as _t
        for i, group in enumerate(groups):
            _merge_one_group(group)
            task.done = i + 1
            task.message = f'merged {i + 1}/{len(groups)} group(s)'
            task.save(update_fields=['done', 'message'])
            _t.sleep(0)   # let other requests (and the progress poll) interleave

    return JsonResponse({'task_id': _start_task('merge', work, total=len(groups),
                                                message='merging…').id})


@require_POST
def ai_tag_all_bg(request):
    def work(task):
        import time as _t
        posts = list(Post.objects.filter(ai_tagged=False))
        task.total = len(posts); task.save(update_fields=['total'])
        for i, post in enumerate(posts):
            cover = post.images.order_by('order', 'id').first()
            if cover:
                try:
                    tags = run_ai_tagger(cover.file_path, cover.thumb_path)
                    add_tags_to_post(post, tags, category='ai')
                    post.ai_tagged = True; post.save(update_fields=['ai_tagged'])
                except Exception as e:
                    print(f'ai-tag error post {post.id}: {e}')
            task.done = i + 1
            task.message = f'tagged {i + 1}/{len(posts)} post(s)'
            task.save(update_fields=['done', 'message'])
            _t.sleep(0)
    return JsonResponse({'task_id': _start_task('ai_tag', work, message='ai tagging…').id})


@require_POST
def scan(request):
    # Remove posts whose ALL images are gone; remove orphan images
    removed = 0
    for photo in Photo.objects.all():
        if not os.path.exists(photo.file_path):
            if photo.thumb_path and os.path.exists(photo.thumb_path):
                try: os.remove(photo.thumb_path)
                except OSError: pass
            photo.delete()
            removed += 1
    # Delete posts that now have no images
    empty = Post.objects.filter(images__isnull=True)
    removed += empty.count()
    empty.delete()

    for tag in Tag.objects.all(): tag.update_count()
    Tag.objects.filter(count=0).delete()

    new_posts, extend_posts = scan_inbox()
    added = 0
    for title, paths in new_posts:
        create_post_from_files(paths, title=title)
        added += len(paths)
    # extend existing multi-posts with new files
    for post, paths in extend_posts:
        start_order = post.images.count()
        for i, path in enumerate(sorted(paths)):
            ingest_photo(path, post, order=start_order + i)
        added += len(paths)

    # Retag any videos/PDFs that still have placeholder thumbs (size check)
    import os as _os
    for photo in Photo.objects.filter(is_video=True):
        if not photo.thumb_path or not _os.path.exists(photo.thumb_path) or            _os.path.getsize(photo.thumb_path) < 5000:
            thumb = make_video_thumb(photo.file_path)
            if thumb:
                photo.thumb_path = thumb
                photo.save(update_fields=['thumb_path'])

    # Also retag PDFs with placeholder thumbs
    for photo in Photo.objects.filter(file_path__iendswith='.pdf'):
        if not photo.thumb_path or not _os.path.exists(photo.thumb_path) or            _os.path.getsize(photo.thumb_path) < 5000:
            from .utils import make_pdf_thumb
            thumb = make_pdf_thumb(photo.file_path)
            if thumb:
                photo.thumb_path = thumb
                photo.save(update_fields=['thumb_path'])

    return JsonResponse({'added': added, 'removed': removed,
                         'total': Post.objects.count()})


@require_POST
def upload(request):
    import uuid
    files      = request.FILES.getlist('photos')  # includes videos
    as_one     = request.POST.get('as_one_post') == '1'
    do_ai_tag  = request.POST.get('ai_tag') == '1'
    inbox      = os.path.join(settings.MEDIA_ROOT, 'inbox')
    saved      = []

    if as_one and files:
        # Save into inbox/_/<random_folder>/ so scan treats it as one post
        folder_name = uuid.uuid4().hex[:12]
        dest_dir    = os.path.join(inbox, '_', folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            dest = os.path.join(dest_dir, f.name)
            base, ext = os.path.splitext(f.name)
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
                i += 1
            with open(dest, 'wb') as out:
                for chunk in f.chunks(): out.write(chunk)
            saved.append(dest)
    else:
        # Save each file flat into inbox/
        for f in files:
            dest = os.path.join(inbox, f.name)
            base, ext = os.path.splitext(f.name)
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(inbox, f"{base}_{i}{ext}")
                i += 1
            with open(dest, 'wb') as out:
                for chunk in f.chunks(): out.write(chunk)
            saved.append(dest)

    posts_created = []
    if as_one and saved:
        from .utils import scan_inbox
        # folder already in right place — just ingest it
        post = create_post_from_files(saved, title='')
        posts_created.append(post)
    else:
        for path in saved:
            post = create_post_from_files([path])
            posts_created.append(post)

    if do_ai_tag:
        from .views import run_ai_tagger
        for post in posts_created:
            try:
                # tag using first image
                cover = post.images.first()
                if cover:
                    tags = run_ai_tagger(cover.file_path, cover.thumb_path)
                    add_tags_to_post(post, tags, category='ai')
                    post.ai_tagged = True
                    post.save(update_fields=['ai_tagged'])
            except Exception as e:
                print(f"AI tag error: {e}")

    return JsonResponse({'added': len(posts_created), 'as_one': as_one})


# ── Post actions ───────────────────────────────────────────────

@require_POST
def tag_post(request, pk):
    post     = get_object_or_404(Post, pk=pk)
    data     = json.loads(request.body)
    action   = data.get('action', 'add')
    tag_name = data.get('tag', '').strip().lower().replace(' ', '_')
    category = data.get('category', 'general')

    if not tag_name:
        return JsonResponse({'error': 'empty tag'}, status=400)

    if action == 'add':
        add_tags_to_post(post, [tag_name], category)
    elif action == 'remove':
        try:
            tag = Tag.objects.get(name=tag_name)
            post.tags.remove(tag)
            tag.update_count()
            if tag.count == 0: tag.delete()
        except Tag.DoesNotExist:
            pass

    tags = list(post.tags.order_by('category', 'name').values('name', 'category'))
    return JsonResponse({'tags': tags})

   # @require_POST
   # def tag_post(request, pk):
   #     post     = get_object_or_404(Post, pk=pk)
   #     data     = json.loads(request.body)
   #     action   = data.get('action', 'add')
   #     tag_name = data.get('tag', '').strip().lower().replace(' ', '_')
   #     category = data.get('category', 'general')

   #     if not tag_name:
   #         return JsonResponse({'error': 'empty tag'}, status=400)

   #     if action == 'add':
   #         add_tags_to_post(post, [tag_name], category)
   #     elif action == 'remove':
   #         try:
   #             tag = Tag.objects.get(name=tag_name)
   #             post.tags.remove(tag)
   #             tag.update_count()
   #             if tag.count == 0: tag.delete()
   #         except Tag.DoesNotExist:
   #             pass

   #     tags = list(post.tags.order_by('category', 'name').values('name', 'category'))
   #     return JsonResponse({'tags': tags})


@require_POST
def rate_post(request, pk):
    post   = get_object_or_404(Post, pk=pk)
    data   = json.loads(request.body)
    rating = data.get('rating', None)
    fav    = data.get('fav', None)
    fields = []
    if rating is not None:
        new_rating = max(0, min(5, int(rating)))
        if new_rating != post.rating:
            post.rated_at = timezone.now()
            fields.append('rated_at')
        post.rating = new_rating
        fields.append('rating')
    if fav is not None:
        new_fav = bool(fav)
        if new_fav and not post.fav:
            post.faved_at = timezone.now()   # stamp only when newly favorited
            fields.append('faved_at')
        post.fav = new_fav
        fields.append('fav')
    post.save(update_fields=fields or ['rating', 'fav'])
    return JsonResponse({'rating': post.rating, 'fav': post.fav})


@require_POST
def delete_post_view(request, pk):
    post      = get_object_or_404(Post, pk=pk)
    also_file = json.loads(request.body).get('delete_file', False)
    delete_post(post, also_files=also_file)
    return JsonResponse({'ok': True})


@require_POST
def ai_tag_post(request, pk):
    post = get_object_or_404(Post, pk=pk)
    try:
        cover = post.images.order_by('order', 'id').first()
        if not cover:
            return JsonResponse({'error': 'no images', 'ok': False}, status=400)
        tags = run_ai_tagger(cover.file_path, cover.thumb_path)
        add_tags_to_post(post, tags, category='ai')
        post.ai_tagged = True
        post.save(update_fields=['ai_tagged'])
        return JsonResponse({'tags': tags, 'ok': True})
    except Exception as e:
        return JsonResponse({'error': str(e), 'ok': False}, status=500)


# ── Bulk actions (operate on posts) ───────────────────────────

@require_POST
def bulk_action(request):
    data   = json.loads(request.body)
    ids    = data.get('ids', [])
    action = data.get('action', '')
    posts  = Post.objects.filter(id__in=ids)

    if action == 'add_tag':
        tag_name = data.get('tag', '').strip().lower().replace(' ', '_')
        category = data.get('category', 'general')
        if not tag_name:
            return JsonResponse({'error': 'empty tag'}, status=400)
        for post in posts:
            add_tags_to_post(post, [tag_name], category)
        return JsonResponse({'ok': True})

    if action == 'remove_tag':
        tag_name = data.get('tag', '').strip().lower().replace(' ', '_')
        try:
            tag = Tag.objects.get(name=tag_name)
            for post in posts: post.tags.remove(tag)
            tag.update_count()
            if tag.count == 0: tag.delete()
        except Tag.DoesNotExist:
            pass
        return JsonResponse({'ok': True})

    if action == 'rate':
        posts.update(rating=max(0, min(5, int(data.get('rating', 0)))))
        return JsonResponse({'ok': True})

    if action == 'fav':
        posts.update(fav=bool(data.get('fav', True)))
        return JsonResponse({'ok': True})

    if action == 'delete':
        also_file = data.get('delete_file', False)
        for post in posts:
            delete_post(post, also_files=also_file)
        return JsonResponse({'ok': True})

    if action == 'add_to_folder':
        try:
            folder = Folder.objects.get(pk=data.get('folder_id'), is_smart=False)
        except Folder.DoesNotExist:
            return JsonResponse({'error': 'folder not found'}, status=404)
        folder.posts.add(*posts)
        return JsonResponse({'ok': True})

    if action == 'remove_from_folder':
        try:
            folder = Folder.objects.get(pk=data.get('folder_id'), is_smart=False)
        except Folder.DoesNotExist:
            return JsonResponse({'error': 'folder not found'}, status=404)
        folder.posts.remove(*posts)
        return JsonResponse({'ok': True})

    return JsonResponse({'error': 'unknown action'}, status=400)


@require_POST
def folder_create(request):
    data     = json.loads(request.body)
    name     = data.get('name', '').strip()
    is_smart = bool(data.get('is_smart', False))
    query    = data.get('query', '').strip() if is_smart else ''
    if not name:
        return JsonResponse({'error': 'name required'}, status=400)
    folder = Folder.objects.create(name=name, is_smart=is_smart, query=query)
    return JsonResponse({'id': folder.id, 'name': folder.name,
                          'is_smart': folder.is_smart, 'query': folder.query})


@require_POST
def folder_delete(request, pk):
    Folder.objects.filter(pk=pk).delete()
    return JsonResponse({'ok': True})


@require_POST
def ai_tag_all(request):
    posts   = Post.objects.filter(ai_tagged=False)
    results = []
    for post in posts:
        cover = post.images.order_by('order', 'id').first()
        if not cover:
            continue
        try:
            tags = run_ai_tagger(cover.file_path, cover.thumb_path)
            add_tags_to_post(post, tags, category='ai')
            post.ai_tagged = True
            post.save(update_fields=['ai_tagged'])
            results.append({'id': post.id, 'tags': tags})
        except Exception as e:
            results.append({'id': post.id, 'error': str(e)})
    return JsonResponse({'results': results, 'count': len(results)})


# ── AI tagger ─────────────────────────────────────────────────

def run_ai_tagger(file_path, thumb_path=''):
    """Run the WD14 tagger on an image.

    If the original can't be opened as an image (e.g. it's an mp4/video or a
    pdf), fall back to the generated thumbnail so video/pdf posts can still be
    auto-tagged from their preview frame.
    """
    from PIL import Image
    import numpy as np

    def _load(path):
        img = Image.open(path).convert('RGBA')
        bg  = Image.new('RGBA', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert('RGB')

    img = None
    ext = os.path.splitext(file_path)[1].lower()
    is_unopenable = ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v',
                            '.m4a', '.3gp', '.pdf'}
    # For known non-image formats, go straight to the thumbnail.
    if is_unopenable and thumb_path and os.path.exists(thumb_path):
        img = _load(thumb_path)
    else:
        try:
            img = _load(file_path)
        except Exception:
            if thumb_path and os.path.exists(thumb_path):
                img = _load(thumb_path)
            else:
                raise

    model, tags_list = _get_wd14_model()
    target = 448
    img.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new('RGB', (target, target), (255, 255, 255))
    canvas.paste(img, ((target-img.width)//2, (target-img.height)//2))
    arr   = np.array(canvas, dtype=np.float32)[:, :, ::-1][np.newaxis, :]
    probs = model.run(None, {model.get_inputs()[0].name: arr})[0][0]
    return [tags_list[i].replace(' ', '_') for i, s in enumerate(probs) if s >= 0.35][:40]


_wd14_cache = None
def _get_wd14_model():
    global _wd14_cache
    if _wd14_cache: return _wd14_cache
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort, csv
    model_path = hf_hub_download('SmilingWolf/wd-vit-tagger-v3', 'model.onnx')
    tags_path  = hf_hub_download('SmilingWolf/wd-vit-tagger-v3', 'selected_tags.csv')
    session = ort.InferenceSession(model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    tags = [row['name'] for row in csv.DictReader(open(tags_path, encoding='utf-8'))]
    _wd14_cache = (session, tags)
    return _wd14_cache


# ── Tag helpers ────────────────────────────────────────────────

def tag_search(request):
    q    = request.GET.get('q', '').lower()
    tags = Tag.objects.filter(name__icontains=q).order_by('-count')[:20]
    return JsonResponse({'tags': [{'name': t.name, 'count': t.count,
                                    'category': t.category} for t in tags]})


def tags_all(request):
    # Same ordering the desktop sidebar uses: pinned first, then by category
    # rank (meta, char, art, gen, ai), then alphabetically by name — so the
    # mobile tag sheet is grouped by category + name instead of name-only.
    from django.db.models import Case, When, IntegerField, Value
    cat_order = Case(
        When(category='meta', then=Value(0)),
        When(category='character', then=Value(1)),
        When(category='artist', then=Value(2)),
        When(category='general', then=Value(3)),
        When(category='ai', then=Value(4)),
        default=Value(9), output_field=IntegerField(),
    )
    tags = (Tag.objects.filter(count__gt=0)
            .annotate(cat_rank=cat_order)
            .order_by('-fav', 'cat_rank', 'name'))
    return JsonResponse({'tags': [{'name': t.name, 'count': t.count,
                                    'category': t.category, 'fav': t.fav} for t in tags]})


def post_neighbors(request, pk):
    # Respect the same filters/sort as the gallery so prev/next match browsing order
    posts, q_tags, sort_by, _, __ = _build_post_qs(request)
    ids = list(posts.values_list('id', flat=True))
    try:    idx = ids.index(pk)
    except ValueError:
        return JsonResponse({'prev': None, 'next': None, 'index': 0, 'total': len(ids)})

    prev_id = ids[idx-1] if idx > 0         else None
    next_id = ids[idx+1] if idx < len(ids)-1 else None

    # Provide ONE lightweight thumbnail per neighbour for preloading + the
    # swipe preview. (Previously this returned up to 3 FULL-size images per
    # side = 6 big downloads per post view, which made phones crawl.)
    def cover_urls(post_id):
        if not post_id:
            return []
        try:
            p = Post.objects.prefetch_related('images').get(pk=post_id)
        except Post.DoesNotExist:
            return []
        cover = p.cover
        return [cover.thumb_url] if cover else []

    return JsonResponse({
        'prev':  prev_id,
        'next':  next_id,
        'index': idx, 'total': len(ids),
        'prev_preload': cover_urls(prev_id),
        'next_preload': cover_urls(next_id),
    })



@require_POST
def delete_photo_view(request, pk):
    """Delete a single image from a post (not the whole post)."""
    from .models import Photo as P
    photo     = get_object_or_404(P, pk=pk)
    also_file = json.loads(request.body).get('delete_file', False)
    post      = photo.post
    if also_file and os.path.exists(photo.file_path):
        try: os.remove(photo.file_path)
        except OSError: pass
    if photo.thumb_path and os.path.exists(photo.thumb_path):
        try: os.remove(photo.thumb_path)
        except OSError: pass
    photo.delete()
    # if post is now empty, delete it too
    if post and post.images.count() == 0:
        post.delete()
    return JsonResponse({'ok': True})

@require_POST
@require_POST
def toggle_tag_fav(request, name):
    """Toggle favourite on a tag."""
    tag = get_object_or_404(Tag, name=name)
    tag.fav = not tag.fav
    tag.save(update_fields=['fav'])
    return JsonResponse({'fav': tag.fav})


def random_post(request):
    """Redirect to a random post, respecting current filters."""
    import random
    posts, q_tags, _, __, ___ = _build_post_qs(request)
    ids = list(posts.values_list('id', flat=True))
    if not ids:
        return JsonResponse({'error': 'no posts'}, status=404)
    pk = random.choice(ids)
    tag_qs = '?' + '&'.join(f'tag={t}' for t in q_tags) if q_tags else ''
    from django.shortcuts import redirect
    return redirect(f'/post/{pk}/{tag_qs}')



@require_POST
def merge_posts(request):
    """Merge one or more posts into a target post, then move all files into
    a single folder under inbox/_/<first_file_name>/."""
    from .utils import move_post_to_folder
    data      = json.loads(request.body)
    target_id = data.get('target')
    source_ids = [i for i in data.get('sources', []) if i != target_id]
    if not target_id or not source_ids:
        return JsonResponse({'error': 'need target and sources'}, status=400)
    target = get_object_or_404(Post, pk=target_id)
    max_order = target.images.count()
    merged = 0
    failed = []
    for src_id in source_ids:
        try:
            src = Post.objects.get(pk=src_id)
        except Post.DoesNotExist:
            failed.append(src_id)
            continue
        # Re-grouping each source in its own try so a single bad post (e.g. a
        # missing file or a DB hiccup) can't abort the whole batch — this is
        # what caused "selected 40+ but only a few migrated".
        try:
            n_imgs = src.images.count()
            for i, photo in enumerate(src.images.order_by('order', 'id')):
                photo.post  = target
                photo.order = max_order + i
                photo.save(update_fields=['post', 'order'])
            max_order += n_imgs
            for tag in src.tags.all():
                target.tags.add(tag)
            src.delete()
            merged += 1
        except Exception as e:
            print(f"merge error for source {src_id}: {e}")
            failed.append(src_id)
            continue
    for tag in target.tags.all():
        tag.update_count()

    # Move all files into a folder named after the first image
    cover = target.cover
    if cover:
        folder_base = os.path.splitext(os.path.basename(cover.file_path))[0]
        if not target.title:
            target.title = folder_base
            target.save(update_fields=['title'])
        try:
            move_post_to_folder(target, folder_base)
        except Exception as e:
            print(f"merge folder move error: {e}")

    return JsonResponse({'ok': True, 'post_id': target.pk,
                         'merged': merged, 'failed': failed})


@require_POST
def organize_existing_merged(request):
    """One-off: move all multi-image posts into their own inbox/_/ folders."""
    from .utils import move_post_to_folder
    from django.db.models import Count
    moved = 0
    multi = Post.objects.annotate(c=Count('images')).filter(c__gt=1)
    for post in multi:
        cover = post.cover
        if not cover:
            continue
        # skip if already in a _/folder/
        if os.sep + '_' + os.sep in cover.file_path:
            continue
        folder_base = post.title or os.path.splitext(os.path.basename(cover.file_path))[0]
        try:
            move_post_to_folder(post, folder_base)
            moved += 1
        except Exception as e:
            print(f"organize error post {post.pk}: {e}")
    return JsonResponse({'ok': True, 'moved': moved})


@require_POST
def organize_singles(request):
    """Tidy loose single-image files sitting directly in the inbox ROOT into
    per-month folders (inbox/YYYY-MM/) so the root isn't one giant directory
    that's slow to open in a file manager. Multi-image posts (inbox/_/…) and
    files already inside a subfolder are left alone. Idempotent + safe: it just
    moves the file and updates its stored path (thumbnails are unaffected)."""
    inbox = os.path.join(settings.MEDIA_ROOT, 'inbox')
    inbox_norm = os.path.normpath(inbox)
    moved = 0
    for post in Post.objects.prefetch_related('images').all():
        imgs = list(post.images.all())
        if len(imgs) != 1:
            continue                       # only loose single-image posts
        photo = imgs[0]
        fp = photo.file_path
        if not fp or not os.path.exists(fp):
            continue
        if os.path.normpath(os.path.dirname(fp)) != inbox_norm:
            continue                       # already in a subfolder (or _/)
        # bucket by the file's own modified date (the real capture/download
        # date). Nested as YYYY-MM/DD so the top level stays a short list of
        # months and each month splits into day folders.
        try:
            import datetime as _dt
            d = _dt.datetime.fromtimestamp(os.path.getmtime(fp))
            bucket = os.path.join(d.strftime('%Y-%m'), d.strftime('%d'))
        except OSError:
            bucket = (post.added_at.strftime('%Y-%m') if post.added_at else 'misc')
        dest_dir = os.path.join(inbox, bucket)
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.basename(fp)
        dest = os.path.join(dest_dir, base)
        if os.path.exists(dest):           # name collision → numeric suffix
            stem, ext = os.path.splitext(base)
            i = 1
            while os.path.exists(os.path.join(dest_dir, f'{stem}_{i}{ext}')):
                i += 1
            dest = os.path.join(dest_dir, f'{stem}_{i}{ext}')
        try:
            os.rename(fp, dest)
            photo.file_path = dest
            photo.save(update_fields=['file_path'])
            moved += 1
        except OSError as e:
            print(f"organize_singles error post {post.pk}: {e}")
    return JsonResponse({'ok': True, 'moved': moved})


@require_POST
def organize_singles_deep(request):
    """One-shot deep retidy: moves files that are ALREADY in subdirectories
    (e.g. inbox/YYYY-MM/ from a previous tidy run) into the full
    inbox/YYYY-MM/DD/ structure. Skips inbox/_/ (multi-post folders) and files
    already in the correct 3-part path. Run once after upgrading to day folders."""
    inbox = os.path.join(settings.MEDIA_ROOT, 'inbox')
    multi_root = os.path.normpath(os.path.join(inbox, '_'))
    moved = 0
    for post in Post.objects.prefetch_related('images').all():
        imgs = list(post.images.all())
        if len(imgs) != 1:
            continue
        photo = imgs[0]
        fp = photo.file_path
        if not fp or not os.path.exists(fp):
            continue
        fp_norm = os.path.normpath(fp)
        fp_dir = os.path.normpath(os.path.dirname(fp_norm))
        # skip anything inside inbox/_/
        if fp_dir.startswith(multi_root + os.sep) or fp_dir == multi_root:
            continue
        try:
            import datetime as _dt
            d = _dt.datetime.fromtimestamp(os.path.getmtime(fp))
            target_dir = os.path.normpath(os.path.join(inbox, d.strftime('%Y-%m'), d.strftime('%d')))
        except OSError:
            continue
        # already in the right place
        if fp_dir == target_dir:
            continue
        os.makedirs(target_dir, exist_ok=True)
        base = os.path.basename(fp)
        dest = os.path.join(target_dir, base)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(base)
            i = 1
            while os.path.exists(os.path.join(target_dir, f'{stem}_{i}{ext}')):
                i += 1
            dest = os.path.join(target_dir, f'{stem}_{i}{ext}')
        try:
            os.rename(fp, dest)
            photo.file_path = dest
            photo.save(update_fields=['file_path'])
            moved += 1
        except OSError as e:
            print(f'organize_singles_deep error post {post.pk}: {e}')
    return JsonResponse({'ok': True, 'moved': moved})


@require_POST
def split_image(request, pk):
    """Move a single image out of its post into a new post."""
    photo = get_object_or_404(Photo, pk=pk)
    post  = photo.post
    if not post or post.images.count() <= 1:
        return JsonResponse({'error': 'cannot split last image'}, status=400)
    new_post = Post.objects.create(title=photo.filename)
    photo.post  = new_post
    photo.order = 0
    photo.save(update_fields=['post', 'order'])
    # copy tags from parent
    for tag in post.tags.all():
        new_post.tags.add(tag)
    for tag in new_post.tags.all():
        tag.update_count()
    return JsonResponse({'ok': True, 'new_post_id': new_post.pk})


@require_POST
def split_images_to_one(request):
    """Move several selected images out of their post into a SINGLE new post
    (keeps them grouped together rather than scattering them into one post
    each)."""
    data = json.loads(request.body)
    ids  = [int(i) for i in data.get('ids', [])]
    if not ids:
        return JsonResponse({'error': 'no images selected'}, status=400)

    photos = list(Photo.objects.filter(pk__in=ids).select_related('post'))
    if not photos:
        return JsonResponse({'error': 'images not found'}, status=404)

    # Source post = the post the selection currently lives in. Don't allow
    # emptying a post completely (that would orphan it) — leave at least one.
    source = photos[0].post
    if source:
        remaining = source.images.exclude(pk__in=ids).count()
        if remaining == 0:
            return JsonResponse({'error': 'cannot split out every image — '
                                          'leave at least one in the post'},
                                status=400)

    new_post = Post.objects.create(title=photos[0].filename)
    for order, photo in enumerate(sorted(photos, key=lambda p: (p.order, p.id))):
        photo.post  = new_post
        photo.order = order
        photo.save(update_fields=['post', 'order'])

    # copy tags from the source post
    if source:
        for tag in source.tags.all():
            new_post.tags.add(tag)
    for tag in new_post.tags.all():
        tag.update_count()

    return JsonResponse({'ok': True, 'new_post_id': new_post.pk, 'count': len(photos)})


@require_POST
def split_images_to_separate(request):
    """Move several selected images out of their post, each into its OWN new
    post (one post per image), as opposed to split-to-one which groups them.
    Leaves at least one image in the source post."""
    data = json.loads(request.body)
    ids  = [int(i) for i in data.get('ids', [])]
    if not ids:
        return JsonResponse({'error': 'no images selected'}, status=400)

    photos = list(Photo.objects.filter(pk__in=ids).select_related('post'))
    if not photos:
        return JsonResponse({'error': 'images not found'}, status=404)

    source = photos[0].post
    if source:
        remaining = source.images.exclude(pk__in=ids).count()
        if remaining == 0:
            return JsonResponse({'error': 'cannot split out every image — '
                                          'leave at least one in the post'},
                                status=400)

    src_tags = list(source.tags.all()) if source else []
    new_ids = []
    for photo in sorted(photos, key=lambda p: (p.order, p.id)):
        new_post = Post.objects.create(title=photo.filename)
        photo.post  = new_post
        photo.order = 0
        photo.save(update_fields=['post', 'order'])
        for tag in src_tags:
            new_post.tags.add(tag)
        new_ids.append(new_post.pk)
    for tag in src_tags:
        tag.update_count()

    return JsonResponse({'ok': True, 'new_post_ids': new_ids, 'count': len(new_ids)})


@require_POST
def reorder_images(request, pk):
    """Reorder images within a post."""
    post    = get_object_or_404(Post, pk=pk)
    data    = json.loads(request.body)
    ordered = data.get('order', [])  # list of photo IDs in new order
    for i, photo_id in enumerate(ordered):
        Photo.objects.filter(pk=photo_id, post=post).update(order=i)
    return JsonResponse({'ok': True})


def mark_not_dupe(request):
    """Mark two posts as not duplicates of each other."""
    data = json.loads(request.body)
    id_a = data.get('a')
    id_b = data.get('b')
    undo = data.get('undo', False)
    try:
        post_a = Post.objects.get(pk=id_a)
        post_b = Post.objects.get(pk=id_b)
        if undo:
            post_a.not_dupes.remove(post_b)
        else:
            post_a.not_dupes.add(post_b)
        return JsonResponse({'ok': True, 'ignored': not undo})
    except Post.DoesNotExist:
        return JsonResponse({'error': 'post not found'}, status=404)


@require_POST
def delete_image_from_post(request, pk):
    """Delete a single image from a multi-image post."""
    photo     = get_object_or_404(Photo, pk=pk)
    post      = photo.post
    also_file = json.loads(request.body).get('delete_file', False)
    if post and post.image_count <= 1:
        return JsonResponse({'error': 'cannot delete last image — delete the post instead'}, status=400)
    if also_file and os.path.exists(photo.file_path):
        try: os.remove(photo.file_path)
        except OSError: pass
    if photo.thumb_path and os.path.exists(photo.thumb_path):
        try: os.remove(photo.thumb_path)
        except OSError: pass
    photo.delete()
    return JsonResponse({'ok': True, 'remaining': post.image_count if post else 0})


def post_not_dupes(request, pk):
    """Return list of post IDs that this post has marked as not-duplicate."""
    post = get_object_or_404(Post, pk=pk)
    ids  = list(post.not_dupes.values_list('id', flat=True))
    return JsonResponse({'not_dupe_ids': ids})


@require_POST
def regen_thumb(request, pk):
    """Regenerate the thumbnail for a single image/video/gif/pdf. Also re-reads
    dimensions + pHash, which fixes files that were still downloading when they
    were first ingested (so the original thumb was a placeholder/blank).
    For videos, an optional `pct` (0-100) picks which frame to grab."""
    photo = get_object_or_404(Photo, pk=pk)
    try:
        pct = float(json.loads(request.body or '{}').get('pct', 0))
    except Exception:
        pct = 0
    from .utils import (make_thumb, compute_phash, _thumb_path_for,
                        is_video as _is_video, is_pdf as _is_pdf)
    if not os.path.exists(photo.file_path):
        return JsonResponse({'ok': False, 'error': 'source file missing'}, status=404)

    # drop any stale/placeholder thumb first so a fresh one is written cleanly
    for tp in {photo.thumb_path, _thumb_path_for(photo.file_path)}:
        try:
            if tp and os.path.exists(tp):
                os.remove(tp)
        except OSError:
            pass

    thumb = make_thumb(photo.file_path, pct=pct)   # dispatches to video/pdf/image
    if not thumb:
        return JsonResponse({'ok': False, 'error': 'could not render thumbnail'}, status=500)

    photo.thumb_path = thumb
    vid = _is_video(photo.file_path)
    pdf = _is_pdf(photo.file_path)
    photo.is_video = vid
    if vid:
        ph = compute_phash(thumb) if thumb else ''
        if ph:
            photo.phash = ph
    elif not pdf:
        try:
            from PIL import Image as _Img
            with _Img.open(photo.file_path) as im:
                photo.width, photo.height = im.size
        except Exception:
            pass
        ph = compute_phash(photo.file_path)
        if ph:
            photo.phash = ph
    photo.save(update_fields=['thumb_path', 'phash', 'is_video', 'width', 'height'])
    return JsonResponse({'ok': True, 'thumb_url': photo.thumb_url})


@require_POST
def bulk_video_thumb(request):
    """Regenerate the COVER thumbnail of each selected post at a given video
    percentage. Only affects posts whose cover is a video."""
    data = json.loads(request.body)
    ids = data.get('ids', [])
    try:
        pct = float(data.get('pct', 0))
    except (TypeError, ValueError):
        pct = 0
    from .utils import make_thumb, is_video as _is_video
    done = 0
    for post in Post.objects.filter(id__in=ids).prefetch_related('images'):
        cover = post.cover
        if not cover or not cover.is_video:
            continue
        if not os.path.exists(cover.file_path):
            continue
        thumb = make_thumb(cover.file_path, pct=pct)
        if thumb:
            cover.thumb_path = thumb
            cover.save(update_fields=['thumb_path'])
            done += 1
    return JsonResponse({'ok': True, 'updated': done})


def similar_posts(request, pk):
    """Rank other posts by visual (cover pHash) + tag overlap similarity.
    Returns an ordered id list the gallery can display via ?ids=…"""
    post   = get_object_or_404(Post, pk=pk)
    cover  = post.cover
    my_ph  = cover.phash if cover else ''
    my_tags = set(post.tags.values_list('name', flat=True))

    scored = []
    for other in (Post.objects.exclude(pk=pk)
                      .prefetch_related('tags', 'images')):
        score = 0.0
        if my_ph:
            oc = other.cover
            if oc and oc.phash:
                dist = phash_distance(my_ph, oc.phash)
                if dist <= 16:
                    score += (16 - dist) * 2.0      # visual closeness
        if my_tags:
            ot = set(other.tags.values_list('name', flat=True))
            inter = len(my_tags & ot)
            if inter:
                union = len(my_tags | ot) or 1
                score += (inter / union) * 20.0     # tag overlap (Jaccard)
        if score > 0:
            scored.append((score, other.pk))

    scored.sort(reverse=True)
    ids = [pid for _, pid in scored[:120]]
    return JsonResponse({'ids': ids, 'count': len(ids)})


# ── Tag management ──────────────────────────────────────────────
@require_POST
def tag_manage(request):
    """Edit a tag globally: rename (merge if target exists), delete, or change
    its category. The tag name is passed in the body to avoid URL-encoding
    issues with special characters."""
    data = json.loads(request.body)
    action = data.get('action')
    name   = (data.get('name') or '').strip()
    tag = Tag.objects.filter(name=name).first()
    if not tag:
        return JsonResponse({'ok': False, 'error': 'tag not found'}, status=404)

    if action == 'delete':
        tag.delete()      # M2M links cascade automatically
        return JsonResponse({'ok': True})

    if action == 'category':
        cat = data.get('category', 'general')
        if cat not in {'general', 'character', 'artist', 'meta', 'ai'}:
            return JsonResponse({'ok': False, 'error': 'bad category'}, status=400)
        tag.category = cat
        tag.save(update_fields=['category'])
        return JsonResponse({'ok': True})

    if action == 'rename':
        new_name = (data.get('new_name') or '').strip().lower().replace(' ', '_')
        if not new_name:
            return JsonResponse({'ok': False, 'error': 'empty name'}, status=400)
        if new_name == tag.name:
            return JsonResponse({'ok': True, 'merged': False, 'name': new_name})
        existing = Tag.objects.filter(name=new_name).first()
        if existing:
            # merge: every post that had the old tag gets the existing one
            for post in tag.posts.all():
                post.tags.add(existing)
            tag.delete()
            existing.update_count()
            return JsonResponse({'ok': True, 'merged': True, 'name': new_name, 'count': existing.count})
        tag.name = new_name
        tag.save(update_fields=['name'])
        return JsonResponse({'ok': True, 'merged': False, 'name': new_name})

    return JsonResponse({'ok': False, 'error': 'bad action'}, status=400)


# ── Validate which post ids still exist (for the recent strip) ──
def posts_exist(request):
    ids = [int(x) for x in request.GET.get('ids', '').split(',') if x.strip().isdigit()]
    alive = set(Post.objects.filter(id__in=ids).values_list('id', flat=True))
    return JsonResponse({'alive': list(alive)})


# ── Admin pages ─────────────────────────────────────────────────
def shortcuts_page(request):
    try:
        with open(_prefs_path()) as f:
            prefs = json.load(f)
    except Exception:
        prefs = {}
    return render(request, 'gallery/shortcuts.html',
                  {'bindings_json': json.dumps(prefs.get('keyBindings', {}))})


def tags_edit_page(request):
    return render(request, 'gallery/tags_edit.html', {})


# ── Cross-device preferences (shared, single-user app) ──────────
def _prefs_path():
    return os.path.join(settings.BASE_DIR, 'prefs.json')

def get_prefs(request):
    try:
        with open(_prefs_path()) as f:
            return JsonResponse({'prefs': json.load(f)})
    except Exception:
        return JsonResponse({'prefs': {}})

@require_POST
def set_pref(request):
    data = json.loads(request.body)
    key, value = data.get('key'), data.get('value')
    if not key:
        return JsonResponse({'error': 'no key'}, status=400)
    try:
        with open(_prefs_path()) as f:
            prefs = json.load(f)
    except Exception:
        prefs = {}
    prefs[key] = value
    try:
        with open(_prefs_path(), 'w') as f:
            json.dump(prefs, f)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'ok': True})


@require_POST
def recent_add(request):
    """Atomically prepend a post-view event to the server-side recents list.
    Because this is a read-modify-write on the server (not a client push of the
    full local array), multiple devices can call it simultaneously without one
    overwriting the other's data — each POST just adds one entry at the front."""
    data = json.loads(request.body)
    entry = {k: data[k] for k in ('id', 'thumb', 'url', 't') if k in data}
    if not entry.get('id'):
        return JsonResponse({'error': 'no id'}, status=400)
    entry['t'] = entry.get('t') or int(__import__('time').time() * 1000)
    try:
        with open(_prefs_path()) as f:
            prefs = json.load(f)
    except Exception:
        prefs = {}
    recents = prefs.get('recentPosts', [])
    if not isinstance(recents, list):
        recents = []
    # remove any existing entry for this post (so it moves to front)
    recents = [r for r in recents if r.get('id') != entry['id']]
    recents.insert(0, entry)
    prefs['recentPosts'] = recents[:50]
    try:
        with open(_prefs_path(), 'w') as f:
            json.dump(prefs, f)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'ok': True, 'recents': prefs['recentPosts']})


def login_view(request):
    from django.conf import settings
    if request.method == 'POST':
        pwd = request.POST.get('password', '')
        if pwd == getattr(settings, 'GALLERY_PASSWORD', ''):
            request.session['authed'] = True
            request.session.set_expiry(60 * 60 * 24 * 30)  # 30 days
            next_url = request.GET.get('next', '/')
            from django.shortcuts import redirect
            return redirect(next_url)
        return render(request, 'gallery/login.html', {'error': True})
    return render(request, 'gallery/login.html', {})


def logout_view(request):
    request.session.flush()
    from django.shortcuts import redirect
    return redirect('/login/')


def service_worker(request):
    from django.http import FileResponse
    path = os.path.join(settings.BASE_DIR, 'static', 'js', 'sw.js')
    return FileResponse(open(path, 'rb'), content_type='application/javascript')
