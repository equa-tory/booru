import os
import re
import tempfile
import json
import random
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, FileResponse, Http404, HttpResponse, HttpResponseNotModified, StreamingHttpResponse
from django.core.paginator import Paginator
from django.views.decorators.http import require_POST
from django.db.models import Q, Case, When, IntegerField, F, Count, Value, OuterRef, Subquery
from django.db.models.functions import Mod, Coalesce
from django.conf import settings

from .models import (Post, Photo, Tag, Task, Folder, CustomConcept, CustomExample,
                     PostFeature, CustomApplied)
from .utils import (scan_inbox, create_post_from_files, ingest_photo,
                    add_tags_to_post, delete_post, phash_distance, make_thumb,
                    make_video_thumb, retag_all_videos, sync_sound_tag,
                    make_gif_from_post, recount_tags, TaskCancelled, check_cancel)
from . import char_tagger, ai_models, custom_model, prefs


# ── Search syntax helpers ──────────────────────────────────────
# Supported in the search box (tokens are split on whitespace):
#   tag1 tag2        AND   — posts having both
#   ( a ~ b )        OR    — posts having at least one (braces + spaces matter)
#   -tag1            NOT   — posts without the tag
#   night~           FUZZY — Levenshtein-close tag names (night/fight/bright…)
#   ta*1             GLOB  — tags starting "ta" and ending "1" (* = anything)
#   pages:3          PAGES — posts with exactly 3 images (also >15, <12, >=, <=)

_PAGES_RE = re.compile(r'^pages:(>=|<=|>|<)?(\d+)$')

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


def _page_count_sq():
    """Correlated subquery giving each post's image count, independent of
    whatever joins the rest of the query has already added. A plain
    annotate(Count('images')) (as multi_only/single_only use below) would be
    wrong here: a tag join that matches multiple rows per post (wildcard,
    fuzzy, OR groups) inflates the count, and a file:/folder: filter's WHERE
    clause on the same 'images' join would UNDER-count instead. A subquery
    sidesteps both."""
    return Coalesce(Subquery(
        Photo.objects.filter(post=OuterRef('pk')).order_by()
             .values('post').annotate(c=Count('id')).values('c')[:1],
        output_field=IntegerField()), Value(0))


def _term_to_q(term):
    """Translate one search term into a Q over Post.tags.
    Returns (Q, multi) — multi=True means the term may match several tag
    names, so it has OR semantics (a post matches if ANY of its tags fit)."""
    term = term.strip()
    if not term:
        return None, False
    if term.startswith('file:') and len(term) > 5:   # search by file name
        return Q(images__rel_path__icontains=term[5:]), True
    if term.startswith('folder:') and len(term) > 7:  # search by folder name
        return Q(images__rel_path__icontains=term[7:]), True
    m = _PAGES_RE.match(term)                          # image-count filter
    if m:
        op, n = m.group(1), int(m.group(2))
        if op == '>':
            return Q(_pages__gt=n), False
        if op == '>=':
            return Q(_pages__gte=n), False
        if op == '<':
            return Q(_pages__lt=n), False
        if op == '<=':
            return Q(_pages__lte=n), False
        return Q(_pages=n), False                       # exact
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
    posts = Post.objects.prefetch_related('tags', 'images', 'folders').filter(id__in=id_list)
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

    posts = Post.objects.prefetch_related('tags', 'images', 'folders').all()

    # Only pay for the extra correlated subquery when a pages: token is
    # actually present (tokens inside "( ... )" OR groups are still separate
    # entries in q_tags, so a flat scan covers those too; strip a leading '-'
    # so "-pages:1" NOT-queries are detected as well).
    if any(t.lstrip('-').startswith('pages:') for t in q_tags):
        posts = posts.annotate(_pages=_page_count_sq())

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
        try:
            _folder = Folder.objects.filter(id=int(folder_id)).first()
            _recurse = bool(_folder and _folder.include_subfolders)
        except Exception:
            _folder, _recurse = None, False  # DB behind on a folder migration
        if _recurse:
            posts = posts.filter(folders__id__in=[_folder.id, *_folder.descendant_ids()]).distinct()
        else:
            posts = posts.filter(folders__id=int(folder_id)).distinct()

    # AI-processing filter (settings -> debug): posts the AI has not looked at yet
    proc = request.GET.get('proc', '')
    if proc == 'none':          # neither the main tagger nor the PixAI character model
        posts = posts.filter(ai_tagged=False).exclude(char_model='pixai')
    elif proc == 'noai':
        posts = posts.filter(ai_tagged=False)
    elif proc == 'nopixai':
        posts = posts.exclude(char_model='pixai')

    if multi_only == '1':
        posts = posts.annotate(_img_count=Count('images')).filter(_img_count__gt=1)

    if single_only == '1':
        if multi_only == '1':
            pass  # conflicting filters
        else:
            posts = posts.annotate(_img_count2=Count('images')).filter(_img_count2__lte=1)

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

# ── Sidebar tag sampler ────────────────────────────────────────
SIDEBAR_TAG_CAP = 150
TAG_CATEGORY_ORDER = ['meta', 'character', 'artist', 'general', 'ai']


def _fair_shares(sizes, cap):
    """Split `cap` slots equally across categories ({category: available}).
    A category with fewer tags than its share gives the unused slots to the
    others; any rounding leftover is handed out one by one."""
    shares = {}
    remaining = cap
    order = sorted(sizes, key=lambda c: sizes[c])          # smallest first
    for i, c in enumerate(order):
        take = min(sizes[c], remaining // (len(order) - i))
        shares[c] = take
        remaining -= take
    for c in order[::-1]:                                  # rounding leftovers
        while remaining > 0 and shares[c] < sizes[c]:
            shares[c] += 1
            remaining -= 1
    return shares


def _weighted_sample(rows, k, rng):
    """k ids from [(id, weight)] without replacement; popular tags are more
    likely (Efraimidis-Spirakis: key = random ** (1 / w) with w = sqrt(count)).
    The square root keeps usage as a preference without letting the few
    thousand-post tags appear on every reload (raw counts: ~half the list
    repeated between two loads; sqrt: ~30%)."""
    if k >= len(rows):
        return [r[0] for r in rows]
    keyed = sorted(rows, key=lambda r: rng.random() ** (1.0 / max(r[1], 1) ** 0.5), reverse=True)
    return [r[0] for r in keyed[:k]]


def _sidebar_tags(cand, cap, by_name=False, filtered=False, rng=random):
    """cand = [(id, category, fav, weight)]. Returns Tag objects to show:
    favorites first (pinned, never sampled away), then an equal random share
    of every tag type, each group sorted by usage (or name). Each object gets
    `.group` ('favorites' or its category) and, when filtering, `.filtered_count`."""
    favs = [c for c in cand if c[2]]
    rest = [c for c in cand if not c[2]]
    by_cat = {}
    for tid, cat, _fav, w in rest:
        by_cat.setdefault(cat, []).append((tid, w))
    shares = _fair_shares({c: len(v) for c, v in by_cat.items()}, max(0, cap - len(favs)))
    picked = {}
    for cat, rows in by_cat.items():
        for tid in _weighted_sample(rows, shares.get(cat, 0), rng):
            picked[tid] = cat
    weight = {c[0]: c[3] for c in cand}
    cat_of = {c[0]: c[1] for c in cand}
    ids = [c[0] for c in favs] + list(picked)
    tags = {t.id: t for t in Tag.objects.filter(pk__in=ids)}
    rank = {c: i for i, c in enumerate(TAG_CATEGORY_ORDER)}
    fav_ids = {c[0] for c in favs}
    out = []
    for tid in ids:
        t = tags.get(tid)
        if t is None:
            continue
        t.group = 'favorites' if tid in fav_ids else cat_of[tid]
        if filtered:
            t.filtered_count = weight[tid]
        out.append(t)
    def key(t):
        g = 0 if t.group == 'favorites' else 1
        return (g, rank.get(t.category, 9), t.name if by_name else -weight[t.id], t.name)
    out.sort(key=key)
    return out


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

    # Sidebar tags: a fair, random sample instead of "top N by category rank"
    # (which let ~850 character tags push every general/ai tag out of the list).
    if q_tags:
        from django.db.models import Count
        post_ids = posts.values_list('id', flat=True)
        cand = (Tag.objects
            .filter(posts__id__in=post_ids)
            .annotate(filtered_count=Count('posts', distinct=True))
            .filter(filtered_count__gt=0)
            .values_list('id', 'category', 'fav', 'filtered_count'))
    else:
        cand = Tag.objects.filter(count__gt=0).values_list('id', 'category', 'fav', 'count')
    cand = list(cand)

    # Only render a handful of tags in the sidebar. The full list lives behind the
    # search box / tag sheet / "edit tags" page. Rendering thousands of <a>
    # tags into every gallery page made paging back to the gallery slow on the
    # phone (the markup is parsed even though the sidebar is hidden on mobile).
    tag_total = len(cand)
    # fast mode (cookie set from the more menu) renders far fewer tags so the
    # gallery page is lighter to parse on a phone.
    tag_cap = 40 if request.COOKIES.get('fastMode') == '1' else SIDEBAR_TAG_CAP
    popular_tags = _sidebar_tags(cand, tag_cap, by_name=(sort_tags == 'name'), filtered=bool(q_tags))
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
        'proc': request.GET.get('proc', ''),
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
            'ai':         post.ai_tagged,
            'chars':      post.char_tagged,
            'char_model': post.char_model,
            'folders':    [f.name for f in post.folders.all()],   # prefetched
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
    elif 'booru_back_url' in request.session:
        back_url = request.session['booru_back_url']
    # save back_url in session for post-to-post navigation
    if back_url:
        request.session['booru_back_url'] = back_url

    # Full search query string (tags + filters + sort) for neighbor navigation
    search_params = []
    for key in ('tag', 'sort', 'min_rating', 'rating', 'fav',
                'multi_only', 'single_only', 'folder', 'seed', 'ids', 'proc'):
        for val in request.GET.getlist(key):
            search_params.append(f'{key}={val}')
    search_qs = '&'.join(search_params)

    return render(request, 'gallery/detail.html', {
        'post': post, 'images': images, 'q_tags': q_tags,
        'back_url': back_url,
        'net_prefix': _net_prefix(request),
        'search_qs': search_qs,
    })


def _lan_ip():
    """This machine's LAN address (the interface that routes outward; nothing is sent)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return '127.0.0.1'


def _auto_net_prefix(request=None):
    r"""\\<host>\<share>\<folders>\ guessed from MEDIA_ROOT: /mnt/mass/Media_MASS/Photo ->
    \\192.168.1.50\mass\Media_MASS\Photo\ (first folder under /mnt = the share name).
    <host> is the name/IP the browser used to reach this server, else the LAN IP."""
    host = ''
    if request is not None:
        h = request.get_host()
        host = '' if h.startswith('[') else h.rsplit(':', 1)[0]
    if not host or host in ('localhost', '0.0.0.0') or host.startswith('127.'):
        host = _lan_ip()
    parts = [p for p in str(settings.MEDIA_ROOT).replace('\\', '/').split('/') if p]
    if parts[:1] in (['mnt'], ['srv']):
        parts = parts[1:]
    elif parts[:1] == ['media'] and len(parts) > 2:        # /media/<user>/<share>/...
        parts = parts[2:]
    return '\\\\' + host + '\\' + '\\'.join(parts) + '\\'


def _net_prefix(request=None):
    """Windows network path of MEDIA_ROOT shown on the post page: the one saved in
    settings (prefs `netPathPrefix`) or the automatic guess."""
    pref = str(_pref('netPathPrefix', '') or '').strip()
    if pref:
        pref = pref.replace('/', '\\')
        if not pref.startswith('\\\\'):
            pref = '\\\\' + pref.lstrip('\\')
        return pref if pref.endswith('\\') else pref + '\\'
    return _auto_net_prefix(request)


def _net_path(file_path, request=None):
    r"""The \\server\share\... path of a file (for "open in explorer" / copy)."""
    rel = os.path.relpath(file_path, settings.MEDIA_ROOT).replace(os.sep, '/')
    return _net_prefix(request) + rel.replace('/', '\\')


def net_path_api(request):
    """GET -> {saved, auto, effective}; POST {prefix} saves it ('' = automatic)."""
    if request.method == 'POST':
        try:
            _set_pref('netPathPrefix', str(json.loads(request.body or '{}').get('prefix', '')).strip())
        except ValueError:
            return JsonResponse({'error': 'bad json'}, status=400)
    return JsonResponse({'saved': str(_pref('netPathPrefix', '') or ''), 'auto': _auto_net_prefix(request),
                         'effective': _net_prefix(request), 'example': _net_path(os.path.join(str(settings.MEDIA_ROOT), 'inbox', 'a', 'b.jpg'), request)})


def _dupe_keeper_id(group):
    """Pick which post in a duplicate group to keep by default: biggest file
    first (assume it's the highest quality), then multi-image posts, then more
    images, then lowest id as a stable tiebreak."""
    ranked = sorted(group, key=lambda p: (
        -p['file_size'], -(1 if p['img_count'] > 1 else 0), -p['img_count'], p['post_id'],
    ))
    return ranked[0]['post_id']


def _compute_dupe_groups(progress=None):
    """Compare every post's cover image pHash against every other post's and
    group the near-matches.
    - Only different posts are ever compared (within-post images are never
      treated as duplicates of each other).
    - GIFs are never compared against non-GIF images (animated vs still).
    - not_dupes relationships between posts are respected.
    Still O(n^2), but each comparison is now a precomputed-int XOR + popcount
    instead of re-parsing two hex strings (phash_distance) on every pair —
    that re-parsing was the main cost once the library grew large.

    progress(done, total), if given, is called periodically while building the
    per-post data (used to drive a Task's progress bar).
    """
    posts = list(Post.objects.prefetch_related('not_dupes', 'images').all())

    from .utils import compute_phash
    post_data = []
    total = len(posts)
    for i, post in enumerate(posts):
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
        try:
            ph_int = int(cover.phash, 16)
        except ValueError:
            continue
        src = cover.thumb_path or cover.file_path
        post_data.append({
            'post_id':   post.pk,
            'phash_int': ph_int,
            'is_gif':    cover.is_gif,
            'is_video':  cover.is_video,
            'thumb_url': Photo._url_for(src),
            'title':     post.title,
            'img_count': post.image_count,
            'file_size': cover.file_size,
            'file_path': _net_path(cover.file_path),
        })
        if progress and i % 25 == 0:
            progress(i, total)
    if progress:
        progress(total, total)

    # Build ignored pairs set for O(1) lookup
    ignored_pairs = set()
    for post in posts:
        for other in post.not_dupes.all():
            ignored_pairs.add(tuple(sorted([post.pk, other.pk])))

    def _comparable(a, b):
        # video only ever compares with video; image/gif cross-compare freely.
        if a['is_video'] or b['is_video']:
            return a['is_video'] and b['is_video']
        return True

    groups = []
    used   = set()
    n = len(post_data)
    for i in range(n):
        p = post_data[i]
        if progress and i % 50 == 0:
            progress(i, n)   # also the cancel point for the O(n^2) phase
        if p['post_id'] in used:
            continue
        group = [p]
        best_dist = None
        for j in range(n):
            if i == j:
                continue
            q = post_data[j]
            if q['post_id'] in used or p['post_id'] == q['post_id']:
                continue
            if not _comparable(p, q):
                continue
            # Video thumbnails (often dark/title frames) collide far too easily,
            # so require a near-exact match for video-vs-video; images/gifs keep
            # the looser perceptual threshold.
            both_video = p['is_video'] and q['is_video']
            threshold = 2 if both_video else 8
            dist = bin(p['phash_int'] ^ q['phash_int']).count('1')
            if dist <= threshold:
                # cheap distance test first; the ignored-pair lookup only for near matches
                if tuple(sorted([p['post_id'], q['post_id']])) in ignored_pairs:
                    continue
                group.append(q)
                used.add(q['post_id'])
                best_dist = dist if best_dist is None else min(best_dist, dist)
        if len(group) > 1:
            used.add(p['post_id'])
            for g in group:
                g.pop('phash_int', None)   # internal only, not needed client-side
            groups.append({
                'min_distance': best_dist,
                'keeper_id':    _dupe_keeper_id(group),
                'posts':        group,
            })

    groups.sort(key=lambda g: g['min_distance'])   # most-certain duplicates first
    return groups


def duplicates(request):
    """Lightweight shell page — the actual comparison runs in the background
    (see dupes_scan/_do_dupes) and results are fetched as JSON, so this view
    never has to render every group's markup into one page (that's what made
    the old all-at-once list unusable on a phone with a large library)."""
    return render(request, 'gallery/duplicates.html', {})


# ── Scan / upload ──────────────────────────────────────────────

# ── Background tasks ────────────────────────────────────────────
import threading
import traceback
from django.utils import timezone


def _start_task(kind, fn, total=0, message='', exclusive=False):
    """Create a Task row and run `fn(task)` in a background thread so the work
    survives the page being closed and its progress is visible to every worker
    (state lives in the DB). With exclusive=True, a live task of the same kind
    is returned instead of starting a second one (tapping "scan inbox" twice
    used to run two scans over the same files at once)."""
    from django.db import transaction
    with transaction.atomic():   # IMMEDIATE write lock: check+create can't race across workers
        if exclusive:
            live = Task.objects.filter(kind=kind, status='running',
                                       updated_at__gte=timezone.now() - TASK_STALE_CANCEL).first()
            if live:
                return live
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
        except TaskCancelled:
            try:
                t = Task.objects.get(pk=tid)
                t.status = 'cancelled'
                t.message = (t.message + ' — stopped')[:300]
                t.finished_at = timezone.now()
                t.save()
            except Exception:
                pass
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


TASK_STALE_CANCEL = timezone.timedelta(minutes=5)    # no progress this long → a stop request force-ends it
TASK_STALE_SWEEP  = timezone.timedelta(minutes=30)   # no progress this long → assume the worker died


def _sweep_stale_tasks():
    """A background thread dies with its gunicorn worker, leaving its Task row
    'running' forever (and undismissable). Mark long-silent ones as errored."""
    Task.objects.filter(status='running', updated_at__lt=timezone.now() - TASK_STALE_SWEEP).update(
        status='error', error='worker stopped responding (orphaned)', finished_at=timezone.now())


def tasks_list(request):
    """Active tasks + recently finished ones (last few minutes), with elapsed
    time. The frontend polls this to show progress / notifications."""
    _sweep_stale_tasks()
    cutoff = timezone.now() - timezone.timedelta(minutes=5)
    qs = Task.objects.filter(Q(status='running') | Q(finished_at__gte=cutoff))[:20]
    out = []
    for t in qs:
        out.append({
            'id': t.id, 'kind': t.kind, 'status': t.status,
            'done': t.done, 'total': t.total, 'message': t.message,
            'error': t.error, 'elapsed': t.elapsed,
            'finished': bool(t.finished_at),
            'cancel_requested': t.cancel_requested,
        })
    return JsonResponse({'tasks': out})


@require_POST
def tasks_clear(request):
    """Dismiss all finished/errored/cancelled tasks from the list."""
    Task.objects.exclude(status='running').delete()
    return JsonResponse({'ok': True})


@require_POST
def task_dismiss(request, pk):
    """Dismiss ONE finished task card."""
    Task.objects.filter(pk=pk).exclude(status='running').delete()
    return JsonResponse({'ok': True})


@require_POST
def task_cancel(request, pk):
    """Ask a running task to stop. The work fn notices at its next cooperative
    check (check_cancel). If the task has shown no progress for a while its
    thread is presumably gone, so it is ended directly instead."""
    t = get_object_or_404(Task, pk=pk)
    if t.status != 'running':
        return JsonResponse({'ok': True, 'status': t.status})
    if timezone.now() - t.updated_at > TASK_STALE_CANCEL:
        t.status = 'cancelled'; t.cancel_requested = True
        t.message = (t.message + ' — stopped')[:300]
        t.finished_at = timezone.now(); t.save()
    else:
        Task.objects.filter(pk=pk).update(cancel_requested=True)
    return JsonResponse({'ok': True})


def _throttled_save(task, interval=0.5, **fields):
    """Set `fields` on the task and persist them, but at most every `interval`
    seconds — a per-file commit made the scan itself a major cost. Pass
    force=True-like behaviour by calling task.save() directly at the end."""
    import time as _t
    for k, v in fields.items():
        setattr(task, k, v)
    now = _t.monotonic()
    if now - getattr(task, '_last_save', 0) >= interval:
        task._last_save = now
        task.save(update_fields=list(fields))


# ── Background variants of the heavy operations ─────────────────
def _existing_paths_by_dir(abs_paths):
    """{dir: set(filenames)} for just the directories involved — one listdir
    per directory instead of one stat per file."""
    listing = {}
    for d in {os.path.dirname(p) for p in abs_paths}:
        try:
            listing[d] = set(os.listdir(d))
        except OSError:
            listing[d] = set()
    return listing


def _do_scan(task):
    import time as _t
    if not os.path.isdir(settings.MEDIA_ROOT):
        # e.g. the drive isn't mounted — every file would look "missing" and the
        # prune below would wipe the whole library from the DB
        raise RuntimeError(f'MEDIA_ROOT not available: {settings.MEDIA_ROOT}')
    # Phase A: prune rows whose file vanished. Cheap: ids/paths only, one
    # listdir per directory, and model instances only for the (rare) missing ones.
    check_cancel(task)
    rows = list(Photo.objects.values_list('id', 'rel_path'))
    abs_of = {pk: (rp if os.path.isabs(rp) else os.path.join(settings.MEDIA_ROOT, rp)) for pk, rp in rows}
    listing = _existing_paths_by_dir(abs_of.values())
    missing_ids = [pk for pk, ap in abs_of.items()
                   if os.path.basename(ap) not in listing.get(os.path.dirname(ap), ())]
    removed = 0
    for photo in Photo.objects.filter(pk__in=missing_ids):
        if photo.thumb_path and os.path.exists(photo.thumb_path):
            try: os.remove(photo.thumb_path)
            except OSError: pass
        photo.delete(); removed += 1
    empty = Post.objects.filter(images__isnull=True)
    n_empty = empty.count()
    if n_empty:
        removed += n_empty; empty.delete()

    task.message = 'scanning inbox…'; task.save(update_fields=['message'])
    new_posts, extend_posts = scan_inbox()
    task.total = len(new_posts) + len(extend_posts); task.save(update_fields=['total'])
    added = 0
    for i, (title, paths) in enumerate(new_posts):
        check_cancel(task)
        create_post_from_files(paths, title=title, recount=False)
        added += len(paths)
        _throttled_save(task, done=i + 1, message=f'added {added} file(s)')
        _t.sleep(0)   # let the progress poll / other requests interleave (gevent)
    base = len(new_posts)
    for j, (post, paths) in enumerate(extend_posts):
        check_cancel(task)
        start_order = post.images.count()
        added_video = False
        for k, path in enumerate(sorted(paths)):
            photo = ingest_photo(path, post, order=start_order + k)
            added_video = added_video or photo.is_video
        added += len(paths)
        if added_video:
            try:
                sync_sound_tag(post, recount=False)
            except Exception as e:
                print(f'sound-tag error post {post.id}: {e}')
        _throttled_save(task, done=base + j + 1, message=f'added {added} file(s)')
        _t.sleep(0)
    task.save(update_fields=['done', 'message'])
    # one batched recount instead of two queries per tag per file
    if added or removed:
        task.message = 'updating tag counts…'; task.save(update_fields=['message'])
        recount_tags()
        Tag.objects.filter(count=0).delete()
    # Phase D: only videos whose thumbnail file is MISSING (a plain stat, no
    # ffmpeg). It used to also re-run ffmpeg on every scan for any thumb under
    # 5 KB — dark/flat clips hit that every time. Use "bulk video thumb" to
    # force-regenerate those.
    check_cancel(task)
    vids = list(Photo.objects.filter(is_video=True).values_list('id', 'rel_thumb_path'))
    vthumb_abs = {pk: (rt if os.path.isabs(rt) else os.path.join(settings.MEDIA_ROOT, rt))
                  for pk, rt in vids if rt}
    vlisting = _existing_paths_by_dir(vthumb_abs.values())
    need = [pk for pk, rt in vids
            if not rt or os.path.basename(vthumb_abs[pk]) not in vlisting.get(os.path.dirname(vthumb_abs[pk]), ())]
    for photo in Photo.objects.filter(pk__in=need):
        check_cancel(task)
        thumb = make_video_thumb(photo.file_path)
        if thumb: photo.thumb_path = thumb; photo.save(update_fields=['rel_thumb_path'])
    task.message = f'added {added}, removed {removed} (total {Post.objects.count()})'
    task.save(update_fields=['message'])


@require_POST
def scan_bg(request):
    return JsonResponse({'task_id': _start_task('scan', _do_scan, message='scan starting…', exclusive=True).id})


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
            check_cancel(task)
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
        label = {'wd14': 'default', 'pixai': 'PixAI', 'custom': 'My model'}[_main_model()]
        learned = 0
        for i, post in enumerate(posts):
            check_cancel(task)
            try:
                res = apply_ai_tags(post)
                learned += len((res or {}).get('learned') or [])
            except Exception as e:
                print(f'ai-tag error post {post.id}: {e}')
            task.done = i + 1
            task.message = f'{label}: tagged {i + 1}/{len(posts)} post(s)' + (f' · {learned} taught tag(s)' if label == 'My model' else '')
            task.save(update_fields=['done', 'message'])
            _t.sleep(0)
    return JsonResponse({'task_id': _start_task('ai_tag', work, message='ai tagging…', exclusive=True).id})


@require_POST
def sound_tag_all_bg(request):
    """Re-check every video post for an audio stream and sync the `sound` tag.
    Idempotent — safe to re-run (adds/removes the tag as clips change)."""
    def work(task):
        import time as _t
        posts = list(Post.objects.filter(images__is_video=True).distinct())
        task.total = len(posts); task.save(update_fields=['total'])
        found = 0
        for i, post in enumerate(posts):
            check_cancel(task)
            try:
                if sync_sound_tag(post):
                    found += 1
            except Exception as e:
                print(f'sound-tag error post {post.id}: {e}')
            task.done = i + 1
            task.message = f'{found} with sound / {i + 1} checked'
            task.save(update_fields=['done', 'message'])
            _t.sleep(0)
    return JsonResponse({'task_id': _start_task('sound_tag', work, message='detecting audio…', exclusive=True).id})


@require_POST
def post_to_gif_bg(request, pk):
    """Render a multi-image post into a full-resolution animated GIF and
    ingest it as a NEW post (tags copied across) — the source post is left
    untouched. Runs in the background: full-res GIF encoding can take
    minutes, well past what a plain request should hold open."""
    post = get_object_or_404(Post, pk=pk)
    if post.image_count < 2:
        return JsonResponse({'error': 'post needs at least 2 images'}, status=400)
    try:
        fps = float(json.loads(request.body or '{}').get('fps') or 2)
    except (ValueError, TypeError):
        fps = 2.0

    def work(task):
        src = Post.objects.get(pk=pk)  # re-fetch: this closure runs in its own thread
        gif_path = make_gif_from_post(src, fps, task=task)
        try:
            check_cancel(task)
        except TaskCancelled:
            try: os.remove(gif_path)   # don't leave a stray file for the next scan to ingest
            except OSError: pass
            raise
        new_post = create_post_from_files([gif_path], title=src.title)
        for tag in src.tags.all():
            new_post.tags.add(tag)
        for tag in new_post.tags.all():
            tag.update_count()
        task.message = f'gif ready — post #{new_post.id}'
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('gif', work, message='building gif…').id})


def _repair_missing_thumbs(task=None):
    """Rebuild thumbnails whose file is gone (e.g. a stored path from before the media
    folder moved: the post shows no preview and its URL 404s). Photos whose source file
    is missing too are left alone. Returns (rebuilt, failed)."""
    from .models import Photo
    todo = [p for p in Photo.objects.exclude(rel_thumb_path='').iterator(chunk_size=5000)
            if not os.path.exists(p.thumb_path)]
    if task:
        task.total = len(todo); task.done = 0
        task.message = f'rebuilding {len(todo)} missing thumbnail(s)…'
        task.save(update_fields=['total', 'done', 'message'])
    fixed = failed = 0
    os.makedirs(os.path.join(str(settings.MEDIA_ROOT), 'thumbs'), exist_ok=True)
    for i, photo in enumerate(todo):
        check_cancel(task)
        try:
            ok, _err = _regen_photo_thumb(photo)
            if ok:
                photo.save(); fixed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f'thumb repair error photo {photo.id}: {e}')
        if task:
            _throttled_save(task, done=i + 1)
    return fixed, failed


@require_POST
def rebase_paths_bg(request):
    """Convert any Photo rows still storing an absolute file/thumb path into
    a path relative to the current MEDIA_ROOT. Run this BEFORE moving the
    media folder (while MEDIA_ROOT still points at the current location) so
    future moves only need a MEDIA_ROOT change in settings.py, not a DB edit.
    Idempotent — safe to re-run."""
    from .utils import rebase_photo_paths
    def work(task):
        converted, already_relative, left_absolute = rebase_photo_paths(task)
        fixed, failed = _repair_missing_thumbs(task)
        task.message = (f'{converted} converted, {already_relative} already relative, '
                         f'{left_absolute} left absolute (outside MEDIA_ROOT); '
                         f'thumbnails: {fixed} rebuilt' + (f', {failed} failed' if failed else ''))
        task.save(update_fields=['message'])
    return JsonResponse({'task_id': _start_task('rebase_paths', work, message='rebasing paths…', exclusive=True).id})


def _dupes_cache_path():
    return os.path.join(settings.BASE_DIR, '.dupes_cache.json')


def _do_dupes(task):
    def on_progress(done, total):
        check_cancel(task)
        task.total = total
        task.done = done
        task.message = f'comparing {done}/{total} post(s)…'
        task.save(update_fields=['total', 'done', 'message'])

    task.message = 'scanning posts…'
    task.save(update_fields=['message'])
    groups = _compute_dupe_groups(progress=on_progress)
    data = {'computed_at': timezone.now().isoformat(), 'groups': groups}
    with open(_dupes_cache_path(), 'w') as f:
        json.dump(data, f)
    task.message = f'found {len(groups)} group(s)'
    task.save(update_fields=['message'])


@require_POST
def dupes_scan(request):
    """Kick off duplicate detection in the background — see _do_dupes. The
    frontend polls the existing /api/tasks/ endpoint for progress, then fetches
    the cached result from dupes_result once the task finishes."""
    return JsonResponse({'task_id': _start_task('dupes', _do_dupes,
                                                 message='scanning for duplicates…', exclusive=True).id})


def dupes_result(request):
    """Last computed duplicate-group results (from _do_dupes), or an empty
    result if a scan has never run."""
    try:
        with open(_dupes_cache_path()) as f:
            data = json.load(f)
    except Exception:
        data = {'computed_at': None, 'groups': []}
    return JsonResponse(data)


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
                photo.save(update_fields=['rel_thumb_path'])

    # Also retag PDFs with placeholder thumbs
    for photo in Photo.objects.filter(rel_path__iendswith='.pdf'):
        if not photo.thumb_path or not _os.path.exists(photo.thumb_path) or            _os.path.getsize(photo.thumb_path) < 5000:
            from .utils import make_pdf_thumb
            thumb = make_pdf_thumb(photo.file_path)
            if thumb:
                photo.thumb_path = thumb
                photo.save(update_fields=['rel_thumb_path'])

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
        for post in posts_created:
            try:
                apply_ai_tags(post)
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
        res = apply_ai_tags(post)
        if res is None:
            return JsonResponse({'error': 'no images', 'ok': False}, status=400)
        return JsonResponse({'tags': res['general'] + res['character'] + [n for n, _p in res.get('learned', [])],
                             'characters': res['character'], 'learned': [n for n, _p in res.get('learned', [])],
                             'model': res.get('model', 'wd14'), 'ok': True})
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
    data      = json.loads(request.body)
    name      = data.get('name', '').strip()
    is_smart  = bool(data.get('is_smart', False))
    query     = data.get('query', '').strip() if is_smart else ''
    parent_id = data.get('parent_id')
    if not name:
        return JsonResponse({'error': 'name required'}, status=400)
    parent = None
    if parent_id:
        parent = Folder.objects.filter(pk=parent_id).first()
        if not parent:
            return JsonResponse({'error': 'parent folder not found'}, status=404)
    folder = Folder.objects.create(name=name, is_smart=is_smart, query=query, parent=parent)
    return JsonResponse({'id': folder.id, 'name': folder.name, 'is_smart': folder.is_smart,
                          'query': folder.query, 'parent_id': folder.parent_id})


@require_POST
def folder_delete(request, pk):
    # cascades to the whole subtree (Folder.parent is on_delete=CASCADE) —
    # posts themselves are never touched, only the folder rows.
    Folder.objects.filter(pk=pk).delete()
    return JsonResponse({'ok': True})


@require_POST
def folder_rename(request, pk):
    data = json.loads(request.body)
    name = data.get('name', '').strip()
    if not name:
        return JsonResponse({'error': 'name required'}, status=400)
    folder = get_object_or_404(Folder, pk=pk)
    folder.name = name
    folder.save(update_fields=['name'])
    return JsonResponse({'ok': True, 'name': folder.name})


@require_POST
def folder_set_subfolders(request, pk):
    """Toggle whether opening this folder also shows its descendant folders' posts."""
    data = json.loads(request.body)
    folder = get_object_or_404(Folder, pk=pk)
    folder.include_subfolders = bool(data.get('include_subfolders'))
    folder.save(update_fields=['include_subfolders'])
    return JsonResponse({'ok': True, 'include_subfolders': folder.include_subfolders})


def folders_list(request):
    """JSON list of all folders — used by the gallery sidebar, the mobile
    "more" sheet, and the detail-page "add to folder" picker. Flat (each row
    carries its own parent_id); the shared renderFolderTree() JS builds the
    tree client-side from that.

    Optional ?ids=1,2,3 — marks each folder 'contains': True if it already
    holds ALL of the given post ids, so the picker can highlight it (and
    switch its button to a remove action) instead of blindly re-adding."""
    from django.db.utils import OperationalError
    ids_param = request.GET.get('ids', '')
    ids = [int(i) for i in ids_param.split(',') if i.strip().isdigit()]
    result = []
    try:
        for f in Folder.objects.all():
            contains = bool(ids) and not f.is_smart and f.posts.filter(pk__in=ids).count() == len(ids)
            result.append({'id': f.id, 'name': f.name, 'is_smart': f.is_smart, 'query': f.query,
                            'parent_id': f.parent_id, 'contains': contains,
                            'include_subfolders': f.include_subfolders})
    except OperationalError as e:
        # almost always a missing column on a DB that predates a folder
        # migration (e.g. an older db.sqlite3 swapped in) — surface the fix
        # instead of a bare 500.
        return JsonResponse(
            {'folders': [], 'error': f'database is behind — run: python manage.py migrate  ({e})'},
            status=503)
    return JsonResponse({'folders': result})


@require_POST
def ai_tag_all(request):
    posts   = Post.objects.filter(ai_tagged=False)
    results = []
    for post in posts:
        try:
            res = apply_ai_tags(post)
            if res is None:
                continue
            results.append({'id': post.id, 'tags': res['general'] + res['character']})
        except Exception as e:
            results.append({'id': post.id, 'error': str(e)})
    return JsonResponse({'results': results, 'count': len(results)})


# ── AI tagger ─────────────────────────────────────────────────

# WD14 emits one probability per tag; the tag list is ordered rating → general →
# character. Characters are scored with their OWN (much higher) threshold: the
# model is confident when it knows a character, and a low bar mostly adds
# wrong guesses. Override in booru/local_settings.py.
AI_GENERAL_THRESHOLD   = getattr(settings, 'AI_GENERAL_THRESHOLD', 0.35)
AI_CHARACTER_THRESHOLD = getattr(settings, 'AI_CHARACTER_THRESHOLD', 0.85)
AI_MAX_GENERAL         = 40
AI_MAX_CHARACTERS      = 12
# multi-image posts: tag up to N items (evenly spaced when there are more) and sum the results
AI_MAX_IMAGES_PER_POST = getattr(settings, 'AI_MAX_IMAGES_PER_POST', 24)
AI_MAX_GENERAL_MULTI   = getattr(settings, 'AI_MAX_GENERAL_MULTI', 80)
AI_MAX_CHARACTERS_MULTI = 20
_RATING_ORDER = ['general', 'sensitive', 'questionable', 'explicit']      # least -> most severe


def _main_model():
    """Which model does the main AI tagging: the selected one if it is ready on
    disk, else the default WD14 (selector in Settings -> AI models)."""
    want = _pref('aiMainModel', 'wd14')
    if want in ('pixai', 'custom') and ai_models.is_ready(want):
        return want
    return 'wd14'


def run_ai_tagger(file_path, thumb_path='', model=None):
    """Tag one image with the selected main model (default WD14 / PixAI / my
    model). Returns {'general': [...], 'character': [...], 'model': key}."""
    model = model or _main_model()
    if model == 'pixai':
        return char_tagger.run_pixai_general(file_path, thumb_path)
    return _run_wd14(file_path, thumb_path, custom=(model == 'custom'))


def _run_wd14(file_path, thumb_path='', custom=False):
    """Run the WD14 tagger on an image. Returns {'general': [...], 'character': [...]}.

    - general: rating + general tags >= AI_GENERAL_THRESHOLD, first AI_MAX_GENERAL
      in the model's (frequency) order — unchanged from the original behaviour.
    - character: character tags >= AI_CHARACTER_THRESHOLD, most confident first.
      They used to share the general list's 40-tag cap, and since characters come
      last in the model's order they were silently cut whenever a picture had
      40+ general tags (about half of the confident character hits).

    If the original can't be opened as an image (e.g. it's an mp4/video or a
    pdf), fall back to the generated thumbnail so video/pdf posts can still be
    auto-tagged from their preview frame.
    """
    from .utils import load_image_for_tagging, wd14_input
    import numpy as np

    arr = wd14_input(load_image_for_tagging(file_path, thumb_path))
    if custom:
        probs, feat = custom_model.run_clone(arr)       # the clone: same default outputs + pooled features
        tags_list, is_char = _wd14_tags()
    else:
        model, tags_list, is_char = _get_wd14_model()
        probs = np.asarray(model.run(None, {model.get_inputs()[0].name: arr})[0][0])

    gen_idx  = np.nonzero((probs >= AI_GENERAL_THRESHOLD) & ~is_char)[0][:AI_MAX_GENERAL]
    char_idx = np.nonzero((probs >= AI_CHARACTER_THRESHOLD) & is_char)[0]
    char_idx = char_idx[np.argsort(-probs[char_idx])][:AI_MAX_CHARACTERS]
    res = {
        'general':   [tags_list[i].replace(' ', '_') for i in gen_idx],
        'character': [tags_list[i].replace(' ', '_') for i in char_idx],
        'model': 'custom' if custom else 'wd14',
    }
    if custom:
        res['learned'] = custom_model.learned_tags(feat)   # [(tag, probability)] from the taught heads
    return res


def _sample_images(images, n):
    """At most n items, evenly spaced, always including the first and the last."""
    if n <= 0 or len(images) <= n:
        return list(images)
    if n == 1:
        return [images[0]]
    idx = sorted({round(i * (len(images) - 1) / (n - 1)) for i in range(n)})
    return [images[i] for i in idx]


def _merge_ai_results(results):
    """Sum the run_ai_tagger results of several images of one post. Tags are ranked
    by how many images carry them; of the (contradicting) rating tags only the most
    severe one survives. One result passes through unchanged."""
    from collections import Counter
    if len(results) == 1:
        return results[0]

    def ranked(key, cap):
        cnt, first = Counter(), {}
        for r in results:
            for t in dict.fromkeys(r.get(key) or []):
                cnt[t] += 1
                first.setdefault(t, len(first))
        return sorted(cnt, key=lambda t: (-cnt[t], first[t]))[:cap], cnt

    general, _ = ranked('general', 10 ** 6)
    ratings = [t for t in general if t in _RATING_ORDER]
    general = [t for t in general if t not in _RATING_ORDER][:AI_MAX_GENERAL_MULTI]
    if ratings:
        general.insert(0, max(ratings, key=_RATING_ORDER.index))
    characters, _ = ranked('character', AI_MAX_CHARACTERS_MULTI)
    learned = {}
    for r in results:
        for name, p in r.get('learned') or []:
            learned[name] = max(p, learned.get(name, 0.0))
    out = {'general': general, 'character': characters, 'model': results[0].get('model', 'wd14')}
    if learned or any('learned' in r for r in results):
        out['learned'] = sorted(learned.items(), key=lambda kv: -kv[1])
    return out


def apply_ai_tags(post, cover=None):
    """Tag one post with the main AI tagger and record the result: general tags in
    category 'ai', characters in category 'character', and both ai_tagged /
    char_tagged flags set. With no `cover` every item of a multi-image post is
    tagged (up to AI_MAX_IMAGES_PER_POST) and the tags are summed; an explicit
    `cover` tags just that image. Returns the merged dict, or None when the post
    has no image."""
    if cover is not None:
        covers, n_images = [cover], 1
    else:
        imgs = list(post.images.order_by('order', 'id'))
        covers, n_images = _sample_images(imgs, AI_MAX_IMAGES_PER_POST), len(imgs)
    if not covers:
        return None
    results, last_err = [], None
    for c in covers:
        try:
            results.append(run_ai_tagger(c.file_path, c.thumb_path))
        except Exception as e:           # one unreadable item must not sink the post
            last_err = e
    if not results:
        raise last_err
    res = _merge_ai_results(results)
    cover = covers[0]
    add_tags_to_post(post, res['general'], category='ai')
    if res['character']:
        touched = add_tags_to_post(post, res['character'], category='character')
        # A character tag created by the old pipeline sits in category 'ai' —
        # promote it. Never touch tags the user categorised themselves.
        Tag.objects.filter(pk__in=[t.pk for t in touched], category='ai').update(category='character')
    if res.get('learned'):
        custom_model.apply_learned(post, [n for n, _p in res['learned']])
    post.ai_tagged = True
    post.char_tagged = True
    if res.get('model') == char_tagger.MODEL_ID:
        post.char_model = char_tagger.MODEL_ID       # PixAI was the main tagger: it already did the characters
    elif post.char_model != char_tagger.MODEL_ID:    # never downgrade a PixAI result
        post.char_model = 'wd14'
    fields = ['ai_tagged', 'char_tagged', 'char_model']
    if n_images > 1 and len(covers) > 1:
        post.ai_multi = True
        fields.append('ai_multi')
    post.save(update_fields=fields)
    if res.get('model') != char_tagger.MODEL_ID and _pref('aiCharAuto') and char_tagger.model_ready():
        try:
            res['character'] = list(dict.fromkeys(res['character'] + apply_character_model(post, cover)))
        except Exception as e:
            print(f'character model error post {post.id}: {e}')
    return res


def _pref(key, default=None):
    """A server-side preference (database; see gallery/prefs.py)."""
    try:
        return prefs.get(key, default)
    except Exception:
        return default


def apply_character_model(post, cover=None):
    """Run the PixAI character model on the post's cover and ADD its character
    tags (category 'character'; nothing is removed). Marks the post
    char_model='pixai'. Returns the character names found (maybe empty), or
    None when the post has no image."""
    cover = cover or post.images.order_by('order', 'id').first()
    if not cover:
        return None
    names = [n for n, _p in char_tagger.run_character_tagger(cover.file_path, cover.thumb_path)]
    if names:
        touched = add_tags_to_post(post, names, category='character')
        Tag.objects.filter(pk__in=[t.pk for t in touched], category='ai').update(category='character')
    post.char_tagged = True
    post.char_model = char_tagger.MODEL_ID
    post.save(update_fields=['char_tagged', 'char_model'])
    return names


_wd14_tags_cache = None


def _wd14_tags():
    """(names, is_character mask) from the model's selected_tags.csv (category 4
    = character, 0 = general, 9 = rating). Cheap — does not load the model."""
    global _wd14_tags_cache
    if _wd14_tags_cache: return _wd14_tags_cache
    import csv
    import numpy as np
    tags_path = ai_models.cached_file('wd14', 'selected_tags.csv')
    if not tags_path:
        raise RuntimeError('default model not downloaded — Settings → AI models')
    with open(tags_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    _wd14_tags_cache = ([r['name'] for r in rows],
                        np.array([r['category'] == '4' for r in rows], dtype=bool))
    return _wd14_tags_cache


def _get_wd14_model():
    """(model, names, is_char). `model` is a ManagedModel (gallery/ai_runtime.py):
    CUDA when available, CPU fallback, unloaded again when idle."""
    from . import ai_runtime
    path = ai_models.cached_file('wd14', 'model.onnx')
    if not path:
        raise RuntimeError('default model not downloaded — Settings → AI models')
    names, is_char = _wd14_tags()
    return ai_runtime.get_model('wd14', path), names, is_char


# ── Tag helpers ────────────────────────────────────────────────

def tag_search(request):
    q    = request.GET.get('q', '').lower().lstrip('-')
    try:
        limit = max(1, min(int(request.GET.get('limit', 20)), 50))
    except ValueError:
        limit = 20
    if not q:
        return JsonResponse({'tags': []})
    # prefix matches first, then everything containing it; unused tags are noise
    tags = (Tag.objects.filter(name__icontains=q, count__gt=0)
            .annotate(_pre=Case(When(name__istartswith=q, then=Value(0)), default=Value(1), output_field=IntegerField()))
            .order_by('_pre', '-count', 'name')[:limit])
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
    # Carry the FULL filter set (tags, rating, fav, sort, ...) into the post
    # page's query string, not just tags — otherwise a second "random" press
    # from the post page (which reads its own URL's query string) silently
    # loses every filter but tags.
    np = request.GET.copy()
    np.pop('page', None)
    np.pop('scroll', None)
    nav_qs = np.urlencode()
    from django.shortcuts import redirect
    return redirect(f'/post/{pk}/{"?" + nav_qs if nav_qs else ""}')



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
            photo.save(update_fields=['rel_path'])
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
            photo.save(update_fields=['rel_path'])
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


def _regen_photo_thumb(photo, pct=0):
    """Regenerate one Photo's thumbnail (image / gif / video / pdf). Also
    re-reads dimensions + pHash, which fixes files that were still downloading
    when first ingested (so the original thumb was a placeholder/blank).
    Returns (ok, error_message)."""
    from .utils import (make_thumb, compute_phash, _thumb_path_for,
                        is_video as _is_video, is_pdf as _is_pdf)
    if not os.path.exists(photo.file_path):
        return False, 'source file missing'

    # drop any stale/placeholder thumb first so a fresh one is written cleanly
    for tp in {photo.thumb_path, _thumb_path_for(photo.file_path)}:
        try:
            if tp and os.path.exists(tp):
                os.remove(tp)
        except OSError:
            pass

    thumb = make_thumb(photo.file_path, pct=pct)   # dispatches to video/pdf/image
    if not thumb:
        return False, 'could not render thumbnail'

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
    photo.save(update_fields=['rel_thumb_path', 'phash', 'is_video', 'width', 'height'])
    return True, ''


@require_POST
def regen_thumb(request, pk):
    """Regenerate the thumbnail for a single image/video/gif/pdf.
    For videos, an optional `pct` (0-100) picks which frame to grab."""
    photo = get_object_or_404(Photo, pk=pk)
    try:
        pct = float(json.loads(request.body or '{}').get('pct', 0))
    except Exception:
        pct = 0
    ok, err = _regen_photo_thumb(photo, pct)
    if not ok:
        return JsonResponse({'ok': False, 'error': err},
                            status=404 if err == 'source file missing' else 500)
    return JsonResponse({'ok': True, 'thumb_url': photo.thumb_url})


@require_POST
def bulk_regen_thumb(request):
    """Regenerate the COVER thumbnail of each selected post whose cover is a
    picture (image / gif / pdf). Video covers are left to bulk_video_thumb,
    which also picks the frame (pct)."""
    ids = json.loads(request.body or '{}').get('ids', [])
    updated = failed = skipped_videos = 0
    for post in Post.objects.filter(id__in=ids).prefetch_related('images'):
        cover = post.cover
        if not cover:
            continue
        if cover.is_video:
            skipped_videos += 1
            continue
        ok, _err = _regen_photo_thumb(cover)
        if ok: updated += 1
        else: failed += 1
    return JsonResponse({'ok': True, 'updated': updated, 'failed': failed,
                         'skipped_videos': skipped_videos})


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
            cover.save(update_fields=['rel_thumb_path'])
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
    return render(request, 'gallery/shortcuts.html',
                  {'bindings_json': json.dumps(prefs.get('keyBindings', {}) or {})})


def tags_edit_page(request):
    return render(request, 'gallery/tags_edit.html', {})


# ── Cross-device preferences (shared, single-user app) ──────────
def _prefs_path():
    """The LEGACY prefs.json (imported into the database on first use, see prefs.py)."""
    return os.path.join(settings.BASE_DIR, 'prefs.json')


def get_prefs(request):
    try:
        return JsonResponse({'prefs': prefs.all()})
    except Exception:
        return JsonResponse({'prefs': {}})


def _set_pref(key, value):
    prefs.set(key, value)


@require_POST
def set_pref(request):
    data = json.loads(request.body)
    key, value = data.get('key'), data.get('value')
    if not key:
        return JsonResponse({'error': 'no key'}, status=400)
    try:
        _set_pref(key, value)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'ok': True})


# ── Quick links: shortcut buttons in the top menu (prefs.json `quickLinks`) ──
QUICK_LINKS_MAX = 20
_URL_TOKEN = re.compile(r'^(https?://\S+|/\S*)$', re.I)


def _quick_links():
    links = _pref('quickLinks', [])
    return [l for l in links if isinstance(l, dict) and l.get('url')] if isinstance(links, list) else []


def _link_label(url):
    from urllib.parse import urlparse
    host = urlparse(url).netloc or url
    return (host[4:] if host.startswith('www.') else host)[:30] or url[:30]


def parse_quick_links(text):
    """Pasted text -> [{'label','url'}]. One link per line; a line with a single URL
    may carry a label around it ("my site https://x.org"); a line with several
    URLs (or a bare URL) labels each one with its host name. Only http(s) and
    site-relative ("/path") URLs are accepted."""
    out = []
    for line in str(text or '').splitlines():
        toks = line.replace(',', ' ').split()
        urls = [t for t in toks if _URL_TOKEN.match(t)]
        if not urls:
            continue
        words = [t for t in toks if t not in urls]
        for u in urls:
            label = ' '.join(words).strip() if len(urls) == 1 and words else _link_label(u)
            out.append({'label': label[:30], 'url': u[:500]})
    return out


def quick_links_api(request):
    """GET -> the list. POST {add: "pasted links"} appends, {remove: index} deletes."""
    links = _quick_links()
    if request.method == 'POST':
        try:
            data = json.loads(request.body or '{}')
        except ValueError:
            return JsonResponse({'error': 'bad json'}, status=400)
        if 'remove' in data:
            try:
                links.pop(int(data['remove']))
            except (ValueError, IndexError, TypeError):
                return JsonResponse({'error': 'no such link'}, status=400)
        added = parse_quick_links(data.get('add', ''))
        if data.get('add') and not added:
            return JsonResponse({'error': 'no http(s) links found in that text'}, status=400)
        have = {l['url'] for l in links}
        links += [l for l in added if l['url'] not in have and not have.add(l['url'])]
        links = links[:QUICK_LINKS_MAX]
        _set_pref('quickLinks', links)
    return JsonResponse({'links': links})


# ── Character model (PixAI) + AI runtime ────────────────────────
def ai_info(request):
    """State for the settings panel's AI section. Loads no model."""
    from . import ai_runtime
    ready = char_tagger.model_ready()
    return JsonResponse({
        'runtime': ai_runtime.runtime_info(),
        'char_model': {
            'name': 'PixAI tagger v0.9', 'ready': ready,
            'mb': round(char_tagger.model_bytes() / 1048576) if ready else 0,
            'threshold': char_tagger.CHAR_THRESHOLD,
        },
        'auto': bool(_pref('aiCharAuto')),
        'pixai_done': Post.objects.filter(char_model=char_tagger.MODEL_ID).count(),
        'pixai_todo': Post.objects.exclude(char_model=char_tagger.MODEL_ID).filter(images__isnull=False).distinct().count(),
        'multi_done': _multi_posts().filter(ai_multi=True).count(),
        'multi_todo': _multi_posts().filter(ai_multi=False).count(),
        'multi_items': sum(min(n, AI_MAX_IMAGES_PER_POST) for n in _multi_posts().filter(ai_multi=False).values_list('_n', flat=True)),
        'multi_cap': AI_MAX_IMAGES_PER_POST,
    })


def _ensure_char_model(task):
    """Download the character model if missing, reporting MB to the task."""
    if char_tagger.model_ready():
        return
    task.message = 'downloading character model (1.3 GB)…'
    task.total = round(char_tagger.EXPECTED_BYTES / 1048576)
    task.save(update_fields=['message', 'total'])

    def prog(done, total):
        _throttled_save(task, interval=2, done=min(round(done / 1048576), task.total))

    char_tagger.download_model(prog)
    task.done = 0
    task.save(update_fields=['done'])


@require_POST
def char_model_download(request):
    def work(task):
        _ensure_char_model(task)
        task.message = 'character model ready'
        task.save(update_fields=['message'])
    return JsonResponse({'task_id': _start_task('model_download', work, message='downloading…', exclusive=True).id})


@require_POST
def post_tag_characters(request, pk):
    """Run the second (PixAI) character model on one post."""
    post = get_object_or_404(Post, pk=pk)
    if not char_tagger.model_ready():
        return JsonResponse({'ok': False, 'error': 'character model not downloaded yet — settings → AI tagging'}, status=409)
    try:
        names = apply_character_model(post)
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)
    if names is None:
        return JsonResponse({'ok': False, 'error': 'no images'}, status=400)
    return JsonResponse({'ok': True, 'characters': names})


def _multi_posts():
    return Post.objects.annotate(_n=Count('images')).filter(_n__gt=1)


@require_POST
def ai_multi_retag(request):
    """Re-run the main AI tagger over EVERY item of the multi-image posts that have
    not had that yet (newest first). Resumable: a processed post is marked
    ai_multi=True. Adds tags only — nothing is removed."""
    def work(task):
        import time as _t
        ids = list(_multi_posts().filter(ai_multi=False).order_by('-id').values_list('id', flat=True))
        label = {'wd14': 'default', 'pixai': 'PixAI', 'custom': 'My model'}[_main_model()]
        task.total = len(ids); task.done = 0
        task.message = f'{label}: {len(ids)} multi-image post(s)…'
        task.save(update_fields=['total', 'done', 'message'])
        added = errors = 0
        for i, pid in enumerate(ids):
            check_cancel(task)
            try:
                post = Post.objects.get(pk=pid)
                before = post.tags.count()
                apply_ai_tags(post)
                added += max(0, post.tags.count() - before)
            except Exception as e:
                errors += 1
                print(f'multi re-tag error post {pid}: {e}')
            _throttled_save(task, done=i + 1,
                            message=f'{label}: {i + 1}/{len(ids)} posts · +{added} tag(s)' + (f' · {errors} error(s)' if errors else ''))
            _t.sleep(0)
        task.save(update_fields=['done', 'message'])

    return JsonResponse({'task_id': _start_task('ai_multi', work, message='starting…', exclusive=True).id})


@require_POST
def char_retag_start(request):
    """Re-tag characters with the PixAI model for up to `limit` posts that it
    has not checked yet (newest first; 0 = all). Resumable: each processed post
    is marked char_model='pixai', so the next run continues where this one
    stopped. Adds tags only — nothing is removed."""
    try:
        limit = max(0, int(json.loads(request.body or '{}').get('limit') or 0))
    except (TypeError, ValueError):
        limit = 0

    def work(task):
        import time as _t
        _ensure_char_model(task)
        qs = (Post.objects.exclude(char_model=char_tagger.MODEL_ID).filter(images__isnull=False)
              .distinct().order_by('-id').values_list('id', flat=True))
        ids = list(qs[:limit] if limit else qs)
        task.total = len(ids); task.done = 0
        task.message = f'checking {len(ids)} post(s)…'
        task.save(update_fields=['total', 'done', 'message'])
        found = errors = 0
        for i, pid in enumerate(ids):
            check_cancel(task)
            try:
                names = apply_character_model(Post.objects.get(pk=pid))
                found += len(names or [])
            except Exception as e:
                errors += 1
                print(f'character re-tag error post {pid}: {e}')
            _throttled_save(task, done=i + 1,
                            message=f'{i + 1}/{len(ids)} · {found} character tag(s) found' + (f' · {errors} error(s)' if errors else ''))
            _t.sleep(0)
        task.save(update_fields=['done', 'message'])

    return JsonResponse({'task_id': _start_task('char_retag', work, message='starting…', exclusive=True).id})


# ── My model: clone, teach, train ───────────────────────────────
import re as _re

CONCEPT_CATEGORIES = {'general', 'character', 'artist', 'meta', 'ai'}
_CONCEPT_NAME_RE = _re.compile(r"^[a-z0-9_().:'!+&-]{1,100}$")


def _concept_dict(c):
    return {
        'id': c.id, 'name': c.name, 'category': c.category, 'threshold': round(c.threshold, 3),
        'enabled': c.enabled, 'metrics': c.metrics, 'trained': bool(c.trained_at),
        'trained_at': c.trained_at.isoformat() if c.trained_at else None,
        'pos': getattr(c, 'pos', None), 'neg': getattr(c, 'neg', None),
        'applied': getattr(c, 'applied_n', 0),
    }


def custom_info(request):
    from django.db.models import Count, Q
    try:
        import onnx  # noqa: F401
        has_onnx = True
    except ImportError:
        has_onnx = False
    concepts = (CustomConcept.objects
                .annotate(pos=Count('examples', filter=Q(examples__label__gt=0)),
                          neg=Count('examples', filter=Q(examples__label__lt=0, examples__auto=False)),
                          applied_n=Count('applied', distinct=True)))
    return JsonResponse({
        'ready': custom_model.is_ready(), 'default_ready': ai_models.is_ready('wd14'), 'onnx': has_onnx,
        'concepts': [_concept_dict(c) for c in concepts],
        'features_cached': PostFeature.objects.filter(model_hash=custom_model.model_hash()).count() if custom_model.is_ready() else 0,
        'posts': Post.objects.count(),
    })


@require_POST
def custom_clone(request):
    if custom_model.is_ready():
        return JsonResponse({'error': 'already cloned — delete it first to re-clone'}, status=409)
    if not ai_models.is_ready('wd14'):
        return JsonResponse({'error': 'download the default model first'}, status=409)

    def work(task):
        def say(msg):
            task.message = msg
            task.save(update_fields=['message'])
        custom_model.clone_default(say)
        task.message = 'my model is ready'
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('model_clone', work, message='cloning…', exclusive=True).id})


def _normalize_concept_name(raw):
    return (raw or '').strip().lower().replace(' ', '_')


@require_POST
def custom_examples(request):
    """Teach by example. body: {ids:[post ids], concept, label: 1|-1, category?, also_tag?}
    +1: these posts show the concept (optionally also tag them now).
    -1: they do NOT (the tag is removed from them — how wrong auto-tags get corrected)."""
    data = json.loads(request.body or '{}')
    name = _normalize_concept_name(data.get('concept'))
    label = 1 if int(data.get('label', 1)) > 0 else -1
    ids = [int(i) for i in data.get('ids', []) if str(i).isdigit()]
    category = data.get('category') or 'ai'
    if not _CONCEPT_NAME_RE.match(name):
        return JsonResponse({'error': 'name: 1-100 chars of a-z 0-9 _ ( ) . : \' ! + & -'}, status=400)
    if category not in CONCEPT_CATEGORIES:
        return JsonResponse({'error': 'bad category'}, status=400)
    if not ids:
        return JsonResponse({'error': 'no posts selected'}, status=400)
    try:
        default_names = {n.strip().lower().replace(' ', '_') for n in _wd14_tags()[0]}
    except RuntimeError:
        default_names = set()
    if name in default_names:
        return JsonResponse({'error': f'"{name}" is already a tag the default model knows — pick a different name'}, status=409)

    concept, created = CustomConcept.objects.get_or_create(name=name, defaults={'category': category})
    posts = list(Post.objects.filter(pk__in=ids))
    for post in posts:
        CustomExample.objects.update_or_create(concept=concept, post=post,
                                               defaults={'label': label, 'auto': False})
    tagged = untagged = 0
    if label > 0 and data.get('also_tag', True):
        for post in posts:
            add_tags_to_post(post, [name], category=concept.category)
            tagged += 1
    elif label < 0:
        tag = Tag.objects.filter(name=name).first()
        if tag:
            for post in posts:
                if post.tags.filter(pk=tag.pk).exists():
                    post.tags.remove(tag)
                    untagged += 1
            CustomApplied.objects.filter(concept=concept, post__in=posts).delete()
            tag.update_count()
    return JsonResponse({'ok': True, 'concept': name, 'created': created, 'examples': len(posts),
                         'tagged': tagged, 'untagged': untagged,
                         'pos': concept.examples.filter(label__gt=0).count(),
                         'neg': concept.examples.filter(label__lt=0, auto=False).count()})


@require_POST
def custom_train(request, pk):
    concept = get_object_or_404(CustomConcept, pk=pk)
    if not custom_model.is_ready():
        return JsonResponse({'error': 'clone the default model first'}, status=409)

    def work(task):
        c = CustomConcept.objects.get(pk=pk)
        task.message = f'training “{c.name}”…'
        task.save(update_fields=['message'])

        def prog(done, total):
            _throttled_save(task, done=done, total=total, message=f'“{c.name}”: features {done}/{total}')

        m = custom_model.train_concept(c, progress=prog, check=lambda: check_cancel(task))
        task.message = (f'“{c.name}”: precision ≈ {round(m["precision_est"] * 100)}%, recall {round(m["recall"] * 100)}% '
                        f'(cross-validated, {m["n_pos"]}+ / {m["n_neg"]}− examples)')
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('custom_train', work, message=f'training “{concept.name}”…', exclusive=True).id})


@require_POST
def custom_scan(request):
    """Index the whole library with my model (features cached; resumable)."""
    if not custom_model.is_ready():
        return JsonResponse({'error': 'clone the default model first'}, status=409)

    def work(task):
        total_posts = Post.objects.count()
        task.message = 'indexing the library…'
        task.save(update_fields=['message'])

        def prog(done, total):
            _throttled_save(task, done=done, total=total, message=f'indexed {done}/{total} new post(s)')

        done, failed = custom_model.scan_library(prog, lambda: check_cancel(task))
        task.message = f'indexed {done} post(s)' + (f', {failed} unreadable' if failed else '') + f' (library: {total_posts})'
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('custom_scan', work, message='indexing…', exclusive=True).id})


def custom_candidates(request, pk):
    """Posts to review for a concept (open them with /?ids=…)."""
    concept = get_object_or_404(CustomConcept, pk=pk)
    try:
        n = max(1, min(200, int(request.GET.get('n', 60))))
    except ValueError:
        n = 60
    mode = 'uncertain' if request.GET.get('mode') == 'uncertain' else 'top'
    ids, scores, n_above = custom_model.candidates(concept, n, mode)
    return JsonResponse({'ids': ids, 'scores': scores, 'mode': mode, 'above_threshold': n_above,
                         'indexed': PostFeature.objects.filter(model_hash=custom_model.model_hash()).count(),
                         'posts': Post.objects.count()})


@require_POST
def custom_apply(request, pk):
    concept = get_object_or_404(CustomConcept, pk=pk)
    if not concept.trained_at:
        return JsonResponse({'error': 'train it first'}, status=409)

    def work(task):
        def prog(done, total):
            _throttled_save(task, done=done, total=total, message=f'tagging “{concept.name}” {done}/{total}')
        n = custom_model.apply_concept(concept, prog, lambda: check_cancel(task))
        task.message = f'“{concept.name}” added to {n} post(s) — undo it in Settings → My model'
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('custom_apply', work, message=f'applying “{concept.name}”…', exclusive=True).id})


@require_POST
def custom_undo(request, pk):
    concept = get_object_or_404(CustomConcept, pk=pk)
    return JsonResponse({'ok': True, 'untagged': custom_model.undo_concept(concept)})


@require_POST
def custom_concept_update(request, pk):
    c = get_object_or_404(CustomConcept, pk=pk)
    data = json.loads(request.body or '{}')
    if 'threshold' in data:
        c.threshold = min(0.999, max(0.05, float(data['threshold'])))
    if 'enabled' in data:
        c.enabled = bool(data['enabled'])
    if data.get('category') in CONCEPT_CATEGORIES and data['category'] != c.category:
        Tag.objects.filter(name=c.name, category__in=['ai', c.category]).update(category=data['category'])
        c.category = data['category']
    c.save()
    custom_model.update_head(c.name, thr=c.threshold, enabled=c.enabled)
    return JsonResponse({'ok': True, 'concept': _concept_dict(c)})


@require_POST
def custom_concept_delete(request, pk):
    """Forget a concept: its head and examples go; tags already on posts stay."""
    c = get_object_or_404(CustomConcept, pk=pk)
    custom_model.remove_head(c.name)
    c.delete()
    return JsonResponse({'ok': True})


# ── AI models panel: status / download / delete / main selector / free VRAM ──
AI_TASK_KINDS = ['ai_tag', 'ai_multi', 'char_retag', 'model_clone', 'custom_train', 'custom_scan', 'custom_apply']


def ai_models_info(request):
    """Everything the settings panel's AI models section shows. Loads no model."""
    from . import ai_runtime
    return JsonResponse({
        'models': [ai_models.status(k) for k in ai_models.KEYS],
        'main': _main_model(),
        'main_pref': _pref('aiMainModel', 'wd14'),
        'taught': list(CustomConcept.objects.filter(enabled=True).order_by('name').values_list('name', flat=True)),
        'gpu': ai_runtime.gpu_info(),
        'runtime': ai_runtime.runtime_info(),
        'busy': Task.objects.filter(status='running', kind__in=AI_TASK_KINDS).exists(),
    })


@require_POST
def ai_model_download(request, key):
    if key not in ai_models.HF_KEYS:
        return JsonResponse({'error': 'unknown or not downloadable model'}, status=400)
    info = ai_models.MODELS[key]

    def work(task):
        task.message = f"downloading {info['name']}…"
        task.total = round(info['bytes'] / 1048576)
        task.save(update_fields=['message', 'total'])

        def prog(done, total):
            _throttled_save(task, interval=2, done=min(round(done / 1048576), task.total))

        ai_models.download(key, prog)
        task.done = task.total
        task.message = f"{info['name']} ready"
        task.save(update_fields=['done', 'message'])

    return JsonResponse({'task_id': _start_task('model_download', work, message='downloading…', exclusive=True).id})


@require_POST
def ai_model_delete(request, key):
    from . import ai_runtime
    if key not in ai_models.KEYS:
        return JsonResponse({'error': 'unknown model'}, status=404)
    if Task.objects.filter(status='running', kind__in=AI_TASK_KINDS + ['model_download']).exists():
        return JsonResponse({'error': 'wait for (or stop) the running AI tasks first'}, status=409)
    if _main_model() == key:
        return JsonResponse({'error': 'it is the selected main tagger — switch to another one first'}, status=409)
    ai_runtime.forget(key)
    ai_runtime.request_unload_all()          # other workers may hold sessions on the files
    freed = ai_models.delete(key)
    if key == 'custom':
        custom_model.on_clone_deleted()      # its caches/heads are meaningless now; examples are kept
    return JsonResponse({'ok': True, 'freed_mb': round(freed / 1048576)})


@require_POST
def ai_main_set(request):
    key = json.loads(request.body or '{}').get('model')
    if key not in ('wd14', 'pixai', 'custom'):
        return JsonResponse({'error': 'unknown model'}, status=400)
    if not ai_models.is_ready(key):
        return JsonResponse({'error': 'that model is not on disk yet — download/clone it first'}, status=409)
    _set_pref('aiMainModel', key)
    return JsonResponse({'ok': True, 'main': key})


@require_POST
def ai_free_vram(request):
    """Unload every AI model in every worker, no matter what: running AI tasks
    are asked to stop (they would just load the model again), a signal makes all
    gunicorn workers drop their sessions within seconds, and this one drops now."""
    from . import ai_runtime
    before = ai_runtime.gpu_info()
    stopped = Task.objects.filter(status='running', kind__in=AI_TASK_KINDS).update(cancel_requested=True)
    ai_runtime.request_unload_all()
    here = ai_runtime.force_unload()
    return JsonResponse({'ok': True, 'tasks_stopped': stopped, 'unloaded_here': here, 'before': before})


@require_POST
def ai_restart_workers(request):
    """Graceful reload of the gunicorn workers: also releases each worker's
    CUDA context (~120 MiB) that model unloading cannot. Running background
    tasks stop with their worker."""
    import signal
    ppid = os.getppid()
    try:
        with open(f'/proc/{ppid}/cmdline', 'rb') as f:
            cmd = f.read().decode(errors='ignore')
    except OSError:
        cmd = ''
    if 'gunicorn' not in cmd:
        return JsonResponse({'error': 'not running under gunicorn'}, status=409)
    threading.Timer(1.0, lambda: os.kill(ppid, signal.SIGHUP)).start()
    return JsonResponse({'ok': True})


# ── Debug overlay helpers ───────────────────────────────────────
def debug_stats(request):
    """Counters for the settings panel's debug section."""
    return JsonResponse({
        'posts': Post.objects.count(),
        'ai_tagged': Post.objects.filter(ai_tagged=True).count(),
        'char_tagged': Post.objects.filter(char_tagged=True).count(),
        'pixai': Post.objects.filter(char_model='pixai').count(),
        'ai_not_char': Post.objects.filter(ai_tagged=True, char_tagged=False).count(),
        'unprocessed': Post.objects.filter(ai_tagged=False).exclude(char_model='pixai').filter(images__isnull=False).distinct().count(),
        'no_ai': Post.objects.filter(ai_tagged=False, images__isnull=False).distinct().count(),
        'no_pixai': Post.objects.exclude(char_model='pixai').filter(images__isnull=False).distinct().count(),
        'in_folder': Post.objects.filter(folders__isnull=False).distinct().count(),
        'character_tags': Tag.objects.filter(category='character').count(),
    })


@require_POST
def recategorize_characters(request):
    """Move tags that are WD14 character names out of category 'ai' into
    'character' (the old tagger saved every AI tag as 'ai'). Only touches tags
    still in 'ai' — anything you categorised yourself is left alone. Pure
    category change: no image is re-tagged."""
    names, is_char = _wd14_tags()
    char_names = {n.strip().lower().replace(' ', '_') for n, c in zip(names, is_char) if c}
    updated = 0
    ids = [i for i, n in Tag.objects.filter(category='ai').values_list('id', 'name') if n in char_names]
    for k in range(0, len(ids), 500):
        updated += Tag.objects.filter(pk__in=ids[k:k + 500]).update(category='character')
    return JsonResponse({'ok': True, 'updated': updated})


# ── Database backups (logic in gallery/backup.py) ───────────────
from . import backup as _backup


def backup_info(request):
    return JsonResponse(_backup.info())


@require_POST
def backup_config_save(request):
    try:
        cfg = _backup.save_config(json.loads(request.body or '{}'))
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'ok': True, 'config': cfg})


@require_POST
def backup_run(request):
    return JsonResponse({'task_id': _start_task('backup', _backup.run_backup, message='backing up…', exclusive=True).id})


def _start_restore(path, label):
    if Task.objects.filter(status='running').exists():
        return JsonResponse({'error': 'wait for (or stop) the running tasks first'}, status=409)
    try:
        _backup.validate_backup(path)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

    def work(task):
        _backup.restore_from(path, task)
        task.message = f'restored from {label}'
        task.save(update_fields=['message'])

    return JsonResponse({'task_id': _start_task('restore', work, message='restoring…').id})


@require_POST
def backup_restore(request):
    """body: {name} — a file currently listed in the backup folder."""
    name = (json.loads(request.body or '{}').get('name') or '')
    if name != os.path.basename(name) or name not in {b['name'] for b in _backup.list_backups()}:
        return JsonResponse({'error': 'no such backup'}, status=404)
    return _start_restore(os.path.join(_backup.get_config()['path'], name), name)


@require_POST
def backup_upload_restore(request):
    """multipart 'file' — a .sqlite3 uploaded from the browser; kept in the
    backup folder (as uploaded-<time>.sqlite3) and then restored."""
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'no file'}, status=400)
    if Task.objects.filter(status='running').exists():
        return JsonResponse({'error': 'wait for (or stop) the running tasks first'}, status=409)
    d = _backup.get_config()['path']
    try:
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, f'uploaded-{timezone.now().strftime("%Y%m%d-%H%M%S")}.sqlite3')
        with open(dest, 'wb') as out:
            for chunk in f.chunks():
                out.write(chunk)
    except OSError as e:
        return JsonResponse({'error': f'cannot save upload: {e.strerror or e}'}, status=500)
    resp = _start_restore(dest, f.name)
    if resp.status_code != 200:
        try: os.remove(dest)
        except OSError: pass
    return resp


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
    def push(recents):
        recents = recents if isinstance(recents, list) else []
        # remove any existing entry for this post (so it moves to front)
        recents = [r for r in recents if r.get('id') != entry['id']]
        recents.insert(0, entry)
        return recents[:50]

    try:
        recents = prefs.update('recentPosts', push, default=[])
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'ok': True, 'recents': recents})


def login_view(request):
    from django.conf import settings
    if request.method == 'POST':
        pwd = request.POST.get('password', '')
        if pwd == getattr(settings, 'GALLERY_PASSWORD', ''):
            request.session['booru_authed'] = True
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


# ── /media/ with HTTP Range support ────────────────────────────
# django.views.static.serve ignores Range, so Chromium could neither seek in a
# video nor stream it progressively (it waits for the whole file). This view
# answers Range requests with 206 and always advertises Accept-Ranges.
_RANGE_RE = re.compile(r'^bytes=(\d*)-(\d*)$')
MEDIA_CHUNK = 1024 * 1024


def _parse_range(header, size):
    """'bytes=a-b' | 'a-' | '-n' -> (start, end) inclusive, None if the header is
    not a single usable range, False if it is syntactically fine but unsatisfiable.
    (Multi-range requests are answered with their first range.)"""
    m = _RANGE_RE.match((header or '').split(',')[0].strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if not m.group(1):                                  # suffix: last n bytes
        n = int(m.group(2))
        if n == 0:
            return False
        return (max(0, size - n), size - 1)
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    if start >= size or end < start:
        return False
    return (start, end)


def _iter_file(path, start, end):
    left = end - start + 1
    with open(path, 'rb') as f:
        f.seek(start)
        while left > 0:
            chunk = f.read(min(MEDIA_CHUNK, left))
            if not chunk:
                break
            left -= len(chunk)
            yield chunk


def media_serve(request, path):
    import mimetypes
    from django.utils._os import safe_join
    from django.utils.http import http_date, parse_http_date_safe
    from django.views.static import was_modified_since
    try:
        full = safe_join(str(settings.MEDIA_ROOT), path)
    except Exception:                                   # traversal attempt
        raise Http404
    if not os.path.isfile(full):
        raise Http404
    st = os.stat(full)
    size = st.st_size
    ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
    if not was_modified_since(request.META.get('HTTP_IF_MODIFIED_SINCE'), st.st_mtime):
        resp = HttpResponseNotModified()
        resp['Last-Modified'] = http_date(st.st_mtime)
        resp['Accept-Ranges'] = 'bytes'
        return resp
    rng = None
    hdr = request.META.get('HTTP_RANGE')
    if hdr:
        # If-Range: only honour the range when the file is unchanged since that date
        ifr = request.META.get('HTTP_IF_RANGE')
        if ifr and (parse_http_date_safe(ifr) is None or int(st.st_mtime) > parse_http_date_safe(ifr)):
            hdr = None
        rng = _parse_range(hdr, size) if hdr else None
    if rng is False:
        resp = HttpResponse(status=416)
        resp['Content-Range'] = f'bytes */{size}'
        resp['Accept-Ranges'] = 'bytes'
        return resp
    if rng is None:
        resp = FileResponse(open(full, 'rb'), content_type=ctype)
        resp['Content-Length'] = str(size)
    else:
        start, end = rng
        resp = StreamingHttpResponse(_iter_file(full, start, end), status=206, content_type=ctype)
        resp['Content-Length'] = str(end - start + 1)
        resp['Content-Range'] = f'bytes {start}-{end}/{size}'
    resp['Accept-Ranges'] = 'bytes'
    resp['Last-Modified'] = http_date(st.st_mtime)
    resp['Content-Disposition'] = f'inline; filename="{os.path.basename(full)}"'.encode('ascii', 'replace').decode()
    return resp
