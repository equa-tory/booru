import os
import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.views.decorators.http import require_POST
from django.conf import settings

from .models import Post, Photo, Tag
from .utils import (scan_inbox, create_post_from_files, ingest_photo,
                    add_tags_to_post, delete_post, phash_distance, make_thumb,
                    make_video_thumb, retag_all_videos)


# ── Helpers ────────────────────────────────────────────────────

def _build_post_qs(request):
    q_tags     = request.GET.getlist('tag')
    min_rating = request.GET.get('min_rating', '')
    exact_rating = request.GET.get('rating', '')  # exact rating filter
    fav_only   = request.GET.get('fav', '')
    multi_only  = request.GET.get('multi_only', '')
    single_only = request.GET.get('single_only', '')
    sort_by     = request.GET.get('sort', 'new')   # new | old | rating | fav | random

    posts = Post.objects.prefetch_related('tags', 'images').all()

    if q_tags:
        for t in q_tags:
            posts = posts.filter(tags__name=t)
        posts = posts.distinct()

    if min_rating.isdigit():
        posts = posts.filter(rating__gte=int(min_rating))

    if exact_rating.isdigit():
        posts = posts.filter(rating=int(exact_rating))

    if fav_only == '1':
        posts = posts.filter(fav=True)

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
    elif sort_by == 'random':
        posts = posts.order_by('?')
    else:  # new (default)
        posts = posts.order_by('-added_at')

    return posts, q_tags, sort_by, multi_only, single_only


# ── Pages ──────────────────────────────────────────────────────

def index(request):
    posts, q_tags, sort_by, multi_only, single_only = _build_post_qs(request)

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

    is_htmx = request.headers.get('HX-Request')
    scroll_mode = request.GET.get('scroll', '0') == '1'
    if is_htmx:
        return render(request, 'gallery/_photo_grid.html', {
            'page_obj': page_obj, 'q_tags': q_tags,
            'scroll_mode': scroll_mode, 'request': request,
        })


    # Build base query string (everything except page) for pagination links
    p = request.GET.copy()
    p.pop('page', None)
    base_qs = ('&' + p.urlencode()) if p else ''

    return render(request, 'gallery/index.html', {
        'page_obj': page_obj,
        'popular_tags': popular_tags,
        'q_tags': q_tags,
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
        'sort_options': [('new','newest'),('old','oldest'),('rating','rating'),('fav','favs'),('random','random')],
    })


def posts_json(request):
    """JSON API for infinite scroll — returns page of posts as JSON."""
    posts, q_tags, sort_by, multi_only, single_only = _build_post_qs(request)
    paginator = Paginator(posts, 40)
    page_obj  = paginator.get_page(request.GET.get('page', 1))

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
            'url':        f'/post/{post.pk}/',
        })

    # build tag query string to embed in post URLs
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
    for key in ('tag', 'sort', 'min_rating', 'rating', 'fav', 'multi_only', 'single_only'):
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
    post_data = []
    for post in posts:
        cover = post.cover
        if not cover or not cover.phash:
            continue
        src = cover.thumb_path or cover.file_path
        rel = os.path.relpath(src, settings.MEDIA_ROOT)
        post_data.append({
            'post_id':   post.pk,
            'post':      post,
            'phash':     cover.phash,
            'is_gif':    cover.is_gif,
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

    groups = []
    used   = set()
    for i, p in enumerate(post_data):
        if p['post_id'] in used: continue
        group = [p]
        for j, q in enumerate(post_data):
            if i == j or q['post_id'] in used: continue
            # never compare two photos from the same post
            if p['post_id'] == q['post_id']: continue
            # don't compare animated GIFs against still images
            if p['is_gif'] != q['is_gif']: continue
            pair = tuple(sorted([p['post_id'], q['post_id']]))
            if pair in ignored_pairs: continue
            if phash_distance(p['phash'], q['phash']) <= 8:
                group.append(q)
                used.add(q['post_id'])
        if len(group) > 1:
            used.add(p['post_id'])
            groups.append(group)

    return render(request, 'gallery/duplicates.html', {'groups': groups})


# ── Scan / upload ──────────────────────────────────────────────

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
    if rating is not None:
        post.rating = max(0, min(5, int(rating)))
    if fav is not None:
        post.fav = bool(fav)
    post.save(update_fields=['rating', 'fav'])
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

    return JsonResponse({'error': 'unknown action'}, status=400)


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
    tags = Tag.objects.filter(count__gt=0).order_by('-fav', '-count')
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

    # Provide cover image URLs of neighbours for preloading
    def cover_urls(post_id):
        if not post_id: return []
        try:
            p = Post.objects.prefetch_related('images').get(pk=post_id)
        except Post.DoesNotExist:
            return []
        return [img.media_url for img in p.images.order_by('order', 'id')[:3]]

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
    for src_id in source_ids:
        try:
            src = Post.objects.get(pk=src_id)
        except Post.DoesNotExist:
            continue
        for i, photo in enumerate(src.images.order_by('order', 'id')):
            photo.post  = target
            photo.order = max_order + i
            photo.save(update_fields=['post', 'order'])
        max_order += src.images.count()
        for tag in src.tags.all():
            target.tags.add(tag)
        src.delete()
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

    return JsonResponse({'ok': True, 'post_id': target.pk})


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
