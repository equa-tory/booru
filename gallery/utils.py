import os
import re
import hashlib
import subprocess
from PIL import Image
from django.conf import settings


def natural_key(path):
    """Sort key for human/natural filename ordering so that
    pageRandName_2 < pageRandName_10, and 2 < 10 (not '10' < '2').
    Only ASCII [0-9] runs are treated as numbers — characters like the
    superscript '⁹' satisfy str.isdigit() but blow up int(), so we guard
    with isascii() and use a tuple key to avoid int-vs-str comparisons."""
    base = os.path.basename(str(path)).lower()
    out = []
    for t in re.split(r'([0-9]+)', base):
        if t.isascii() and t.isdigit():
            out.append((1, int(t), ''))
        else:
            out.append((0, 0, t))
    return out

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
VIDEO_EXTS     = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.m4a', '.3gp'}
PDF_EXTS       = {'.pdf'}
ALL_EXTS       = SUPPORTED_EXTS | VIDEO_EXTS | PDF_EXTS

GIF_EXTS       = {'.gif'}
AUTO_TAGS      = {
    '.gif':  ['gif', 'animated'],
    '.mp4':  ['video', 'mp4'],
    '.mov':  ['video', 'mov'],
    '.webm': ['video', 'webm'],
    '.mkv':  ['video', 'mkv'],
    '.avi':  ['video', 'avi'],
    '.m4v':  ['video', 'm4v'],
    '.m4a':  ['audio', 'm4a'],
    '.3gp':  ['video', '3gp'],
    '.pdf':  ['pdf'],
}


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS

def is_pdf(path):
    return os.path.splitext(path)[1].lower() in PDF_EXTS

def make_pdf_thumb(src_path, max_size=360):
    """Render first page of a PDF to a JPEG thumbnail.

    Tries PyMuPDF (pip install pymupdf) first — it needs no system binary —
    then falls back to the `pdftoppm` CLI (poppler-utils). If neither is
    available a placeholder is produced instead of raising.
    """
    thumb_path = _thumb_path_for(src_path)
    if not os.path.exists(src_path):
        print(f"PDF thumb skip (missing file): {src_path}")
        return _make_placeholder(src_path, max_size)

    # 1) PyMuPDF (fitz) — pure wheel, no system dependency.
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(src_path)
        if doc.page_count > 0:
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=96)
            from PIL import Image as PILImage
            import io
            img = PILImage.open(io.BytesIO(pix.tobytes('png')))
            img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(thumb_path, 'JPEG', quality=88)
            doc.close()
            return thumb_path
        doc.close()
    except ImportError:
        pass  # PyMuPDF not installed — fall through to pdftoppm
    except Exception as e:
        print(f"PyMuPDF thumb error {src_path}: {e}")

    # 2) pdftoppm CLI (poppler-utils).
    try:
        h = hashlib.md5(src_path.encode()).hexdigest()[:12]
        tmp_prefix = f'/tmp/booru_pdft_{h}'
        result = subprocess.run(
            ['pdftoppm', '-jpeg', '-r', '96', '-l', '1', src_path, tmp_prefix],
            capture_output=True, timeout=60
        )
        # pdftoppm writes tmp_prefix-1.jpg
        tmp_file = f'{tmp_prefix}-1.jpg'
        if result.returncode == 0 and os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 100:
            from PIL import Image as PILImage
            img = PILImage.open(tmp_file)
            img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            if img.mode != 'RGB': img = img.convert('RGB')
            img.save(thumb_path, 'JPEG', quality=88)
            try: os.remove(tmp_file)
            except OSError: pass
            return thumb_path
        print(f"pdftoppm failed ({result.returncode}) for {src_path}: {result.stderr[:200].decode(errors='ignore')}")
    except FileNotFoundError:
        # Neither PyMuPDF nor poppler-utils available.
        print("PDF thumb: no renderer found. Install one of: "
              "`pip install pymupdf`  OR  `apt install poppler-utils`.")
    except Exception as e:
        print(f"PDF thumb error {src_path}: {e}")
    return _make_placeholder(src_path, max_size)


def auto_tags_for(path):
    """Return format-based tags for a file."""
    ext = os.path.splitext(path)[1].lower()
    return AUTO_TAGS.get(ext, [])


def compute_phash(path, hash_size=8):
    try:
        img = Image.open(path).convert('L').resize((hash_size*2, hash_size*2), Image.LANCZOS)
        import numpy as np
        arr  = np.array(img, dtype=float)
        diff = arr[:, 1:] > arr[:, :-1]
        bits = diff.flatten()[:hash_size*hash_size]
        h = 0
        for b in bits: h = (h << 1) | int(b)
        return format(h, '016x')
    except Exception:
        return ''


def phash_distance(a, b):
    if not a or not b or len(a) != len(b): return 999
    return bin(int(a, 16) ^ int(b, 16)).count('1')


def _thumb_path_for(src_path):
    """Canonical thumb path for any file — always the same name."""
    h = hashlib.md5(src_path.encode()).hexdigest()[:16]
    return os.path.join(settings.MEDIA_ROOT, 'thumbs', f'{h}.jpg')


def _video_duration(src_path):
    """Video duration in seconds via ffprobe (0.0 if unknown)."""
    try:
        r = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', src_path
        ], capture_output=True, timeout=30)
        return float(r.stdout.decode().strip())
    except Exception:
        return 0.0


def make_video_thumb(src_path, max_size=360, pct=0):
    """Extract a frame from the video for the thumbnail.
    pct = where in the video to grab the frame (0-100). 0 → first usable frame."""
    thumb_path = _thumb_path_for(src_path)
    seeks = []
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        pct = 0
    if pct and pct > 0:
        dur = _video_duration(src_path)
        if dur > 0:
            t = max(0.0, min(dur - 0.05, dur * pct / 100.0))
            seeks.append(f'{t:.3f}')
    # fall back to the usual first-usable-frame seeks
    seeks += ['00:00:01', '00:00:00']
    result = None
    for seek in seeks:
        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-ss', seek,          # BEFORE -i = fast input seek (jumps to the
                '-i', src_path,       # nearest keyframe instead of decoding the
                '-frames:v', '1',     # whole file up to the timestamp, which was
                '-an',                # timing out on long videos)
                '-vf', f'scale={max_size}:{max_size}:force_original_aspect_ratio=decrease',
                '-q:v', '2',
                thumb_path
            ], capture_output=True, timeout=30)
        except FileNotFoundError:
            print("Video thumb: ffmpeg not found. Install it with `apt install ffmpeg`.")
            return _make_placeholder(src_path, max_size)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 500:
            return thumb_path
    if result is not None:
        print(f"ffmpeg failed for {src_path}: {result.stderr[-300:].decode(errors='ignore')}")
    return _make_placeholder(src_path, max_size)


def _make_placeholder(src_path, max_size=360):
    """Dark placeholder with play icon — uses same canonical path as ffmpeg."""
    thumb_path = _thumb_path_for(src_path)
    try:
        from PIL import ImageDraw
        img = Image.new('RGB', (max_size, max_size), (20, 20, 26))
        d = ImageDraw.Draw(img)
        cx, cy = max_size // 2, max_size // 2
        r = max_size // 4
        pts = [(cx - r//2, cy - r), (cx - r//2, cy + r), (cx + r, cy)]
        d.polygon(pts, fill=(124, 106, 247))
        img.save(thumb_path, 'JPEG', quality=85)
        return thumb_path
    except Exception:
        return ''


def make_thumb(src_path, max_size=360, pct=0):
    if is_video(src_path):
        return make_video_thumb(src_path, max_size, pct=pct)
    if is_pdf(src_path):
        return make_pdf_thumb(src_path, max_size)
    try:
        img = Image.open(src_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (20, 20, 26))
            if img.mode == 'P': img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        thumb_path = _thumb_path_for(src_path)
        img.save(thumb_path, 'JPEG', quality=85)
        return thumb_path
    except Exception as e:
        print(f"Thumb error {src_path}: {e}")
        return ''


def ingest_photo(path, post, order=0):
    from gallery.models import Photo
    if Photo.objects.filter(file_path=path).exists():
        return Photo.objects.get(file_path=path)
    vid = is_video(path)
    pdf = is_pdf(path)
    if vid or pdf:
        w, h = 0, 0
    else:
        try:
            img = Image.open(path)
            w, h = img.size
        except Exception:
            w, h = 0, 0
    size  = os.path.getsize(path)
    thumb = make_thumb(path)
    # still images: phash the file. videos: phash the generated thumbnail so
    # they can take part in duplicate detection (video-vs-video).
    if vid:
        ph = compute_phash(thumb) if thumb else ''
    elif pdf:
        ph = ''
    else:
        ph = compute_phash(path)
    return Photo.objects.create(
        post=post, order=order,
        file_path=path, thumb_path=thumb,
        width=w, height=h, file_size=size, phash=ph,
        is_video=vid,
    )


def create_post_from_files(paths, title=''):
    from gallery.models import Post
    post = Post.objects.create(title=title)
    for i, path in enumerate(sorted(paths, key=natural_key)):
        ingest_photo(path, post, order=i)
    # add format-based auto tags
    for path in paths:
        tags = auto_tags_for(path)
        if tags:
            add_tags_to_post(post, tags, category='meta')
    return post


def scan_inbox():
    from gallery.models import Photo
    inbox     = os.path.join(settings.MEDIA_ROOT, 'inbox')
    multi_dir = os.path.join(inbox, '_')
    existing  = set(Photo.objects.values_list('file_path', flat=True))

    folder_to_post = {}
    for photo in Photo.objects.filter(
        file_path__contains=os.sep + '_' + os.sep
    ).select_related('post'):
        folder = os.path.dirname(photo.file_path)
        if folder not in folder_to_post and photo.post:
            folder_to_post[folder] = photo.post

    new_posts   = []
    extend_post = []

    if os.path.isdir(multi_dir):
        for folder_name in sorted(os.listdir(multi_dir), key=natural_key):
            folder_path = os.path.join(multi_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            new_files = []
            for root, _, fnames in os.walk(folder_path):
                for f in sorted(fnames, key=natural_key):
                    if os.path.splitext(f)[1].lower() in ALL_EXTS:
                        full = os.path.join(root, f)
                        if full not in existing:
                            new_files.append(full)
            if not new_files:
                continue
            new_files.sort(key=natural_key)
            if folder_path in folder_to_post:
                extend_post.append((folder_to_post[folder_path], new_files))
            else:
                new_posts.append((folder_name, new_files))

    for root, dirs, files in os.walk(inbox):
        dirs[:] = [d for d in dirs if d != '_']
        for f in sorted(files, key=natural_key):
            if os.path.splitext(f)[1].lower() in ALL_EXTS:
                full = os.path.join(root, f)
                if full not in existing:
                    new_posts.append((os.path.splitext(f)[0], [full]))

    return new_posts, extend_post


def retag_all_videos():
    """Regenerate thumbnails for all video Photos. Returns count updated."""
    from gallery.models import Photo
    updated = 0
    for photo in Photo.objects.filter(is_video=True):
        thumb = make_video_thumb(photo.file_path)
        if thumb and thumb != photo.thumb_path:
            photo.thumb_path = thumb
            photo.save(update_fields=['thumb_path'])
        updated += 1
    return updated


def add_tags_to_post(post, tag_names, category='general'):
    from gallery.models import Tag
    for name in tag_names:
        name = name.strip().lower().replace(' ', '_')
        if not name: continue
        tag, _ = Tag.objects.get_or_create(name=name, defaults={'category': category})
        post.tags.add(tag)
    for tag in post.tags.all():
        tag.update_count()


def delete_post(post, also_files=False):
    from gallery.models import Tag
    tags = list(post.tags.all())
    for photo in post.images.all():
        if also_files and os.path.exists(photo.file_path):
            try: os.remove(photo.file_path)
            except OSError: pass
        if photo.thumb_path and os.path.exists(photo.thumb_path):
            try: os.remove(photo.thumb_path)
            except OSError: pass
    post.delete()
    for tag in tags:
        tag.update_count()
    from gallery.models import Tag as T
    T.objects.filter(count=0).delete()


def _safe_folder_name(name):
    """Make a string safe for use as a folder name."""
    import re
    name = re.sub(r'[<>:"/\\|?*]', '_', name or '')
    name = name.strip().strip('.')
    return name[:120] or 'post'


def move_post_to_folder(post, folder_basename):
    """
    Move all of a post's files into MEDIA_ROOT/inbox/_/<folder_basename>/.
    Updates Photo.file_path in DB. Thumbnails are left where they are
    (they're keyed by hash of the original path, so we regenerate path key).
    Returns the new folder path.
    """
    folder_name = _safe_folder_name(folder_basename)
    dest_dir = os.path.join(settings.MEDIA_ROOT, 'inbox', '_', folder_name)

    # avoid collision: if dir exists and belongs to a different post, append number
    base_dest = dest_dir
    n = 1
    while os.path.isdir(dest_dir) and not _dir_belongs_to_post(dest_dir, post):
        dest_dir = f'{base_dest}_{n}'
        n += 1

    os.makedirs(dest_dir, exist_ok=True)

    for photo in post.images.order_by('order', 'id'):
        src = photo.file_path
        if not os.path.exists(src):
            continue
        fname = os.path.basename(src)
        dst = os.path.join(dest_dir, fname)
        # avoid overwriting different file with same name
        if os.path.exists(dst) and os.path.abspath(dst) != os.path.abspath(src):
            stem, ext = os.path.splitext(fname)
            k = 1
            while os.path.exists(dst):
                dst = os.path.join(dest_dir, f'{stem}_{k}{ext}')
                k += 1
        if os.path.abspath(dst) != os.path.abspath(src):
            try:
                import shutil
                shutil.move(src, dst)
            except Exception as e:
                print(f"move error {src} -> {dst}: {e}")
                continue
        # regenerate thumb at new path key
        new_thumb = make_thumb(dst)
        photo.file_path = dst
        if new_thumb:
            photo.thumb_path = new_thumb
        photo.save(update_fields=['file_path', 'thumb_path'])

    return dest_dir


def _dir_belongs_to_post(dir_path, post):
    """Check if any of post's photos already live in dir_path."""
    photo_dirs = {os.path.dirname(p) for p in post.images.values_list('file_path', flat=True)}
    return os.path.abspath(dir_path) in {os.path.abspath(d) for d in photo_dirs}
