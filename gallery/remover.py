"""Remove painted objects from a picture (✨ create → 🖌 remove) with LaMa, an object-removal
model (512×512 ONNX, ~200 MB in AI_MODELS_DIR). Only the painted pixels change; the rest of the
picture keeps its original pixels and full resolution.

Steps per painted blob: if it is a light (bright inside, e.g. a street lamp), its halo is
subtracted first (`deglow`: per-radius brightness above the far surroundings), otherwise a
hole would sit inside a bright ring; then the blob is filled tile by tile (square crops around
it, each scaled to 512) so big pictures keep detail. Tested on a 2682×1790 night photo:
11 lamps + poles gone, nothing else changed (~45 s on the CPU).
"""
import io
import os
from collections import deque

import numpy as np
from django.conf import settings
from PIL import Image, ImageFilter

LAMA_URL = 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx'
LAMA_SIZE = 208044816
TILE_MIN, TILE_MAX = 384, 2048


def model_path():
    d = str(getattr(settings, 'AI_MODELS_DIR', os.path.join(settings.BASE_DIR, 'ai_models')))
    return os.path.join(d, 'lama_fp32.onnx')


def is_ready():
    p = model_path()
    return os.path.isfile(p) and os.path.getsize(p) > 100_000_000


def download(progress=None):
    """Fetch the model (Xet off, like the other HF downloads); atomic rename at the end."""
    import httpx
    p = model_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.part'
    with httpx.Client(timeout=60.0, follow_redirects=True) as c, c.stream('GET', LAMA_URL) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length') or LAMA_SIZE)
        got = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
    os.replace(tmp, p)


def _lama(sess, img, m):
    a = np.asarray(img.resize((512, 512), Image.LANCZOS), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    mm = (np.asarray(m.resize((512, 512), Image.NEAREST), dtype=np.float32) > 127).astype(np.float32)[None, None]
    out = sess.run(None, {'image': a, 'mask': mm})[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize(img.size, Image.LANCZOS)


def blobs(mask, scale=4):
    """Connected painted areas -> [(x0, y0, x1, y1)] in full-size pixels (labelled on a
    1/scale copy so a big brush stroke stays cheap)."""
    small = mask.resize((max(1, mask.width // scale), max(1, mask.height // scale)), Image.BILINEAR)
    m = np.asarray(small) > 40
    seen = np.zeros_like(m)
    H, W = m.shape
    out = []
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        y0 = y1 = y
        x0 = x1 = x
        while q:
            cy, cx = q.popleft()
            y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        out.append((x0 * scale, y0 * scale, (x1 + 1) * scale, (y1 + 1) * scale))
    return out


def deglow_at(a, x, y, g, reach=3.0, pct=35):
    """Subtract the halo of a light at (x, y) with core radius g from float array `a` (in place)."""
    H, W = a.shape[:2]
    R = int(max(g, 8) * reach)
    x0, x1, y0, y1 = max(0, x - R), min(W, x + R + 1), max(0, y - R), min(H, y + R + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    sub = a[y0:y1, x0:x1]
    step = max(2, R // 80)
    bins = np.arange(0, R + step, step)
    prof = np.zeros((len(bins), 3))
    for i, b in enumerate(bins):
        ring = (d >= b) & (d < b + step)
        prof[i] = np.percentile(sub[ring], pct, axis=0) if ring.sum() > 3 else (prof[i - 1] if i else 0)
    base = prof[-max(3, len(bins) // 10):].mean(0)
    glow = np.maximum.accumulate(np.maximum(prof - base, 0)[::-1], axis=0)[::-1]
    glow *= np.clip((R - bins) / (R * 0.25), 0, 1)[:, None]
    idx = np.clip((d / step).astype(int), 0, len(bins) - 1)
    sub -= glow[idx] * (d <= R)[..., None]


def _lights(img_arr, mask_arr, box):
    """A painted blob that IS a light: [(x, y, core radius)], else []. Both must hold: a near-white
    core that is big for the blob (a lamp head, not the white collar of a painted person:
    blob short side <= 7 core radii), and a real halo (the unpainted ring just around it clearly
    brighter than further out). A false positive is costly: the halo pass wipes a whole disc."""
    x0, y0, x1, y1 = box
    sub = img_arr[y0:y1, x0:x1]
    lum = sub.mean(2) * (mask_arr[y0:y1, x0:x1] > 127)
    if lum.size == 0 or lum.max() < 225:
        return []
    core = int((lum > 215).sum())
    r = max(6, int((core / np.pi) ** 0.5))
    if min(x1 - x0, y1 - y0) > 7 * r:
        return []
    yy, xx = np.unravel_index(np.argmax(lum), lum.shape)
    x, y = x0 + int(xx), y0 + int(yy)
    H, W = img_arr.shape[:2]
    g = max(4 * r, min(x1 - x0, y1 - y0) // 2)
    R = int(3 * g)
    ys, xs = np.mgrid[max(0, y - R):min(H, y + R + 1), max(0, x - R):min(W, x + R + 1)]
    d = np.hypot(xs - x, ys - y)
    L = img_arr[max(0, y - R):min(H, y + R + 1), max(0, x - R):min(W, x + R + 1)].mean(2)
    free = mask_arr[max(0, y - R):min(H, y + R + 1), max(0, x - R):min(W, x + R + 1)] < 128
    near = free & (d >= g * 0.8) & (d < g * 1.4)
    far = free & (d >= g * 2.3) & (d < g * 3.0)
    need = 2 if r < 15 else 8               # small lamps have faint halos; a wrong small disc costs little
    if near.sum() < 20 or far.sum() < 20 or np.median(L[near]) - np.median(L[far]) < need:
        return []
    return [(x, y, r)]


def tiles(box, W, H):
    """Square crops covering a blob with context: side ~3.5× the blob's short side (min 384,
    max 2048; more context filled a removed person far better than 2.4×), stepping 75% along
    the long side (a lamp pole or a standing person = a column of squares)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    T = int(min(TILE_MAX, max(TILE_MIN, 3.5 * min(bw, bh) + 120)))
    T = min(T, W, H)
    out = []
    step = int(T * 0.75)
    ys = range(y0 + bh // 2, y0 + bh // 2 + 1) if bh + 120 <= T else range(y0 - 60 + T // 2, y1 + 60 - T // 2 + step, step)
    xs = range(x0 + bw // 2, x0 + bw // 2 + 1) if bw + 120 <= T else range(x0 - 60 + T // 2, x1 + 60 - T // 2 + step, step)
    for cy in ys:
        for cx in xs:
            cx, cy = min(max(cx, T // 2), W - T // 2), min(max(cy, T // 2), H - T // 2)
            out.append((cx - T // 2, cy - T // 2, cx - T // 2 + T, cy - T // 2 + T))
    return out


def remove(src_path, mask_img, progress=None, check=None):
    """The picture at `src_path` with the painted (white) parts of `mask_img` removed ->
    PNG bytes at the source's full size. `progress(done, total)`, `check()` may raise to stop."""
    from . import ai_runtime
    sess = ai_runtime.get_model('lama', model_path())
    with Image.open(src_path) as im:
        from PIL import ImageOps
        src = ImageOps.exif_transpose(im).convert('RGB')
    W, H = src.size
    m = mask_img.convert('L').resize((W, H), Image.BILINEAR).point(lambda v: 255 if v > 40 else 0)
    m = m.filter(ImageFilter.MaxFilter(7))                 # a few pixels past the brush edge
    mask_arr = np.asarray(m)
    if not mask_arr.any():
        raise ValueError('nothing painted')
    work_arr = np.asarray(src, dtype=np.float64).copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(m)
    for b in blobs(m):                                     # lights: take their halo away first
        for x, y, r in _lights(work_arr, mask_arr, b):
            g = max(4 * r, min(b[2] - b[0], b[3] - b[1]) // 2)    # halo ≈ 4× the white core
            deglow_at(work_arr, x, y, g)
            k = int(g * 1.3)                               # its rays reach past the paint: fill them too
            draw.ellipse([x - k, y - k, x + k, y + k], fill=255)
    work = Image.fromarray(np.clip(work_arr, 0, 255).astype(np.uint8))
    boxes = blobs(m)
    hole = m.copy()
    jobs = [t for b in boxes for t in tiles(b, W, H)]
    for i, box in enumerate(jobs):
        if check:
            check()
        mc = hole.crop(box)
        if np.asarray(mc).any():
            filled = _lama(sess, work.crop(box), mc)
            work.paste(filled, box[:2], mc.filter(ImageFilter.GaussianBlur(3)))
            hole.paste(0, box, mc)                         # filled: context for the next tile
        if progress:
            progress(i + 1, len(jobs))
    # only the painted area (+ its feathered edge) and the removed halos differ from the source
    buf = io.BytesIO()
    work.save(buf, 'PNG')
    return buf.getvalue()
