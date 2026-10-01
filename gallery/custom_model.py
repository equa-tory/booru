"""My model: a CLONE of the default WD14 tagger that knows a few more things.

How it works (frozen network + learned heads):
- The clone is the default ONNX with ONE extra graph output: the 768-d pooled
  feature vector that feeds the classifier. Nothing else is touched, so the
  default model's 10,861 outputs are bit-identical (verified when cloning).
- What you teach it is stored as small linear "heads" on those features
  (heads.npz): logit = features @ W + b. Learning one is a numpy logistic
  regression on a few dozen of your posts — seconds, no PyTorch, and nothing
  the default model already knows can change or be forgotten.
- At tagging time the clone runs once; the default tags are chosen exactly as
  before and the heads add the learned tags.

Files (settings.AI_MODELS_DIR/custom/): model.onnx, selected_tags.csv (copy),
meta.json (source hash, feature tensor name), heads.npz.
"""
import hashlib
import json
import os
import shutil
import threading
import time

import numpy as np
from django.conf import settings
from django.utils import timezone

from . import ai_models

FEATURE_DIM = 768
HEADS_FILE = 'heads.npz'
PRIOR = float(getattr(settings, 'CUSTOM_PREVALENCE_PRIOR', 0.02))   # how common a taught concept is assumed to be in the library
MIN_EXAMPLES = 10
L2_GRID = (1.0, 10.0, 100.0)

_heads_cache = {'mtime': None, 'heads': None}
_heads_lock = threading.Lock()


def _p(name):
    return os.path.join(ai_models.custom_dir(), name)


def meta():
    try:
        with open(_p('meta.json')) as f:
            return json.load(f)
    except Exception:                         # noqa: BLE001
        return None


def model_hash():
    """Identifies the clone's backbone; cached features are only valid for it."""
    return ((meta() or {}).get('source_sha256') or '')[:40]


def is_ready():
    return ai_models.is_ready('custom')


# ── clone ───────────────────────────────────────────────────────
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clone_default(say=None):
    """Create the clone from the downloaded default model. `say(msg)` reports
    progress. Raises RuntimeError with a user-facing message on any problem; a
    clone that does not reproduce the default outputs EXACTLY is never kept."""
    say = say or (lambda m: None)
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        raise RuntimeError('the "onnx" Python package is not installed — run ./install.sh')
    src = ai_models.cached_file('wd14', 'model.onnx')
    tags_csv = ai_models.cached_file('wd14', 'selected_tags.csv')
    if not (src and tags_csv):
        raise RuntimeError('download the default model first (Settings → AI models)')

    final = ai_models.custom_dir()
    tmp = final + '.partial'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        say('reading the default model…')
        m = onnx.load(src)
        g = m.graph
        prod = {o: n for n in g.node for o in n.output}
        sig = prod.get(g.output[0].name)
        head = prod.get(sig.input[0]) if sig is not None else None
        if sig is None or sig.op_type != 'Sigmoid' or head is None or head.op_type != 'Gemm':
            raise RuntimeError('unexpected default-model structure (expected …→Gemm→Sigmoid); cannot clone it')
        feat = head.input[0]
        w = next((i for i in g.initializer if i.name == head.input[1]), None)
        if w is None:
            raise RuntimeError('unexpected default-model structure (classifier weights not found)')
        trans_b = next((a.i for a in head.attribute if a.name == 'transB'), 0)
        dim = int(w.dims[1] if trans_b else w.dims[0])
        if dim != FEATURE_DIM:
            raise RuntimeError(f'unexpected feature size {dim} (expected {FEATURE_DIM})')
        g.output.append(helper.make_tensor_value_info(feat, TensorProto.FLOAT, ['batch_size', dim]))
        say('writing the clone…')
        onnx.save(m, os.path.join(tmp, 'model.onnx'))
        del m, g

        say('verifying the clone reproduces the default exactly…')
        import onnxruntime as ort
        a = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
        b = ort.InferenceSession(os.path.join(tmp, 'model.onnx'), providers=['CPUExecutionProvider'])
        rng = np.random.default_rng(0)
        for _ in range(2):
            x = (rng.random((1, 448, 448, 3)) * 255).astype(np.float32)
            ra = a.run(None, {a.get_inputs()[0].name: x})[0]
            rb = b.run(None, {b.get_inputs()[0].name: x})
            if not np.array_equal(ra, rb[0]):
                raise RuntimeError('the clone does not reproduce the default model exactly — discarded')
            if tuple(rb[1].shape) != (1, dim):
                raise RuntimeError(f'unexpected feature output shape {rb[1].shape}')
        del a, b

        shutil.copy(tags_csv, os.path.join(tmp, 'selected_tags.csv'))
        sha = _sha256(src)
        with open(os.path.join(tmp, 'meta.json'), 'w') as f:
            json.dump({'source_sha256': sha, 'feature_tensor': feat, 'dim': dim,
                       'cloned_at': timezone.now().isoformat()}, f)
        # keep already-learned heads when the default model is unchanged (still valid)
        old = meta()
        old_heads = _p(HEADS_FILE)
        if old and old.get('source_sha256') == sha and os.path.exists(old_heads):
            shutil.copy(old_heads, os.path.join(tmp, HEADS_FILE))
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        with _heads_lock:
            _heads_cache['mtime'] = None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ── running the clone ───────────────────────────────────────────
def clone_session():
    from . import ai_runtime
    return ai_runtime.get_model('custom', ai_models.cached_file('custom', 'model.onnx'))


def run_clone(arr):
    """(default-tag probabilities [10861], pooled features [768]) for one preprocessed image."""
    m = clone_session()
    out, feat = m.run(['output', meta()['feature_tensor']], {m.get_inputs()[0].name: arr})
    return np.asarray(out[0]), np.asarray(feat[0], dtype=np.float32)


def extract_features(file_path, thumb_path=''):
    from .utils import load_image_for_tagging, wd14_input
    if not is_ready():
        raise RuntimeError('my model is not cloned yet — Settings → AI models')
    return run_clone(wd14_input(load_image_for_tagging(file_path, thumb_path)))[1]


# ── heads (learned classifiers) ─────────────────────────────────
def load_heads():
    """{'names','W'[K,768],'b'[K],'thr'[K],'enabled'[K]} or None. Cached by file mtime."""
    path = _p(HEADS_FILE)
    try:
        mt = os.stat(path).st_mtime_ns
    except OSError:
        return None
    with _heads_lock:
        if _heads_cache['mtime'] != mt:
            with np.load(path, allow_pickle=False) as z:
                _heads_cache['heads'] = {'names': [str(n) for n in z['names']], 'W': z['W'].astype(np.float32),
                                         'b': z['b'].astype(np.float32), 'thr': z['thr'].astype(np.float32),
                                         'enabled': z['enabled'].astype(bool)}
            _heads_cache['mtime'] = mt
        return _heads_cache['heads']


def _save_heads(names, W, b, thr, enabled):
    path = _p(HEADS_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, names=np.array(names, dtype=str), W=np.asarray(W, np.float32).reshape(len(names), FEATURE_DIM),
             b=np.asarray(b, np.float32), thr=np.asarray(thr, np.float32), enabled=np.asarray(enabled, bool))
    os.replace(tmp, path)


def set_head(name, w, b, thr, enabled=True):
    """Add or replace one concept's head."""
    h = load_heads() or {'names': [], 'W': np.zeros((0, FEATURE_DIM), np.float32), 'b': np.zeros(0, np.float32),
                         'thr': np.zeros(0, np.float32), 'enabled': np.zeros(0, bool)}
    names = list(h['names'])
    W, bb, tt, ee = h['W'].copy(), h['b'].copy(), h['thr'].copy(), h['enabled'].copy()
    if name in names:
        i = names.index(name)
        W[i], bb[i], tt[i], ee[i] = w, b, thr, enabled
    else:
        names.append(name)
        W, bb, tt, ee = np.vstack([W, w]), np.append(bb, b), np.append(tt, thr), np.append(ee, enabled)
    _save_heads(names, W, bb, tt, ee)


def remove_head(name):
    h = load_heads()
    if not h or name not in h['names']:
        return
    keep = [i for i, n in enumerate(h['names']) if n != name]
    _save_heads([h['names'][i] for i in keep], h['W'][keep], h['b'][keep], h['thr'][keep], h['enabled'][keep])


def update_head(name, thr=None, enabled=None):
    h = load_heads()
    if not h or name not in h['names']:
        return
    i = h['names'].index(name)
    thr_a, en = h['thr'].copy(), h['enabled'].copy()
    if thr is not None: thr_a[i] = thr
    if enabled is not None: en[i] = enabled
    _save_heads(h['names'], h['W'], h['b'], thr_a, en)


def score_all(feats):
    """probabilities [n, K] of every head for features [n, 768]; None without heads."""
    h = load_heads()
    if not h or not h['names']:
        return None, None
    return h['names'], _sigmoid(np.asarray(feats, np.float32) @ h['W'].T + h['b'])


def learned_tags(feat):
    """[(name, probability)] of enabled heads at/above their threshold."""
    h = load_heads()
    if not h or not h['names']:
        return []
    p = _sigmoid(np.asarray(feat, np.float32) @ h['W'].T + h['b'])
    return [(n, float(p[i])) for i, n in enumerate(h['names']) if h['enabled'][i] and p[i] >= h['thr'][i]]


# ── numerics: logistic regression, CV, threshold ────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2, sw, iters=40):
    """L2-regularised weighted logistic regression by Newton's method.
    X [n,d] (standardised), y in {0,1}, sw per-sample weights. Returns theta [d+1] (bias last)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 1e-6                                   # the bias is not regularised
    for _ in range(iters):
        p = _sigmoid(Xb @ theta)
        grad = Xb.T @ (sw * (p - y)) + reg @ theta
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + reg + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.abs(step).max() < 1e-7:
            break
    return theta


def _weights(y):
    n, npos = len(y), int(y.sum())
    return np.where(y == 1, n / (2.0 * npos), n / (2.0 * (n - npos)))     # balanced classes


def fit_standardised(X, y, l2):
    """Fit on standardised features and fold the scaling back, so the returned
    (w, b) apply to RAW features: logit = x @ w + b."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    theta = fit_logreg((X - mu) / sd, y, l2, _weights(y))
    w = theta[:-1] / sd
    return w, float(theta[-1] - np.dot(w, mu))


def stratified_folds(y, k, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), int)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def cv_predict(X, y, l2, k, seed=0):
    folds = stratified_folds(y, k, seed)
    oof = np.zeros(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        w, b = fit_standardised(X[tr], y[tr], l2)
        oof[te] = _sigmoid(X[te] @ w + b)
    return oof


def auc(y, p):
    """Rank-based AUC (Mann-Whitney)."""
    order = np.argsort(p, kind='mergesort')
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def choose_threshold(y, p, prior=PRIOR):
    """Pick the cut-off on out-of-fold probabilities. The training set is
    class-balanced but a real concept is rare, so precision is re-estimated for
    `prior` (share of the library showing the concept) and F0.5 is maximised —
    false positives hurt more than misses here. Two robustness tweaks, chosen on
    real library data (4 concepts, 0.2-2.7% prevalence):
    - the false-positive rate is Laplace-smoothed ((fp+1)/(n+2));
    - the highest-scoring ~2.5% of the negatives are ignored: the negatives are
      random unlabeled library posts, and a few of them genuinely show the concept
      (they scored 0.98 while truly clean held-out posts never exceeded 0.06).
    Returns (threshold, precision_est, recall)."""
    pos = p[y == 1]
    neg = np.sort(p[y == 0])[::-1]
    neg = neg[max(1, int(round(0.025 * len(neg)))):] if len(neg) >= 20 else neg
    best = (-1.0, 0.9, 0.0, 0.0)
    for t in np.arange(0.1, 0.995, 0.005):
        tpr = float((pos >= t).mean())
        fpr = float(((neg >= t).sum() + 1) / (len(neg) + 2))
        denom = tpr * prior + fpr * (1 - prior)
        prec = tpr * prior / denom if denom > 0 else 1.0
        f05 = (1.25 * prec * tpr) / (0.25 * prec + tpr) if (prec + tpr) > 0 else 0.0
        if f05 >= best[0]:
            best = (f05, float(t), prec, tpr)
    return best[1], best[2], best[3]


def train(X, y, seed=0):
    """Full fit with cross-validation. Returns (w, b, threshold, metrics)."""
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos < MIN_EXAMPLES or nneg < MIN_EXAMPLES:
        raise RuntimeError(f'need at least {MIN_EXAMPLES} examples of each kind (have {npos} positive / {nneg} negative)')
    k = max(2, min(5, npos, nneg))
    best_l2, best_ll, best_oof = None, None, None
    for l2 in L2_GRID:
        oof = cv_predict(X, y, l2, k, seed)
        pe = np.clip(oof, 1e-6, 1 - 1e-6)
        ll = float(-(_weights(y) * (y * np.log(pe) + (1 - y) * np.log(1 - pe))).mean())
        if best_ll is None or ll < best_ll:
            best_l2, best_ll, best_oof = l2, ll, oof
    thr, prec_est, rec = choose_threshold(y, best_oof)
    w, b = fit_standardised(X, y, best_l2)
    return w, b, thr, {
        'precision_est': round(prec_est, 3), 'recall': round(rec, 3), 'auc': round(auc(y, best_oof), 3),
        'n_pos': npos, 'n_neg': nneg, 'l2': best_l2, 'folds': k, 'prior': PRIOR, 'threshold': round(thr, 3),
    }


# ── features for posts (example cache → library cache → run the clone) ──
def encode_feat(vec):
    return np.asarray(vec, np.float16).tobytes()


def decode_feat(blob):
    return np.frombuffer(bytes(blob), dtype=np.float16).astype(np.float32)


def feature_for_post(post_id):
    """Pooled features of a post's cover, preferring the library cache."""
    from .models import Photo, PostFeature
    h = model_hash()
    pf = PostFeature.objects.filter(post_id=post_id, model_hash=h).values_list('vec', flat=True).first()
    if pf is not None:
        return decode_feat(pf)
    cover = Photo.objects.filter(post_id=post_id).order_by('order', 'id').first()
    if cover is None:
        raise RuntimeError('post has no image')
    if not os.path.exists(cover.file_path):
        raise RuntimeError('source file missing')
    return extract_features(cover.file_path, cover.thumb_path)


def fill_example_features(examples, progress=None, check=None):
    """Make sure every CustomExample has cached features from the CURRENT clone.
    Unusable examples (missing file…) are returned in `failed`."""
    h = model_hash()
    todo = [e for e in examples if e.feat is None or e.model_hash != h]
    failed = []
    for i, e in enumerate(todo):
        if check: check()
        try:
            e.feat = encode_feat(feature_for_post(e.post_id))
            e.model_hash = h
            e.save(update_fields=['feat', 'model_hash'])
        except Exception as ex:                # noqa: BLE001
            failed.append((e.post_id, str(ex)))
        if progress: progress(i + 1, len(todo))
    return failed


def train_concept(concept, progress=None, check=None, rng=None):
    """Train (or retrain) one concept from its examples + auto-sampled
    negatives, store the head and metrics. Returns the metrics dict."""
    import random
    from .models import CustomExample, Post
    rng = rng or random.Random(0)
    if not is_ready():
        raise RuntimeError('my model is not cloned yet')
    h = model_hash()

    # stale auto-negatives (from another clone) are useless — drop them
    concept.examples.filter(auto=True).exclude(model_hash=h).delete()
    exs = list(concept.examples.select_related(None).all())
    npos = sum(1 for e in exs if e.label > 0)
    explicit_neg = sum(1 for e in exs if e.label < 0 and not e.auto)
    auto_have = sum(1 for e in exs if e.auto)
    want_auto = min(600, max(40, 4 * npos)) - explicit_neg - auto_have
    if want_auto > 0:
        used = {e.post_id for e in exs}
        pool = list(Post.objects.filter(ai_tagged=True).exclude(pk__in=used)
                    .exclude(tags__name=concept.name).values_list('id', flat=True))
        rng.shuffle(pool)
        for pid in pool[:want_auto]:
            exs.append(CustomExample.objects.create(concept=concept, post_id=pid, label=-1, auto=True))
    failed = fill_example_features(exs, progress, check)
    bad = {pid for pid, _ in failed}
    use = [e for e in exs if e.post_id not in bad and e.feat is not None]
    X = np.stack([decode_feat(e.feat) for e in use])
    y = np.array([1 if e.label > 0 else 0 for e in use])
    w, b, thr, metrics = train(X, y)
    metrics['skipped'] = len(failed)
    set_head(concept.name, w, b, thr, concept.enabled)
    concept.threshold = thr
    concept.n_pos, concept.n_neg = int(y.sum()), int(len(y) - y.sum())
    concept.metrics = metrics
    concept.trained_at = timezone.now()
    concept.save()
    return metrics


def apply_learned(post, names):
    """Add taught tags to a post, in each concept's own category (an existing
    tag keeps whatever category it already has)."""
    from .models import CustomConcept
    from .utils import add_tags_to_post
    cats = dict(CustomConcept.objects.filter(name__in=names).values_list('name', 'category'))
    for n in names:
        add_tags_to_post(post, [n], category=cats.get(n, 'ai'))


def on_clone_deleted():
    """The clone is gone: its heads/caches are meaningless. Examples stay so
    the user can clone again and retrain."""
    from .models import CustomApplied, CustomConcept, CustomExample, PostFeature
    PostFeature.objects.all().delete()
    CustomExample.objects.update(feat=None, model_hash='')
    CustomExample.objects.filter(auto=True).delete()
    CustomConcept.objects.update(trained_at=None, metrics={}, n_pos=0, n_neg=0)


# ── library index: features of every post, so previews/applies are matrix ops ──
def scan_library(progress=None, check=None, batch=50):
    """Compute and cache the clone's features for every post that has none yet
    (newest first). Resumable: cached posts are skipped; features from an
    older clone are dropped. Returns (indexed_now, failed)."""
    from .models import Post, PostFeature
    h = model_hash()
    PostFeature.objects.exclude(model_hash=h).delete()
    have = set(PostFeature.objects.values_list('post_id', flat=True))
    todo = [i for i in Post.objects.order_by('-id').values_list('id', flat=True) if i not in have]
    pending, done, failed = [], 0, 0
    for n, pid in enumerate(todo):
        if check: check()
        try:
            pending.append(PostFeature(post_id=pid, vec=encode_feat(feature_for_post(pid)), model_hash=h))
            done += 1
        except Exception:                      # noqa: BLE001
            failed += 1
        if len(pending) >= batch:
            PostFeature.objects.bulk_create(pending, ignore_conflicts=True)
            pending = []
        if progress: progress(n + 1, len(todo))
    if pending:
        PostFeature.objects.bulk_create(pending, ignore_conflicts=True)
    return done, failed


def feature_matrix():
    """(post ids [n], features [n,768] float32) of the library index for the current clone."""
    from .models import PostFeature
    rows = list(PostFeature.objects.filter(model_hash=model_hash()).values_list('post_id', 'vec'))
    if not rows:
        return np.zeros(0, int), np.zeros((0, FEATURE_DIM), np.float32)
    return (np.array([r[0] for r in rows]),
            np.stack([np.frombuffer(bytes(r[1]), dtype=np.float16) for r in rows]).astype(np.float32))


def concept_scores(concept, ids, M):
    h = load_heads()
    if not h or concept.name not in h['names'] or not len(ids):
        return np.zeros(len(ids))
    i = h['names'].index(concept.name)
    return _sigmoid(M @ h['W'][i] + h['b'][i])


def candidates(concept, n=60, mode='top'):
    """Library posts to review for a concept, excluding posts already labeled or
    tagged. mode='top': highest scores (what it would tag). mode='uncertain':
    closest to the threshold (what teaching more would help most).
    Returns (ids, scores, n_above): n_above = how many unlabeled/untagged posts
    score >= the threshold, i.e. what "apply to library" would tag."""
    from .models import Tag
    ids, M = feature_matrix()
    if not len(ids):
        return [], [], 0
    sc = concept_scores(concept, ids, M)
    skip = set(concept.examples.filter(auto=False).values_list('post_id', flat=True))
    tag = Tag.objects.filter(name=concept.name).first()
    if tag:
        skip |= set(tag.posts.values_list('id', flat=True))
    keep = np.array([pid not in skip for pid in ids])
    ids, sc = ids[keep], sc[keep]
    order = np.argsort(-sc) if mode == 'top' else np.argsort(np.abs(sc - concept.threshold))
    order = order[:n]
    return [int(ids[i]) for i in order], [round(float(sc[i]), 3) for i in order], int((sc >= concept.threshold).sum())


def apply_concept(concept, progress=None, check=None):
    """Tag every indexed post scoring >= the threshold (skipping posts already
    tagged or labeled 'no'), recording them so it can be undone. Returns the number tagged."""
    from .models import CustomApplied, Post, Tag
    from .utils import add_tags_to_post, recount_tags
    ids, M = feature_matrix()
    if not len(ids):
        raise RuntimeError('index the library first (Settings → My model → index library)')
    sc = concept_scores(concept, ids, M)
    neg = set(concept.examples.filter(label__lt=0, auto=False).values_list('post_id', flat=True))
    tag = Tag.objects.filter(name=concept.name).first()
    have = set(tag.posts.values_list('id', flat=True)) if tag else set()
    todo = [int(i) for i, s_ in zip(ids, sc) if s_ >= concept.threshold and int(i) not in neg and int(i) not in have]
    touched = []
    for n, pid in enumerate(todo):
        if check: check()
        post = Post.objects.get(pk=pid)
        touched = add_tags_to_post(post, [concept.name], category=concept.category, recount=False) or touched
        CustomApplied.objects.get_or_create(concept=concept, post_id=pid)
        if progress: progress(n + 1, len(todo))
    recount_tags([t.pk for t in touched] if touched else None)
    return len(todo)


def undo_concept(concept):
    """Remove the concept's tag from the posts apply_concept tagged (never from
    posts you tagged yourself). Returns how many were untagged."""
    from .models import Tag
    from .utils import recount_tags
    tag = Tag.objects.filter(name=concept.name).first()
    ids = list(concept.applied.values_list('post_id', flat=True))
    n = 0
    if tag and ids:
        n = tag.posts.through.objects.filter(tag_id=tag.pk, post_id__in=ids).delete()[0]
        recount_tags([tag.pk])
    concept.applied.all().delete()
    return n
