import os
from django.utils import timezone
from django.conf import settings
from django.db import models


class Tag(models.Model):
    name = models.CharField(max_length=200, unique=True)
    category = models.CharField(max_length=50, default='general',
        choices=[('general','general'),('character','character'),
                 ('artist','artist'),('meta','meta'),('ai','ai')])
    count = models.IntegerField(default=0)
    fav   = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    def update_count(self):
        self.count = self.posts.count()
        self.save(update_fields=['count'])


class Post(models.Model):
    title      = models.CharField(max_length=500, blank=True)
    tags       = models.ManyToManyField(Tag, blank=True, related_name='posts')
    ai_tagged  = models.BooleanField(default=False)
    # True once the post went through the fixed tagger (per-category thresholds,
    # characters kept + saved as category 'character'). Posts AI-tagged before
    # that have ai_tagged=True but char_tagged=False — the candidates for a
    # future characters-only re-tag.
    char_tagged = models.BooleanField(default=False)
    # which model last tagged this post's characters: '' (old tagger / never), 'wd14', 'pixai'
    char_model  = models.CharField(max_length=16, blank=True, default='')
    ai_multi    = models.BooleanField(default=False)   # every item of this multi-image post was AI-tagged, not just the cover
    rating     = models.SmallIntegerField(default=0, db_index=True)
    fav        = models.BooleanField(default=False, db_index=True)
    rated_at   = models.DateTimeField(null=True, blank=True, db_index=True)
    faved_at   = models.DateTimeField(null=True, blank=True, db_index=True)
    added_at   = models.DateTimeField(auto_now_add=True, db_index=True)
    # posts that this post has been marked "not a duplicate of"
    not_dupes  = models.ManyToManyField('self', blank=True, symmetrical=True)

    class Meta:
        ordering = ['-added_at']

    def __str__(self):
        return self.title or f'post-{self.pk}'

    @property
    def cover(self):
        # Use the prefetched images cache when available (avoids an N+1 query
        # on the gallery grid). Falls back to a single query otherwise.
        imgs = list(self.images.all())
        if not imgs:
            return None
        return min(imgs, key=lambda i: (i.order, i.id))

    @property
    def image_count(self):
        # len() on the (possibly prefetched) cache — no extra COUNT query.
        return len(self.images.all())

    @property
    def has_video(self):
        # True if ANY image in the post is a video (even a multi-image post
        # with one clip). Uses the prefetched cache.
        return any(i.is_video for i in self.images.all())

    @property
    def has_gif(self):
        return any(i.is_gif for i in self.images.all())


class Photo(models.Model):
    post           = models.ForeignKey(Post, on_delete=models.CASCADE,
                                       related_name='images', null=True, blank=True)
    order          = models.IntegerField(default=0)
    # Stored relative to settings.MEDIA_ROOT (so moving the whole media folder
    # only requires updating MEDIA_ROOT, not every row in the DB) — accessed
    # as absolute paths via the file_path/thumb_path properties below. Rows
    # from before this became relative may still hold a legacy absolute value;
    # both properties handle that transparently, and `rebase_photo_paths()` in
    # utils.py (wired to the "rebase paths" button) converts them in bulk.
    rel_path       = models.CharField(max_length=1000, unique=True)
    rel_thumb_path = models.CharField(max_length=1000, blank=True)
    width      = models.IntegerField(default=0)
    height     = models.IntegerField(default=0)
    file_size  = models.BigIntegerField(default=0)
    phash      = models.CharField(max_length=64, blank=True, db_index=True)
    is_video   = models.BooleanField(default=False)

    class Meta:
        ordering = ['order', 'id']

    def __str__(self):
        return os.path.basename(self.file_path)

    @staticmethod
    def _to_rel(value):
        """Convert an absolute path to one relative to MEDIA_ROOT. Values that
        are already relative, empty, or fall outside MEDIA_ROOT (or are on a
        different drive on Windows) are returned unchanged."""
        if not value or not os.path.isabs(value):
            return value
        try:
            rel = os.path.relpath(value, settings.MEDIA_ROOT)
        except ValueError:
            return value
        return value if rel.startswith('..') else rel

    @property
    def file_path(self):
        p = self.rel_path
        if not p or os.path.isabs(p):
            return p
        return os.path.join(settings.MEDIA_ROOT, p)

    @file_path.setter
    def file_path(self, value):
        self.rel_path = self._to_rel(value)

    @property
    def thumb_path(self):
        p = self.rel_thumb_path
        if not p or os.path.isabs(p):
            return p
        return os.path.join(settings.MEDIA_ROOT, p)

    @thumb_path.setter
    def thumb_path(self, value):
        self.rel_thumb_path = self._to_rel(value) if value else ''

    @property
    def is_pdf(self):
        return os.path.splitext(self.file_path)[1].lower() == '.pdf'

    @property
    def is_gif(self):
        return os.path.splitext(self.file_path)[1].lower() == '.gif'

    @property
    def filename(self):
        return os.path.basename(self.file_path)

    @staticmethod
    def _url_for(path):
        """/media/ URL of a file under MEDIA_ROOT, percent-encoded (a '#', '?' or '%' in a
        file name used to cut the URL short -> 404). '' when the file is outside MEDIA_ROOT."""
        from urllib.parse import quote
        rel = os.path.relpath(path, settings.MEDIA_ROOT)
        if rel.startswith('..'):
            return ''
        return '/media/' + quote(rel.replace(os.sep, '/'), safe='/')

    @property
    def media_url(self):
        return self._url_for(self.file_path)

    @property
    def thumb_url(self):
        if self.thumb_path:
            url = self._url_for(self.thumb_path)
            if not url:
                return ''
            # version by mtime: unchanged thumbs keep the same URL (cache hit),
            # a regenerated thumb gets a new URL (cache bust) automatically.
            try:
                url += f'?v={int(os.path.getmtime(self.thumb_path))}'
            except OSError:
                pass
            return url
        return self.media_url


class Folder(models.Model):
    """A named collection of posts shown in the sidebar, arranged in a tree.

    Manual folders (is_smart=False) hold an explicit set of posts (added via
    the select-mode "folder" bulk action). Smart folders (is_smart=True)
    instead store a gallery query string (tags/rating/fav/sort) captured at
    creation time; opening one just re-runs that query, so its contents
    update automatically as posts are added/edited — no separate filter
    engine needed, it reuses `_build_post_qs`.

    `parent` nests folders into a tree (root folders have parent=None); it is
    an organizational hierarchy for browsing/creating. By default opening a
    folder (`?folder=<id>`) only shows that folder's own directly assigned
    posts, not its descendants'. Set `include_subfolders=True` (the "⊞" toggle
    on a folder row) to also fold in every descendant folder's posts when the
    folder is opened (see `_build_post_qs`).
    Deleting a folder cascades to its whole subtree (on_delete=CASCADE) —
    the posts inside are never deleted, only the folder groupings.
    """
    name       = models.CharField(max_length=200)
    is_smart   = models.BooleanField(default=False)
    query      = models.CharField(max_length=500, blank=True)  # smart folders only
    posts      = models.ManyToManyField(Post, blank=True, related_name='folders')  # manual folders only
    parent     = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='children')
    include_subfolders = models.BooleanField(default=False)  # opening this folder also shows descendants' posts
    order      = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', 'name']

    def __str__(self):
        return self.name

    def descendant_ids(self):
        """All folder ids strictly below this one in the tree (BFS, cycle-guarded)."""
        seen = set()
        frontier = [self.id]
        while frontier:
            batch = list(
                Folder.objects.filter(parent_id__in=frontier)
                .exclude(id__in=seen)
                .values_list('id', flat=True)
            )
            frontier = [i for i in batch if i not in seen]
            seen.update(frontier)
        seen.discard(self.id)
        return list(seen)


class Task(models.Model):
    """A long-running background job (merge / scan / ai_tag / dupes) whose
    state is stored in the DB so any gunicorn worker (and any device) can see
    its progress, and so it survives the user closing the page."""
    kind        = models.CharField(max_length=32)                  # merge / scan / ai_tag / dupes
    status      = models.CharField(max_length=16, default='running')  # running / done / error / cancelled
    cancel_requested = models.BooleanField(default=False)  # set by the stop button; checked cooperatively by the work fn
    done        = models.IntegerField(default=0)
    total       = models.IntegerField(default=0)
    message     = models.CharField(max_length=300, blank=True)
    error       = models.TextField(blank=True)
    started_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    def save(self, *args, **kwargs):
        # Progress is saved with save(update_fields=[...]), which Django does NOT
        # treat as touching an auto_now field — so updated_at never moved while a
        # task ran. It doubles as the heartbeat the stale-task sweep and the stop
        # button rely on, so always include it.
        uf = kwargs.get('update_fields')
        if uf is not None and 'updated_at' not in uf:
            kwargs['update_fields'] = [*uf, 'updated_at']
        super().save(*args, **kwargs)

    @property
    def elapsed(self):
        end = self.finished_at or timezone.now()
        return max(0, int((end - self.started_at).total_seconds()))


# ── "My model": things taught to the cloned default tagger ──────
class CustomConcept(models.Model):
    """One thing the user taught the cloned tagger to recognise. `name` is the
    tag it assigns. The learned classifier itself lives in
    <AI_MODELS_DIR>/custom/heads.npz; this row holds its settings and metrics."""
    name       = models.CharField(max_length=200, unique=True)
    category   = models.CharField(max_length=50, default='ai')       # category of the tag it creates
    threshold  = models.FloatField(default=0.7)
    enabled    = models.BooleanField(default=True)
    n_pos      = models.IntegerField(default=0)                      # examples used by the last training
    n_neg      = models.IntegerField(default=0)
    metrics    = models.JSONField(default=dict, blank=True)          # cross-validated precision/recall/...
    trained_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class CustomExample(models.Model):
    """A post the user marked as showing (+1) / not showing (-1) a concept, or an
    auto-sampled negative. `feat` caches the clone's 768-d pooled features
    (float16) so retraining never has to re-run the image through the model."""
    concept = models.ForeignKey(CustomConcept, on_delete=models.CASCADE, related_name='examples')
    post    = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='+')
    label   = models.SmallIntegerField()                              # +1 / -1
    auto    = models.BooleanField(default=False)                     # sampled by the trainer, not chosen by the user
    feat    = models.BinaryField(null=True, blank=True)
    model_hash = models.CharField(max_length=40, blank=True, default='')   # clone the cached features came from

    class Meta:
        unique_together = [('concept', 'post')]


class PostFeature(models.Model):
    """Library-wide feature cache for the clone (filled by the 'scan library'
    task) — makes previewing/applying a concept a matrix multiply."""
    post       = models.OneToOneField(Post, primary_key=True, on_delete=models.CASCADE, related_name='+')
    vec        = models.BinaryField()                                 # float16[768]
    model_hash = models.CharField(max_length=40, db_index=True)


class CustomApplied(models.Model):
    """Posts that 'apply to library' tagged with a concept, so it can be undone
    without touching tags the user added by hand."""
    concept = models.ForeignKey(CustomConcept, on_delete=models.CASCADE, related_name='applied')
    post    = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='+')

    class Meta:
        unique_together = [('concept', 'post')]
