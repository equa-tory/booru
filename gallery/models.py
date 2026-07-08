import os
from django.utils import timezone
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
    post       = models.ForeignKey(Post, on_delete=models.CASCADE,
                                   related_name='images', null=True, blank=True)
    order      = models.IntegerField(default=0)
    file_path  = models.CharField(max_length=1000, unique=True)
    thumb_path = models.CharField(max_length=1000, blank=True)
    width      = models.IntegerField(default=0)
    height     = models.IntegerField(default=0)
    file_size  = models.BigIntegerField(default=0)
    phash      = models.CharField(max_length=64, blank=True, db_index=True)
    is_video   = models.BooleanField(default=False)

    class Meta:
        ordering = ['order', 'id']

    def __str__(self):
        return os.path.basename(self.file_path)

    @property
    def is_pdf(self):
        return os.path.splitext(self.file_path)[1].lower() == '.pdf'

    @property
    def is_gif(self):
        return os.path.splitext(self.file_path)[1].lower() == '.gif'

    @property
    def filename(self):
        return os.path.basename(self.file_path)

    @property
    def media_url(self):
        from django.conf import settings
        rel = os.path.relpath(self.file_path, settings.MEDIA_ROOT)
        return '/media/' + rel.replace(os.sep, '/')

    @property
    def thumb_url(self):
        from django.conf import settings
        if self.thumb_path:
            rel = os.path.relpath(self.thumb_path, settings.MEDIA_ROOT)
            url = '/media/' + rel.replace(os.sep, '/')
            # version by mtime: unchanged thumbs keep the same URL (cache hit),
            # a regenerated thumb gets a new URL (cache bust) automatically.
            try:
                url += f'?v={int(os.path.getmtime(self.thumb_path))}'
            except OSError:
                pass
            return url
        return self.media_url


class Folder(models.Model):
    """A named collection of posts shown in the sidebar.

    Manual folders (is_smart=False) hold an explicit set of posts (added via
    the select-mode "folder" bulk action). Smart folders (is_smart=True)
    instead store a gallery query string (tags/rating/fav/sort) captured at
    creation time; opening one just re-runs that query, so its contents
    update automatically as posts are added/edited — no separate filter
    engine needed, it reuses `_build_post_qs`.
    """
    name       = models.CharField(max_length=200)
    is_smart   = models.BooleanField(default=False)
    query      = models.CharField(max_length=500, blank=True)  # smart folders only
    posts      = models.ManyToManyField(Post, blank=True, related_name='folders')  # manual folders only
    order      = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', 'name']

    def __str__(self):
        return self.name


class Task(models.Model):
    """A long-running background job (merge / scan / ai_tag / dupes) whose
    state is stored in the DB so any gunicorn worker (and any device) can see
    its progress, and so it survives the user closing the page."""
    kind        = models.CharField(max_length=32)                  # merge / scan / ai_tag / dupes
    status      = models.CharField(max_length=16, default='running')  # running / done / error
    done        = models.IntegerField(default=0)
    total       = models.IntegerField(default=0)
    message     = models.CharField(max_length=300, blank=True)
    error       = models.TextField(blank=True)
    started_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    @property
    def elapsed(self):
        end = self.finished_at or timezone.now()
        return max(0, int((end - self.started_at).total_seconds()))
