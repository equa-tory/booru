"""Ratings change from 0-5 stars to a signed score (like +1 / dislike -1).

Old stars become 5 -> +2, 4 -> +1, 3 -> 0, 2 -> -1, 1 -> -2 (0 = unrated stays 0); saved
smart-folder queries with min_rating/rating are converted the same way.
"""
import re

from django.db import migrations
from django.db.models import Case, IntegerField, Value, When

FORWARD = {1: -2, 2: -1, 3: 0, 4: 1, 5: 2}
BACKWARD = {-2: 1, -1: 2, 1: 4, 2: 5}


def _remap(posts, table, fallback):
    whens = [When(rating=old, then=Value(new)) for old, new in table.items()]
    posts.update(rating=Case(*whens, *fallback, default='rating', output_field=IntegerField()))


def _query_param(query, table):
    def sub(m):
        n = int(m.group(2))
        return f'{m.group(1)}={table.get(n, n)}'
    return re.sub(r'\b(min_rating|rating)=(-?\d+)', sub, query)


def forwards(apps, schema_editor):
    Post = apps.get_model('gallery', 'Post')
    Folder = apps.get_model('gallery', 'Folder')
    _remap(Post.objects.all(), FORWARD, [])
    for f in Folder.objects.filter(query__contains='rating='):
        new = _query_param(f.query, FORWARD)
        if new != f.query:
            Folder.objects.filter(pk=f.pk).update(query=new)


def backwards(apps, schema_editor):
    Post = apps.get_model('gallery', 'Post')
    Folder = apps.get_model('gallery', 'Folder')
    _remap(Post.objects.all(), BACKWARD,
           [When(rating__gte=3, then=Value(5)), When(rating__lte=-3, then=Value(1))])
    for f in Folder.objects.filter(query__contains='rating='):
        new = _query_param(f.query, BACKWARD)
        if new != f.query:
            Folder.objects.filter(pk=f.pk).update(query=new)


class Migration(migrations.Migration):

    dependencies = [
        ('gallery', '0021_photo_duration'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
