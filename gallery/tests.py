import os
import shutil
import tempfile
from unittest import mock

from django.conf import settings
from django.test import TestCase, override_settings
from django.utils import timezone
from PIL import Image

from . import views
from .models import Photo, Post, Tag, Task
from .utils import (TaskCancelled, add_tags_to_post, check_cancel,
                    recount_tags)


class AuthTests(TestCase):
    def test_login_uses_unique_cookie_and_session_key(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 302)
        r = self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.assertIn('booru_sessionid', r.cookies)
        self.assertNotIn('sessionid', r.cookies)
        self.assertTrue(self.client.session.get('booru_authed'))
        self.assertEqual(self.client.get('/').status_code, 200)


class TaskApiTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def test_check_cancel(self):
        t = Task.objects.create(kind='scan')
        check_cancel(t)                      # not requested: no-op
        Task.objects.filter(pk=t.pk).update(cancel_requested=True)
        with self.assertRaises(TaskCancelled):
            check_cancel(t)                  # re-reads the DB, not the in-memory copy
        check_cancel(None)

    def test_cancel_sets_flag_on_live_task(self):
        t = Task.objects.create(kind='scan')
        self.client.post(f'/api/tasks/{t.pk}/cancel/')
        t.refresh_from_db()
        self.assertTrue(t.cancel_requested)
        self.assertEqual(t.status, 'running')   # the work fn ends it, not the endpoint

    def test_cancel_stale_task_ends_it_directly(self):
        t = Task.objects.create(kind='scan')
        Task.objects.filter(pk=t.pk).update(updated_at=timezone.now() - timezone.timedelta(minutes=5))
        self.client.post(f'/api/tasks/{t.pk}/cancel/')
        t.refresh_from_db()
        self.assertEqual(t.status, 'cancelled')
        self.assertIsNotNone(t.finished_at)

    def test_dismiss_one_not_running(self):
        running = Task.objects.create(kind='scan')
        done = Task.objects.create(kind='merge', status='done', finished_at=timezone.now())
        other = Task.objects.create(kind='dupes', status='cancelled', finished_at=timezone.now())
        self.client.post(f'/api/tasks/{running.pk}/dismiss/')
        self.client.post(f'/api/tasks/{done.pk}/dismiss/')
        self.assertTrue(Task.objects.filter(pk=running.pk).exists())
        self.assertFalse(Task.objects.filter(pk=done.pk).exists())
        self.assertTrue(Task.objects.filter(pk=other.pk).exists())
        self.client.post('/api/tasks/clear/')
        self.assertFalse(Task.objects.filter(pk=other.pk).exists())
        self.assertTrue(Task.objects.filter(pk=running.pk).exists())

    def test_orphaned_running_task_is_swept(self):
        t = Task.objects.create(kind='scan')
        Task.objects.filter(pk=t.pk).update(updated_at=timezone.now() - timezone.timedelta(hours=2))
        self.client.get('/api/tasks/')
        t.refresh_from_db()
        self.assertEqual(t.status, 'error')


class TagCountTests(TestCase):
    def test_recount_and_deferred_recount(self):
        a, b = Post.objects.create(), Post.objects.create()
        add_tags_to_post(a, ['x', 'y', 'x'], recount=False)
        add_tags_to_post(b, ['x'], recount=False)
        self.assertEqual(Tag.objects.get(name='x').count, 0)   # deferred
        recount_tags()
        self.assertEqual(Tag.objects.get(name='x').count, 2)
        self.assertEqual(Tag.objects.get(name='y').count, 1)


class ScanTests(TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.override = override_settings(MEDIA_ROOT=self.root)
        self.override.enable()
        self.addCleanup(self.override.disable)
        os.makedirs(os.path.join(self.root, 'inbox', '_', 'set1'))
        os.makedirs(os.path.join(self.root, 'thumbs'))
        for rel in ('inbox/a.gif', 'inbox/b.gif', 'inbox/_/set1/1.gif', 'inbox/_/set1/2.gif'):
            Image.new('RGB', (40, 30), (len(rel) * 5 % 255, 80, 120)).save(os.path.join(self.root, rel))

    def _scan(self):
        t = Task.objects.create(kind='scan')
        views._do_scan(t)
        return t

    def test_scan_adds_then_second_scan_is_a_noop(self):
        self._scan()
        self.assertEqual(Post.objects.count(), 3)       # a, b, set1
        self.assertEqual(Photo.objects.count(), 4)
        gif = Tag.objects.get(name='gif')
        self.assertEqual(gif.count, 3)                  # batched recount is correct
        with mock.patch.object(views, 'make_video_thumb') as mvt, \
             mock.patch.object(views, 'create_post_from_files') as cpf:
            self._scan()
            mvt.assert_not_called()
            cpf.assert_not_called()
        self.assertEqual(Post.objects.count(), 3)

    def test_scan_prunes_deleted_files_and_empty_posts(self):
        self._scan()
        os.remove(os.path.join(self.root, 'inbox', 'a.gif'))
        self._scan()
        self.assertEqual(Post.objects.count(), 2)
        self.assertEqual(Tag.objects.get(name='gif').count, 2)

    def test_scan_cancel_stops_before_partial_post(self):
        t = Task.objects.create(kind='scan', cancel_requested=True)
        with self.assertRaises(TaskCancelled):
            views._do_scan(t)
        self.assertEqual(Post.objects.count(), 0)

    def test_scan_refuses_when_media_root_missing(self):
        with override_settings(MEDIA_ROOT=os.path.join(self.root, 'nope')):
            with self.assertRaises(RuntimeError):
                self._scan()
        self.assertEqual(Post.objects.count(), 0)
