import os
import unittest
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


# ── Backups ─────────────────────────────────────────────────────
import sqlite3
import time

from . import backup


def _make_db(path, rows=3, session='sess-A'):
    con = sqlite3.connect(path)
    con.execute('PRAGMA journal_mode=WAL')
    for t in ('gallery_post', 'gallery_photo', 'gallery_tag'):
        con.execute(f'CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY, v TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS django_session (session_key TEXT PRIMARY KEY, session_data TEXT, expire_date TEXT)')
    con.execute('DELETE FROM gallery_post')
    con.executemany('INSERT INTO gallery_post (v) VALUES (?)', [(f'p{i}',) for i in range(rows)])
    con.execute('INSERT OR REPLACE INTO django_session VALUES (?,?,?)', (session, 'd', '2099-01-01'))
    con.commit()
    con.close()


class BackupTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, 'live.sqlite3')
        self.dir = os.path.join(self.tmp, 'bk')
        _make_db(self.db)
        self.cfg = {'enabled': True, 'max_backups': 2, 'interval_hours': 48, 'path': self.dir}
        p = mock.patch.object(backup, 'db_path', return_value=self.db); p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(backup, 'get_config', side_effect=lambda: dict(self.cfg)); p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(backup, '_prefs_path', return_value=os.path.join(self.tmp, 'prefs.json')); p.start(); self.addCleanup(p.stop)

    def _rows(self, path):
        con = sqlite3.connect(path)
        try:
            return con.execute('SELECT COUNT(*) FROM gallery_post').fetchone()[0]
        finally:
            con.close()

    def test_backup_creates_standalone_file_and_rotates(self):
        names = []
        for i in range(3):
            names.append(backup.run_backup())
            # distinct, ordered timestamps in the file names
            p = os.path.join(self.dir, names[-1]); os.utime(p, (time.time() + i, time.time() + i))
            time.sleep(1.05)
        left = sorted(os.listdir(self.dir))
        left = [n for n in left if n.startswith('booru-')]
        self.assertEqual(left, sorted(names)[-2:])               # max_backups=2 kept, newest
        self.assertEqual(self._rows(os.path.join(self.dir, left[-1])), 3)
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(('-wal', '-shm', '.partial'))])

    def test_info_and_due(self):
        self.assertTrue(backup._due(self.cfg))                   # no backup yet
        backup.run_backup()
        self.assertFalse(backup._due(self.cfg))
        self.cfg['interval_hours'] = 0.0000001
        self.assertTrue(backup._due(dict(self.cfg)))
        self.cfg['enabled'] = False
        self.assertFalse(backup._due(dict(self.cfg)))
        i = backup.info()
        self.assertEqual(len(i['backups']), 1)
        self.assertTrue(i['dir_exists'])

    def test_restore_replaces_data_keeps_sessions_and_snapshots(self):
        name = backup.run_backup()
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO gallery_post (v) VALUES ('later')")
        con.execute("INSERT INTO django_session VALUES ('sess-NOW','d','2099-01-01')")
        con.commit(); con.close()
        self.assertEqual(self._rows(self.db), 4)
        backup.restore_from(os.path.join(self.dir, name))
        self.assertEqual(self._rows(self.db), 3)                 # rolled back
        con = sqlite3.connect(self.db)
        keys = {r[0] for r in con.execute('SELECT session_key FROM django_session')}
        con.close()
        self.assertIn('sess-NOW', keys)                          # still logged in
        self.assertEqual(self._rows(os.path.join(self.dir, backup.SNAPSHOT_NAME)), 4)   # safety copy

    def test_validate_rejects_garbage(self):
        bad = os.path.join(self.tmp, 'bad.sqlite3')
        with open(bad, 'wb') as f:
            f.write(b'this is not a database' * 100)
        with self.assertRaises(ValueError):
            backup.validate_backup(bad)
        other = os.path.join(self.tmp, 'other.sqlite3')
        con = sqlite3.connect(other); con.execute('CREATE TABLE x (a)'); con.commit(); con.close()
        with self.assertRaises(ValueError):
            backup.validate_backup(other)

    def test_save_config_validation(self):
        good = {'enabled': True, 'max_backups': 1, 'interval_hours': 48, 'path': self.dir}
        cfg = backup.save_config(good)
        self.assertEqual((cfg['max_backups'], cfg['interval_hours']), (1, 48))
        for bad in ({'max_backups': 0}, {'interval_hours': 0}, {'path': 'relative/dir'}, {'max_backups': 'x'}):
            with self.assertRaises(ValueError):
                backup.save_config(dict(good, **bad))

    def test_busy_lock(self):
        with backup._lock(self.dir) as got:
            self.assertTrue(got)
            with self.assertRaises(RuntimeError):
                backup.run_backup()


class BackupApiTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        open(os.path.join(self.tmp, 'booru-20250101-000000.sqlite3'), 'wb').close()
        p = mock.patch.object(backup, 'get_config', return_value={
            'enabled': True, 'max_backups': 1, 'interval_hours': 48, 'path': self.tmp}); p.start(); self.addCleanup(p.stop)

    def test_info_lists_files(self):
        d = self.client.get('/api/backup/').json()
        self.assertEqual([b['name'] for b in d['backups']], ['booru-20250101-000000.sqlite3'])

    def test_restore_rejects_unknown_and_traversal(self):
        for name in ('nope.sqlite3', '../../etc/passwd', ''):
            r = self.client.post('/api/backup/restore/', {'name': name}, content_type='application/json')
            self.assertEqual(r.status_code, 404)

    def test_restore_refused_while_tasks_running(self):
        Task.objects.create(kind='scan')
        r = self.client.post('/api/backup/restore/', {'name': 'booru-20250101-000000.sqlite3'},
                             content_type='application/json')
        self.assertEqual(r.status_code, 409)

    def test_restore_rejects_invalid_file(self):
        r = self.client.post('/api/backup/restore/', {'name': 'booru-20250101-000000.sqlite3'},
                             content_type='application/json')
        self.assertEqual(r.status_code, 400)       # empty file isn't a booru DB


class TaskHeartbeatTests(TestCase):
    def test_progress_save_bumps_updated_at(self):
        t = Task.objects.create(kind='scan')
        old = timezone.now() - timezone.timedelta(hours=1)
        Task.objects.filter(pk=t.pk).update(updated_at=old)
        t.done = 5
        t.save(update_fields=['done'])          # how every work fn reports progress
        t.refresh_from_db()
        self.assertGreater(t.updated_at, old + timezone.timedelta(minutes=30))

    def test_live_long_running_task_is_not_swept_or_force_cancelled(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        t = Task.objects.create(kind='scan')
        Task.objects.filter(pk=t.pk).update(started_at=timezone.now() - timezone.timedelta(days=1))
        t.done = 1; t.save(update_fields=['done'])      # heartbeat just now
        self.client.get('/api/tasks/')
        self.client.post(f'/api/tasks/{t.pk}/cancel/')
        t.refresh_from_db()
        self.assertEqual(t.status, 'running')           # a day old, but alive
        self.assertTrue(t.cancel_requested)

    def test_second_scan_tap_reuses_the_running_scan(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        with mock.patch.object(views.threading, 'Thread') as th:
            a = self.client.post('/api/scan-bg/').json()['task_id']
            b = self.client.post('/api/scan-bg/').json()['task_id']
        self.assertEqual(a, b)
        self.assertEqual(Task.objects.filter(kind='scan').count(), 1)
        self.assertEqual(th.call_count, 1)              # one worker thread, not two

    def test_dead_scan_does_not_block_a_new_one(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        dead = Task.objects.create(kind='scan')
        Task.objects.filter(pk=dead.pk).update(updated_at=timezone.now() - timezone.timedelta(minutes=20))
        with mock.patch.object(views.threading, 'Thread'):
            b = self.client.post('/api/scan-bg/').json()['task_id']
        self.assertNotEqual(b, dead.pk)


# ── AI tagger: characters ───────────────────────────────────────
import numpy as np


class _FakeInput:
    name = 'input'


class _FakeModel:
    def __init__(self, probs): self.probs = np.array(probs, dtype=np.float32)
    def get_inputs(self): return [_FakeInput()]
    def run(self, _out, _feed): return [self.probs[None, :]]


def _fake_wd14(probs_by_name):
    """names: 2 ratings, 60 general, 3 characters (like the real CSV order)."""
    names = ['rating_a', 'rating_b'] + [f'g{i}' for i in range(60)] + ['char_a', 'char_b', 'char_c']
    is_char = np.array([n.startswith('char_') for n in names])
    probs = [probs_by_name.get(n, 0.0) for n in names]
    return (_FakeModel(probs), names, is_char)


class AiTaggerTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.img = os.path.join(self.tmp, 'a.png')
        Image.new('RGB', (30, 30), (10, 20, 30)).save(self.img)

    def _probs(self):
        p = {'rating_a': 0.9, 'rating_b': 0.1}
        p.update({f'g{i}': 0.5 for i in range(60)})          # 60 general tags over the bar
        p.update({'char_a': 0.95, 'char_b': 0.5, 'char_c': 0.90})
        return p

    def test_characters_are_not_cut_by_the_general_cap(self):
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14(self._probs())):
            res = views.run_ai_tagger(self.img)
        self.assertEqual(len(res['general']), 40)                   # general cap unchanged
        self.assertEqual(res['general'][0], 'rating_a')              # same index order as before
        self.assertEqual(res['character'], ['char_a', 'char_c'])     # most confident first; char_b (0.5) < 0.85

    def test_character_threshold_is_configurable(self):
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14(self._probs())), \
             mock.patch.object(views, 'AI_CHARACTER_THRESHOLD', 0.4):
            res = views.run_ai_tagger(self.img)
        self.assertEqual(res['character'], ['char_a', 'char_c', 'char_b'])

    def test_apply_ai_tags_sets_categories_and_flags(self):
        post = Post.objects.create()
        photo = Photo(post=post, order=0); photo.file_path = self.img; photo.thumb_path = ''; photo.save()
        # an old-pipeline character tag stuck in category 'ai', and one the user categorised themselves
        Tag.objects.create(name='char_a', category='ai')
        Tag.objects.create(name='char_c', category='artist')
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14(self._probs())):
            res = views.apply_ai_tags(post)
        post.refresh_from_db()
        self.assertTrue(post.ai_tagged and post.char_tagged)
        self.assertEqual(Tag.objects.get(name='char_a').category, 'character')   # promoted from 'ai'
        self.assertEqual(Tag.objects.get(name='char_c').category, 'artist')      # left alone
        self.assertEqual(Tag.objects.get(name='g0').category, 'ai')
        self.assertEqual(set(post.tags.values_list('name', flat=True)) & {'char_a', 'char_c', 'g0'},
                         {'char_a', 'char_c', 'g0'})

    def test_ai_tag_endpoint_reports_characters(self):
        post = Post.objects.create()
        photo = Photo(post=post, order=0); photo.file_path = self.img; photo.thumb_path = ''; photo.save()
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14(self._probs())):
            d = self.client.post(f'/api/post/{post.pk}/ai-tag/').json()
        self.assertTrue(d['ok'])
        self.assertEqual(d['characters'], ['char_a', 'char_c'])
        self.assertEqual(len(d['tags']), 42)

    def test_recategorize_only_moves_ai_character_names(self):
        names = ['hatsune_miku', 'smile']
        with mock.patch.object(views, '_wd14_tags', return_value=(names, np.array([True, False]))):
            Tag.objects.create(name='hatsune_miku', category='ai')
            Tag.objects.create(name='smile', category='ai')
            Tag.objects.create(name='hatsune_miku_(vocaloid)', category='ai')
            r = self.client.post('/api/tags/recategorize-characters/').json()
        self.assertEqual(r['updated'], 1)
        self.assertEqual(Tag.objects.get(name='hatsune_miku').category, 'character')
        self.assertEqual(Tag.objects.get(name='smile').category, 'ai')


class OverlayAndThumbTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        o = override_settings(MEDIA_ROOT=self.root); o.enable(); self.addCleanup(o.disable)
        os.makedirs(os.path.join(self.root, 'thumbs'))

    def _post(self, name='a.png', **kw):
        path = os.path.join(self.root, name)
        Image.new('RGB', (200, 100), (200, 50, 50)).save(path)
        from .utils import create_post_from_files
        return create_post_from_files([path], **kw)

    def test_grid_and_json_carry_overlay_state(self):
        a = self._post('a.png'); b = self._post('b.png')
        Post.objects.filter(pk=a.pk).update(ai_tagged=True, char_tagged=True)
        from .models import Folder
        f = Folder.objects.create(name='faves'); f.posts.add(a)
        j = {p['id']: p for p in self.client.get('/api/posts/').json()['posts']}
        self.assertEqual((j[a.pk]['ai'], j[a.pk]['chars'], j[a.pk]['folders']), (True, True, ['faves']))
        self.assertEqual((j[b.pk]['ai'], j[b.pk]['chars'], j[b.pk]['folders']), (False, False, []))
        html = self.client.get('/').content.decode()
        self.assertIn('class="d-folder on"', html)
        self.assertIn('class="d-folder off"', html)
        self.assertIn('class="d-ai off"', html)
        self.assertIn('faves', html)

    def test_debug_stats(self):
        a = self._post('a.png'); self._post('b.png')
        Post.objects.filter(pk=a.pk).update(ai_tagged=True)
        d = self.client.get('/api/debug/stats/').json()
        self.assertEqual((d['posts'], d['ai_tagged'], d['char_tagged'], d['ai_not_char']), (2, 1, 0, 1))

    def test_bulk_regen_thumb_handles_pictures_and_skips_videos(self):
        pic = self._post('a.png')
        vid = self._post('v.png')
        Photo.objects.filter(post=vid).update(is_video=True)       # pretend this cover is a video
        thumb = pic.cover.thumb_path
        self.assertTrue(os.path.exists(thumb))
        os.remove(thumb)                                           # broken/missing thumbnail
        r = self.client.post('/api/bulk-regen-thumb/', {'ids': [pic.pk, vid.pk]},
                             content_type='application/json').json()
        self.assertEqual((r['updated'], r['failed'], r['skipped_videos']), (1, 0, 1))
        self.assertTrue(os.path.exists(Post.objects.get(pk=pic.pk).cover.thumb_path))

    def test_bulk_regen_thumb_reports_missing_source(self):
        pic = self._post('a.png')
        os.remove(pic.cover.file_path)
        r = self.client.post('/api/bulk-regen-thumb/', {'ids': [pic.pk]}, content_type='application/json').json()
        self.assertEqual((r['updated'], r['failed']), (0, 1))


# ── Template JavaScript parses ──────────────────────────────────
import re
import subprocess


@unittest.skipUnless(shutil.which('node'), 'node not installed')
class TemplateJsSyntaxTests(TestCase):
    """The pages are big hand-written templates with lots of inline JS; a stray
    typo silently kills the whole page, so syntax-check every inline <script>."""
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def _check(self, url):
        html = self.client.get(url).content.decode()
        scripts = [s for s in re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html, re.S) if s.strip()]
        self.assertTrue(scripts)
        for i, src in enumerate(scripts):
            with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
                f.write(src)
            self.addCleanup(os.remove, f.name)
            r = subprocess.run(['node', '--check', f.name], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f'{url} script #{i}: {r.stderr[:400]}')

    def test_index(self):
        self._check('/')

    def test_post_detail(self):
        post = Post.objects.create()
        self._check(f'/post/{post.pk}/')
