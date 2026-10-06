import json
import os
import unittest
import shutil
import tempfile
from unittest import mock

from django.conf import settings
from django.test import TestCase, override_settings
from django.utils import timezone
from PIL import Image

from . import prefs, views
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


# ── AI runtime (CUDA/CPU fallback, idle unload) ─────────────────
import types

from . import ai_runtime, char_tagger


class _Sess:
    def __init__(self, provider, fail=False):
        self.provider, self.fail, self.calls = provider, fail, 0
    def get_inputs(self): return [_FakeInput()]
    def run(self, _o, _f):
        self.calls += 1
        if self.fail:
            raise RuntimeError('CUDNN_FE failure 11')
        return [np.array([[0.1, 0.9]], dtype=np.float32)]


class AiRuntimeTests(TestCase):
    def setUp(self):
        ai_runtime._models.clear()
        ai_runtime._cuda_off_until = 0.0
        self.addCleanup(ai_runtime._models.clear)
        self.addCleanup(setattr, ai_runtime, '_cuda_off_until', 0.0)
        p = mock.patch.object(ai_runtime, '_start_reaper'); p.start(); self.addCleanup(p.stop)

    def test_cuda_failure_falls_back_to_cpu_and_keeps_working(self):
        made = []
        def fake_make(path, use_cuda):
            s = _Sess('CUDA', fail=True) if use_cuda else _Sess('CPU')
            made.append((use_cuda, s)); return s, s.provider
        with mock.patch.object(ai_runtime, '_make_session', side_effect=fake_make):
            m = ai_runtime.get_model('t', '/x.onnx')
            out = m.run(None, {'input': 1})
            self.assertAlmostEqual(float(out[0][0][1]), 0.9, places=5)   # result came from the CPU retry
            self.assertEqual(m.provider, 'CPU')
            self.assertFalse(ai_runtime.cuda_enabled())              # CUDA parked for a while
            self.assertEqual([c for c, _ in made], [True, False])
            m.run(None, {'input': 1})
            self.assertEqual(len(made), 2)                           # no re-creation on the next call

    def test_cpu_error_is_not_swallowed(self):
        with mock.patch.object(ai_runtime, '_make_session', return_value=(_Sess('CPU', fail=True), 'CPU')):
            with self.assertRaises(RuntimeError):
                ai_runtime.get_model('t', '/x.onnx').run(None, {})

    def test_idle_models_are_unloaded_and_reloaded_on_demand(self):
        with mock.patch.object(ai_runtime, '_make_session', side_effect=lambda p, c: (_Sess('CPU'), 'CPU')) as mk:
            m = ai_runtime.get_model('t', '/x.onnx')
            m.run(None, {})
            self.assertEqual(ai_runtime.unload_idle(ttl=10_000), 0)  # recently used: stays
            self.assertEqual(ai_runtime.unload_idle(ttl=0), 1)       # idle: freed
            self.assertIsNone(m._sess)
            m.run(None, {})
            self.assertEqual(mk.call_count, 2)                       # transparently reloaded

    def test_runtime_info_does_not_load_models(self):
        info = ai_runtime.runtime_info()
        self.assertIn('cuda_available', info)
        self.assertEqual(info['loaded_in_this_worker'], {})


class CharTaggerTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.img = os.path.join(self.tmp, 'a.png')
        Image.new('RGBA', (60, 30), (255, 0, 0, 128)).save(self.img)
        # 5 general + 4 characters; character tags are interleaved, like the real csv
        self.names = ['1girl', 'char_a', 'solo', 'char_b', 'smile', 'char_c', 'x', 'char_d', 'y']
        self.mask = np.array([n.startswith('char_') for n in self.names])

    def _patch(self, probs):
        model = _FakeModel(probs)
        ps = [mock.patch.object(char_tagger, 'model_ready', return_value=True),
              mock.patch.object(char_tagger, '_cached', return_value='/fake/model.onnx'),
              mock.patch.object(char_tagger, '_tags', return_value=(self.names, self.mask)),
              mock.patch.object(ai_runtime, 'get_model', return_value=model)]
        for p in ps: p.start(); self.addCleanup(p.stop)
        return model

    def test_preprocess_matches_preprocess_json(self):
        arr = char_tagger._preprocess(Image.new('RGB', (100, 40), (255, 0, 128)))
        self.assertEqual(arr.shape, (1, 3, 448, 448))                # squashed to 448x448, NCHW
        self.assertAlmostEqual(float(arr[0, 0, 0, 0]), 1.0, places=4)    # R=255 -> +1
        self.assertAlmostEqual(float(arr[0, 1, 0, 0]), -1.0, places=4)   # G=0   -> -1

    def test_only_characters_above_threshold_most_confident_first(self):
        self._patch([0.99, 0.90, 0.99, 0.95, 0.99, 0.50, 0.99, 0.86, 0.1])
        got = char_tagger.run_character_tagger(self.img)
        self.assertEqual([n for n, _ in got], ['char_b', 'char_a', 'char_d'])    # char_c 0.5 < 0.85; no general tags
        self.assertAlmostEqual(got[0][1], 0.95, places=5)

    def test_logits_are_converted_to_probabilities(self):
        self._patch([5, 4.0, 5, -3.0, 5, 0.0, 5, 2.0, -5])           # logits
        got = [n for n, _ in char_tagger.run_character_tagger(self.img)]
        self.assertEqual(got, ['char_a', 'char_d'])                  # sigmoid(4)=.98, sigmoid(2)=.88 pass 0.85

    def test_refuses_when_model_missing(self):
        with mock.patch.object(char_tagger, 'model_ready', return_value=False):
            with self.assertRaises(RuntimeError):
                char_tagger.run_character_tagger(self.img)

    # -- views
    def _post_with_image(self):
        post = Post.objects.create()
        path = os.path.join(self.tmp, f'img{post.pk}.png')       # rel_path is unique per photo
        shutil.copy(self.img, path)
        ph = Photo(post=post, order=0); ph.file_path = path; ph.thumb_path = ''; ph.save()
        return post

    def test_post_endpoint_needs_downloaded_model(self):
        post = self._post_with_image()
        with mock.patch.object(char_tagger, 'model_ready', return_value=False):
            r = self.client.post(f'/api/post/{post.pk}/tag-characters/')
        self.assertEqual(r.status_code, 409)

    def test_post_endpoint_adds_characters_and_marks_pixai(self):
        post = self._post_with_image()
        Tag.objects.create(name='char_a', category='ai')              # old-pipeline tag gets promoted
        with mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', return_value=[('char_a', .9), ('char_z', .8)]):
            d = self.client.post(f'/api/post/{post.pk}/tag-characters/').json()
        self.assertEqual(d['characters'], ['char_a', 'char_z'])
        post.refresh_from_db()
        self.assertEqual((post.char_tagged, post.char_model), (True, 'pixai'))
        self.assertEqual(Tag.objects.get(name='char_a').category, 'character')
        self.assertEqual(Tag.objects.get(name='char_z').category, 'character')

    def test_wd14_tagging_never_downgrades_a_pixai_post(self):
        post = self._post_with_image()
        Post.objects.filter(pk=post.pk).update(char_model='pixai')
        post.refresh_from_db()
        probs = {'rating_a': 0.9}
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14(probs)):
            views.apply_ai_tags(post)
        post.refresh_from_db()
        self.assertEqual(post.char_model, 'pixai')

    def test_auto_option_runs_the_character_model_after_wd14(self):
        post = self._post_with_image()
        with mock.patch.object(views, '_get_wd14_model', return_value=_fake_wd14({'char_a': 0.95})), \
             mock.patch.object(views, '_pref', side_effect=lambda k, d=None: True if k == 'aiCharAuto' else d), \
             mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', return_value=[('char_z', .9)]):
            res = views.apply_ai_tags(post)
        post.refresh_from_db()
        self.assertEqual(post.char_model, 'pixai')
        self.assertEqual(res['character'], ['char_a', 'char_z'])      # WD14's + PixAI's, de-duplicated

    def test_retag_work_is_resumable_cancellable_and_skips_done_posts(self):
        done = Post.objects.create(char_model='pixai'); Photo.objects.create(post=done, order=0, rel_path='x.png')
        todo = [self._post_with_image() for _ in range(3)]
        captured = {}
        def fake_start(kind, fn, **kw):
            captured['fn'] = fn; captured['kind'] = kind
            return types.SimpleNamespace(id=1)
        with mock.patch.object(views, '_start_task', side_effect=fake_start):
            self.client.post('/api/ai/char-retag/', {'limit': 2}, content_type='application/json')
        self.assertEqual(captured['kind'], 'char_retag')
        task = Task.objects.create(kind='char_retag')
        seen = []
        def fake_run(path, thumb=''):
            seen.append(path); return [('char_q', .9)]
        with mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', side_effect=fake_run):
            captured['fn'](task)
        self.assertEqual(len(seen), 2)                                # limit=2, newest first
        marked = list(Post.objects.filter(char_model='pixai').values_list('pk', flat=True))
        self.assertEqual(set(marked), {done.pk, todo[2].pk, todo[1].pk})   # the oldest todo post is left for the next run
        # resume: the next run takes the remaining one
        task2 = Task.objects.create(kind='char_retag')
        with mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', return_value=[]):
            captured['fn'](task2)
        self.assertTrue(Post.objects.get(pk=todo[0].pk).char_model == 'pixai')
        # cancel
        extra = self._post_with_image()
        t3 = Task.objects.create(kind='char_retag', cancel_requested=True)
        with mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', return_value=[]):
            with self.assertRaises(TaskCancelled):
                captured['fn'](t3)
        self.assertEqual(Post.objects.get(pk=extra.pk).char_model, '')

    def test_ai_info(self):
        with mock.patch.object(char_tagger, 'model_ready', return_value=False):
            d = self.client.get('/api/ai/info/').json()
        self.assertFalse(d['char_model']['ready'])
        self.assertIn('runtime', d)
        self.assertIn('pixai_todo', d)


# ── Sidebar tag sampler ─────────────────────────────────────────
import random as _random
from collections import Counter


class SidebarTagTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def _make_tags(self, sizes):
        rows = []
        for cat, n in sizes.items():
            for i in range(n):
                t = Tag.objects.create(name=f'{cat}_{i}', category=cat, count=1 + i % 50)
                rows.append((t.id, cat, False, t.count))
        return rows

    def test_fair_shares_redistributes_unused_slots(self):
        self.assertEqual(views._fair_shares({'a': 3, 'b': 5, 'c': 40, 'd': 60, 'e': 300}, 150),
                         {'a': 3, 'b': 5, 'c': 40, 'd': 51, 'e': 51})
        self.assertEqual(views._fair_shares({'a': 1000}, 150), {'a': 150})
        self.assertEqual(views._fair_shares({'a': 10, 'b': 10}, 150), {'a': 10, 'b': 10})   # fewer tags than slots
        sh = views._fair_shares({'a': 100, 'b': 100, 'c': 100}, 100)                          # rounding leftover handed out
        self.assertEqual(sum(sh.values()), 100)
        self.assertEqual(views._fair_shares({}, 150), {})

    def test_every_type_gets_an_equal_share(self):
        cand = self._make_tags({'meta': 3, 'character': 300, 'artist': 5, 'general': 60, 'ai': 40})
        out = views._sidebar_tags(cand, 150, rng=_random.Random(1))
        self.assertEqual(len(out), 150)
        self.assertEqual(Counter(t.category for t in out),
                         {'meta': 3, 'artist': 5, 'ai': 40, 'general': 51, 'character': 51})

    def test_display_order_groups_by_type_then_usage(self):
        cand = self._make_tags({'character': 80, 'general': 80, 'ai': 80})
        out = views._sidebar_tags(cand, 90, rng=_random.Random(2))
        order = [t.category for t in out]
        self.assertEqual(order, sorted(order, key=['meta', 'character', 'artist', 'general', 'ai'].index))
        for cat in ('character', 'general', 'ai'):
            counts = [t.count for t in out if t.category == cat]
            self.assertEqual(counts, sorted(counts, reverse=True))

    def test_favorites_are_pinned_first_and_count_toward_the_cap(self):
        cand = self._make_tags({'character': 200, 'general': 200})
        fav_ids = [cand[0][0], cand[250][0]]
        Tag.objects.filter(pk__in=fav_ids).update(fav=True)
        cand = [(i, c, i in fav_ids, w) for i, c, _f, w in cand]
        out = views._sidebar_tags(cand, 50, rng=_random.Random(3))
        self.assertEqual(len(out), 50)
        self.assertEqual({t.id for t in out[:2]}, set(fav_ids))
        self.assertTrue(all(t.group == 'favorites' for t in out[:2]))

    def test_different_seeds_give_different_lists_and_popular_tags_are_likelier(self):
        cand = self._make_tags({'character': 300})
        a = {t.id for t in views._sidebar_tags(cand, 60, rng=_random.Random(1))}
        b = {t.id for t in views._sidebar_tags(cand, 60, rng=_random.Random(2))}
        self.assertNotEqual(a, b)
        hits = Counter()
        for seed in range(200):
            for t in views._sidebar_tags(cand, 60, rng=_random.Random(seed)):
                hits[t.count >= 40] += 1                    # counts are 1..50: top fifth vs the rest
        self.assertGreater(hits[True] / 200 / 60, 0.2)       # popular fifth shows up more than its 20% share

    def test_page_shows_balanced_random_sidebar(self):
        self._make_tags({'character': 400, 'general': 120, 'ai': 120, 'artist': 20, 'meta': 5})
        h1 = self.client.get('/').content.decode()
        h2 = self.client.get('/').content.decode()
        self.assertEqual(h1.count('class="tag-item tag-entry"'), 150)
        for cat in ('character', 'general', 'ai'):
            self.assertGreater(h1.count(f'data-cat="{cat}"'), 25)
        self.assertEqual(h1.count('tag-group-title">'), 5)           # one heading per type
        self.assertNotEqual(h1.split('<div class="tag-list" id="tag-list">')[1][:6000],
                            h2.split('<div class="tag-list" id="tag-list">')[1][:6000])   # random on every reload

    def test_search_still_lists_related_tags_with_filtered_counts(self):
        a = Post.objects.create(); b = Post.objects.create()
        t1 = Tag.objects.create(name='alpha', category='general', count=2)
        t2 = Tag.objects.create(name='beta', category='character', count=1)
        a.tags.add(t1, t2); b.tags.add(t1)
        out = views._sidebar_tags([(t1.id, 'general', False, 2), (t2.id, 'character', False, 1)], 10, filtered=True)
        self.assertEqual({t.name: t.filtered_count for t in out}, {'alpha': 2, 'beta': 1})
        html = self.client.get('/?tag=alpha').content.decode()
        self.assertIn('data-name="beta"', html)


# ── Part B: models panel, selector, free VRAM ───────────────────
_prefs_dir = None
_prefs_patch = None


def setUpModule():
    """Never let a test read or write the real prefs.json (it holds the user's
    aiMainModel / aiCharAuto / backup settings)."""
    global _prefs_dir, _prefs_patch
    _prefs_dir = tempfile.mkdtemp()
    _prefs_patch = mock.patch.object(prefs, 'legacy_path', return_value=os.path.join(_prefs_dir, 'prefs.json'))
    _prefs_patch.start()


def tearDownModule():
    _prefs_patch.stop()
    shutil.rmtree(_prefs_dir, ignore_errors=True)


from . import ai_models


class AiModelsPanelTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        o = override_settings(AI_MODELS_DIR=self.tmp); o.enable(); self.addCleanup(o.disable)
        views._set_pref('aiMainModel', 'wd14')
        ai_runtime._models.clear()
        self.addCleanup(ai_runtime._models.clear)

    def _ready(self, **ready):
        """Pretend which models are on disk."""
        return mock.patch.object(ai_models, 'is_ready', side_effect=lambda k: ready.get(k, False))

    def _custom_files(self):
        d = ai_models.custom_dir(); os.makedirs(d, exist_ok=True)
        for n in ('model.onnx', 'meta.json'):
            with open(os.path.join(d, n), 'wb') as f: f.write(b'x' * 10)

    def test_info_lists_all_models(self):
        with self._ready(wd14=True):
            d = self.client.get('/api/ai/models/').json()
        self.assertEqual([m['key'] for m in d['models']], ['wd14', 'pixai', 'custom'])
        self.assertTrue(d['models'][0]['ready'])
        self.assertEqual(d['main'], 'wd14')
        self.assertIn('gpu', d)

    def test_main_falls_back_to_default_when_selected_model_is_missing(self):
        views._set_pref('aiMainModel', 'pixai')
        with self._ready(wd14=True, pixai=False):
            self.assertEqual(views._main_model(), 'wd14')
        with self._ready(wd14=True, pixai=True):
            self.assertEqual(views._main_model(), 'pixai')

    def test_set_main_validates_and_persists(self):
        post = lambda m: self.client.post('/api/ai/main/', {'model': m}, content_type='application/json')
        self.assertEqual(post('nonsense').status_code, 400)
        with self._ready(wd14=True, pixai=False):
            self.assertEqual(post('pixai').status_code, 409)          # not downloaded
            self.assertEqual(post('wd14').status_code, 200)
        with self._ready(wd14=True, pixai=True):
            self.assertEqual(post('pixai').json()['main'], 'pixai')
            self.assertEqual(views._pref('aiMainModel'), 'pixai')

    def test_delete_guards(self):
        self._custom_files()
        with self._ready(wd14=True, custom=True):
            Task.objects.create(kind='ai_tag')
            self.assertEqual(self.client.post('/api/ai/models/custom/delete/').status_code, 409)   # AI task running
            Task.objects.all().delete()
            views._set_pref('aiMainModel', 'custom')
            self.assertEqual(self.client.post('/api/ai/models/custom/delete/').status_code, 409)   # it is the main tagger
            views._set_pref('aiMainModel', 'wd14')
            self.assertEqual(self.client.post('/api/ai/models/nope/delete/').status_code, 404)
            r = self.client.post('/api/ai/models/custom/delete/').json()
        self.assertTrue(r['ok'])
        self.assertFalse(os.path.exists(ai_models.custom_dir()))

    def test_download_endpoint_is_exclusive_and_rejects_custom(self):
        self.assertEqual(self.client.post('/api/ai/models/custom/download/').status_code, 400)
        with mock.patch.object(views.threading, 'Thread'):
            a = self.client.post('/api/ai/models/pixai/download/').json()['task_id']
            b = self.client.post('/api/ai/models/wd14/download/').json()['task_id']
        self.assertEqual(a, b)                                   # one download at a time

    def test_download_task_reports_progress_and_finishes(self):
        captured = {}
        with mock.patch.object(views, '_start_task', side_effect=lambda k, fn, **kw: captured.update(fn=fn) or types.SimpleNamespace(id=1)):
            self.client.post('/api/ai/models/pixai/download/')
        task = Task.objects.create(kind='model_download')
        def fake_download(key, on_progress):
            on_progress(500 * 1048576, 1)
        with mock.patch.object(ai_models, 'download', side_effect=fake_download) as dl:
            captured['fn'](task)
        dl.assert_called_once()
        task.refresh_from_db()
        self.assertEqual(task.done, task.total)
        self.assertIn('ready', task.message)

    def test_free_vram_stops_ai_tasks_and_signals_every_worker(self):
        ai_runtime._seen_signal = 0.0
        t = Task.objects.create(kind='char_retag')
        other = Task.objects.create(kind='scan')
        m = ai_runtime.get_model('t', '/x.onnx'); m._sess = object()
        with mock.patch.object(ai_runtime, 'gpu_info', return_value={'name': 'GPU', 'used_mb': 2000, 'total_mb': 8000}):
            d = self.client.post('/api/ai/free-vram/').json()
        self.assertEqual((d['tasks_stopped'], d['unloaded_here'], d['before']['used_mb']), (1, 1, 2000))
        t.refresh_from_db(); other.refresh_from_db()
        self.assertTrue(t.cancel_requested)
        self.assertFalse(other.cancel_requested)                 # only AI tasks are stopped
        self.assertIsNone(m._sess)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, '.unload')))

    def test_other_worker_unloads_when_it_sees_the_signal(self):
        # this process stands in for ANOTHER worker: it holds a model and has not seen the signal yet
        m = ai_runtime.get_model('t', '/x.onnx'); m._sess = object(); m.last_used = ai_runtime._now()
        ai_runtime._seen_signal = ai_runtime._signal_mtime()
        ai_runtime._force_until = 0.0
        ai_runtime._reaper_tick()
        self.assertIsNotNone(m._sess)                            # no signal, recently used: stays loaded
        time.sleep(0.02)
        ai_runtime.request_unload_all()
        ai_runtime._reaper_tick()
        self.assertIsNone(m._sess)                               # signal seen -> force-unloaded
        m._sess = object(); m.last_used = ai_runtime._now()      # an in-flight task reloads it...
        ai_runtime._reaper_tick()
        self.assertIsNone(m._sess)                               # ...and the force window frees it again
        ai_runtime._force_until = 0.0

    def test_old_signal_is_ignored_by_a_fresh_worker(self):
        ai_runtime.request_unload_all()
        ai_runtime._seen_signal = ai_runtime._signal_mtime()     # what _reaper_loop does at start
        m = ai_runtime.get_model('t', '/x.onnx'); m._sess = object(); m.last_used = ai_runtime._now()
        ai_runtime._force_until = 0.0
        ai_runtime._reaper_tick()
        self.assertIsNotNone(m._sess)

    def test_restart_workers_refuses_outside_gunicorn(self):
        self.assertEqual(self.client.post('/api/ai/restart-workers/').status_code, 409)

    def test_gpu_info_parses_nvidia_smi(self):
        fake = types.SimpleNamespace(stdout='Tesla P4, 2269, 8192\n')
        with mock.patch.object(ai_runtime.subprocess, 'run', return_value=fake):
            self.assertEqual(ai_runtime.gpu_info(), {'name': 'Tesla P4', 'used_mb': 2269, 'total_mb': 8192})
        with mock.patch.object(ai_runtime.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertIsNone(ai_runtime.gpu_info())

    def test_pixai_as_main_tagger_dispatch_and_flags(self):
        post = Post.objects.create()
        path = os.path.join(self.tmp, 'a.png'); Image.new('RGB', (20, 20)).save(path)
        ph = Photo(post=post, order=0); ph.file_path = path; ph.thumb_path = ''; ph.save()
        res = {'general': ['solo', 'smile'], 'character': ['char_q'], 'model': 'pixai'}
        with mock.patch.object(views, '_main_model', return_value='pixai'), \
             mock.patch.object(char_tagger, 'run_pixai_general', return_value=res) as px, \
             mock.patch.object(views, '_pref', side_effect=lambda k, d=None: True if k == 'aiCharAuto' else d), \
             mock.patch.object(char_tagger, 'model_ready', return_value=True), \
             mock.patch.object(char_tagger, 'run_character_tagger', side_effect=AssertionError('char pass must be skipped')):
            views.apply_ai_tags(post)
        px.assert_called_once()
        post.refresh_from_db()
        self.assertEqual((post.ai_tagged, post.char_tagged, post.char_model), (True, True, 'pixai'))
        self.assertEqual(Tag.objects.get(name='char_q').category, 'character')

    def test_run_pixai_general_filters_orders_and_caps(self):
        names = [f'g{i}' for i in range(60)] + ['char_a', 'char_b']
        mask = np.array([n.startswith('char_') for n in names])
        probs = np.array([0.31 + i * 0.001 for i in range(60)] + [0.9, 0.5], dtype=np.float32)
        probs[5] = 0.1                                           # below the 0.30 bar
        with mock.patch.object(char_tagger, '_infer', return_value=(probs, names, mask)):
            res = char_tagger.run_pixai_general('x')
        self.assertEqual(len(res['general']), 40)                # cap
        self.assertEqual(res['general'][0], 'g59')               # most confident first
        self.assertEqual(res['general'][-1], 'g20')              # the cap cuts the least confident
        self.assertNotIn('g5', res['general'])
        self.assertEqual(res['character'], ['char_a'])           # char_b 0.5 < 0.85
        self.assertEqual(res['model'], 'pixai')


# ── Part C: my model ────────────────────────────────────────────
from . import custom_model
from .models import CustomApplied, CustomConcept, CustomExample, PostFeature


def _planted(n, d=768, shift=2.0, seed=0, positive=True):
    """Synthetic 768-d 'features': positives are shifted along a fixed direction."""
    rng = np.random.default_rng(seed)
    direction = np.random.default_rng(123).normal(size=d); direction /= np.linalg.norm(direction)
    X = rng.normal(scale=0.4, size=(n, d)).astype(np.float32)
    if positive:
        X += (shift * direction).astype(np.float32)
    return X


class CustomNumericsTests(TestCase):
    def test_train_recovers_a_planted_direction_and_generalises(self):
        Xp, Xn = _planted(30, seed=1), _planted(120, seed=2, positive=False)
        X = np.vstack([Xp, Xn]); y = np.array([1] * 30 + [0] * 120)
        w, b, thr, m = custom_model.train(X, y)
        self.assertGreater(m['auc'], 0.95)
        self.assertGreaterEqual(thr, 0.1); self.assertLessEqual(thr, 0.995)
        # held-out examples
        tp = custom_model._sigmoid(_planted(200, seed=7) @ w + b)
        tn = custom_model._sigmoid(_planted(200, seed=8, positive=False) @ w + b)
        # in 768-d with only 150 examples the head is under-confident on NEW positives, so the
        # out-of-fold threshold lands below 0.5; it must still find most positives and almost never fire on negatives.
        self.assertLess(thr, 0.6)
        self.assertGreater((tp >= thr).mean(), 0.7)
        self.assertLess((tn >= thr).mean(), 0.06)                # (the cut-off ignores the top ~2.5% of negatives: real ones hide true positives)

    def test_weights_apply_to_raw_features(self):
        """fit_standardised must fold the scaling back: logit = x @ w + b on RAW x."""
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=3.0, size=(80, 768)); y = (rng.random(80) > 0.5).astype(int)
        X[y == 1, 0] += 4
        w, b = custom_model.fit_standardised(X, y, 10.0)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        theta = custom_model.fit_logreg((X - mu) / sd, y, 10.0, custom_model._weights(y))
        np.testing.assert_allclose(X @ w + b, ((X - mu) / sd) @ theta[:-1] + theta[-1], rtol=1e-6, atol=1e-6)

    def test_too_few_examples_is_refused(self):
        with self.assertRaises(RuntimeError):
            custom_model.train(_planted(5), np.array([1, 1, 1, 0, 0]))

    def test_threshold_accounts_for_a_rare_concept(self):
        y = np.array([1] * 50 + [0] * 50)
        p = np.concatenate([np.linspace(0.6, 0.99, 50), np.linspace(0.05, 0.75, 50)])   # overlapping scores
        t_rare, _, _ = custom_model.choose_threshold(y, p, prior=0.01)
        t_common, _, _ = custom_model.choose_threshold(y, p, prior=0.5)
        self.assertGreaterEqual(t_rare, t_common)               # rarer concept -> stricter cut-off

    def test_auc(self):
        self.assertAlmostEqual(custom_model.auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])), 1.0)
        self.assertAlmostEqual(custom_model.auc(np.array([0, 1, 0, 1]), np.array([.9, .1, .8, .2])), 0.0)


class CustomHeadsAndTeachingTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        o = override_settings(AI_MODELS_DIR=self.tmp); o.enable(); self.addCleanup(o.disable)
        custom_model._heads_cache.update(mtime=None, heads=None)
        self.addCleanup(custom_model._heads_cache.update, mtime=None, heads=None)

    def _posts(self, n):
        out = []
        for i in range(n):
            post = Post.objects.create(ai_tagged=True)
            path = os.path.join(self.tmp, f'p{post.pk}.png'); Image.new('RGB', (20, 20)).save(path)
            ph = Photo(post=post, order=0); ph.file_path = path; ph.thumb_path = ''; ph.save()
            out.append(post)
        return out

    def test_heads_roundtrip_threshold_and_enabled(self):
        w = np.zeros(768, np.float32); w[0] = 4.0
        custom_model.set_head('my_style', w, -2.0, 0.8)
        f = np.zeros(768, np.float32); f[0] = 1.0                 # logit = 4 - 2 = 2 -> p = 0.88
        self.assertEqual([n for n, _ in custom_model.learned_tags(f)], ['my_style'])
        custom_model.update_head('my_style', thr=0.95)
        self.assertEqual(custom_model.learned_tags(f), [])        # below the new threshold
        custom_model.update_head('my_style', thr=0.5, enabled=False)
        self.assertEqual(custom_model.learned_tags(f), [])        # disabled
        custom_model.update_head('my_style', enabled=True)
        custom_model.set_head('other', -w, 0.0, 0.5)
        self.assertEqual(sorted(custom_model.load_heads()['names']), ['my_style', 'other'])
        custom_model.remove_head('my_style')
        self.assertEqual(custom_model.load_heads()['names'], ['other'])

    def test_teach_endpoint_plus_minus_and_validation(self):
        a, b, c = self._posts(3)
        post = lambda **kw: self.client.post('/api/custom/examples/', kw, content_type='application/json')
        with mock.patch.object(views, '_wd14_tags', return_value=(['smile', '1girl'], np.array([False, False]))):
            self.assertEqual(post(ids=[a.pk], concept='Bad Name!?', label=1).status_code, 400)
            self.assertEqual(post(ids=[], concept='ok', label=1).status_code, 400)
            self.assertEqual(post(ids=[a.pk], concept='ok', label=1, category='nope').status_code, 400)
            self.assertEqual(post(ids=[a.pk], concept='smile', label=1).status_code, 409)     # default model already knows it
            r = post(ids=[a.pk, b.pk], concept='My Style', label=1, category='general').json()
            self.assertEqual((r['concept'], r['created'], r['pos'], r['tagged']), ('my_style', True, 2, 2))
            self.assertEqual(set(Post.objects.get(pk=a.pk).tags.values_list('name', 'category')), {('my_style', 'general')})
            self.assertEqual(Tag.objects.get(name='my_style').count, 2)
            # a wrong one: mark b as NOT showing it -> tag removed, negative example kept
            r = post(ids=[b.pk], concept='my_style', label=-1).json()
            self.assertEqual((r['untagged'], r['pos'], r['neg']), (1, 1, 1))
            self.assertFalse(Post.objects.get(pk=b.pk).tags.filter(name='my_style').exists())
            self.assertEqual(Tag.objects.get(name='my_style').count, 1)
            # changing your mind on a post flips its label instead of duplicating it
            post(ids=[b.pk], concept='my_style', label=1)
            self.assertEqual(CustomExample.objects.filter(concept__name='my_style', post=b).count(), 1)
            self.assertEqual(CustomExample.objects.get(concept__name='my_style', post=b).label, 1)
        self.assertEqual(CustomExample.objects.filter(concept__name='my_style').count(), 2)

    def _fake_features(self, positives):
        pos_ids = {p.pk for p in positives}
        def fake(post_id):
            return _planted(1, seed=post_id, positive=post_id in pos_ids)[0]
        return fake

    def test_train_concept_end_to_end_with_fake_features(self):
        pos = self._posts(24)
        library = self._posts(150)                                  # the rest of the library (negatives are sampled from here)
        c = CustomConcept.objects.create(name='my_style', category='general')
        for p in pos:
            CustomExample.objects.create(concept=c, post=p, label=1)
        fake = self._fake_features(pos)
        with mock.patch.object(custom_model, 'is_ready', return_value=True), \
             mock.patch.object(custom_model, 'model_hash', return_value='h1'), \
             mock.patch.object(custom_model, 'feature_for_post', side_effect=fake):
            m = custom_model.train_concept(c)
        c.refresh_from_db()
        self.assertGreater(m['auc'], 0.95)
        self.assertEqual((c.n_pos, c.n_neg), (24, 96))              # 4x auto negatives
        self.assertIsNotNone(c.trained_at)
        self.assertEqual(c.examples.filter(auto=True).count(), 96)
        self.assertTrue(all(e.feat is not None for e in c.examples.all()))   # features cached for retraining
        # the learned head recognises a fresh positive and rejects a fresh negative
        self.assertEqual([n for n, _ in custom_model.learned_tags(_planted(1, seed=999)[0])], ['my_style'])
        self.assertEqual(custom_model.learned_tags(_planted(1, seed=998, positive=False)[0]), [])
        # retraining re-uses the cached features: the clone is never asked again
        with mock.patch.object(custom_model, 'is_ready', return_value=True), \
             mock.patch.object(custom_model, 'model_hash', return_value='h1'), \
             mock.patch.object(custom_model, 'feature_for_post', side_effect=AssertionError('features should be cached')):
            custom_model.train_concept(c)

    def test_train_endpoint_requires_clone_and_reports_in_task(self):
        c = CustomConcept.objects.create(name='x')
        with mock.patch.object(custom_model, 'is_ready', return_value=False):
            self.assertEqual(self.client.post(f'/api/custom/concept/{c.pk}/train/').status_code, 409)

    def test_my_model_tagging_is_identical_to_default_plus_learned_tags(self):
        post = self._posts(1)[0]
        p = {'rating_a': 0.9}
        p.update({f'g{i}': 0.5 for i in range(60)}); p.update({'char_a': 0.95, 'char_c': 0.9})
        fake_model, names, is_char = _fake_wd14(p)
        probs = fake_model.probs
        w = np.zeros(768, np.float32); w[0] = 6.0
        custom_model.set_head('my_style', w, -3.0, 0.8)
        CustomConcept.objects.create(name='my_style', category='general')
        feat = np.zeros(768, np.float32); feat[0] = 1.0            # logit 3 -> p 0.95
        with mock.patch.object(views, '_get_wd14_model', return_value=(fake_model, names, is_char)):
            default = views.run_ai_tagger(self.tmp + '/p%d.png' % post.pk, model='wd14')
        with mock.patch.object(views, '_wd14_tags', return_value=(names, is_char)), \
             mock.patch.object(custom_model, 'run_clone', return_value=(probs, feat)):
            mine = views.run_ai_tagger(self.tmp + '/p%d.png' % post.pk, model='custom')
        self.assertEqual((mine['general'], mine['character']), (default['general'], default['character']))   # exact same
        self.assertEqual([n for n, _ in mine['learned']], ['my_style'])
        # and apply_ai_tags stores the learned tag in its concept's category
        with mock.patch.object(views, '_main_model', return_value='custom'), \
             mock.patch.object(views, '_wd14_tags', return_value=(names, is_char)), \
             mock.patch.object(custom_model, 'run_clone', return_value=(probs, feat)):
            views.apply_ai_tags(post)
        self.assertEqual(Tag.objects.get(name='my_style').category, 'general')
        self.assertTrue(post.tags.filter(name='my_style').exists())

    def test_update_and_delete_concept(self):
        c = CustomConcept.objects.create(name='my_style', category='ai')
        custom_model.set_head('my_style', np.zeros(768, np.float32), 0.0, 0.7)
        Tag.objects.create(name='my_style', category='ai', count=1)
        r = self.client.post(f'/api/custom/concept/{c.pk}/update/', {'threshold': 0.9, 'category': 'general', 'enabled': False},
                             content_type='application/json').json()
        self.assertAlmostEqual(r['concept']['threshold'], 0.9)
        self.assertEqual(Tag.objects.get(name='my_style').category, 'general')          # tag moved with the concept
        self.assertAlmostEqual(float(custom_model.load_heads()['thr'][0]), 0.9, places=5)
        self.assertFalse(bool(custom_model.load_heads()['enabled'][0]))
        self.client.post(f'/api/custom/concept/{c.pk}/delete/')
        self.assertFalse(CustomConcept.objects.exists())
        self.assertEqual(custom_model.load_heads()['names'], [])
        self.assertTrue(Tag.objects.filter(name='my_style').exists())                   # tags on posts stay

    def test_clone_endpoint_guards_and_info(self):
        with mock.patch.object(ai_models, 'is_ready', side_effect=lambda k: k == 'custom'):
            self.assertEqual(self.client.post('/api/custom/clone/').status_code, 409)   # already cloned
        with mock.patch.object(ai_models, 'is_ready', return_value=False):
            self.assertEqual(self.client.post('/api/custom/clone/').status_code, 409)   # default missing
            d = self.client.get('/api/custom/').json()
        self.assertFalse(d['ready']); self.assertEqual(d['concepts'], [])

    def test_deleting_the_clone_keeps_examples_but_drops_caches(self):
        p = self._posts(1)[0]
        c = CustomConcept.objects.create(name='a', n_pos=5, metrics={'x': 1}, trained_at=timezone.now())
        CustomExample.objects.create(concept=c, post=p, label=1, feat=b'xx', model_hash='h')
        CustomExample.objects.create(concept=c, post=self._posts(1)[0], label=-1, auto=True, feat=b'yy', model_hash='h')
        PostFeature.objects.create(post=p, vec=b'zz', model_hash='h')
        custom_model.on_clone_deleted()
        c.refresh_from_db()
        self.assertIsNone(c.trained_at); self.assertEqual(c.metrics, {})
        self.assertEqual(CustomExample.objects.count(), 1)                              # user's example kept, auto-negative dropped
        self.assertIsNone(CustomExample.objects.get().feat)
        self.assertFalse(PostFeature.objects.exists())


@unittest.skipUnless(
    __import__('importlib').util.find_spec('onnx') and ai_models.cached_file('wd14', 'model.onnx'),
    'needs the onnx package and the downloaded default model')
class RealCloneTests(TestCase):
    def test_clone_reproduces_default_exactly_and_exposes_features(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with override_settings(AI_MODELS_DIR=tmp):
            msgs = []
            custom_model.clone_default(msgs.append)
            self.assertTrue(custom_model.is_ready())
            self.assertEqual(custom_model.meta()['dim'], 768)
            self.assertEqual(len(custom_model.model_hash()), 40)
            import onnxruntime as ort
            src = ort.InferenceSession(ai_models.cached_file('wd14', 'model.onnx'), providers=['CPUExecutionProvider'])
            clone = ort.InferenceSession(ai_models.cached_file('custom', 'model.onnx'), providers=['CPUExecutionProvider'])
            x = (np.random.default_rng(5).random((1, 448, 448, 3)) * 255).astype(np.float32)
            a = src.run(None, {src.get_inputs()[0].name: x})[0]
            out, feat = clone.run(['output', custom_model.meta()['feature_tensor']], {clone.get_inputs()[0].name: x})
            self.assertTrue(np.array_equal(a, out))                 # bit-identical default outputs
            self.assertEqual(feat.shape, (1, 768))
            self.assertTrue(any('verif' in m for m in msgs))
            # deleting it
            ai_models.delete('custom')
            self.assertFalse(custom_model.is_ready())


class CustomLibraryTests(TestCase):
    """Phase C2: index the library once, then preview / apply / undo are matrix ops."""
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        o = override_settings(AI_MODELS_DIR=self.tmp); o.enable(); self.addCleanup(o.disable)
        custom_model._heads_cache.update(mtime=None, heads=None)
        self.addCleanup(custom_model._heads_cache.update, mtime=None, heads=None)
        p1 = mock.patch.object(custom_model, 'model_hash', return_value='h1'); p1.start(); self.addCleanup(p1.stop)
        p2 = mock.patch.object(custom_model, 'is_ready', return_value=True); p2.start(); self.addCleanup(p2.stop)
        self.posts = [Post.objects.create(ai_tagged=True) for _ in range(40)]
        self.pos = {p.pk for p in self.posts[:10]}                     # the 10 "true" positives
        d = np.random.default_rng(123).normal(size=768); d /= np.linalg.norm(d)
        self.w = (6.0 * d).astype(np.float32)                           # head along the planted direction
        custom_model.set_head('my_style', self.w, -6.0, 0.5)
        self.c = CustomConcept.objects.create(name='my_style', category='general', threshold=0.5, trained_at=timezone.now())
        self.feat = lambda pid: _planted(1, seed=pid, positive=pid in self.pos)[0]

    def _index(self):
        with mock.patch.object(custom_model, 'feature_for_post', side_effect=self.feat):
            return custom_model.scan_library()

    def test_scan_is_resumable_counts_failures_and_drops_stale_features(self):
        bad = self.posts[5].pk
        def flaky(pid):
            if pid == bad: raise RuntimeError('source file missing')
            return self.feat(pid)
        PostFeature.objects.create(post=self.posts[0], vec=b'old', model_hash='OLD')       # from an older clone
        with mock.patch.object(custom_model, 'feature_for_post', side_effect=flaky):
            done, failed = custom_model.scan_library()
        self.assertEqual((done, failed), (39, 1))
        self.assertEqual(PostFeature.objects.filter(model_hash='h1').count(), 39)
        self.assertFalse(PostFeature.objects.filter(model_hash='OLD').exists())
        with mock.patch.object(custom_model, 'feature_for_post', side_effect=AssertionError('already indexed')) as f:
            done, failed = custom_model.scan_library()                                      # only the failed one is retried
        self.assertEqual(done + failed, 1)

    def test_scan_can_be_cancelled(self):
        task = Task.objects.create(kind='custom_scan', cancel_requested=True)
        with mock.patch.object(custom_model, 'feature_for_post', side_effect=self.feat):
            with self.assertRaises(TaskCancelled):
                custom_model.scan_library(check=lambda: check_cancel(task))
        self.assertEqual(PostFeature.objects.count(), 0)

    def test_candidates_rank_exclude_labeled_and_tagged(self):
        self._index()
        labeled = self.posts[0]; tagged = self.posts[1]
        CustomExample.objects.create(concept=self.c, post=labeled, label=1)
        tag = Tag.objects.create(name='my_style', category='general'); tagged.tags.add(tag)
        ids, scores, n_above = custom_model.candidates(self.c, n=30)
        self.assertNotIn(labeled.pk, ids); self.assertNotIn(tagged.pk, ids)
        top8 = ids[:8]
        self.assertTrue(set(top8) <= self.pos)                           # the 8 remaining true positives come first
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(n_above, 8)                                     # what 'apply' would tag: the 8 unlabeled/untagged true positives
        unc, usc, _ = custom_model.candidates(self.c, n=5, mode='uncertain')
        self.assertEqual(len(unc), 5)
        self.assertEqual([abs(s - 0.5) for s in usc], sorted(abs(s - 0.5) for s in usc))    # closest to the threshold first

    def test_apply_tags_only_confident_unlabeled_posts_and_undo_spares_manual_tags(self):
        self._index()
        manual = self.posts[0]; neg = self.posts[1]
        tag = Tag.objects.create(name='my_style', category='general'); manual.tags.add(tag)        # tagged by hand
        CustomExample.objects.create(concept=self.c, post=neg, label=-1)                           # user said "no"
        n = custom_model.apply_concept(self.c)
        self.assertEqual(n, 8)                                           # 10 positives - manual - labeled "no"
        tagged = set(Post.objects.filter(tags__name='my_style').values_list('id', flat=True))
        self.assertEqual(tagged, (self.pos - {neg.pk}))
        self.assertEqual(Tag.objects.get(name='my_style').count, len(tagged))
        self.assertEqual(self.c.applied.count(), 8)
        self.assertEqual(custom_model.undo_concept(self.c), 8)
        left = set(Post.objects.filter(tags__name='my_style').values_list('id', flat=True))
        self.assertEqual(left, {manual.pk})                              # the hand-made tag survives
        self.assertEqual(Tag.objects.get(name='my_style').count, 1)
        self.assertEqual(self.c.applied.count(), 0)

    def test_apply_requires_an_index(self):
        with self.assertRaises(RuntimeError):
            custom_model.apply_concept(self.c)

    def test_endpoints(self):
        with mock.patch.object(custom_model, 'is_ready', return_value=False):
            self.assertEqual(self.client.post('/api/custom/scan/').status_code, 409)
        self._index()
        d = self.client.get(f'/api/custom/concept/{self.c.pk}/candidates/?n=5').json()
        self.assertEqual((len(d['ids']), d['indexed'], d['posts'], d['above_threshold']), (5, 40, 40, 10))
        self.assertTrue(set(d['ids']) <= self.pos)
        untrained = CustomConcept.objects.create(name='other')
        self.assertEqual(self.client.post(f'/api/custom/concept/{untrained.pk}/apply/').status_code, 409)
        info = self.client.get('/api/custom/').json()
        self.assertEqual((info['features_cached'], info['posts']), (40, 40))
        r = self.client.post(f'/api/custom/concept/{self.c.pk}/undo/').json()
        self.assertEqual(r['untagged'], 0)


# ── search suggestions, exclude buttons, quick links, tagger label ──
class TagSearchApiTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        for name, count in [('blue_hair', 50), ('hair_ribbon', 90), ('long_hair', 500), ('hairy', 3), ('hair_unused', 0)]:
            Tag.objects.create(name=name, count=count)

    def test_prefix_first_then_count_and_no_zero_count(self):
        names = [t['name'] for t in self.client.get('/api/tag-search/?q=hair').json()['tags']]
        self.assertEqual(names, ['hair_ribbon', 'hairy', 'long_hair', 'blue_hair'])   # prefix by count, then contains by count

    def test_leading_minus_and_empty_query(self):
        names = [t['name'] for t in self.client.get('/api/tag-search/?q=-hair_r').json()['tags']]
        self.assertEqual(names, ['hair_ribbon'])
        self.assertEqual(self.client.get('/api/tag-search/?q=').json()['tags'], [])

    def test_limit(self):
        self.assertEqual(len(self.client.get('/api/tag-search/?q=hair&limit=2').json()['tags']), 2)


@unittest.skipUnless(shutil.which('node'), 'node not installed')
class SearchTokenJsTests(TestCase):
    """searchTokenAt/applySuggestion live inline in base.html between markers;
    run the real source in Node."""
    def _run(self, expr):
        src = open(os.path.join(settings.BASE_DIR, 'templates/gallery/base.html'), encoding='utf-8').read()
        js = src[src.index('// <searchtok>'):src.index('// </searchtok>')]
        r = subprocess.run(['node', '-e', js + f'\nconsole.log(JSON.stringify({expr}))'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:300])
        return json.loads(r.stdout)

    def test_negated_last_token(self):
        t = self._run("searchTokenAt('tag1 -typ', 8)")
        self.assertEqual((t['prefix'], t['query'], t['others']), ('-', 'typ', ['tag1']))

    def test_trailing_space_has_no_active_token(self):
        self.assertIsNone(self._run("searchTokenAt('tag1 ', 5)"))

    def test_structural_and_operator_tokens(self):
        for v in ("( ", "a ~", "file:abc", "folder:x", "wild*", "fuzzy~", ">3"):
            self.assertIsNone(self._run(f"searchTokenAt({json.dumps(v)}, {len(v)})"), v)

    def test_caret_in_the_middle_picks_that_token(self):
        t = self._run("searchTokenAt('aa bb cc', 4)")
        self.assertEqual((t['query'], t['start'], t['end']), ('bb', 3, 5))

    def test_apply_keeps_minus_and_other_tokens(self):
        r = self._run("applySuggestion('a -typ c', searchTokenAt('a -typ c', 6), 'typhoon')")
        self.assertEqual(r['value'], 'a -typhoon c')
        self.assertEqual(r['caret'], len('a -typhoon '))


class ExcludeAndSortButtonsTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def test_sidebar_has_exclude_and_remember_sort(self):
        t = Tag.objects.create(name='cat', count=1)
        post = Post.objects.create(); post.tags.add(t)
        html = self.client.get('/').content.decode()
        self.assertIn("addTagToSearch('cat', true)", html)
        self.assertIn('remember-sort-btn', html.split('<aside>')[1].split('</aside>')[0])      # desktop sidebar, not just the phone sheet

    def test_detail_chip_has_exclude(self):
        t = Tag.objects.create(name='cat', count=1)
        post = Post.objects.create(); post.tags.add(t)
        html = self.client.get(f'/post/{post.pk}/').content.decode()
        self.assertIn("addTagToSearch('cat', true)", html)

    def test_negative_tag_filter_still_works(self):
        a, b = Post.objects.create(), Post.objects.create()
        for i, p in enumerate((a, b)):
            Photo.objects.create(post=p, order=0, rel_path=f'neg{i}.png')
        t = Tag.objects.create(name='cat', count=1); a.tags.add(t)
        ids = [p['id'] for p in self.client.get('/api/posts/?tag=-cat').json()['posts']]
        self.assertEqual(ids, [b.id])


class QuickLinksTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        views._set_pref('quickLinks', [])

    def _add(self, text):
        return self.client.post('/api/quick-links/', json.dumps({'add': text}), content_type='application/json')

    def test_parse(self):
        got = views.parse_quick_links('my site https://www.example.org/a\nhttps://b.io/x http://c.net\njunk line\n/duplicates/')
        self.assertEqual([(l['label'], l['url']) for l in got], [
            ('my site', 'https://www.example.org/a'), ('b.io', 'https://b.io/x'), ('c.net', 'http://c.net'), ('/duplicates/', '/duplicates/')])

    def test_rejects_non_http_schemes(self):
        self.assertEqual(self._add('javascript:alert(1)').status_code, 400)
        self.assertEqual(self.client.get('/api/quick-links/').json()['links'], [])

    def test_add_dedupe_remove_and_header(self):
        self.assertEqual(len(self._add('home https://example.org\nhttps://example.org').json()['links']), 1)
        html = self.client.get('/').content.decode()
        self.assertIn('class="btn header-desktop quick-link" href="https://example.org"', html)
        d = self.client.post('/api/quick-links/', json.dumps({'remove': 0}), content_type='application/json').json()
        self.assertEqual(d['links'], [])
        self.assertNotIn('quick-link" href', self.client.get('/').content.decode())

    def test_cap(self):
        self._add('\n'.join(f'https://e{i}.org' for i in range(30)))
        self.assertEqual(len(self.client.get('/api/quick-links/').json()['links']), views.QUICK_LINKS_MAX)


class TaggerLabelTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_custom_run_is_labelled_and_learned_tags_reach_the_endpoint(self):
        img = os.path.join(self.tmp, 'a.png'); Image.new('RGB', (20, 20)).save(img)
        post = Post.objects.create(); Photo.objects.create(post=post, rel_path=img, rel_thumb_path='', width=20, height=20)
        fake = {'general': ['a'], 'character': [], 'model': 'custom', 'learned': [('my_style', 0.9)]}
        with mock.patch.object(views, 'run_ai_tagger', return_value=fake), \
             mock.patch.object(views.custom_model, 'apply_learned') as al:
            d = self.client.post(f'/api/post/{post.pk}/ai-tag/').json()
        self.assertEqual((d['model'], d['learned']), ('custom', ['my_style']))
        al.assert_called_once()

    def test_wd14_run_with_custom_main_reports_custom(self):
        img = os.path.join(self.tmp, 'a.png'); Image.new('RGB', (20, 20)).save(img)
        probs = {'rating_a': 0.9}
        with mock.patch.object(views.custom_model, 'run_clone', return_value=(_fake_wd14(probs)[0].probs, None)), \
             mock.patch.object(views, '_wd14_tags', return_value=_fake_wd14(probs)[1:]), \
             mock.patch.object(views.custom_model, 'learned_tags', return_value=[]):
            res = views.run_ai_tagger(img, model='custom')
        self.assertEqual(res['model'], 'custom')


class PrefsStoreTests(TestCase):
    def test_set_get_all_update(self):
        prefs.set('a', {'x': 1}); prefs.set('b', [1, 2])
        self.assertEqual(prefs.get('a'), {'x': 1})
        self.assertEqual(prefs.get('missing', 'dflt'), 'dflt')
        self.assertEqual(prefs.all(), {'a': {'x': 1}, 'b': [1, 2]})
        self.assertEqual(prefs.update('b', lambda v: v + [3]), [1, 2, 3])
        self.assertEqual(prefs.update('new', lambda v: v + 1, default=0), 1)

    def test_legacy_prefs_json_is_imported_once_without_overwriting(self):
        legacy = os.path.join(tempfile.mkdtemp(), 'prefs.json')
        self.addCleanup(shutil.rmtree, os.path.dirname(legacy), ignore_errors=True)
        with open(legacy, 'w') as f:
            json.dump({'quickLinks': [{'label': 'a', 'url': 'https://a.io'}], 'aiMainModel': 'custom'}, f)
        with mock.patch.object(prefs, 'legacy_path', return_value=legacy), mock.patch.object(prefs, '_legacy_checked', False):
            self.assertEqual(prefs.get('aiMainModel'), 'custom')
            self.assertEqual(prefs.get('quickLinks')[0]['url'], 'https://a.io')
        self.assertFalse(os.path.exists(legacy))
        self.assertTrue(os.path.exists(legacy + '.imported'))

    def test_recent_posts_push_does_not_wipe_other_settings(self):
        """The bug: every post view rewrote the whole prefs file from a stale copy."""
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        prefs.set('quickLinks', [{'label': 'x', 'url': 'https://x.io'}])
        prefs.set('aiCharAuto', True)
        prefs.set('aiMainModel', 'pixai')
        for i in range(1, 4):
            r = self.client.post('/api/recent/add/', json.dumps({'id': i, 'thumb': 't', 'url': f'/post/{i}/'}), content_type='application/json')
            self.assertEqual(r.status_code, 200)
        self.assertEqual((prefs.get('aiCharAuto'), prefs.get('aiMainModel')), (True, 'pixai'))
        self.assertEqual(len(prefs.get('quickLinks')), 1)
        self.assertEqual([r['id'] for r in prefs.get('recentPosts')], [3, 2, 1])

    def test_char_model_toggle_roundtrip_through_the_api(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        self.client.post('/api/pref/set/', json.dumps({'key': 'aiCharAuto', 'value': True}), content_type='application/json')
        self.assertTrue(self.client.get('/api/ai/info/').json()['auto'])
        self.client.post('/api/pref/set/', json.dumps({'key': 'aiCharAuto', 'value': False}), content_type='application/json')
        self.assertFalse(self.client.get('/api/ai/info/').json()['auto'])


class UnprocessedFilterTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        mk = lambda **kw: Post.objects.create(**kw)
        self.none = mk(); self.ai_only = mk(ai_tagged=True, char_model='wd14')
        self.px_only = mk(char_model='pixai'); self.both = mk(ai_tagged=True, char_model='pixai')
        for i, p in enumerate((self.none, self.ai_only, self.px_only, self.both)):
            Photo.objects.create(post=p, order=0, rel_path=f'u{i}.png')

    def _ids(self, proc):
        return {p['id'] for p in self.client.get(f'/api/posts/?proc={proc}').json()['posts']}

    def test_filters(self):
        self.assertEqual(self._ids('none'), {self.none.id, self.ai_only.id} - {self.ai_only.id})   # neither AI nor PixAI
        self.assertEqual(self._ids('noai'), {self.none.id, self.px_only.id})
        self.assertEqual(self._ids('nopixai'), {self.none.id, self.ai_only.id})

    def test_stats_and_pill(self):
        d = self.client.get('/api/debug/stats/').json()
        self.assertEqual((d['unprocessed'], d['no_ai'], d['no_pixai']), (1, 2, 2))
        self.assertIn('not AI-tagged &amp; not PixAI-checked', self.client.get('/?proc=none').content.decode())

    def test_filter_survives_into_post_navigation(self):
        html = self.client.get(f'/post/{self.none.id}/?proc=none').content.decode()
        self.assertIn('proc=none', html)


class SettingsPanelLayoutTests(TestCase):
    def test_retag_buttons_live_in_maintenance_with_tooltips(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        html = self.client.get('/').content.decode()
        sec = html.split('<h3>maintenance</h3>')[1].split('<h3>ai models</h3>')[0]
        for needle in ('startCharRetag()', 'startMultiRetag()', 'rebasePaths()', 'soundTagAll()', 'organizeSinglesDeep()'):
            self.assertIn(needle, sec)
        self.assertEqual(len(re.findall(r'<button class="btn"[^>]*title="', sec)) + len(re.findall(r'<a class="btn"[^>]*title="', sec)), 6)
        tagging = html.split('<h3>ai tagging</h3>')[1].split('<h3>debug overlays')[0]
        self.assertNotIn('startCharRetag', tagging)
        self.assertIn('id="ai-auto"', tagging)


from django.test import TransactionTestCase


class PrefsConcurrencyTests(TransactionTestCase):
    def test_parallel_writers_neither_fail_nor_lose_keys(self):
        import threading
        from django.db import connection
        errs = []

        def w(i):
            try:
                for j in range(10):
                    prefs.set(f'k{i}_{j}', j)
                    prefs.update('shared', lambda v: v + 1, default=0)
            except Exception as e:      # noqa: BLE001
                errs.append(e)
            finally:
                connection.close()
        ts = [threading.Thread(target=w, args=(i,)) for i in range(5)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errs, [])
        data = prefs.all()
        self.assertEqual(sum(1 for k in data if k.startswith('k')), 50)
        self.assertEqual(data['shared'], 50)          # every atomic increment counted


# ── multi-image AI tagging ──────────────────────────────────────
class MultiImageTaggingTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def _post(self, n):
        post = Post.objects.create()
        for i in range(n):
            Photo.objects.create(post=post, order=i, rel_path=f'mt_{post.pk}_{i}.png')
        return post

    def _fake(self, per_image):
        """run_ai_tagger replacement: result i for the i-th call."""
        calls = iter(per_image)
        return mock.patch.object(views, 'run_ai_tagger', side_effect=lambda *a, **k: next(calls))

    def test_merge_ranks_by_image_count_and_keeps_strictest_rating(self):
        res = views._merge_ai_results([
            {'general': ['general', '1girl', 'smile'], 'character': ['a'], 'model': 'wd14'},
            {'general': ['explicit', '1girl', 'hat'], 'character': ['a', 'b'], 'model': 'wd14'},
            {'general': ['sensitive', '1girl', 'hat'], 'character': [], 'model': 'wd14'},
        ])
        self.assertEqual(res['general'], ['explicit', '1girl', 'hat', 'smile'])   # rating first; 3x, 2x, 1x
        self.assertEqual(res['character'], ['a', 'b'])

    def test_merge_of_one_result_is_unchanged(self):
        r = {'general': ['x', 'general'], 'character': ['c'], 'model': 'wd14'}
        self.assertIs(views._merge_ai_results([r]), r)

    def test_merge_learned_keeps_best_probability(self):
        res = views._merge_ai_results([
            {'general': [], 'character': [], 'learned': [('s', 0.8)], 'model': 'custom'},
            {'general': [], 'character': [], 'learned': [('s', 0.95), ('t', 0.9)], 'model': 'custom'}])
        self.assertEqual(res['learned'], [('s', 0.95), ('t', 0.9)])

    def test_sampling_keeps_first_and_last(self):
        imgs = list(range(100))
        got = views._sample_images(imgs, 24)
        self.assertEqual((len(got), got[0], got[-1]), (24, 0, 99))
        self.assertEqual(views._sample_images(imgs[:10], 24), imgs[:10])

    def test_apply_tags_every_item_and_marks_post(self):
        post = self._post(3)
        with self._fake([{'general': ['a'], 'character': [], 'model': 'wd14'},
                         {'general': ['b'], 'character': [], 'model': 'wd14'},
                         {'general': ['b', 'c'], 'character': [], 'model': 'wd14'}]) as m:
            views.apply_ai_tags(post)
        self.assertEqual(m.call_count, 3)
        self.assertEqual(set(post.tags.values_list('name', flat=True)), {'a', 'b', 'c'})
        post.refresh_from_db()
        self.assertTrue(post.ai_multi and post.ai_tagged)

    def test_explicit_cover_tags_just_that_image(self):
        post = self._post(3)
        with self._fake([{'general': ['only'], 'character': [], 'model': 'wd14'}]) as m:
            views.apply_ai_tags(post, post.images.first())
        self.assertEqual(m.call_count, 1)
        post.refresh_from_db()
        self.assertFalse(post.ai_multi)

    def test_unreadable_item_is_skipped_but_all_failing_raises(self):
        post = self._post(2)
        seq = iter([RuntimeError('bad'), {'general': ['ok'], 'character': [], 'model': 'wd14'}])

        def run(*a, **k):
            r = next(seq)
            if isinstance(r, Exception):
                raise r
            return r
        with mock.patch.object(views, 'run_ai_tagger', side_effect=run):
            views.apply_ai_tags(post)
        self.assertEqual(list(post.tags.values_list('name', flat=True)), ['ok'])
        with mock.patch.object(views, 'run_ai_tagger', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                views.apply_ai_tags(self._post(2))

    def test_single_image_post_behaves_as_before(self):
        post = self._post(1)
        with self._fake([{'general': ['a'], 'character': ['c'], 'model': 'wd14'}]):
            res = views.apply_ai_tags(post)
        self.assertEqual((res['general'], res['character']), (['a'], ['c']))
        post.refresh_from_db()
        self.assertFalse(post.ai_multi)

    def test_retag_task_skips_done_posts_and_info_counts(self):
        todo, done, single = self._post(2), self._post(2), self._post(1)
        Post.objects.filter(pk=done.pk).update(ai_multi=True)
        info = self.client.get('/api/ai/info/').json()
        self.assertEqual((info['multi_todo'], info['multi_done']), (1, 1))
        with self._fake([{'general': ['t1'], 'character': [], 'model': 'wd14'},
                         {'general': ['t2'], 'character': [], 'model': 'wd14'}]) as m, \
             mock.patch.object(views, '_start_task', side_effect=lambda kind, fn, **kw: _run_now(kind, fn)):
            self.client.post('/api/ai/multi-retag/')
        self.assertEqual(m.call_count, 2)                      # only `todo` (2 items), not `done` / `single`
        self.assertEqual(set(todo.tags.values_list('name', flat=True)), {'t1', 't2'})
        self.assertEqual(done.tags.count() + single.tags.count(), 0)
        self.assertEqual(self.client.get('/api/ai/info/').json()['multi_todo'], 0)


def _run_now(kind, fn):
    task = Task.objects.create(kind=kind, status='running')
    fn(task)
    return task


# ── /media/ Range support ──────────────────────────────────────
class MediaRangeTests(TestCase):
    DATA = bytes(range(256)) * 40            # 10,240 bytes

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        with open(os.path.join(self.tmp, 'v.mp4'), 'wb') as f:
            f.write(self.DATA)
        os.makedirs(os.path.join(self.tmp, 'dir'))
        ov = override_settings(MEDIA_ROOT=self.tmp)
        ov.enable(); self.addCleanup(ov.disable)
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def _get(self, **hdr):
        return self.client.get('/media/v.mp4', **hdr)

    def _body(self, r):
        return b''.join(r.streaming_content)

    def test_plain_get_advertises_ranges(self):
        r = self._get()
        self.assertEqual((r.status_code, r['Accept-Ranges'], r['Content-Length'], r['Content-Type']), (200, 'bytes', '10240', 'video/mp4'))
        self.assertEqual(self._body(r), self.DATA)

    def test_partial_ranges(self):
        r = self._get(HTTP_RANGE='bytes=0-99')
        self.assertEqual((r.status_code, r['Content-Range'], r['Content-Length']), (206, 'bytes 0-99/10240', '100'))
        self.assertEqual(self._body(r), self.DATA[:100])
        r = self._get(HTTP_RANGE='bytes=10000-')
        self.assertEqual((r.status_code, r['Content-Range']), (206, 'bytes 10000-10239/10240'))
        self.assertEqual(self._body(r), self.DATA[10000:])
        r = self._get(HTTP_RANGE='bytes=-16')
        self.assertEqual(self._body(r), self.DATA[-16:])
        r = self._get(HTTP_RANGE='bytes=100-99999')                # end past EOF is clamped
        self.assertEqual(r['Content-Range'], 'bytes 100-10239/10240')

    def test_unsatisfiable_and_garbage_ranges(self):
        r = self._get(HTTP_RANGE='bytes=20000-')
        self.assertEqual((r.status_code, r['Content-Range']), (416, 'bytes */10240'))
        self.assertEqual(self._get(HTTP_RANGE='lines=1-2').status_code, 200)    # not a byte range: ignore

    def test_if_range_mismatch_serves_everything(self):
        r = self._get(HTTP_RANGE='bytes=0-9', HTTP_IF_RANGE='Mon, 01 Jan 2001 00:00:00 GMT')
        self.assertEqual(r.status_code, 200)
        lm = self._get()['Last-Modified']
        self.assertEqual(self._get(HTTP_RANGE='bytes=0-9', HTTP_IF_RANGE=lm).status_code, 206)

    def test_not_modified(self):
        lm = self._get()['Last-Modified']
        self.assertEqual(self._get(HTTP_IF_MODIFIED_SINCE=lm).status_code, 304)

    def test_traversal_and_directories_are_404(self):
        for u in ('/media/../db.sqlite3', '/media/%2e%2e/db.sqlite3', '/media/dir', '/media/nope.mp4', '/media/dir/../../x'):
            self.assertEqual(self.client.get(u).status_code, 404, u)

    def test_login_still_required(self):
        self.client.get('/logout/')
        self.assertEqual(self._get().status_code, 302)


@unittest.skipUnless(shutil.which('node'), 'node not installed')
class GalleryPageForJsTests(TestCase):
    def test_page_math_matches_server_page_size(self):
        src = open(os.path.join(settings.BASE_DIR, 'templates/gallery/detail.html'), encoding='utf-8').read()
        fn = re.search(r'function galleryPageFor\(.*?\n', src).group(0)
        js = 'const GALLERY_PAGE_SIZE = 40;\n' + fn + 'console.log(JSON.stringify([0,39,40,81].map(galleryPageFor)))'
        r = subprocess.run(['node', '-e', js], capture_output=True, text=True)
        self.assertEqual(json.loads(r.stdout), [1, 1, 2, 3])


# ── URLs of odd file names, thumbnail repair, network path, search selector ──
class MediaUrlTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ov = override_settings(MEDIA_ROOT=self.tmp)
        ov.enable(); self.addCleanup(ov.disable)
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def test_special_characters_are_percent_encoded_and_served(self):
        rel = 'inbox/_/Fond #2 100% ?x/a #1.png'
        os.makedirs(os.path.dirname(os.path.join(self.tmp, rel)))
        Image.new('RGB', (4, 4)).save(os.path.join(self.tmp, rel))
        post = Post.objects.create(); ph = Photo.objects.create(post=post, rel_path=rel)
        self.assertEqual(ph.media_url, '/media/inbox/_/Fond%20%232%20100%25%20%3Fx/a%20%231.png')
        self.assertEqual(self.client.get(ph.media_url).status_code, 200)

    def test_outside_media_root_gives_no_url_instead_of_a_broken_one(self):
        ph = Photo(rel_path='/elsewhere/x.png', rel_thumb_path='/ssd/old/thumbs/t.jpg')
        self.assertEqual(ph.media_url, '')
        self.assertEqual(ph.thumb_url, '')

    def test_missing_thumbnail_is_rebuilt(self):
        rel = 'inbox/a.png'
        os.makedirs(os.path.join(self.tmp, 'inbox'))
        Image.new('RGB', (40, 30), (200, 0, 0)).save(os.path.join(self.tmp, rel))
        post = Post.objects.create()
        ph = Photo.objects.create(post=post, rel_path=rel, rel_thumb_path='/mnt/ssd/old_root/thumbs/gone.jpg')
        self.assertEqual(views._repair_missing_thumbs(), (1, 0))
        ph.refresh_from_db()
        self.assertFalse(os.path.isabs(ph.rel_thumb_path))
        self.assertTrue(os.path.exists(ph.thumb_path))
        self.assertTrue(ph.thumb_url.startswith('/media/thumbs/'))


class NetPathTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        views._set_pref('netPathPrefix', '')

    def test_auto_prefix_from_mnt_media_root_and_request_host(self):
        class Req:
            def get_host(self): return '192.168.1.50:3002'
        with override_settings(MEDIA_ROOT='/mnt/mass/Media_MASS/Photo'):
            self.assertEqual(views._auto_net_prefix(Req()), '\\\\192.168.1.50\\mass\\Media_MASS\\Photo\\')
            self.assertEqual(views._net_path('/mnt/mass/Media_MASS/Photo/inbox/x y.jpg', Req()),
                             '\\\\192.168.1.50\\mass\\Media_MASS\\Photo\\inbox\\x y.jpg')

    def test_localhost_falls_back_to_the_lan_ip(self):
        class Req:
            def get_host(self): return 'localhost:3002'
        with override_settings(MEDIA_ROOT='/mnt/mass/Photo'), mock.patch.object(views, '_lan_ip', return_value='10.0.0.7'):
            self.assertEqual(views._auto_net_prefix(Req()), '\\\\10.0.0.7\\mass\\Photo\\')

    def test_saved_prefix_wins_and_is_normalised(self):
        d = self.client.post('/api/net-path/', json.dumps({'prefix': '//nas/share/Photo'}), content_type='application/json').json()
        self.assertEqual(d['effective'], '\\\\nas\\share\\Photo\\')
        self.assertEqual(d['saved'], '//nas/share/Photo')
        d = self.client.post('/api/net-path/', json.dumps({'prefix': ''}), content_type='application/json').json()
        self.assertTrue(d['effective'].startswith('\\\\') and d['effective'] == d['auto'])

    def test_detail_page_gets_the_prefix(self):
        views._set_pref('netPathPrefix', r'\\nas\share\\')
        post = Post.objects.create()
        html = self.client.get(f'/post/{post.pk}/').content.decode()
        self.assertIn("const NET_PREFIX = '", html)
        self.assertNotIn('192.168.1.50\\\\@', html)


class SearchSelectorRegressionTests(TestCase):
    def test_tag_filter_ignores_rows_without_data_name(self):
        """Folder tree rows carry .tag-entry but no data-name; reading it threw before the
        suggestion request, so the search box never suggested anything."""
        src = open(os.path.join(settings.BASE_DIR, 'templates/gallery/index.html'), encoding='utf-8').read()
        self.assertIn("querySelectorAll('#tag-list .tag-entry')", src)
        self.assertNotIn("querySelectorAll('.tag-entry')", src)


# ── AI edit: ComfyUI img2img + Ollama prompt ───────────────────
import io as _io

import httpx

from . import comfy


def _png(size=(32, 24), color=(200, 30, 30)):
    buf = _io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


class ComfyWorkflowTests(TestCase):
    def test_every_link_resolves_and_chain_is_wired(self):
        wf = comfy.build_img2img('a.png', 'pos', 'neg', 5, comfy.config(), 0.6, temp_node=True)
        self.assertEqual(comfy.workflow_problems(wf), [])
        ks = wf['8']['inputs']
        self.assertEqual(ks['latent_image'], ['7', 0])                 # temperature gate sits between encode and sampler
        self.assertEqual(wf['7']['inputs']['passthrough'], ['4', 0])
        self.assertEqual(wf['10']['class_type'], 'PreviewImage')       # nothing lands in ComfyUI's output folder
        self.assertEqual((ks['denoise'], ks['seed']), (0.6, 5))

    def test_no_temp_node_when_unavailable_or_off(self):
        wf = comfy.build_img2img('a.png', 'p', 'n', 1, comfy.config(), temp_node=False)
        self.assertNotIn('7', wf)
        self.assertEqual(wf['8']['inputs']['latent_image'], ['4', 0])
        cfg = dict(comfy.config(), max_temp=0)
        self.assertNotIn('7', comfy.build_img2img('a.png', 'p', 'n', 1, cfg, temp_node=True))

    def test_denoise_is_clamped_and_never_one(self):
        for asked, want in ((1.0, comfy.DENOISE_MAX), (0.0, comfy.DENOISE_MIN), (0.55, 0.55)):
            wf = comfy.build_img2img('a.png', 'p', 'n', 1, comfy.config(), asked)
            self.assertEqual(wf['8']['inputs']['denoise'], want)

    def test_lora_chain(self):
        wf = comfy.build_img2img('a.png', 'p', 'n', 1, comfy.config(), loras=[('x.safetensors', 0.8), ('y.safetensors', 1)])
        self.assertEqual(comfy.workflow_problems(wf), [])
        self.assertEqual(wf['20']['inputs']['model'], ['1', 0])
        self.assertEqual(wf['21']['inputs']['model'], ['20', 0])
        self.assertEqual(wf['8']['inputs']['model'], ['21', 0])
        self.assertEqual(wf['5']['inputs']['clip'], ['21', 1])
        self.assertEqual(wf['6']['inputs']['clip'], ['21', 1])

    def test_shipped_api_json_matches_the_builder(self):
        path = os.path.join(settings.BASE_DIR, 'docs', 'comfyui', 'booru_img2img_api.json')
        with open(path) as f:
            self.assertEqual(json.load(f), comfy.example_workflow())

    def test_a_broken_link_is_reported(self):
        wf = comfy.build_img2img('a.png', 'p', 'n', 1, comfy.config())
        wf['9']['inputs']['samples'] = ['99', 0]
        self.assertEqual(len(comfy.workflow_problems(wf)), 1)


class ComfyPromptTests(TestCase):
    def test_escape_and_clean(self):
        self.assertEqual(comfy.escape_tag('nakano_miku_(go-toubun_no_hanayome)'), 'nakano miku \\(go-toubun no hanayome\\)')
        self.assertEqual(comfy.escape_tag('already \\(escaped\\)'), 'already \\(escaped\\)')
        got = comfy.clean_tags(['Long_Hair', 'long hair', 'explicit', 'masterpiece', ' 1girl ,', '', 'blue eyes'])
        self.assertEqual(got, ['long hair', '1girl', 'blue eyes'])
        self.assertEqual(comfy.join_prompt('masterpiece, best quality', ['a_b', 'c (d)']), 'masterpiece, best quality, a b, c \\(d\\)')

    def test_parse_llm_answer(self):
        ok = comfy.parse_llm_prompt('```json\n{"add": ["Blue_Hair", "best quality"], "remove": "brown hair", "denoise": 0.95, "note": "x"}\n```', 0.5)
        self.assertEqual((ok['add'], ok['remove'], ok['denoise']), (['blue hair'], ['brown hair'], comfy.DENOISE_MAX))
        self.assertEqual(comfy.parse_llm_prompt('{"add": ["a"], "remove": [], "denoise": "oops"}', 0.5)['denoise'], 0.5)
        self.assertEqual(comfy.parse_llm_prompt('{"add": [], "remove": ["glasses"], "denoise": 0.7}', 0.5)['remove'], ['glasses'])
        for bad in ('', 'no json here', '{"add": [], "remove": []}', '[1,2]'):
            with self.assertRaises(ValueError):
                comfy.parse_llm_prompt(bad, 0.5)

    def test_diff_builds_the_prompt(self):
        """#37111: the new outfit must replace the old one, not sit unweighted among 40 old tags."""
        add, keep, neg = comfy.apply_diff(['1girl', 'school_uniform', 'pleated skirt', 'Blonde_Hair', 'smile'],
                                          ['astronaut suit', 'helmet'], ['school uniform', 'Pleated_Skirt', 'not_in_picture', 'helmet'])
        self.assertEqual(keep, ['1girl', 'blonde hair', 'smile'])
        self.assertEqual(neg, ['school uniform', 'pleated skirt', 'not in picture'])      # added wins over removed
        pos, negt = comfy.prompt_texts(add, keep, neg)
        self.assertTrue(pos.startswith('(astronaut suit:1.3), (helmet:1.3), 1girl'))
        self.assertEqual(comfy.prompt_texts(['x (y)'], [], [])[0], '(x \\(y\\):1.3)')
        self.assertNotIn('school uniform', pos)

    def _client(self, handler):
        return mock.patch.object(comfy, '_client', lambda timeout=10.0: httpx.Client(transport=httpx.MockTransport(handler)))

    def test_to_prompt_uses_the_model_answer(self):
        seen = {}

        def handler(req):
            seen['body'] = json.loads(req.content)
            return httpx.Response(200, json={'message': {'content': json.dumps(
                {'add': ['blue hair'], 'remove': ['brown hair'], 'denoise': 0.6, 'note': 'hair'})}})
        with self._client(handler):
            r = comfy.to_prompt('blue hair', ['1girl', 'brown_hair', 'general'])
        self.assertEqual((r['source'], r['positive'], r['negative'], r['denoise']), ('llm', '(blue hair:1.3), 1girl', 'brown hair', 0.6))
        self.assertIn('Current: 1girl, brown hair | Request: blue hair', seen['body']['messages'][1]['content'])   # rating tag dropped
        self.assertIs(seen['body']['think'], False)
        self.assertEqual(seen['body']['format']['required'], ['add', 'remove', 'denoise', 'note'])   # schema-enforced: any model, no tools
        self.assertEqual(seen['body']['model'], comfy.config()['ollama_model'])

    def test_to_prompt_falls_back_when_ollama_is_down_or_rambles(self):
        def down(req):
            raise httpx.ConnectError('refused')
        with self._client(down):
            r = comfy.to_prompt('add a hat', ['1girl'])
        self.assertEqual((r['source'], r['positive']), ('fallback', '(add a hat:1.3), 1girl'))
        with self._client(lambda req: httpx.Response(200, json={'message': {'content': 'sure! here you go'}})):
            self.assertEqual(comfy.to_prompt('x', ['a'])['source'], 'fallback')
        with self._client(lambda req: httpx.Response(200, json={'message': {'content': '{"add": [], "remove": [], "denoise": 0.5, "note": ""}'}})):
            self.assertEqual(comfy.to_prompt('x', ['a'])['source'], 'fallback')     # "no change" answers are not trusted

    def test_config_clamps_and_roundtrips(self):
        saved = comfy.save_config({'steps': 9999, 'cfg': 'abc', 'denoise': 1.0, 'megapixels': 0.01, 'quality': '',
                                   'comfy_url': ' http://h:1/ ', 'unknown': 1})
        self.assertEqual((saved['steps'], saved['cfg'], saved['denoise'], saved['megapixels']), (150, comfy.DEFAULTS['cfg'], comfy.DENOISE_MAX, 0.25))
        self.assertEqual((saved['comfy_url'], saved['quality']), ('http://h:1', ''))
        self.assertEqual(comfy.config()['quality'], '')               # an empty prefix is a valid choice
        self.assertNotIn('unknown', comfy.config())


class ComfyHttpFlowTests(TestCase):
    """upload -> queue -> poll -> download against a fake ComfyUI."""
    def setUp(self):
        self.calls = []
        self.history_polls = 0
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, 'src.png')
        with open(self.src, 'wb') as f:
            f.write(_png((64, 48)))
        for patch in (mock.patch.object(comfy, '_client', lambda timeout=10.0: httpx.Client(transport=httpx.MockTransport(self.handler))),
                      mock.patch.object(comfy.time, 'sleep', lambda s: None)):
            patch.start(); self.addCleanup(patch.stop)
        self.temp_node = {}
        self.history = lambda n: {'p1': {'status': {'status_str': 'success', 'completed': True, 'messages': []},
                                         'outputs': {'10': {'images': [{'filename': 'o.png', 'subfolder': '', 'type': 'temp'}]}}}} if n >= 3 else {}

    def handler(self, req):
        path = req.url.path
        self.calls.append((req.method, path))
        if path == '/system_stats':
            return httpx.Response(200, json={'system': {}, 'devices': []})
        if path == '/object_info/WaitForGPUTemperature':
            return httpx.Response(200, json=self.temp_node)
        if path == '/upload/image':
            self.upload = req.content
            return httpx.Response(200, json={'name': 'src_x.png', 'subfolder': 'booru', 'type': 'input'})
        if path == '/prompt':
            self.prompt = json.loads(req.content)['prompt']
            return httpx.Response(200, json={'prompt_id': 'p1'})
        if path == '/history/p1':
            self.history_polls += 1
            return httpx.Response(200, json=self.history(self.history_polls))
        if path == '/queue' and req.method == 'GET':
            return httpx.Response(200, json={'queue_pending': [[1, 'other', {}], [2, 'p1', {}]]})
        if path == '/view':
            return httpx.Response(200, content=_png())
        if path == '/api/ps':
            return httpx.Response(200, json={'models': [{'name': 'cpu-model', 'size_vram': 0}]})
        return httpx.Response(200, json={})

    def test_generate_happy_path(self):
        msgs = []
        png, wf = comfy.generate(self.src, 'pos', 'neg', 42, 0.55, check=lambda: None, progress=msgs.append)
        self.assertEqual(Image.open(_io.BytesIO(png)).size, (32, 24))
        self.assertEqual(self.prompt['2']['inputs']['image'], 'booru/src_x.png')          # LoadImage points at the uploaded file
        self.assertNotIn('7', self.prompt)                                                 # ComfyUI has no temperature node here
        self.assertEqual(self.prompt['8']['inputs']['denoise'], 0.55)
        self.assertTrue(any('queue' in m for m in msgs))
        self.assertNotIn(('POST', '/api/generate'), self.calls)                            # CPU-only Ollama is left loaded

    def test_temperature_node_used_when_installed(self):
        self.temp_node = {'WaitForGPUTemperature': {}}
        comfy.generate(self.src, 'p', 'n', 1, 0.5)
        self.assertEqual(self.prompt['7']['class_type'], 'WaitForGPUTemperature')

    def test_upload_is_deduplicated_by_content_name(self):
        a = comfy.upload_image(self.src)
        self.assertEqual(a, 'booru/src_x.png')
        self.assertIn(b'name="image"; filename="src_', self.upload)

    def test_node_errors_are_shown(self):
        orig = self.handler

        def bad(req):
            if req.url.path == '/prompt':
                return httpx.Response(400, json={'error': {'message': 'Prompt outputs failed validation'},
                                                 'node_errors': {'1': {'class_type': 'CheckpointLoaderSimple',
                                                                       'errors': [{'message': 'Value not in list', 'details': 'ckpt_name: x'}]}}})
            return orig(req)
        self.handler = bad
        with self.assertRaises(comfy.ComfyError) as cm:
            comfy.generate(self.src, 'p', 'n', 1, 0.5)
        self.assertIn('Value not in list', str(cm.exception))

    def test_execution_error_is_reported(self):
        self.history = lambda n: {'p1': {'status': {'status_str': 'error', 'completed': False, 'messages': [
            ['execution_error', {'node_type': 'KSampler', 'exception_message': 'CUDA out of memory'}]]}, 'outputs': {}}}
        with self.assertRaises(comfy.ComfyError) as cm:
            comfy.generate(self.src, 'p', 'n', 1, 0.5)
        self.assertIn('CUDA out of memory', str(cm.exception))

    def test_cancel_interrupts_comfyui_and_reraises(self):
        self.history = lambda n: {}
        n = {'i': 0}

        def check():
            n['i'] += 1
            if n['i'] > 2:
                raise TaskCancelled()
        with self.assertRaises(TaskCancelled):
            comfy.generate(self.src, 'p', 'n', 1, 0.5, check=check)
        self.assertIn(('POST', '/interrupt'), self.calls)
        self.assertIn(('POST', '/queue'), self.calls)

    def test_ollama_model_in_vram_is_unloaded_before_comfyui(self):
        orig = self.handler
        seen = []

        def h(req):
            if req.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': 'big', 'size_vram': 5 << 30}]})
            if req.url.path == '/api/generate':
                seen.append(json.loads(req.content))
            return orig(req)
        self.handler = h
        comfy.unload_ollama()
        self.assertEqual(seen, [{'model': 'big', 'keep_alive': 0}])

    def test_status_summarises_both_servers(self):
        def h(req):
            p = req.url.path
            if p == '/system_stats':
                return httpx.Response(200, json={'system': {'comfyui_version': '9.9'}, 'devices': [{'name': 'gpu', 'vram_free': 1, 'vram_total': 2}]})
            if p == '/object_info/KSampler':
                return httpx.Response(200, json={'KSampler': {'input': {'required': {'sampler_name': [['euler']], 'scheduler': [['normal']]}}}})
            if p == '/object_info/CheckpointLoaderSimple':
                return httpx.Response(200, json={'CheckpointLoaderSimple': {'input': {'required': {'ckpt_name': [['a.safetensors']]}}}})
            if p == '/object_info/WaitForGPUTemperature':
                return httpx.Response(200, json={})
            if p == '/api/version':
                return httpx.Response(200, json={'version': '0.1'})
            if p == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': 'z'}, {'name': 'a'}]})
            return httpx.Response(404)
        self.handler = h
        st = comfy.status()
        self.assertEqual((st['comfy']['ok'], st['comfy']['version'], st['comfy']['checkpoints'], st['comfy']['temp_node']), (True, '9.9', ['a.safetensors'], False))
        self.assertEqual(st['ollama']['models'], ['a', 'z'])

    def test_status_when_nothing_runs(self):
        def down(req):
            raise httpx.ConnectError('refused')
        self.handler = down
        st = comfy.status()
        self.assertFalse(st['comfy']['ok']); self.assertFalse(st['ollama']['ok'])
        self.assertIn('not reachable', st['comfy']['error'])


class ComfyEditEndpointTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ov = override_settings(MEDIA_ROOT=self.tmp)
        ov.enable(); self.addCleanup(ov.disable)
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        comfy.save_config({'enabled': True})
        self.post = Post.objects.create()
        self.photo = self._photo('a.png', self.post)

    def _photo(self, name, post, data=None):
        with open(os.path.join(self.tmp, name), 'wb') as f:
            f.write(data or _png((64, 48)))
        return Photo.objects.create(post=post, order=0, rel_path=name, width=64, height=48, is_video=name.endswith('.mp4'))

    def _post(self, url, body):
        return self.client.post(url, json.dumps(body), content_type='application/json')

    def _run_edit(self, body, **kw):
        pr = {'positive': '(blue hair:1.3), 1girl', 'negative': 'brown hair', 'add': ['blue hair'], 'remove': ['brown hair'],
              'denoise': 0.6, 'note': 'hair', 'source': 'llm'}
        with mock.patch.object(comfy, 'to_prompt', return_value=pr) as tp, \
             mock.patch.object(comfy, 'generate', return_value=(_png((48, 36)), {})) as gen, \
             mock.patch.object(comfy, 'free_comfy', return_value=True) as fr, \
             mock.patch.object(views, 'run_ai_tagger', return_value={'general': ['1girl'], 'character': [], 'model': 'wd14'}), \
             mock.patch.object(views, 'apply_ai_tags') as tag, \
             mock.patch('gallery.ai_runtime.force_unload', return_value=0), \
             mock.patch.object(views, '_start_task', side_effect=lambda kind, fn, **k: _run_now(kind, fn)):
            r = self._post('/api/comfy/edit/', body)
        return r, tp, gen, fr, tag

    def test_creates_a_linked_post(self):
        r, tp, gen, fr, tag = self._run_edit({'photo_id': self.photo.pk, 'request': 'make her hair blue', 'seed': '77'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['seed'], 77)
        new = Post.objects.exclude(pk=self.post.pk).get()
        self.assertEqual((new.source_post_id, new.source_photo_id), (self.post.pk, self.photo.pk))
        self.assertEqual(new.gen_info['request'], 'make her hair blue')
        self.assertEqual((new.gen_info['seed'], new.gen_info['denoise']), (77, 0.6))
        self.assertIn('blue hair', new.gen_info['positive'])
        self.assertTrue(new.title.startswith(f'AI edit of #{self.post.pk}: make her hair blue'))
        self.assertTrue(new.tags.filter(name='ai_edit', category='meta').exists())
        self.assertEqual(new.images.count(), 1)
        self.assertTrue(os.path.isfile(new.images.first().file_path))
        self.assertIn(os.sep + 'ai_edits' + os.sep, new.images.first().file_path)
        tag.assert_called_once()                                  # the new post is AI-tagged
        fr.assert_called_once()                                   # ComfyUI memory released
        args = gen.call_args[0]
        self.assertTrue(args[1].startswith(comfy.config()['quality']))     # quality prefix in front
        self.assertIn('brown hair', args[2]); self.assertIn(comfy.config()['negative'], args[2])
        self.assertTrue(Task.objects.get(kind='comfy_edit').message.endswith(f'→ post #{new.pk}'))
        self.assertEqual(list(self.post.derived_posts.values_list('pk', flat=True)), [new.pk])

    def test_previewed_prompt_skips_the_llm(self):
        r, tp, gen, *_ = self._run_edit({'photo_id': self.photo.pk, 'request': 'x', 'positive': 'my own tags', 'negative': 'bad', 'denoise': '0.7'})
        tp.assert_not_called()
        self.assertEqual(gen.call_args[0][4], 0.7)
        self.assertIn('my own tags', gen.call_args[0][1])

    def test_ollama_model_choice_is_remembered(self):
        self._run_edit({'photo_id': self.photo.pk, 'request': 'x', 'ollama_model': 'gemma2:9b'})
        self.assertEqual(comfy.config()['ollama_model'], 'gemma2:9b')

    def test_cancel_creates_nothing(self):
        with mock.patch.object(comfy, 'to_prompt', return_value={'positive': 'a', 'negative': '', 'add': [], 'remove': [], 'denoise': .5, 'note': '', 'source': 'llm'}), \
             mock.patch.object(comfy, 'generate', side_effect=TaskCancelled()), \
             mock.patch.object(views, 'run_ai_tagger', return_value={'general': [], 'character': []}), \
             mock.patch('gallery.ai_runtime.force_unload', return_value=0), \
             mock.patch.object(views, '_start_task', side_effect=lambda kind, fn, **k: _run_now(kind, fn)):
            with self.assertRaises(TaskCancelled):
                self._post('/api/comfy/edit/', {'photo_id': self.photo.pk, 'request': 'x'})
        self.assertEqual(Post.objects.count(), 1)

    def test_refuses_video_pdf_gif_missing_and_empty_request(self):
        for name in ('v.mp4', 'd.pdf', 'g.gif'):
            ph = self._photo(name, Post.objects.create(), b'x')
            r = self._post('/api/comfy/edit/', {'photo_id': ph.pk, 'request': 'x'})
            self.assertEqual(r.status_code, 400, name)
            self.assertEqual(self._post('/api/comfy/prompt/', {'photo_id': ph.pk, 'request': 'x'}).status_code, 400, name)
        gone = Photo.objects.create(post=self.post, order=1, rel_path='missing.png')
        self.assertEqual(self._post('/api/comfy/edit/', {'photo_id': gone.pk, 'request': 'x'}).status_code, 404)
        self.assertEqual(self._post('/api/comfy/edit/', {'photo_id': 999999, 'request': 'x'}).status_code, 404)
        self.assertEqual(self._post('/api/comfy/edit/', {'photo_id': self.photo.pk, 'request': '  '}).status_code, 400)
        self.assertEqual(self._post('/api/comfy/prompt/', {'photo_id': self.photo.pk, 'request': ''}).status_code, 400)

    def test_one_edit_at_a_time(self):
        t = Task.objects.create(kind='comfy_edit')
        r = self._post('/api/comfy/edit/', {'photo_id': self.photo.pk, 'request': 'x'})
        self.assertEqual((r.status_code, r.json()['busy'], r.json()['task_id']), (409, True, t.pk))

    def test_switched_off(self):
        comfy.save_config({'enabled': False})
        self.assertEqual(self._post('/api/comfy/edit/', {'photo_id': self.photo.pk, 'request': 'x'}).status_code, 400)

    def test_prompt_preview_returns_editable_text(self):
        pr = {'positive': '(x:1.3), 1girl, nakano miku \\(go-toubun\\)', 'negative': '', 'add': ['x'], 'remove': [], 'denoise': 0.55, 'note': 'n', 'source': 'llm'}
        with mock.patch.object(comfy, 'to_prompt', return_value=pr), \
             mock.patch.object(views, 'run_ai_tagger', return_value={'general': ['1girl'], 'character': [], 'model': 'wd14'}):
            d = self._post('/api/comfy/prompt/', {'photo_id': self.photo.pk, 'request': 'x'}).json()
        self.assertEqual((d['positive'], d['add']), ('(x:1.3), 1girl, nakano miku \\(go-toubun\\)', ['x']))
        self.assertEqual(d['denoise'], 0.55)

    def test_current_tags_include_post_characters_and_tagger_output(self):
        add_tags_to_post(self.post, ['miku'], category='character')
        add_tags_to_post(self.post, ['smile'], category='general')
        with mock.patch.object(views, 'run_ai_tagger', return_value={'general': ['1girl'], 'character': ['rin'], 'model': 'wd14'}):
            names = views._edit_current_tags(self.photo)
        self.assertEqual(names[:2], ['rin', '1girl'])
        self.assertIn('miku', names); self.assertIn('smile', names)        # single-image post: general tags too

    def test_settings_endpoint_and_status_endpoint(self):
        d = self._post('/api/comfy/settings/', {'steps': 12, 'enabled': False}).json()
        self.assertEqual((d['config']['steps'], d['config']['enabled']), (12, False))
        with mock.patch.object(comfy, 'status', return_value={'comfy': {'ok': False}, 'ollama': {'ok': False}, 'config': comfy.config()}):
            self.assertIn('config', self.client.get('/api/comfy/status/').json())

    def test_free_vram_also_frees_comfyui(self):
        with mock.patch.object(comfy, 'free_comfy', return_value=True) as fr, \
             mock.patch('gallery.ai_runtime.request_unload_all'), mock.patch('gallery.ai_runtime.force_unload', return_value=0), \
             mock.patch('gallery.ai_runtime.gpu_info', return_value={}):
            d = self.client.post('/api/ai/free-vram/').json()
        fr.assert_called_once()
        self.assertTrue(d['comfy_freed'])


class ComfyEditPageTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        comfy.save_config({'enabled': True})

    def _post(self, *names):
        post = Post.objects.create()
        for i, n in enumerate(names):
            Photo.objects.create(post=post, order=i, rel_path=n, is_video=n.endswith('.mp4'))
        return post

    def test_button_only_on_still_images(self):
        post = self._post('a.png', 'b.mp4', 'c.gif', 'd.pdf', 'e.jpg')
        html = self.client.get(f'/post/{post.pk}/').content.decode()
        self.assertEqual(html.count('class="ai-edit-btn"'), 2)             # a.png and e.jpg only
        self.assertIn('aria-label="AI edit">&#9998;</button>', html)        # icon only, no text
        self.assertIn('id="aiedit-backdrop"', html)
        for ph in Photo.objects.filter(post=post, rel_path__in=['a.png', 'e.jpg']):
            self.assertIn(f'data-photo-id="{ph.pk}"', html)

    def test_nothing_rendered_when_switched_off(self):
        comfy.save_config({'enabled': False})
        html = self.client.get(f'/post/{self._post("a.png").pk}/').content.decode()
        self.assertNotIn('class="ai-edit-btn"', html)
        self.assertNotIn('id="aiedit-backdrop"', html)

    def test_source_and_derived_links_and_generation_block(self):
        src = self._post('a.png', 'b.png')
        item = src.images.order_by('order').last()
        new = self._post('n.png')
        new.source_post, new.source_photo = src, item
        new.gen_info = {'request': 'add <glasses>', 'positive': 'p', 'negative': 'n', 'denoise': 0.6, 'seed': 5, 'steps': 20,
                        'cfg': 5.5, 'sampler': 'euler', 'checkpoint': 'ck', 'seconds': 99, 'note': ''}
        new.save()
        html = self.client.get(f'/post/{new.pk}/').content.decode()
        self.assertIn(f'post #{src.pk} (item 2)</a>', html)
        self.assertIn(f'href="/post/{src.pk}/" style="font-size:.75rem;" title="open the post this AI edit was made from">&#10548; original #{src.pk}</a>', html)
        self.assertIn(f"location.href='/post/{src.pk}/'\" title=\"open the original post\"", html)
        self.assertIn('add &lt;glasses&gt;', html)                          # request is escaped
        self.assertIn(f'aiedit={item.pk}&req=add%20%3Cglasses%3E', html)    # "edit again" prefilled
        self.assertIn(f'#{new.pk}</a>', self.client.get(f'/post/{src.pk}/').content.decode())

    def test_deleting_the_source_keeps_the_edit(self):
        src = self._post('a.png'); new = self._post('n.png')
        new.source_post = src; new.save()
        src.delete()
        new.refresh_from_db()
        self.assertIsNone(new.source_post_id)


class DoubleTapJsTests(TestCase):
    """makeDoubleTap (detail.html): scrolling right after a zoom double-tap must not count as a tap."""
    def _run(self, steps):
        src = open(os.path.join(settings.BASE_DIR, 'templates/gallery/detail.html'), encoding='utf-8').read()
        js = src[src.index('function makeDoubleTap()'):src.index('function attachZoom(')]
        prog = js + """
let now = 1000; Date.now = () => now;
const d = makeDoubleTap(), out = [];
const T = (x, y) => ({clientX: x, clientY: y});
const down = (x, y, n = 1) => d.start({touches: Array.from({length: n}, () => T(x, y))});
const up = (x, y, n = 1) => out.push(d.end({changedTouches: Array.from({length: n}, () => T(x, y))}));
""" + steps + "\nconsole.log(JSON.stringify(out));"
        r = subprocess.run(['node', '-e', prog], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:300])
        return json.loads(r.stdout)

    def test_two_quick_taps_are_a_double_tap(self):
        self.assertEqual(self._run("down(100,100); now+=40; up(100,100); now+=120; down(102,101); now+=40; up(102,101);"),
                         [None, {'x': 102, 'y': 101}])

    def test_a_quick_swipe_after_the_zoom_tap_is_not_a_second_double_tap(self):
        steps = ("down(100,100); now+=40; up(100,100); now+=100; down(100,100); now+=40; up(100,100);"      # zoom in
                 "now+=60; down(100,300); now+=90; up(100,180);"                                              # scroll right away
                 "now+=60; down(100,300); now+=90; up(100,180);")
        got = self._run(steps)
        self.assertEqual(got[1], {'x': 100, 'y': 100})      # the zoom itself
        self.assertEqual(got[2:], [None, None])             # the swipes never toggle it again

    def test_tap_after_a_zoom_is_a_fresh_pair(self):
        steps = ("down(5,5); now+=30; up(5,5); now+=100; down(5,5); now+=30; up(5,5);"
                 "now+=100; down(5,5); now+=30; up(5,5);")
        self.assertEqual([bool(x) for x in self._run(steps)], [False, True, False])   # third tap does NOT undo the zoom

    def test_slow_or_distant_second_tap_and_multitouch_do_not_count(self):
        self.assertEqual(self._run("down(0,0); now+=30; up(0,0); now+=400; down(0,0); now+=30; up(0,0);"), [None, None])
        self.assertEqual(self._run("down(0,0); now+=30; up(0,0); now+=100; down(90,0); now+=30; up(90,0);"), [None, None])
        self.assertEqual(self._run("down(0,0); now+=30; up(0,0); now+=100; down(0,0,2); now+=30; up(0,0,2);"), [None, None])

    def test_long_press_is_not_a_tap(self):
        self.assertEqual(self._run("down(0,0); now+=400; up(0,0); now+=50; down(0,0); now+=30; up(0,0);"), [None, None])


# ── AI edit: LoRAs, timings, notifications ──────────────────────
def _lm_item(folder, file, title, base='Illustrious', tags=(), groups=(), preview='x.jpeg'):
    return {'folder': folder, 'file_name': file, 'model_name': title, 'base_model': base, 'tags': list(tags),
            'file_path': f'/l/{folder}/{file}.safetensors',
            'preview_url': f'/api/lm/previews?path=%2Fl%2F{file}.{preview.split(".")[-1]}' if preview else '',
            'civitai': {'trainedWords': list(groups)}}


LM_PAGES = [
    {'items': [
        _lm_item('Illustrious/character', 'Hayasaka_Ai', 'Hayasaka Ai [Kaguya-sama]', tags=['character', 'hayasaka ai'],
                 groups=['1girl, hayasaka ai, blonde hair, side ponytail,', 'school uniform, white shirt, <lora:x:1>', 'maid, maid headdress']),
        _lm_item('Illustrious/character', 'AiChan', 'Ai-chan', tags=['character'], groups=['aichan']),
        _lm_item('Illustrious/clothing', 'bikini_v2', 'Micro Bikini', tags=['clothing', 'bikini'], groups=['micro bikini, bikini', 'string bikini'], preview='p.mp4'),
    ], 'total_pages': 2},
    {'items': [
        _lm_item('Pony/style', 'ponystyle', 'Pony style', base='Pony', groups=['score_9']),
        _lm_item('Illustrious/concept', 'nakano', 'Nakano Miku', tags=['character'], groups=['nakano miku, headphones']),
    ], 'total_pages': 2},
]


class ComfyLoraTests(TestCase):
    def setUp(self):
        comfy._catalog_cache.update(t=0.0, key=None, items=None)
        self.addCleanup(comfy._catalog_cache.update, t=0.0, key=None, items=None)
        self.pages = 0

        def handler(req):
            if req.url.path == '/api/lm/loras/list':
                self.pages += 1
                return httpx.Response(200, json=LM_PAGES[int(req.url.params['page']) - 1])
            if req.url.path == '/models/loras':
                return httpx.Response(200, json=['Illustrious/character/x.safetensors', 'y.safetensors'])
            return httpx.Response(404)
        self.handler = handler
        p = mock.patch.object(comfy, '_client', lambda timeout=10.0: httpx.Client(transport=httpx.MockTransport(lambda r: self.handler(r))))
        p.start(); self.addCleanup(p.stop)

    def test_catalog_pages_groups_base_filter_and_cache(self):
        cat = comfy.lora_catalog()
        self.assertEqual(self.pages, 2)
        names = [i['name'] for i in cat]
        self.assertNotIn('Pony/style/ponystyle.safetensors', names)                 # base model does not fit
        hay = next(i for i in cat if i['file'] == 'Hayasaka_Ai')
        self.assertEqual(hay['name'], 'Illustrious/character/Hayasaka_Ai.safetensors')
        self.assertEqual(hay['groups'], ['1girl, hayasaka ai, blonde hair, side ponytail', 'school uniform, white shirt', 'maid, maid headdress'])
        self.assertEqual((hay['kind'], hay['preview']), ('character', '/l/Hayasaka_Ai.jpeg'))
        self.assertEqual(next(i for i in cat if i['file'] == 'bikini_v2')['preview'], '')   # video previews skipped
        comfy.lora_catalog()
        self.assertEqual(self.pages, 2)                                                    # cached
        comfy.lora_catalog(force=True)
        self.assertEqual(self.pages, 4)

    def test_catalog_falls_back_to_plain_names(self):
        def no_lm(req):
            if req.url.path == '/api/lm/loras/list':
                return httpx.Response(404)
            return httpx.Response(200, json=['Illustrious/character/x.safetensors', 'y.safetensors'])
        self.handler = no_lm
        cat = comfy.lora_catalog(force=True)
        self.assertEqual([(i['name'], i['kind'], i['groups']) for i in cat],
                         [('Illustrious/character/x.safetensors', 'character', []), ('y.safetensors', 'other', [])])

    def test_suggestions(self):
        cat = comfy.lora_catalog()
        got = comfy.suggest_loras(['hayasaka_ai'], ['make her wear a bikini', 'bikini'], cat)
        self.assertEqual([g['file'] for g in got], ['Hayasaka_Ai', 'bikini_v2'])           # not the "ai"-only Ai-chan
        self.assertEqual(got[0]['default_groups'], [0])                                     # character: identity group only
        self.assertEqual(got[1]['default_groups'], [0, 1])                                  # 2 groups: both
        self.assertIn('character: hayasaka ai', got[0]['reason'])
        got = comfy.suggest_loras(['nakano_miku_(go-toubun_no_hanayome)'], [], cat)
        self.assertEqual([g['file'] for g in got], ['nakano'])
        self.assertEqual(comfy.suggest_loras([], ['school uniform'], cat), [])              # generic word deep in a group: no match

    def test_search(self):
        cat = comfy.lora_catalog()
        self.assertEqual([i['file'] for i in comfy.search_loras('bikini', cat)], ['bikini_v2'])
        self.assertEqual([i['file'] for i in comfy.search_loras('hayasaka', cat)], ['Hayasaka_Ai'])

    def test_resolve_and_triggers(self):
        cat = comfy.lora_catalog()
        got = comfy.resolve_loras([{'name': 'Illustrious/character/Hayasaka_Ai.safetensors', 'strength': 5,
                                    'words': ['1girl, hayasaka ai', 'pom pom (clothes), <lora:z:1>']}], cat)
        self.assertEqual(got[0]['strength'], 2.0)
        self.assertEqual(comfy.trigger_text(got[0]['words']), '1girl, hayasaka ai, pom pom \\(clothes\\)')
        with self.assertRaises(comfy.ComfyError):
            comfy.resolve_loras([{'name': '../../etc/passwd'}], cat)
        with self.assertRaises(comfy.ComfyError):
            comfy.resolve_loras([{'name': 'x'}] * 7, cat)


class ComfyLoraEndpointTests(TestCase):
    CAT = [{'name': 'Illustrious/character/Hayasaka_Ai.safetensors', 'title': 'Hayasaka Ai', 'file': 'Hayasaka_Ai',
            'folder': 'Illustrious/character', 'base': 'Illustrious', 'tags': ['hayasaka ai', 'character'],
            'groups': ['hayasaka ai, side ponytail', 'maid'], 'kind': 'character', 'preview': '/l/h.jpeg'}]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ov = override_settings(MEDIA_ROOT=self.tmp)
        ov.enable(); self.addCleanup(ov.disable)
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        comfy.save_config({'enabled': True})
        self.post = Post.objects.create()
        with open(os.path.join(self.tmp, 'a.png'), 'wb') as f:
            f.write(_png((64, 48)))
        self.photo = Photo.objects.create(post=self.post, order=0, rel_path='a.png', width=64, height=48)
        add_tags_to_post(self.post, ['hayasaka_ai'], category='character')
        p = mock.patch.object(comfy, 'lora_catalog', return_value=self.CAT)
        p.start(); self.addCleanup(p.stop)

    def test_suggest_and_search_and_exact(self):
        with mock.patch.object(views, 'run_ai_tagger', return_value={'general': [], 'character': [], 'model': 'wd14'}):
            d = self.client.get(f'/api/comfy/loras/suggest/?photo_id={self.photo.pk}&request=x').json()
        self.assertEqual([l['name'] for l in d['loras']], [self.CAT[0]['name']])
        self.assertEqual(d['loras'][0]['preview'], '/api/comfy/lora-preview/?path=/l/h.jpeg')
        self.assertEqual(d['loras'][0]['default_groups'], [0])
        self.assertEqual(len(self.client.get('/api/comfy/loras/?q=haya').json()['loras']), 1)
        self.assertEqual(len(self.client.get('/api/comfy/loras/?name=' + self.CAT[0]['name']).json()['loras']), 1)
        self.assertEqual(self.client.get('/api/comfy/loras/?name=nope').json()['loras'], [])

    def test_preview_proxy_only_images(self):
        with mock.patch.object(comfy, 'fetch_lora_preview', return_value=(b'img', 'image/jpeg')):
            r = self.client.get('/api/comfy/lora-preview/?path=/l/h.jpeg')
        self.assertEqual((r.status_code, r.content, r['Content-Type']), (200, b'img', 'image/jpeg'))
        self.assertIsNone(comfy.fetch_lora_preview('/etc/passwd'))
        self.assertEqual(self.client.get('/api/comfy/lora-preview/?path=/etc/passwd').status_code, 404)

    def _edit(self, body):
        with mock.patch.object(comfy, 'generate', return_value=(_png((48, 36)), {})) as gen, \
             mock.patch.object(comfy, 'free_comfy', return_value=True), \
             mock.patch.object(views, 'run_ai_tagger', return_value={'general': ['1girl'], 'character': [], 'model': 'wd14'}), \
             mock.patch.object(views, 'apply_ai_tags'), mock.patch('gallery.ai_runtime.force_unload', return_value=0), \
             mock.patch.object(views, '_start_task', side_effect=lambda kind, fn, **k: _run_now(kind, fn)):
            return self.client.post('/api/comfy/edit/', json.dumps(body), content_type='application/json'), gen

    def test_edit_with_lora_puts_triggers_first_and_records_everything(self):
        r, gen = self._edit({'photo_id': self.photo.pk, 'request': 'x', 'positive': '(maid:1.3), 1girl', 'negative': '',
                             'loras': [{'name': self.CAT[0]['name'], 'strength': 0.9, 'words': ['hayasaka ai, side ponytail']}]})
        self.assertEqual(r.status_code, 200)
        pos = gen.call_args[0][1]
        q = comfy.config()['quality']
        self.assertEqual(pos, q + ', hayasaka ai, side ponytail, (maid:1.3), 1girl')            # quality, triggers, prompt
        self.assertEqual(gen.call_args.kwargs['loras'], [(self.CAT[0]['name'], 0.9)])
        new = Post.objects.exclude(pk=self.post.pk).get()
        self.assertEqual(new.gen_info['loras'][0]['title'], 'Hayasaka Ai')
        self.assertEqual(set(new.gen_info['timings']), {'prompt', 'generate', 'save', 'tagging', 'total'})
        d = self.client.get(f'/api/post/{new.pk}/gen-info/').json()
        self.assertEqual((d['gen_info']['loras'][0]['words'], d['source_photo']), (['hayasaka ai, side ponytail'], self.photo.pk))
        html = self.client.get(f'/post/{new.pk}/').content.decode()
        self.assertIn('<b>LoRA</b> Hayasaka Ai × 0.9 — hayasaka ai, side ponytail', html)
        self.assertIn(f'&from={new.pk}', html)

    def test_unknown_lora_is_refused(self):
        r, gen = self._edit({'photo_id': self.photo.pk, 'request': 'x', 'loras': [{'name': 'evil.safetensors'}]})
        self.assertEqual(r.status_code, 400)
        gen.assert_not_called()


class GenerationTimeTests(TestCase):
    def setUp(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})

    def test_summary_shows_how_long_it_took(self):
        src = Post.objects.create()
        new = Post.objects.create(source_post=src, gen_info={'request': 'r', 'positive': 'p', 'denoise': 0.6, 'seed': 1,
                                                             'timings': {'prompt': 55, 'generate': 150, 'save': 1, 'tagging': 12, 'total': 218}})
        html = self.client.get(f'/post/{new.pk}/').content.decode()
        self.assertIn('generation · took 3m 38s</summary>', html)
        self.assertIn('(prompt 55s · drawing 2m 30s · saving 1s · tagging 12s)', html)
        old = Post.objects.create(source_post=src, gen_info={'request': 'r', 'positive': 'p', 'seconds': 292})   # #37111-style, no timings
        self.assertIn('generation · took 4m 52s</summary>', self.client.get(f'/post/{old.pk}/').content.decode())


class TaskRetentionTests(TestCase):
    def test_finished_ai_edit_stays_for_a_day_other_tasks_five_minutes(self):
        self.client.post('/login/', {'password': settings.GALLERY_PASSWORD})
        ago = timezone.now() - timezone.timedelta(hours=2)
        Task.objects.create(kind='comfy_edit', status='done', message='done → post #5', finished_at=ago)
        Task.objects.create(kind='scan', status='done', finished_at=ago)
        Task.objects.create(kind='comfy_edit', status='done', finished_at=timezone.now() - timezone.timedelta(hours=25))
        kinds = [t['kind'] for t in self.client.get('/api/tasks/').json()['tasks']]
        self.assertEqual(kinds, ['comfy_edit'])


class WatchedTaskJsTests(TestCase):
    """base.html: a watched task is polled even in fast mode and toasts when it finishes."""
    def test_fast_mode_still_polls_a_watched_task(self):
        src = open(os.path.join(settings.BASE_DIR, 'templates/gallery/base.html'), encoding='utf-8').read()
        a = src.index('// A task the user is waiting for'); b = src.index('async function cancelTask')
        prog = """
const store = {fastMode: '1'}; const localStorage = {getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; }, removeItem: k => { delete store[k]; }};
const TASK_LABELS = {comfy_edit: 'AI edit (ComfyUI)'}; let toasts = [], fetched = 0, _taskTimer, _hadRunning = false;
function toast(m, t) { toasts.push(m); } function renderTasks() {} function setTimeout() {} function clearTimeout() {}
let reply = {tasks: [{id: 7, kind: 'comfy_edit', status: 'running', message: 'generating…'}]};
async function fetch() { fetched++; return {json: async () => reply}; }
const document = {getElementById: () => null};
""" + src[a:b] + """
(async () => {
  await pollTasks(); const before = fetched;               // fast mode, nothing watched: no request
  watchTask(7); await pollTasks();                          // watched: polled anyway
  reply = {tasks: [{id: 7, kind: 'comfy_edit', status: 'done', message: 'done in 200s → post #42'}]};
  await pollTasks(); await pollTasks();
  console.log(JSON.stringify({before, fetched, toasts, watch: watchedTask()}));
})();"""
        r = subprocess.run(['node', '-e', prog], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:400])
        out = json.loads(r.stdout)
        self.assertEqual(out['before'], 0)
        self.assertEqual(out['fetched'], 2)                   # third call: watch cleared → fast mode skips again
        self.assertEqual(out['toasts'], ['AI edit (ComfyUI) done → post #42 (open it from the task card)'])
        self.assertIsNone(out['watch'])


class TriggerNegativeConflictTests(TestCase):
    def test_selected_trigger_words_leave_the_negative(self):
        self.assertEqual(comfy.drop_terms('white shirt, school uniform, pom pom \\(clothes\\)', 'Hayasaka, white_shirt, pom pom \\(clothes\\)'),
                         'school uniform')
        self.assertEqual(comfy.drop_terms('a, b', ''), 'a, b')
