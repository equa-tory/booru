import os
import sys

from django.apps import AppConfig


class GalleryConfig(AppConfig):
    name = 'gallery'

    def ready(self):
        # Scheduled DB backups: only inside a real server process (gunicorn, or
        # the runserver child) — never for migrate/shell/test/etc.
        argv0 = os.path.basename(sys.argv[0]) if sys.argv else ''
        serving = 'gunicorn' in argv0 or ('runserver' in sys.argv and os.environ.get('RUN_MAIN') == 'true')
        if serving:
            from . import backup
            backup.start_scheduler()
