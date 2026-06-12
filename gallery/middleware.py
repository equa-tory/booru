from django.conf import settings
from django.shortcuts import redirect
from django.http import HttpResponse


class LoginRequiredMiddleware:
    """Simple password gate — stores auth in session."""

    EXEMPT_PATHS = {'/login/', '/logout/'}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        pwd = getattr(settings, 'GALLERY_PASSWORD', None)
        if not pwd:
            return self.get_response(request)

        # service worker must always be accessible
        if request.path.endswith('sw.js'):
            return self.get_response(request)

        if request.path not in self.EXEMPT_PATHS and not request.session.get('authed'):
            return redirect(f'/login/?next={request.path}')

        return self.get_response(request)


class CacheHeadersMiddleware:
    """Let the browser cache static assets and media aggressively so the app
    feels faster on repeat visits. Dynamic HTML/JSON is never cached here.

    - /static/  : versioned-ish assets → cache a year.
    - /media/   : images/thumbs → cache a day. Thumbnails can be regenerated to
                  the SAME path, so we keep it modest and the 'regenerate
                  thumbnail' button cache-busts with a ?v= query param.
    - sw.js     : never cached, or you could never ship a service-worker update.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        resp = self.get_response(request)
        path = request.path
        if path.endswith('sw.js'):
            resp['Cache-Control'] = 'no-cache'
        elif path.startswith('/static/'):
            resp['Cache-Control'] = 'public, max-age=31536000, immutable'
        elif path.startswith('/media/'):
            resp.setdefault('Cache-Control', 'public, max-age=86400')
        return resp
