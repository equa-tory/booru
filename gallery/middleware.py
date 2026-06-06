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
