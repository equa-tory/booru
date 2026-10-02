from django.urls import path, re_path, include
from django.conf import settings
from django.conf.urls.static import static

from gallery import views

urlpatterns = [
    path('', include('gallery.urls')),
    # media with HTTP Range support (Chromium needs 206 to seek in / stream videos)
    re_path(r'^media/(?P<path>.*)$', views.media_serve, name='media'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
