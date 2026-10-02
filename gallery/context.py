def quick_links(request):
    """Header shortcut buttons (Settings -> quick links), stored in prefs.json."""
    from .views import _quick_links
    return {'quick_links': _quick_links()}
