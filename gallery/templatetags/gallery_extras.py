from django import template

register = template.Library()

# Order the user wants: meta, char, art, gen (then ai last)
CAT_ORDER = {'meta': 0, 'character': 1, 'artist': 2, 'general': 3, 'ai': 4}


@register.filter
def sort_by_category(tags):
    """Sort a tag queryset/list by category (meta, char, art, gen) then name."""
    try:
        items = list(tags)
    except TypeError:
        return tags
    return sorted(items, key=lambda t: (CAT_ORDER.get(getattr(t, 'category', 'general'), 9),
                                        (getattr(t, 'name', '') or '').lower()))
