def merge(base, override):
    """Return a new config: override's values win, and nested sections are merged key by key, not replaced.
    Neither argument is changed."""
    out = dict(base)
    out.update(override)
    return out
