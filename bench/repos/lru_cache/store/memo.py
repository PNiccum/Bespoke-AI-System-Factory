from store.lru import LRUCache


def memoize(capacity):
    """Cache a one-argument function's results in an LRU cache of the given size."""
    def wrap(fn):
        cache = LRUCache(capacity)
        missing = object()

        def inner(x):
            hit = cache.get(x, missing)
            if hit is not missing:
                return hit
            value = fn(x)
            cache.put(x, value)
            return value
        inner.cache = cache
        return inner
    return wrap
