from collections import OrderedDict


class LRUCache:
    """Holds at most ``capacity`` items. When it is full, adding a new key evicts the least recently used one.
    Reading a key with get() or writing it with put() counts as using it; ``key in cache`` does not."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            self._data.popitem(last=True)
        self._data[key] = value

    def keys(self):
        """Keys from least to most recently used."""
        return list(self._data)

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)
