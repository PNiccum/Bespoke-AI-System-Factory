import pytest

from store.lru import LRUCache
from store.memo import memoize


def test_put_existing_key_counts_as_use_and_does_not_evict():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)
    assert len(c) == 2 and c.keys() == ["b", "a"]
    c.put("c", 3)
    assert c.keys() == ["a", "c"] and c.get("a") == 10


def test_contains_does_not_count_as_use():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert "a" in c
    c.put("c", 3)
    assert c.keys() == ["b", "c"]


def test_capacity_one_and_bad_capacity():
    c = LRUCache(1)
    for i in range(5):
        c.put(i, i)
    assert c.keys() == [4]
    with pytest.raises(ValueError):
        LRUCache(0)


def test_memoize_keeps_hot_keys():
    calls = []

    @memoize(2)
    def sq(x):
        calls.append(x)
        return x * x

    for x in [1, 2, 1, 3, 1, 2]:
        sq(x)
    assert calls == [1, 2, 3, 2]
