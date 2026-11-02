from store.lru import LRUCache


def test_evicts_least_recently_used():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert "a" not in c and c.get("b") == 2 and c.get("c") == 3


def test_get_counts_as_use():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert "b" not in c and c.get("a") == 1


def test_missing_key():
    assert LRUCache(1).get("x", 0) == 0
