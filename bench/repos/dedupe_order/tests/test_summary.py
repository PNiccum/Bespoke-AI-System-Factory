from report.clean import dedupe
from report.summary import visited


def test_dedupe_keeps_first_order():
    assert dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_visited():
    log = ["u1 home", "u1 search", "u1 home", "u1 cart"]
    assert visited(log) == "home > search > cart"


def test_empty():
    assert dedupe([]) == []
