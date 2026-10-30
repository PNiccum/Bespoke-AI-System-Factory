from report.clean import dedupe
from report.summary import visited


def test_dedupe_keeps_first_occurrence():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert dedupe(["z", "y", "z"]) == ["z", "y"]


def test_visited_other_log():
    assert visited(["u1 z", "u2 y", "u1 z", "u3 x"]) == "z > y > x"
