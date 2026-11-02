from spans.coverage import total_covered
from spans.merge import merge


def test_contained_interval_does_not_shrink_the_outer_one():
    assert merge([[1, 10], [2, 3]]) == [[1, 10]]
    assert merge([[2, 3], [1, 10], [9, 12]]) == [[1, 12]]


def test_empty_and_single():
    assert merge([]) == []
    assert merge([[3, 3]]) == [[3, 3]]


def test_input_is_not_changed():
    data = [[4, 6], [1, 5]]
    merge(data)
    assert data == [[4, 6], [1, 5]]


def test_total_covered():
    assert total_covered([[5, 7], [1, 3], [2, 4], [6, 10]]) == 8
