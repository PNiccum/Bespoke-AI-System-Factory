from spans.merge import merge


def test_unsorted_input():
    assert merge([[5, 7], [1, 3], [2, 4]]) == [[1, 4], [5, 7]]


def test_touching_intervals_merge():
    assert merge([[1, 2], [2, 3]]) == [[1, 3]]


def test_disjoint():
    assert merge([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]
