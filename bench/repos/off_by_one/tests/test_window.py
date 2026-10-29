from lists.window import first_n, last_n


def test_last_n():
    assert last_n([1, 2, 3, 4], 2) == [3, 4]


def test_last_n_zero():
    assert last_n([1, 2], 0) == []


def test_first_n():
    assert first_n([1, 2, 3], 2) == [1, 2]
