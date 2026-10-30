from lists.window import last_n


def test_other_sizes():
    assert last_n([1, 2, 3, 4, 5], 3) == [3, 4, 5]
    assert last_n([1, 2, 3, 4, 5], 1) == [5]
    assert last_n([7], 1) == [7]


def test_n_larger_than_the_list():
    assert last_n([1, 2], 5) == [1, 2]
