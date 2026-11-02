from money.split import split


def test_shares_add_up():
    assert split(100, 3) == [3334, 3333, 3333]


def test_float_total():
    assert split(0.29, 1) == [29]
