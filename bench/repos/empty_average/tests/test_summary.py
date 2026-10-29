from stats.summary import average, spread


def test_average():
    assert average([2, 4]) == 3


def test_average_of_nothing():
    assert average([]) == 0.0


def test_spread():
    assert spread([1, 5, 3]) == 4
