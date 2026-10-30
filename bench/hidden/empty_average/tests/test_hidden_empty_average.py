import pytest

from stats.summary import average


def test_means():
    assert average([1, 2]) == pytest.approx(1.5)
    assert average([1, 2, 3, 4]) == pytest.approx(2.5)


def test_empty_is_float_zero():
    assert average([]) == 0.0 and type(average([])) is float


def test_bad_input_still_raises():
    with pytest.raises(TypeError):
        average(["a", "b"])
