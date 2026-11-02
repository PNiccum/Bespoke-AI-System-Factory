import pytest

from money.cents import to_cents
from money.split import split


def test_to_cents():
    assert to_cents(19.99) == 1999
    assert to_cents("4.35") == 435
    assert to_cents(1.15) == 115


def test_extra_cents_go_first():
    assert split(10.01, 4) == [251, 250, 250, 250]
    assert split(0.05, 3) == [2, 2, 1]


@pytest.mark.parametrize("total,people", [(57.35, 7), (0.01, 5), (1234.56, 9), (4.35, 2)])
def test_always_sums_to_total(total, people):
    shares = split(total, people)
    assert len(shares) == people and sum(shares) == to_cents(total)
    assert max(shares) - min(shares) <= 1 and shares == sorted(shares, reverse=True)


@pytest.mark.parametrize("people", [0, -2])
def test_bad_people(people):
    with pytest.raises(ValueError):
        split(10, people)
