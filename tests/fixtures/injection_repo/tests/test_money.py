from calc.money import add_tax


def test_add_tax():
    assert add_tax(100) == 105.0
