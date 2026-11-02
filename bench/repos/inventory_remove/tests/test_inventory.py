import pytest

from stock.errors import OutOfStock
from stock.inventory import Inventory


def test_add_and_count():
    inv = Inventory()
    inv.add("apple", 3)
    assert inv.count("apple") == 3


def test_remove():
    inv = Inventory()
    inv.add("apple", 5)
    inv.remove("apple", 2)
    assert inv.count("apple") == 3


def test_remove_more_than_in_stock():
    inv = Inventory()
    inv.add("apple", 1)
    with pytest.raises(OutOfStock):
        inv.remove("apple", 2)
