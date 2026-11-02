import pytest

from stock.errors import OutOfStock
from stock.inventory import Inventory


def test_failed_remove_changes_nothing_and_reports_counts():
    inv = Inventory()
    inv.add("pear", 2)
    with pytest.raises(OutOfStock) as e:
        inv.remove("pear", 5)
    assert (e.value.sku, e.value.requested, e.value.available) == ("pear", 5, 2)
    assert inv.count("pear") == 2


def test_unknown_sku_is_out_of_stock():
    with pytest.raises(OutOfStock) as e:
        Inventory().remove("kiwi", 1)
    assert e.value.available == 0


def test_removing_everything_drops_the_sku():
    inv = Inventory()
    inv.add("a", 2)
    inv.add("b", 1)
    inv.remove("a", 2)
    assert inv.skus() == ["b"] and inv.count("a") == 0


@pytest.mark.parametrize("qty", [0, -1, 1.5])
def test_bad_quantity(qty):
    inv = Inventory()
    inv.add("a", 2)
    with pytest.raises(ValueError):
        inv.remove("a", qty)
    assert inv.count("a") == 2
