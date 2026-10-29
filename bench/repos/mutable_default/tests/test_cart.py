from shop.cart import add_item, total


def test_new_carts_start_empty():
    assert add_item("apple") == ["apple"]
    assert add_item("pear") == ["pear"]


def test_existing_cart():
    assert add_item("b", ["a"]) == ["a", "b"]


def test_total():
    assert total([1.10, 2.20]) == 3.3
