from shop.cart import add_item


def test_every_new_cart_is_fresh():
    first = add_item("apple")
    second = add_item("pear")
    assert first == ["apple"] and second == ["pear"] and first is not second


def test_an_existing_cart_is_added_to_in_place():
    cart = ["a"]
    assert add_item("b", cart) is cart and cart == ["a", "b"]
