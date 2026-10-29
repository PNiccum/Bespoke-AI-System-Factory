def add_item(name, cart=[]):
    """Add an item to a cart and return the cart. With no cart given, start a new one."""
    cart.append(name)
    return cart


def total(prices):
    return round(sum(prices), 2)
