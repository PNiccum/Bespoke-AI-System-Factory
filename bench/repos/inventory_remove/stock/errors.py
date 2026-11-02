class OutOfStock(Exception):
    """Raised when a removal asks for more units than are in stock. Carries ``sku``, ``requested`` and
    ``available``."""

    def __init__(self, sku, requested, available):
        super().__init__(f"{sku}: asked for {requested}, only {available} in stock")
        self.sku = sku
        self.requested = requested
        self.available = available
