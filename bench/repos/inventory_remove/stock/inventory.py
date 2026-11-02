class Inventory:
    """Units on hand per SKU."""

    def __init__(self):
        self._units = {}

    def add(self, sku, qty):
        """Add ``qty`` units (a positive int) of ``sku``."""
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError("qty must be a positive integer")
        self._units[sku] = self._units.get(sku, 0) + qty

    def count(self, sku):
        return self._units.get(sku, 0)

    def skus(self):
        """SKUs with at least one unit, sorted."""
        return sorted(self._units)
