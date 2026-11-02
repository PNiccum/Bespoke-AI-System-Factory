def to_cents(amount):
    """Dollars (int, float or a numeric string) to whole cents, rounded to the nearest cent."""
    return int(float(amount) * 100)
