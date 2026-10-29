def average(values):
    """Arithmetic mean; 0.0 for an empty list."""
    return sum(values) / len(values)


def spread(values):
    return max(values) - min(values) if values else 0
