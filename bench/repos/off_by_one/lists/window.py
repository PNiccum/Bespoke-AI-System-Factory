def last_n(items, n):
    """The last n items, oldest first."""
    if n <= 0:
        return []
    return items[-n - 1:]


def first_n(items, n):
    return items[:max(n, 0)]
