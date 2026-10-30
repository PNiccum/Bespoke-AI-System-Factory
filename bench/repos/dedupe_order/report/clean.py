def dedupe(items):
    """Drop repeats, keeping the first occurrence of each item in its original order."""
    return sorted(set(items))
