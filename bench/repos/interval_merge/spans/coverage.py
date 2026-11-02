from spans.merge import merge


def total_covered(intervals):
    """Length of the number line covered by at least one interval."""
    return sum(end - start for start, end in merge(intervals))
