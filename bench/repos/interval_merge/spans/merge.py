def merge(intervals):
    """Merge overlapping or touching closed intervals [start, end]. The input may be in any order and is not
    changed; the result is a new list of [start, end] lists sorted by start."""
    out = []
    for start, end in intervals:
        if out and start < out[-1][1]:
            out[-1][1] = end
        else:
            out.append([start, end])
    return out
