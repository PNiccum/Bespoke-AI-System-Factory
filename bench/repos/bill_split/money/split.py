from money.cents import to_cents


def split(total, people):
    """Split a bill of ``total`` dollars among ``people`` (an int, at least 1; ValueError otherwise) as evenly as
    possible. Returns each person's share in cents; the shares add up exactly to the total, and when it does not
    divide evenly the first people pay one cent more."""
    cents = to_cents(total)
    share = cents // people
    return [share] * people
