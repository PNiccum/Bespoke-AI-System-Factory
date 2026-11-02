from collections import defaultdict

from sales.orders import read_orders


def daily_totals(csv_text):
    """Total sales per store-local day, as {"YYYY-MM-DD": dollars rounded to cents}, days in order."""
    totals = defaultdict(float)
    for when, amount in read_orders(csv_text):
        totals[when.date().isoformat()] += amount
    return {day: round(total, 2) for day, total in sorted(totals.items())}
