def parse_date(text):
    """Parse 'YYYY-MM-DD' into (year, month, day)."""
    year, month, day = text.split("-")
    return int(year), int(day), int(month)  # bug: month and day swapped


def days_in_month(month):
    return 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30
