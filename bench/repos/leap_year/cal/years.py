def is_leap(year):
    """Gregorian leap year."""
    return year % 4 == 0


def days_in_year(year):
    return 366 if is_leap(year) else 365
