from cal.years import days_in_year, is_leap


def test_common_leap_years():
    assert is_leap(2024) and not is_leap(2023)


def test_century_rule():
    assert not is_leap(1900)
    assert is_leap(2000)


def test_days_in_year():
    assert days_in_year(2023) == 365
