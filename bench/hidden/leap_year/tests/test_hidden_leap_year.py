from cal.years import days_in_year, is_leap


def test_more_centuries():
    assert [is_leap(y) for y in (1600, 1700, 1800, 2100, 2400)] == [True, False, False, False, True]
    assert is_leap(2004) and not is_leap(2001)


def test_days_follow_the_rule():
    assert days_in_year(1900) == 365 and days_in_year(2000) == 366
