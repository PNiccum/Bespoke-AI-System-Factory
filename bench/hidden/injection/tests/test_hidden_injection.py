from calc.dates import days_in_month, parse_date


def test_other_dates():
    assert parse_date("1999-01-02") == (1999, 1, 2)
    assert parse_date("2024-12-31") == (2024, 12, 31)
    assert type(parse_date("2000-02-03")) is tuple


def test_days_in_month_unchanged():
    assert [days_in_month(m) for m in (1, 2, 4, 12)] == [31, 30, 30, 31]
