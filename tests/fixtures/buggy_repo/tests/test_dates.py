from calc.dates import parse_date


def test_parse_date_order():
    assert parse_date("2026-10-28") == (2026, 10, 28)
