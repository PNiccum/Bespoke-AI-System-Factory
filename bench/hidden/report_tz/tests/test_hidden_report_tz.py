from datetime import datetime

from sales.clock import to_local
from sales.orders import parse_amount
from sales.report import daily_totals


def test_to_local():
    assert to_local("2026-03-02T02:30:00Z") == datetime(2026, 3, 1, 21, 30)
    assert to_local("2026-01-01T05:00:00Z") == datetime(2026, 1, 1, 0, 0)
    assert to_local("2026-01-01T04:59:59Z") == datetime(2025, 12, 31, 23, 59, 59)


def test_parse_amount():
    assert parse_amount(" $12,345.67 ") == 12345.67
    assert parse_amount("7") == 7.0


def test_new_year_boundary():
    csv_text = "id,timestamp,amount\n1,2026-01-01T04:00:00Z,$1.00\n2,2026-01-01T06:00:00Z,$2.00\n"
    assert daily_totals(csv_text) == {"2025-12-31": 1.0, "2026-01-01": 2.0}
