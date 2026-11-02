from sales.report import daily_totals

CSV = """id,timestamp,amount
1,2026-03-01T15:00:00Z,$20.00
2,2026-03-02T02:30:00Z,$5.50
3,2026-03-02T16:00:00Z,"$1,200.00"
"""


def test_late_evening_order_counts_on_its_local_day():
    assert daily_totals(CSV) == {"2026-03-01": 25.5, "2026-03-02": 1200.0}
