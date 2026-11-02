from datetime import datetime, timedelta, timezone

STORE_UTC_OFFSET_HOURS = -5  # the store's clock is five hours behind UTC


def to_local(ts_utc):
    """'2026-03-02T02:30:00Z' (a UTC timestamp) as a naive datetime on the store's clock."""
    t = datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (t - timedelta(hours=STORE_UTC_OFFSET_HOURS)).replace(tzinfo=None)
