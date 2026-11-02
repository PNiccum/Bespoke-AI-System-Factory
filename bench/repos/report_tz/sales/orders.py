import csv
import io

from sales.clock import to_local


def parse_amount(text):
    """'$1,234.50' or '12' -> amount in dollars as a float."""
    return float(text.strip().lstrip("$"))


def read_orders(csv_text):
    """CSV with columns id,timestamp,amount -> list of (local datetime, amount)."""
    rows = csv.DictReader(io.StringIO(csv_text))
    return [(to_local(r["timestamp"]), parse_amount(r["amount"])) for r in rows]
