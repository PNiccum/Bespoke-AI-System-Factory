import subprocess
import sys

import pytest
from conftest import ROOT
from csvprofile import profile, to_markdown

CSV = '''id,price,city,note,blank
1,2.50,"Blacksburg, VA",x,
2,3,Roanoke,,
3,,Roanoke,y,
4,10.25,"Blacksburg, VA",z,
'''


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


def test_types(data):
    p = profile(str(data))
    assert list(p) == ["id", "price", "city", "note", "blank"]
    assert {k: v["type"] for k, v in p.items()} == {"id": "int", "price": "float", "city": "str", "note": "str",
                                                     "blank": "empty"}


def test_missing_and_unique(data):
    p = profile(str(data))
    assert [p[c]["missing"] for c in p] == [0, 1, 0, 1, 4]
    assert p["city"]["unique"] == 2 and p["id"]["unique"] == 4


def test_numeric_stats(data):
    p = profile(str(data))
    assert (p["id"]["min"], p["id"]["max"]) == (1, 4) and p["id"]["mean"] == pytest.approx(2.5)
    assert p["price"]["min"] == pytest.approx(2.5) and p["price"]["max"] == pytest.approx(10.25)
    assert p["price"]["mean"] == pytest.approx((2.5 + 3 + 10.25) / 3)
    assert "mean" not in p["city"]


def test_markdown_has_a_row_per_column(data):
    md = to_markdown(profile(str(data)))
    lines = [line for line in md.splitlines() if line.startswith("|")]
    assert len(lines) == 2 + 5  # header, separator, one row per column
    assert "column" in lines[0] and "type" in lines[0] and "missing" in lines[0]
    assert any("price" in line and "float" in line for line in lines)


def test_cli_writes_report(data, tmp_path):
    out = tmp_path / "r.md"
    r = subprocess.run([sys.executable, str(ROOT / "csvprofile.py"), str(data), "-o", str(out)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "city" in out.read_text(encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "csvprofile.py"), str(data)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "blank" in r.stdout
