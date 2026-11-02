import json
import re
import subprocess
import sys

from conftest import ROOT
from digest import rank, reconstruct_abstract, render

WORKS = json.loads((ROOT / "acceptance" / "works.json").read_text(encoding="utf-8"))["results"]


def titles(works):
    return [w["title"] for w in works]


def test_rank_by_citations_then_newer_first_and_skip_untitled():
    assert titles(rank(WORKS, top=10)) == ["Old PCR survey", "A newer primer tool", "Primer design at scale",
                                           "Melting temperature models"]


def test_top_and_since():
    assert titles(rank(WORKS, top=2)) == ["Old PCR survey", "A newer primer tool"]
    assert titles(rank(WORKS, top=10, since_year=2020)) == ["A newer primer tool", "Melting temperature models"]


def test_reconstruct_abstract():
    assert reconstruct_abstract(WORKS[0]["abstract_inverted_index"]) == "Designing primers is hard; primers matter."
    assert reconstruct_abstract(None) == ""


def numbered(md):
    return [line for line in md.splitlines() if re.match(r"^\d+\. ", line)]


def test_render():
    md = render(rank(WORKS, top=3), "PCR primers")
    assert md.splitlines()[0].startswith("#") and "PCR primers" in md.splitlines()[0]
    items = numbered(md)
    assert len(items) == 3 and items[0].startswith("1. ") and "Old PCR survey" in items[0]
    assert "1998" in items[0] and "Kary Mullis" in items[0] and "https://doi.org/10.1000/e5" in items[0]
    assert "Primer design at scale" in items[2] and "Ada Lovelace" in items[2]


def test_cli_from_file(tmp_path):
    out = tmp_path / "digest.md"
    r = subprocess.run([sys.executable, str(ROOT / "digest.py"), "PCR primers", "--from-file",
                        str(ROOT / "acceptance" / "works.json"), "--top", "2", "--since", "2000", "-o", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    items = numbered(out.read_text(encoding="utf-8"))
    assert len(items) == 2 and "A newer primer tool" in items[0] and "Primer design at scale" in items[1]
