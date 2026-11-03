"""Every text-mode subprocess call names its encoding: on Windows the default is cp1252, and a generated program's
UTF-8 output (a curly quote, byte 0x9d) killed the reader thread and left stdout None (2026-11-03 projects run)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "factory"


def test_text_mode_subprocess_calls_name_utf8():
    bad = [f"{p.relative_to(ROOT)}:{n}" for p in ROOT.rglob("*.py")
           for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if re.search(r"\btext=True\b", line) and 'encoding="utf-8"' not in line]
    assert not bad, bad
