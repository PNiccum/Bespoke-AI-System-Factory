import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

from conftest import ROOT
from logspikes import error_windows, parse_line

BASE = datetime(2026, 11, 2, 10, 0, 0, tzinfo=UTC)


def entry(t, status, path="/index.html", offset="+0000"):
    tz = timedelta(hours=int(offset[:3]), minutes=int(offset[0] + offset[3:]))
    local = (t + tz).strftime("%d/%b/%Y:%H:%M:%S")
    return f'203.0.113.7 - - [{local} {offset}] "GET {path} HTTP/1.1" {status} 512 "-" "curl/8.0"'


def log():
    lines = []
    for i in range(20):  # 10:00-10:05: 20 requests, 1 error (5%)
        lines.append(entry(BASE + timedelta(seconds=i * 10), 500 if i == 0 else 200))
    for i in range(20):  # 10:05-10:10: 20 requests, 8 errors (40%)
        lines.append(entry(BASE + timedelta(minutes=5, seconds=i * 10), 503 if i % 5 < 2 else 200))
    for i in range(4):  # 10:10-10:15: 4 requests, all errors, but too few to count
        lines.append(entry(BASE + timedelta(minutes=10, seconds=i), 500))
    lines += ["garbage line", ""]
    return lines


def test_parse_line():
    hit = parse_line(entry(BASE, 404, "/a?b=1", offset="-0500"))
    assert hit["status"] == 404 and hit["path"] == "/a?b=1"
    assert hit["time"] == BASE  # the same instant, whatever offset the log used
    assert parse_line("not a log line") is None


def starts(spikes):
    return [datetime.fromisoformat(s["start"]) for s in spikes]


def test_spike_windows():
    spikes = error_windows(log(), window_minutes=5, threshold=0.2, min_requests=10)
    assert starts(spikes) == [BASE + timedelta(minutes=5)]
    s = spikes[0]
    assert (s["requests"], s["errors"]) == (20, 8) and abs(s["rate"] - 0.4) < 1e-9


def test_threshold_and_min_requests():
    assert len(error_windows(log(), threshold=0.04, min_requests=10)) == 2
    assert len(error_windows(log(), threshold=0.2, min_requests=1)) == 2


def test_offsets_are_windowed_in_utc():
    lines = [entry(BASE + timedelta(minutes=5, seconds=i), 500, offset="+0200") for i in range(10)]
    assert starts(error_windows(lines, min_requests=10)) == [BASE + timedelta(minutes=5)]


def test_cli_json(tmp_path):
    (tmp_path / "access.log").write_text("\n".join(log()) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "logspikes.py"), "access.log", "--json"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert starts(json.loads(r.stdout)) == [BASE + timedelta(minutes=5)]
