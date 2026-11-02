"""Find time windows where a web server's error rate spikes, from an access log in combined log format."""

import argparse
import json
import re
import sys
from datetime import UTC, datetime

LINE = re.compile(r'^\S+ \S+ \S+ \[(?P<time>[^\]]+)\] "(?:\S+) (?P<path>\S+)[^"]*" (?P<status>\d{3}) ')


def parse_line(line):
    m = LINE.match(line)
    if not m:
        return None
    try:
        t = datetime.strptime(m["time"], "%d/%b/%Y:%H:%M:%S %z")
    except ValueError:
        return None
    return {"time": t, "status": int(m["status"]), "path": m["path"]}


def error_windows(lines, window_minutes=5, threshold=0.2, min_requests=10):
    size = window_minutes * 60
    windows = {}
    for line in lines:
        hit = parse_line(line)
        if hit is None:
            continue
        ts = int(hit["time"].timestamp())
        start = ts - ts % size
        n, e = windows.get(start, (0, 0))
        windows[start] = (n + 1, e + (hit["status"] >= 500))
    out = []
    for start, (n, e) in sorted(windows.items()):
        if n >= min_requests and e / n >= threshold:
            out.append({"start": datetime.fromtimestamp(start, UTC).isoformat(), "requests": n, "errors": e,
                        "rate": e / n})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-requests", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        with open(args.file, encoding="utf-8") as f:
            spikes = error_windows(f, args.window, args.threshold, args.min_requests)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(spikes))
    else:
        for s in spikes:
            print(f"{s['start']}  {s['errors']}/{s['requests']} errors ({s['rate']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
