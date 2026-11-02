"""Profile each column of a CSV file and write the profile as a markdown report."""

import argparse
import csv
import sys


def _kind(values):
    if not values:
        return "empty"
    for kind, cast in (("int", int), ("float", float)):
        try:
            for v in values:
                cast(v)
            return kind
        except ValueError:
            continue
    return "str"


def profile(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        names = list(rows[0].keys()) if rows else []
    out = {}
    for name in names:
        cells = [(r.get(name) or "").strip() for r in rows]
        values = [c for c in cells if c]
        kind = _kind(values)
        col = {"type": kind, "missing": len(cells) - len(values), "unique": len(set(values))}
        if kind in ("int", "float"):
            nums = [int(v) if kind == "int" else float(v) for v in values]
            col |= {"min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}
        out[name] = col
    return out


def to_markdown(prof):
    lines = ["| column | type | missing | unique | min | max | mean |", "|---|---|---|---|---|---|---|"]
    for name, c in prof.items():
        mean = f"{c['mean']:.2f}" if "mean" in c else ""
        lines.append(f"| {name} | {c['type']} | {c['missing']} | {c['unique']} | {c.get('min', '')} | "
                     f"{c.get('max', '')} | {mean} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    try:
        report = to_markdown(profile(args.file))
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
