"""Where a run's time went, from its ``llm_calls.jsonl`` logs. No model is called.

    python -m factory.profile runs/20261104T142930Z-build runs/20261104T063324Z-research
    python -m factory.profile runs            # every run under the folder

For each run: wall-clock time (first call start to last call end), the summed time of its model calls, and their
ratio, which is the average number of calls in flight (1.0 means every call waited for the one before). Then, over
all the runs given: time by stage (the tag's first word: build, verify, wiki, review, plan...), time by model with
its output speed, the time lost to replies that hit max_tokens (thrown away and asked again), and transport errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage(tag: str) -> str:
    return re.split(r"[-_ ]", tag or "", maxsplit=1)[0] or "untagged"


def logs(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_file():
            found.append(p)
        elif (p / "llm_calls.jsonl").is_file():
            found.append(p / "llm_calls.jsonl")
        else:
            found += sorted(p.glob("*/llm_calls.jsonl"))
    return found


def profile(paths: list[Path]) -> dict:
    runs, by_stage, by_model = [], Counter(), Counter()
    stage_calls, model_calls, model_out = Counter(), Counter(), Counter()
    cut_off, cut_off_s, errors = Counter(), Counter(), Counter()
    for log in logs(paths):
        rows = load(log)
        timed = [r for r in rows if r.get("seconds")]
        if not timed:
            continue
        ends = [datetime.fromisoformat(r["at"]) for r in timed]
        starts = [e - timedelta(seconds=r["seconds"]) for e, r in zip(ends, timed, strict=True)]
        wall = (max(ends) - min(starts)).total_seconds()
        busy = sum(r["seconds"] for r in timed)
        runs.append({"run": log.parent.name, "calls": len(timed), "wall_s": round(wall, 1),
                     "model_s": round(busy, 1), "in_flight": round(busy / wall, 2) if wall else 0.0})
        for r in rows:
            if "error" in r:
                errors[r["error"].split(":")[0]] += 1
            if not r.get("seconds"):
                continue
            s, m = stage(r.get("tag", "")), r.get("model", "?")
            by_stage[s] += r["seconds"]
            stage_calls[s] += 1
            by_model[m] += r["seconds"]
            model_calls[m] += 1
            model_out[m] += r.get("output_tokens", 0)
            if r.get("stop_reason") == "max_tokens":
                cut_off[(m, s)] += 1
                cut_off_s[(m, s)] += r["seconds"]
    total = sum(by_stage.values())
    return {
        "runs": runs,
        "model_s": round(total, 1),
        "stages": [{"stage": k, "calls": stage_calls[k], "seconds": round(v, 1), "share": round(v / total, 3),
                    "mean_s": round(v / stage_calls[k], 1)} for k, v in by_stage.most_common()],
        "models": [{"model": k, "calls": model_calls[k], "seconds": round(v, 1), "mean_s": round(v / model_calls[k], 1),
                    "out_tok_per_s": round(model_out[k] / v, 1) if v else 0.0} for k, v in by_model.most_common()],
        "cut_off": [{"model": m, "stage": s, "calls": cut_off[(m, s)], "seconds": round(cut_off_s[(m, s)], 1)}
                    for (m, s) in sorted(cut_off, key=lambda k: -cut_off_s[k])],
        "cut_off_s": round(sum(cut_off_s.values()), 1),
        "errors": dict(errors.most_common()),
    }


def report(p: dict) -> str:
    lines = ["| Run | Calls | Wall min | Model min | In flight |", "|---|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['calls']} | {r['wall_s'] / 60:.1f} | {r['model_s'] / 60:.1f} | {r['in_flight']:.2f} |"
              for r in p["runs"]]
    lines += ["", "| Stage | Calls | Minutes | Share | Mean s |", "|---|---|---|---|---|"]
    lines += [f"| {s['stage']} | {s['calls']} | {s['seconds'] / 60:.1f} | {s['share']:.0%} | {s['mean_s']:.0f} |"
              for s in p["stages"]]
    lines += ["", "| Model | Calls | Minutes | Mean s | Output tok/s |", "|---|---|---|---|---|"]
    lines += [f"| {m['model']} | {m['calls']} | {m['seconds'] / 60:.1f} | {m['mean_s']:.0f} | {m['out_tok_per_s']} |"
              for m in p["models"]]
    share = p["cut_off_s"] / p["model_s"] if p["model_s"] else 0
    lines += ["", f"Replies cut off at max_tokens (thrown away): {p['cut_off_s'] / 60:.1f} min, {share:.0%} of model "
                  "time", "", "| Model | Stage | Calls | Minutes |", "|---|---|---|---|"]
    lines += [f"| {c['model']} | {c['stage']} | {c['calls']} | {c['seconds'] / 60:.1f} |" for c in p["cut_off"]]
    if p["errors"]:
        lines += ["", "Transport errors: " + ", ".join(f"{k} x{v}" for k, v in p["errors"].items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.profile", description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="run folders, llm_calls.jsonl files, or a folder of runs")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown tables")
    args = ap.parse_args(argv)
    p = profile(args.paths)
    if not p["runs"]:
        print("no timed model calls found", file=sys.stderr)
        return 1
    print(json.dumps(p, indent=1) if args.json else report(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
