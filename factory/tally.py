"""A per-stage model tally, built from the calls the model client already logs (``<run>/llm_calls.jsonl``).

A fallback is a natural A/B test: for one stage, model A could not answer (or ran out of output room) and model B did.
Code counts, for every (role, stage, model): calls, replies that finished, replies cut off at the output limit, calls
that failed over to the next model, seconds and tokens. It also lists each *hand-off* (A to B, and why) with whether B
then finished, across every run folder. No model is involved.

The tally only says who is ahead when both models have ``MIN_CALLS`` calls on the stage; below that it says "not enough
data", so a lucky streak is never read as a winner. Cached replies are not counted (no model ran).

    python -m factory.tally            # every run -> <workspace>/tally/model_tally.md and .json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from factory.config import load_config

MIN_CALLS = 5
LOG = "llm_calls.jsonl"


class ModelStats(BaseModel):
    model: str
    calls: int = 0
    finished: int = 0  # a reply came back and was not cut off
    cut_off: int = 0  # hit max_tokens
    failed_over: int = 0  # unavailable: the call went on to the next model
    seconds: float = 0.0
    tokens: int = 0

    @property
    def finish_rate(self) -> float:
        return self.finished / self.calls if self.calls else 0.0

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.finished if self.finished else 0.0


class Handoff(BaseModel):
    role: str
    stage: str
    from_model: str
    to_model: str
    reason: str  # unavailable | cut off
    to_finished: int = 0
    count: int = 0


class StageTally(BaseModel):
    role: str
    stage: str
    models: list[ModelStats]
    verdict: str  # who is ahead, or why nothing is said


def stage_of(tag: str, role: str) -> str:
    """'solve-2-first' and 'solve-3' are both the stage 'solve'; paid rescues keep their own '-paid' stage."""
    base = re.sub(r"[-_]?\d+", "", tag or role)
    return re.sub(r"-first$", "", base) or role


def _rows(path: Path):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a line cut short by a stopped run
            if isinstance(row, dict) and row.get("model") and not row.get("cached"):
                yield row


def _verdict(models: list[ModelStats]) -> str:
    if len(models) < 2:
        return "one model only: nothing to compare"
    ready = [m for m in models if m.calls >= MIN_CALLS]
    if len(ready) < 2:
        return f"not enough data (needs {MIN_CALLS}+ calls on each of two models)"
    ready.sort(key=lambda m: (-m.finish_rate, m.mean_seconds or 1e9))
    a, b = ready[0], ready[1]
    if abs(a.finish_rate - b.finish_rate) < 0.10:
        return (f"no clear difference in finishing ({a.model} {a.finish_rate:.0%}, {b.model} {b.finish_rate:.0%})"
                + (f"; {a.model if a.mean_seconds < b.mean_seconds else b.model} is faster" if a.mean_seconds and b.mean_seconds
                   and abs(a.mean_seconds - b.mean_seconds) / max(a.mean_seconds, b.mean_seconds) > 0.25 else ""))
    return f"{a.model} finishes more often ({a.finish_rate:.0%} against {b.model} {b.finish_rate:.0%})"


def build_tally(runs_base: Path) -> tuple[list[StageTally], list[Handoff]]:
    stats: dict[tuple[str, str, str], ModelStats] = {}
    hand: dict[tuple[str, str, str, str, str], Handoff] = {}
    if runs_base.is_dir():
        for run in sorted(p for p in runs_base.iterdir() if p.is_dir()):
            by_tag: dict[str, list[dict]] = defaultdict(list)
            for r in _rows(run / LOG):
                by_tag[r.get("tag") or r.get("role", "")].append(r)
            for tag, rows in by_tag.items():
                for i, r in enumerate(rows):
                    role, stage = r.get("role", ""), stage_of(tag, r.get("role", ""))
                    s = stats.setdefault((role, stage, r["model"]), ModelStats(model=r["model"]))
                    s.calls += 1
                    failed = bool(r.get("fallback_to")) or "error" in r
                    cut = r.get("stop_reason") == "max_tokens"
                    if r.get("fallback_to"):
                        s.failed_over += 1
                    elif cut:
                        s.cut_off += 1
                    elif not failed:
                        s.finished += 1
                        s.seconds += float(r.get("seconds") or 0)
                    s.tokens += sum(int(r.get(k) or 0) for k in ("input_tokens", "output_tokens"))
                    # the next call under this tag, on another model, is the other side of the comparison
                    nxt = rows[i + 1] if i + 1 < len(rows) else None
                    if nxt and nxt["model"] != r["model"] and (r.get("fallback_to") or cut):
                        reason = "unavailable" if r.get("fallback_to") else "cut off"
                        h = hand.setdefault((role, stage, r["model"], nxt["model"], reason),
                                            Handoff(role=role, stage=stage, from_model=r["model"],
                                                    to_model=nxt["model"], reason=reason))
                        h.count += 1
                        if nxt.get("stop_reason") != "max_tokens" and not nxt.get("fallback_to") and "error" not in nxt:
                            h.to_finished += 1
    by_stage: dict[tuple[str, str], list[ModelStats]] = defaultdict(list)
    for (role, stage, _), s in stats.items():
        by_stage[(role, stage)].append(s)
    tallies = [StageTally(role=role, stage=stage, models=sorted(ms, key=lambda m: -m.calls), verdict=_verdict(ms))
               for (role, stage), ms in sorted(by_stage.items())]
    return tallies, sorted(hand.values(), key=lambda h: (-h.count, h.role, h.stage))


def report(tallies: list[StageTally], handoffs: list[Handoff]) -> str:
    out = ["# Model tally per stage", "",
           "Counted by code from each run's `llm_calls.jsonl`. A model is only called ahead when two models each have "
           f"{MIN_CALLS}+ calls on the stage.", ""]
    for t in tallies:
        out += [f"## {t.role} / {t.stage}", "", f"Verdict: {t.verdict}", "",
                "| model | calls | finished | cut off | failed over | mean s | tokens |", "|---|---|---|---|---|---|---|"]
        out += [f"| {m.model} | {m.calls} | {m.finished} ({m.finish_rate:.0%}) | {m.cut_off} | {m.failed_over} | "
                f"{m.mean_seconds:.1f} | {m.tokens} |" for m in t.models]
        out.append("")
    out += ["## Hand-offs (one model could not finish, another took over)", ""]
    if handoffs:
        out += ["| role / stage | from | to | why | times | to finished |", "|---|---|---|---|---|---|"]
        out += [f"| {h.role} / {h.stage} | {h.from_model} | {h.to_model} | {h.reason} | {h.count} | {h.to_finished} |"
                for h in handoffs]
    else:
        out.append("None recorded.")
    return "\n".join(out) + "\n"


def write_tally(runs_base: Path, out_dir: Path) -> Path | None:
    tallies, handoffs = build_tally(runs_base)
    if not tallies:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_tally.json").write_text(json.dumps(
        {"stages": [t.model_dump() for t in tallies], "handoffs": [h.model_dump() for h in handoffs]}, indent=1),
        encoding="utf-8")
    path = out_dir / "model_tally.md"
    path.write_text(report(tallies, handoffs), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.tally", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/factory.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = write_tally(cfg.root / cfg.runs_dir, cfg.work_dir / "tally")
    print(path or "No model calls recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
