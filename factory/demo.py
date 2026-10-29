"""Recorded runs: replay a real run with no network and no model, for a showcase that cannot be stopped by a slow
gateway or a dropped connection.

A recording holds the task and every model reply the run received, keyed exactly as the response cache keys them.
A replay runs the real orchestrator, agents and checks on the same task; only the model calls are answered from the
recording. Tests, the sandbox, the review and the bundle all run for real, so the bundle a replay exports is as
checkable as the original. A call the recording does not hold fails (it is never sent anywhere).

    python -m factory.demo record runs/20261030T140000Z-fix demos/parse_date.json
    python -m factory.demo replay demos/parse_date.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from factory.config import FactoryConfig, load_config
from factory.llm import LLM, new_run_dir
from factory.llm.cache import ResponseCache
from factory.llm.gate import deny_all
from factory.llm.providers import TransientHTTPError
from factory.orchestrator import ASPIRE_ROOT, Orchestrator, RunResult, Task
from factory.records import ProvenanceLog
from factory.registry import Registry
from factory.secrets import SecretFound, scan_text


class NotRecorded(TransientHTTPError):
    """A replay asked for a reply the recording does not hold."""


class Offline:
    """The provider a replay uses: it answers nothing, so every reply must come from the recording."""

    def complete(self, req):
        raise NotRecorded(f"offline replay: no recorded reply for this {req.model} request")


def record(run_dir: Path, out: Path, cfg: FactoryConfig) -> dict:
    """Write a recording of a finished run: its task, its model replies, and what it cost when it ran."""
    first = next(e for e in ProvenanceLog(run_dir).events() if e.kind == "source")
    task = dict(first.detail["task"])
    repo = Path(task["repo_path"]).resolve()
    if repo.is_relative_to(ASPIRE_ROOT):  # a fixture inside ASPIRE: replayable from any checkout of it
        task["repo_path"] = repo.relative_to(ASPIRE_ROOT).as_posix()
    calls = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines() if line.strip()]
    cache = ResponseCache(cfg.work_dir / "cache.sqlite")
    replies = {c["prompt_sha256"]: cache.get(c["prompt_sha256"]) for c in calls if c.get("prompt_sha256")}
    if missing := [k for k, v in replies.items() if v is None]:
        raise ValueError(f"{len(missing)} replies are not in the cache; record a run made with the cache on")
    summary = run_dir / "cost_summary.json"
    recording = {"task": task, "recorded_at": datetime.now(UTC).isoformat(), "from_run": run_dir.name,
                 "models": sorted({c["model"] for c in calls if c.get("model")}),
                 "original_cost": json.loads(summary.read_text()) if summary.exists() else {}, "replies": replies}
    text = json.dumps(recording, indent=1)
    if hits := scan_text(text):  # recordings are committed to demos/; a key must never ride along
        raise SecretFound(f"the recording holds {', '.join(hits)}; not written")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return recording


def replay_llm(cfg: FactoryConfig, run_dir: Path, recording: dict) -> LLM:
    """An LLM that answers only from the recording, never retries, and never asks to spend."""
    cfg = cfg.model_copy(deep=True)
    for provider in cfg.providers.values():
        provider.stream_retries = 0
    llm = LLM(cfg, run_dir, use_cache=False, providers={name: Offline() for name in cfg.providers}, approver=deny_all)
    llm.cache = ResponseCache(run_dir / "replay.sqlite")
    for key, reply in recording["replies"].items():
        llm.cache.put(key, reply)
    return llm


def recorded_task(recording: dict) -> Task:
    task = Task(**recording["task"])
    if not os.path.isabs(task.repo_path):
        task.repo_path = str(ASPIRE_ROOT / task.repo_path)
    return task


def replay(cfg: FactoryConfig, recording: dict, on_event=None) -> tuple[RunResult, Path]:
    run_dir = new_run_dir(cfg, "replay")
    llm = replay_llm(cfg, run_dir, recording)
    result = Orchestrator(llm, Registry.load(cfg), on_event=on_event).run(recorded_task(recording))
    return result, run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.demo")
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="save a finished run's task and model replies")
    rec.add_argument("run_dir")
    rec.add_argument("out")
    rep = sub.add_parser("replay", help="rerun a recording with no network and no model")
    rep.add_argument("recording")
    for p in (rec, rep):
        p.add_argument("--config", default="configs/factory.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "record":
        r = record(Path(args.run_dir), Path(args.out), cfg)
        print(f"Recorded {len(r['replies'])} model replies from {r['from_run']} to {args.out}")
        return 0
    from factory.__main__ import show
    from factory.export import EXPORTABLE, export_bundle
    recording = json.loads(Path(args.recording).read_text())
    result, run_dir = replay(cfg, recording, on_event=show)
    print(f"\nResult: {result.status} (replayed; the original run cost {recording['original_cost']})")
    if result.status in EXPORTABLE:
        print(f"Bundle: {export_bundle(result, recorded_task(recording), run_dir, cfg.root / 'exports')}")
    return 0 if result.status in ("fixed", "tested", "reviewed") else 1


if __name__ == "__main__":
    sys.exit(main())
