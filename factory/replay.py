"""Replay a recorded run's model replies by stage, so pytest can rerun the pipeline's code offline in seconds.

``factory.demo`` replays by exact request hash: change one word of a prompt and every reply is missing. That suits a
showcase of one frozen run. This module is for catching bugs in the code around the models (gates, loops, parsing):
replies are filed by the ``tag`` of the stage that asked (``plan-recheck-b``, ``wiki-check``, ...) and handed back in
the order they were received, so prompts and code can change while the recorded replies stay valid. A stage that
asks for more replies than the recording holds fails with ``NotRecorded``; nothing is ever sent anywhere.

    python -m factory.replay record runs/<run> tests/fixtures/replays/<name>.json --public-data [--deny-list FILE]
    python -m factory.replay check tests/fixtures/replays/<name>.json

A run's ``llm_calls.jsonl`` holds tags, hashes and token counts but no reply text; the text is read from the
response cache on the machine that made the run (``work/cache.sqlite``). Replies cut off at the output limit were
never cached, so they are recorded as empty cut-off replies (their stop reason and token counts come from the log).

Recordings are committed to the repository, so recording refuses by default:
- unless ``--public-data`` says the run read only public data (lab lists, e.g. the MCF7 protein list, never go in);
- if a reply holds a secret (factory/secrets.py);
- if a reply holds any line of a ``--deny-list`` file (each line of 6 or more characters), for data that must not
  leave the machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from factory.config import load_config
from factory.demo import NotRecorded
from factory.llm.cache import ResponseCache
from factory.llm.cost import Usage
from factory.llm.providers import RawResult
from factory.secrets import SecretFound, scan_text

MIN_DENY = 6  # shorter deny-list lines (gene symbols and the like) would match ordinary words


class RefusedRecording(RuntimeError):
    """The recording would hold data that must not be committed."""


def record_run(run_dir: Path, out: Path, cache: ResponseCache, *, public_data: bool = False,
               deny_list: Path | None = None, note: str = "") -> dict:
    if not public_data:
        raise RefusedRecording("recordings are committed to the repository; pass public_data=True (--public-data) "
                               "to say this run read only public data")
    lines = [json.loads(x) for x in (run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    replies: dict[str, list[dict]] = {}
    skipped_cached = 0
    for c in lines:
        if "prompt_sha256" not in c:
            skipped_cached += bool(c.get("cached"))  # logs from before cache hits logged their hash
            continue  # otherwise an event, an error or a fallback note, not a reply
        usage = {k: c.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        hit = cache.get(c["prompt_sha256"])
        if hit is None and c.get("stop_reason") not in ("max_tokens", "refusal"):
            raise ValueError(f"the reply for {c.get('tag') or c.get('role')} is not in the cache; record on the "
                             "machine that made the run")
        reply = hit or {"text": "", "thinking": "", "stop_reason": c["stop_reason"], "usage": usage}
        reply = {"model": c.get("model", ""), "role": c.get("role", ""), **reply}
        replies.setdefault(c.get("tag", ""), []).append(reply)
    recording = {"recorded_at": datetime.now(UTC).isoformat(), "from_run": run_dir.name, "note": note,
                 "cache_hits_not_recorded": skipped_cached, "replies": replies}
    text = json.dumps(recording, indent=1)
    if hits := scan_text(text):
        raise SecretFound(f"the recording holds {', '.join(hits)}; not written")
    if deny_list:
        for line in deny_list.read_text(encoding="utf-8").splitlines():
            if len(line.strip()) >= MIN_DENY and line.strip().lower() in text.lower():
                raise RefusedRecording("a reply holds a line of the deny list; not written (the line is not shown)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return recording


class ReplayProvider:
    """A provider that answers from a recording, by stage tag, in recorded order. Thread-safe; sends nothing."""

    def __init__(self, recording: dict):
        self._left = {tag: list(r) for tag, r in recording["replies"].items()}
        self._lock = threading.Lock()
        self.served: list[str] = []

    def complete(self, req) -> RawResult:
        with self._lock:
            queue = self._left.get(req.tag)
            if not queue:
                raise NotRecorded(f"replay: no recorded reply left for stage {req.tag!r} ({req.model})")
            r = queue.pop(0)
            self.served.append(req.tag)
        return RawResult(r["text"], r.get("thinking", ""), r.get("stop_reason"), Usage(**r["usage"]),
                         r.get("tool_calls", []))

    def unused(self) -> dict[str, int]:
        """Recorded replies the replay never asked for: the code now makes fewer calls than the run did."""
        return {t: len(q) for t, q in self._left.items() if q}


def replay_providers(cfg, recording: dict) -> tuple[dict, ReplayProvider]:
    """One shared replay provider under every configured provider name, and the config with retries off."""
    provider = ReplayProvider(recording)
    for p in cfg.providers.values():
        p.stream_retries = 0  # a missing reply is not transient; never sleep and ask again
    return {name: provider for name in cfg.providers}, provider


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.replay")
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="save a finished run's replies, filed by stage")
    rec.add_argument("run_dir")
    rec.add_argument("out")
    rec.add_argument("--config", default="configs/factory.yaml")
    rec.add_argument("--public-data", action="store_true", help="the run read only public data")
    rec.add_argument("--deny-list", help="file of strings (one per line) that must not appear in the recording")
    rec.add_argument("--note", default="")
    chk = sub.add_parser("check", help="summarize a recording")
    chk.add_argument("recording")
    args = ap.parse_args(argv)
    if args.command == "record":
        cfg = load_config(args.config)
        r = record_run(Path(args.run_dir), Path(args.out), ResponseCache(cfg.work_dir / "cache.sqlite"),
                       public_data=args.public_data, deny_list=Path(args.deny_list) if args.deny_list else None,
                       note=args.note)
        n = sum(len(v) for v in r["replies"].values())
        print(f"Recorded {n} replies in {len(r['replies'])} stages to {args.out} "
              f"({r['cache_hits_not_recorded']} old-format cache hits not recorded)")
        return 0
    r = load(Path(args.recording))
    for tag, replies in r["replies"].items():
        print(f"{tag or '(no tag)'}: {len(replies)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
