"""A deterministic event log: every error a run meets is written to ``<run>/events.jsonl``, with a status of its own.

No model is involved here. Code records an event where it happens (the model client's retry and fallback path, the
pipeline's stage runner, final QA, re-design) and code, never an agent, decides when an event is *corrected*, from a
real outcome: a retry that succeeded, a fallback model that answered, a QA repair that was approved, a re-design change
that passed its gates. An event nothing has corrected stays *open*. The historian (``factory/historian.py``) only
explains what this log says.

The file is append-only: an event line, and later a resolution line that points at it. Status is read from the pair, so
a crash between the two never loses the error. ``meta_analysis`` reads many runs and groups events by fingerprint, so a
problem that keeps coming back shows up as one row.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["warning", "error"]
FILE = "events.jsonl"


def fingerprint(source: str, stage: str, message: str) -> str:
    """Same problem, same hash: digits, hex ids and paths are blanked so a repeat in another run still matches."""
    text = re.sub(r"0x[0-9a-f]+|[0-9a-f]{8,}|\d+|(?:/[\w.\-]+)+", "#", message.lower())
    return hashlib.sha256(f"{source}|{stage}|{' '.join(text.split())[:200]}".encode()).hexdigest()[:12]


class Event(BaseModel):
    id: str
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run: str = ""
    domain: str = ""  # the project's subject area, set once per project by code (factory/pipeline/domain.py)
    source: str  # llm | pipeline | qa | redesign | code: the part of the system that met it
    stage: str = ""
    severity: Severity = "error"
    message: str
    fingerprint: str
    refs: list[str] = []  # blackboard entry ids, tags, request hashes
    # Filled from the resolution line, if any:
    status: Literal["open", "corrected"] = "open"
    how: str = ""  # what corrected it, in words
    evidence: list[str] = []  # ids that show it: a retry, a case, a blackboard entry
    corrected_at: str = ""


class EventLog:
    def __init__(self, path: Path, run: str = ""):
        self.path = path
        self.run = run or path.parent.name
        self.domain = ""  # copied onto every event recorded from now on
        self._lock = threading.Lock()

    def _write(self, row: dict) -> None:  # caller holds the lock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append(self, row: dict) -> None:
        with self._lock:
            self._write(row)

    def record(self, source: str, message: str, *, stage: str = "", severity: Severity = "error",
               refs: list[str] | tuple[str, ...] = (), once: bool = False) -> str:
        """Write an event and return its id. Never raises: a full disk must not turn a recoverable error fatal.
        ``once``: skip it (returning "") while an open event with the same fingerprint exists, for a check that runs
        again and again and would otherwise stack the same failure up."""
        fp = fingerprint(source, stage, str(message))
        try:
            with self._lock:  # the id counts lines, so the count and the write happen under one hold of the lock
                if once and any(e.fingerprint == fp for e in self.events() if e.status == "open"):
                    return ""
                n = sum(1 for _ in self._rows()) + 1  # ids count lines, so they stay unique with resolutions in
                ev = Event(id=f"E{n:04d}", run=self.run, domain=self.domain, source=source, stage=stage,
                           severity=severity, message=" ".join(str(message).split())[:600], fingerprint=fp,
                           refs=list(refs))
                self._write({"type": "event", **ev.model_dump(exclude={"status", "how", "evidence", "corrected_at"})})
            return ev.id
        except OSError:
            return ""

    def resolve(self, ids: str | list[str], how: str, evidence: list[str] | tuple[str, ...] = ()) -> int:
        """Mark events corrected. Called by code that saw the fix happen, never by a model. Returns how many were
        open and are now corrected."""
        want = {ids} if isinstance(ids, str) else set(ids)
        todo = [e for e in self.events() if e.id in want and e.status == "open"]
        for e in todo:
            self._append({"type": "resolution", "event": e.id, "how": how, "evidence": list(evidence),
                          "at": datetime.now(UTC).isoformat()})
        return len(todo)

    def resolve_open(self, how: str, evidence: list[str] | tuple[str, ...] = (), *, source: str | None = None,
                     stage: str | None = None, since: str = "", ref: str | None = None) -> int:
        """Correct every open event matching, e.g. all the events of one stage once that stage went through, or
        every event carrying one ref (a topic) once a verified page covers it."""
        return self.resolve([e.id for e in self.events()
                             if e.status == "open" and (source is None or e.source == source)
                             and (stage is None or e.stage == stage) and e.at >= since
                             and (ref is None or ref in e.refs)], how, evidence)

    def _rows(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                yield json.loads(line)
            except ValueError:
                continue  # a line cut short by a run that was stopped mid-write

    def events(self) -> list[Event]:
        out: dict[str, Event] = {}
        for row in self._rows():
            if row.get("type") == "event":
                out[row["id"]] = Event(**{k: v for k, v in row.items() if k != "type"})
            elif row.get("type") == "resolution" and row.get("event") in out:
                e = out[row["event"]]
                e.status, e.how, e.evidence, e.corrected_at = "corrected", row.get("how", ""), row.get("evidence", []), \
                    row.get("at", "")
        return list(out.values())

    def open_events(self) -> list[Event]:
        return [e for e in self.events() if e.status == "open"]

    @contextmanager
    def capture(self, source: str, stage: str = ""):
        """Whatever escapes this block is recorded (open) and still raised, so nothing leaves a run unlogged."""
        try:
            yield
        except Exception as e:
            self.record(source, f"{type(e).__name__}: {e}", stage=stage)
            raise


class NullLog(EventLog):
    """For code handed an LLM stand-in that has no run folder: events are dropped."""

    def __init__(self):
        super().__init__(Path("/dev/null"), "")

    def record(self, *a, **k) -> str:
        return ""

    def resolve(self, *a, **k) -> int:
        return 0

    def events(self) -> list[Event]:
        return []


def log_of(llm, project=None) -> EventLog:
    """The run's event log behind an LLM client (a NullLog for a stand-in without one). With a ``project``, the log's
    domain is set from it first, so every stage that tags events does it the same way (factory/pipeline/domain.py)."""
    log = getattr(llm, "events", None) or NullLog()
    if project is not None:
        from factory.pipeline.domain import project_domain
        log.domain = project_domain(project)
    return log


# --------------------------------------------------------------------------
# Meta analysis across runs
# --------------------------------------------------------------------------

class Group(BaseModel):
    """Every event with one fingerprint, across runs."""

    fingerprint: str
    source: str
    stage: str
    message: str  # the first one seen
    count: int
    runs: list[str]
    corrected: int
    open: int
    how: list[str]  # the distinct ways it was corrected
    first_seen: str
    last_seen: str
    # It was corrected in an earlier run and showed up again later: the fix did not hold, or was never general.
    returned_after_fix: bool

    @property
    def rate(self) -> float:
        return self.corrected / self.count if self.count else 0.0


def run_events(base: Path) -> list[Event]:
    out = []
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            out += EventLog(d / FILE, d.name).events()
    return out


def meta_analysis(events: list[Event]) -> list[Group]:
    """Group events by fingerprint, most frequent first. Pure code: the historian narrates this, it does not compute it."""
    by: dict[str, list[Event]] = {}
    for e in sorted(events, key=lambda e: e.at):
        by.setdefault(e.fingerprint, []).append(e)
    groups = []
    for fp, es in by.items():
        fixed_runs = {e.run for e in es if e.status == "corrected"}
        later = [e for e in es if any(r < e.run for r in fixed_runs)]
        groups.append(Group(fingerprint=fp, source=es[0].source, stage=es[0].stage, message=es[0].message,
                            count=len(es), runs=sorted({e.run for e in es}),
                            corrected=sum(e.status == "corrected" for e in es), open=sum(e.status == "open" for e in es),
                            how=sorted({e.how for e in es if e.how}), first_seen=es[0].at, last_seen=es[-1].at,
                            returned_after_fix=bool(later)))
    return sorted(groups, key=lambda g: (-g.count, g.fingerprint))


def by_domain(events: list[Event]) -> list[dict]:
    """Events per domain: how many, how many corrected, and the most frequent problem. "" is an untagged run."""
    rows: dict[str, list[Event]] = {}
    for e in events:
        rows.setdefault(e.domain, []).append(e)
    out = []
    for d, es in sorted(rows.items()):
        top = meta_analysis(es)[0]
        out.append({"domain": d or "(untagged)", "runs": len({e.run for e in es}), "events": len(es),
                    "corrected": sum(e.status == "corrected" for e in es), "open": sum(e.status == "open" for e in es),
                    "most frequent": f"{top.stage}: {top.message[:100]} ({top.count}x)"})
    return out


# Runs that are not builds: self-extension and replays write their own run folders (and their own events).
NOT_BUILDS = ("-extend-", "-replay-")
WINDOW = 10  # recurrence looks at this many of the newest runs
MIN_RUNS = 3  # an open fingerprint in this many of them is a recurring error


class Recurrence(BaseModel):
    """One problem that is still open in several of the newest runs."""

    fingerprint: str
    source: str
    stage: str
    message: str
    runs: list[str]  # the runs in the window where it is open, oldest first
    window: int  # how many runs were looked at


def recurring(base: Path, *, window: int = WINDOW, min_runs: int = MIN_RUNS,
              since: dict[str, str] | None = None) -> list[Recurrence]:
    """Open fingerprints that appear in at least ``min_runs`` of the newest ``window`` runs, most runs first.

    Plain code, no model. Only *open* events count: one that code corrected (a retry, a fallback, a QA repair) is the
    system working, not a problem. ``since`` maps a fingerprint to a run stamp (``YYYYmmddTHHMMSSZ``); runs at or
    before it are ignored for that fingerprint, so a problem a re-design already took up must come back in
    ``min_runs`` *later* runs before it is taken up again, which also shows whether the change held."""
    if not base.is_dir():
        return []
    runs = sorted(d for d in base.iterdir() if d.is_dir() and not any(m in d.name for m in NOT_BUILDS))[-window:]
    seen: dict[str, list[tuple[str, Event]]] = {}
    for d in runs:
        for e in EventLog(d / FILE, d.name).open_events():
            seen.setdefault(e.fingerprint, []).append((d.name, e))
    out = []
    for fp, rows in seen.items():
        after = (since or {}).get(fp, "")
        names = sorted({name for name, _ in rows if name[:16] > after})
        if len(names) >= min_runs:
            first = next(e for name, e in rows if name == names[0])
            out.append(Recurrence(fingerprint=fp, source=first.source, stage=first.stage, message=first.message,
                                  runs=names, window=len(runs)))
    return sorted(out, key=lambda r: (-len(r.runs), r.fingerprint))


def event_count(run_dir: str | Path) -> int:
    """How many events one run wrote (0 for a run with no log)."""
    return len(EventLog(Path(run_dir) / FILE).events()) if str(run_dir) else 0


def summary(events: list[Event]) -> dict:
    return {"events": len(events), "open": sum(e.status == "open" for e in events),
            "corrected": sum(e.status == "corrected" for e in events),
            "runs": len({e.run for e in events})}
