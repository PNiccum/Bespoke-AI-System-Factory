"""What past projects teach the next one: a lessons page, and a list of sources that failed before.

Both live in ``<workspace>/lessons/`` beside the shared wiki, so every run on a machine reads and adds to them.

**Bad sources** (``bad_sources.json``) is plain code. When the research orchestrator rejects a document for a topic,
code records it under the task's *domain* and the topic. The next time research for that topic finds the same
document, code skips it before a model reads it, and the orchestrator is told which sources failed and why, so it
searches elsewhere; a source rejected ``SKIP_AFTER`` times is skipped outright (one rejection only warns, so a
source that was judged wrongly once can still be approved, which clears its entry). An entry expires after ``TTL_DAYS`` (a page can be fixed), and a document approved later removes
its entry. A rejection is about one topic only: the same paper can be right for another topic.

**Lessons** (``lessons.jsonl`` and a page per domain, ``<domain>.md``) are written after a project is delivered or
re-designed, by one free ARC call. The call sees the blackboard's entries by id and the run's event log, and must
cite blackboard ids for every lesson. A project that holds any ARC-only entry writes no lessons, because the page is shared
across the machine. Code drops a lesson that cites no entry that exists, so a lesson cannot float
free of what happened. The page is rendered from the saved rows by code; the model never writes the file.

The domain is the project's tag from ``factory/pipeline/domain.py``, so lessons from a bio task do not steer a code task.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from factory.events import FILE, Event, EventLog
from factory.llm import LLM
from factory.llm.client import complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.domain import project_domain

ROLE = "historian"  # a free ARC role; nothing here is ever sent to a paid model
TTL_DAYS = 30
MAX_LESSONS = 8
MAX_ENTRIES = 60  # blackboard entries one prompt carries

SKIP_AFTER = 2  # rejections before a source is skipped without a model looking; one rejection only warns


def domain_of(project: Project) -> str:
    """The project's domain tag (factory/pipeline/domain.py): the one on the request entry, else one from its words."""
    return project_domain(project) or "other"


def lessons_dir(workspace_or_wiki_root: Path) -> Path:
    return workspace_or_wiki_root / "lessons"


def _key(topic: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", topic.casefold()).split()) or topic.casefold().strip()


def _now() -> datetime:
    return datetime.now(UTC)



def _take_over_stale(lock: Path, stale: float) -> bool:
    """Remove a crashed run's lock. The lock is first renamed to a name of our own, so two runs that both judged it
    stale cannot delete each other's fresh lock; if what we renamed turns out to be fresh after all, it is put back."""
    try:
        if time.time() - lock.stat().st_mtime <= stale:
            return False  # fresh: a live run's, never touched (renaming it even briefly would let a third run in)
    except OSError:
        return False
    mine = lock.with_name(f"{lock.name}.{os.getpid()}.{threading.get_ident()}.old")
    try:
        os.rename(lock, mine)
        old = time.time() - mine.stat().st_mtime > stale
    except OSError:
        return False
    if old:
        with contextlib.suppress(OSError):
            mine.unlink()
        return True
    with contextlib.suppress(OSError):
        os.link(mine, lock)  # put the live run's lock back (fails if another run took the name meanwhile)
    with contextlib.suppress(OSError):
        mine.unlink()
    return False


@contextlib.contextmanager
def _exclusive(lock: Path, threads: threading.Lock, timeout: float = 5.0, stale: float = 60.0):
    """Yields whether the lock is held. Threads share ``threads``; separate runs on one machine share the lock file (created exclusively, so it works on
    Windows too). A lock file older than ``stale`` seconds (longer than any wait) belongs to a crashed run and is removed. If a live run still holds
    it after ``timeout``, the write goes ahead unlocked (these files are by-products) and that run's lock is left alone."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    with threads:
        held = False
        start = time.monotonic()
        while not held:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                held = True
            except FileExistsError:
                if _take_over_stale(lock, stale):
                    continue
                if time.monotonic() - start > timeout:
                    break
                time.sleep(0.02)
        try:
            yield held
        finally:
            if held:
                with contextlib.suppress(OSError):
                    lock.unlink()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # its own name, so runs never share one
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)  # a run stopped mid-write never leaves half a file
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------
# Known-bad sources
# --------------------------------------------------------------------------

class BadSources:
    def __init__(self, path: Path, enabled: bool = True):
        """``enabled=False`` (benchmarks and failure replays): nothing is read or written, so deliberately bad test
        documents never become real projects' skips."""
        self.path, self.enabled = path, enabled
        self._lock = threading.Lock()  # topics are researched in parallel threads

    def _load(self) -> dict:
        """The saved list. A missing file is an empty list; an unreadable or damaged one is moved aside (so a save
        never wipes it) and reads as empty."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError:
            return {}  # missing, or briefly unreadable while another run replaces it (Windows): never rename for that
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        with contextlib.suppress(OSError):
            os.replace(self.path, self.path.with_suffix(f".damaged-{int(time.time())}"))
        return {}

    def _save(self, data: dict) -> None:
        _write_atomic(self.path, json.dumps(data, indent=1, ensure_ascii=False))

    def _locked(self):
        return _exclusive(self.path.with_suffix(".lock"), self._lock)

    def record(self, domain: str, topic: str, doc_id: str, why: str, *, title: str = "", url: str = "",
               project: str = "", now: datetime | None = None) -> None:
        if not self.enabled:
            return
        now = now or _now()
        with self._locked() as held:
            if not held:
                return  # another run is writing; one lost note in a by-product list is better than a torn file
            data = self._load()
            rows = data.setdefault(domain, {}).setdefault(_key(topic), {})
            stale = rows.get(doc_id, {}).get("last", "9") < (now - timedelta(days=TTL_DAYS)).isoformat()
            if stale:
                del rows[doc_id]  # an expired entry starts again, so one new rejection only warns
            row = rows.setdefault(
                doc_id, {"title": title[:160], "url": url, "why": why, "count": 0, "projects": [], "first": now.isoformat()})
            row.update(why=" ".join(why.split())[:300] or row["why"], last=now.isoformat())
            if project and project in row["projects"]:
                pass  # the same project rejecting it again (a rerun, two topic names that match) is one rejection
            else:
                row["count"] += 1
                if project:
                    row["projects"].append(project)
            self._save(data)

    def known(self, domain: str, topic: str, *, now: datetime | None = None) -> dict[str, dict]:
        """The unexpired bad sources for this topic in this domain, by document id."""
        if not self.enabled:
            return {}
        cutoff = ((now or _now()) - timedelta(days=TTL_DAYS)).isoformat()
        try:
            rows = self._load().get(domain, {}).get(_key(topic), {})
            return {i: r for i, r in rows.items() if r.get("last", "") >= cutoff}
        except (AttributeError, TypeError):
            return {}  # a file of the wrong shape reads as empty; the next save (under a lock) replaces it

    def clear(self, domain: str, topic: str, doc_id: str) -> bool:
        """A document that was approved after all is no longer a bad source."""
        if not self.enabled or doc_id not in self._load().get(domain, {}).get(_key(topic), {}):
            return False  # nothing to clear: no lock, no write
        with self._locked() as held:
            if not held:
                return False
            data = self._load()
            rows = data.get(domain, {}).get(_key(topic), {})
            if doc_id not in rows:
                return False
            del rows[doc_id]
            self._save(data)
            return True

    def rows(self, *, now: datetime | None = None) -> list[dict]:
        """Every unexpired entry, for the lessons page."""
        cutoff = ((now or _now()) - timedelta(days=TTL_DAYS)).isoformat()
        return [{"domain": d, "topic": t, "id": i, **r}
                for d, topics in self._load().items() for t, docs in topics.items() for i, r in docs.items()
                if r.get("last", "") >= cutoff]


def _plain(text: str, width: int) -> str:
    """Text that came from a web page or a model, made safe to show in a later prompt: letters, digits and light
    punctuation only (no quotes, brackets, newlines or markup), and short."""
    return " ".join(re.sub(r"[^A-Za-z0-9 .,;:'()/_-]", " ", text).split())[:width]


def bad_sources_text(known: dict[str, dict]) -> str:
    """For the orchestrator's prompt: the document id and a short plain reason. The page title is not repeated here."""
    return "; ".join(f"{_plain(i, 60)} ({_plain(r.get('why', ''), 80)})" for i, r in list(known.items())[:8])


# --------------------------------------------------------------------------
# Lessons
# --------------------------------------------------------------------------

SYSTEM = ("You write lessons for an automated system that builds software from plain-language requests. You are given "
          "what happened in one project: its blackboard entries (each has an id) and its error log. A lesson is one "
          "thing the next project in the same domain should do or avoid, in a single plain sentence, and it must rest "
          "on entries you cite by id. Never cite an id that is not listed. Do not repeat what the task asked for. "
          "Reply with JSON only.")

PROMPT = """Domain: {domain}
Outcome: {outcome}

Blackboard entries of the project (id | kind | author | summary):
{entries}

Error log (id | stage | status | message | corrected by):
{events}

Write at most {n} lessons. Reply with JSON:
{{"lessons": [{{"lesson": "one sentence", "kind": "worked|failed|watch",
"evidence": ["0007-source", "0012-fixes"]}}]}}
`worked` = a choice that paid off, `failed` = what went wrong and why, `watch` = a risk to check next time.
Every lesson needs at least one blackboard id in `evidence`."""


LESSON_KEYS = frozenset({"domain", "project", "outcome", "at", "lesson", "kind", "evidence"})


class Lesson(BaseModel):
    lesson: str
    kind: str = "watch"
    evidence: list[str] = []


class Lessons(BaseModel):
    lessons: list[Lesson] = []


def _summary(data: dict, width: int = 200) -> str:
    for key in ("text", "topic", "title", "summary", "verdict", "reason", "why", "path"):
        if isinstance(data.get(key), str) and data[key].strip():
            return _plain(data[key], width)  # web titles and model text: plain words only, never markup or quotes
    return _plain(json.dumps(data, ensure_ascii=False), width)


def _entries_text(project: Project) -> str:
    entries = project.board.entries()
    keep = entries if len(entries) <= MAX_ENTRIES else entries[:MAX_ENTRIES // 3] + entries[-(MAX_ENTRIES * 2 // 3):]
    # an ARC-only entry is named but its content is not copied into a page that people read and share
    return "\n".join(f"{e.id} | {e.kind} | {e.author} | {'(withheld: arc_only)' if e.license == 'arc_only' else _summary(e.data)}"
                     for e in keep)


def _events_text(events: list[Event]) -> str:
    return "\n".join(f"{e.id} | {e.stage} | {e.status} | {e.message[:160]} | {e.how[:100]}" for e in events[:30]) or "none"


def checked(found: Lessons, project: Project) -> list[Lesson]:
    """Keep a lesson only if it cites at least one entry that is really on this project's blackboard. Unknown ids are
    removed from what is kept; duplicates are dropped."""
    ids = {e.id for e in project.board.entries()}
    out, seen = [], set()
    for les in found.lessons[:MAX_LESSONS]:
        ev = [i for i in dict.fromkeys(les.evidence) if i in ids]
        text = " ".join(les.lesson.split())
        if not ev or not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(Lesson(lesson=text[:400], kind=les.kind if les.kind in ("worked", "failed", "watch") else "watch",
                          evidence=ev))
    return out


def read_lessons(directory: Path, domain: str | None = None) -> list[dict]:
    path = directory / "lessons.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (not isinstance(row, dict) or not LESSON_KEYS <= row.keys()
                    or not all(isinstance(row[k], str) for k in LESSON_KEYS - {"evidence"})
                    or not isinstance(row["evidence"], list) or not all(isinstance(i, str) for i in row["evidence"])):
                continue  # a row from an older version or a damaged line
            if domain is None or row["domain"] == domain:
                rows.append(row)
    return rows


def render_page(domain: str, rows: list[dict]) -> str:
    lines = [f"# Lessons: {domain}", "",
             "Written after projects in this domain were delivered or re-designed. Each lesson cites the blackboard "
             "entries it rests on (project id, entry id).", ""]
    by_project: dict[str, list[dict]] = {}
    for r in rows:
        by_project.setdefault((r["project"], r["outcome"]), []).append(r)
    for (project, _), items in reversed(by_project.items()):
        lines += [f"## {project} ({items[0]['outcome']}, {items[0]['at'][:10]})", ""]
        lines += [f"- **{r['kind']}**: {r['lesson']} ({', '.join(r['evidence'])})" for r in items] + [""]
    return "\n".join(lines)


_LESSONS_LOCK = threading.Lock()


def is_free_role(cfg) -> bool:
    """The historian role, its fallbacks and its escalation step all run on ARC (free), and it has no paid step."""
    role = cfg.roles.get(ROLE)
    steps = [role, cfg.escalations.get(ROLE)]
    return bool(role) and role.quality is None and all(r is None or r.provider == "arc" for r in steps)


def write_lessons(llm: LLM, project: Project, outcome: str, directory: Path) -> Path | None:
    """Ask for lessons once, keep the ones that cite real entries, and re-render the domain's page. None when the
    model gave nothing usable or nothing survived the check (a failure here never changes the run's result)."""
    if not is_free_role(llm.cfg) or not llm.cfg.data_public:
        return None  # a free by-product: never spend on it, and nothing goes to a shared page unless the data is public
    domain = domain_of(project)
    restricted = any(e.license == "arc_only" for e in project.board.entries())
    if restricted:  # lessons go to a machine-wide shared page; a project with ARC-only data teaches nothing there
        return None
    log = EventLog(llm.run_dir / FILE, llm.run_dir.name) if getattr(llm, "run_dir", None) else None
    try:
        found, _, _ = complete_validated(
            llm, ROLE, PROMPT.format(domain=domain, outcome=outcome, entries=_entries_text(project),
                                     events=_events_text(log.events() if log else []), n=MAX_LESSONS),
            SYSTEM, Lessons, tag="lessons", rescue=False)
    except Exception:  # noqa: BLE001 (no usable answer: no lessons this time)
        return None
    keep = checked(found, project)
    at = _now().isoformat()
    with _exclusive(directory / "lessons.lock", _LESSONS_LOCK) as held:
        if not held:
            return None  # another run is writing the lessons; skip rather than interleave
        known = {(r["domain"], r["lesson"].lower()) for r in read_lessons(directory)}
        new = [{"domain": domain, "project": project.id, "outcome": outcome, "at": at, **les.model_dump()}
               for les in keep if (domain, les.lesson.lower()) not in known]  # the same lesson twice adds nothing
        if not new:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lessons.jsonl").open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
        page = directory / f"{domain}.md"
        _write_atomic(page, render_page(domain, read_lessons(directory, domain)))
    return page


def lessons_after_build(llm: LLM, project: Project, status: str, workspace: Path) -> Path | None:
    """For the entry points (CLI, Interviewer page): lessons after a project is delivered or re-designed. Never raises,
    and a build that stopped for any other reason (a declined spend, the cap) teaches nothing."""
    outcome = {"delivered": "delivered", "redesign": "re-designed (not delivered)"}.get(status)
    if outcome is None:
        return None
    try:
        return write_lessons(llm, project, outcome, lessons_dir(workspace))
    except Exception:  # noqa: BLE001 (a by-product: it must not change the run's result)
        return None
