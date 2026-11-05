"""The historian: explains a run's event log (and the logs of many runs) in plain words.

``factory/events.py`` is code and decides everything that is a fact: what happened, and whether it was corrected. The
historian is a model on a free ARC role that only writes the prose around those facts. It cannot change a status, and
what it writes is checked against the log before it is kept: an entry that names an event or fingerprint that does not
exist is dropped, and the status printed beside every entry comes from the log, not from the model. If the model gives
no usable answer, the report is still written, as the plain table.

    python -m factory.historian runs/<run>        # one run -> <run>/history.md
    python -m factory.historian --all             # every run -> <workspace>/history/meta.md (the meta analysis)
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

from pydantic import BaseModel

from factory.config import load_config
from factory.events import FILE, Event, EventLog, Group, meta_analysis, run_events, summary
from factory.llm import LLM, new_run_dir
from factory.llm.client import complete_validated

ROLE = "historian"
MAX_GROUPS = 40  # what one prompt carries; the table in the report always lists them all

SYSTEM = ("You are the historian of an automated system. You are given its error log as plain code recorded it. "
          "Explain it to a person who was not watching: what went wrong, in plain words, and, only where the log says an "
          "event was corrected, how it was fixed and why that fix makes sense. Use only the facts in the log; never claim "
          "something was fixed that the log shows as open; refer to events only by the ids given. Reply with JSON only.")

RUN_PROMPT = """Error log of one run, grouped by problem. Each group lists its event ids, its status (from code) and, if
corrected, what corrected it (from code).

{groups}

Reply with JSON: {{"summary": "two or three plain sentences on how the run went",
"entries": [{{"ids": ["E0001"], "title": "short title", "what_happened": "plain words",
"why_it_happened": "likely cause, or 'not known' when the log does not say",
"how_it_was_fixed": "for corrected events: the explanation, grounded in the 'corrected by' text; else empty"}}]}}"""

META_PROMPT = """Error groups across {runs} runs. 'returned after fix' means it was corrected in an earlier run and happened
again later. Counts and statuses come from code.

{groups}

Reply with JSON: {{"summary": "three or four plain sentences on the overall picture",
"patterns": [{{"fingerprints": ["abc123"], "finding": "what the pattern is", "suggestion": "what a person could do about it"}}]}}"""


class Entry(BaseModel):
    ids: list[str]
    title: str
    what_happened: str
    why_it_happened: str = ""
    how_it_was_fixed: str = ""


class RunStory(BaseModel):
    summary: str
    entries: list[Entry] = []


class Pattern(BaseModel):
    fingerprints: list[str]
    finding: str
    suggestion: str = ""


class MetaStory(BaseModel):
    summary: str
    patterns: list[Pattern] = []


def _by_problem(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {}
    for e in events:
        out.setdefault(e.fingerprint, []).append(e)
    return out


def _table(events: list[Event]) -> list[str]:
    rows = ["| id | stage | severity | status | message | corrected by |", "|---|---|---|---|---|---|"]
    for e in events:
        rows.append(f"| {e.id} | {e.stage} | {e.severity} | **{e.status}** | {e.message[:140].replace('|', '/')} | "
                    f"{e.how.replace('|', '/')} |")
    return rows


def run_report(events: list[Event], story: RunStory | None, run: str) -> str:
    s = summary(events)
    known = {e.id: e for e in events}
    lines = [f"# History of run {run}", "",
             f"{s['events']} event(s): **{s['corrected']} corrected**, **{s['open']} not corrected**.", ""]
    if story:
        lines += [story.summary.strip(), ""]
        shown = set()
        for en in story.entries:
            ids = [i for i in en.ids if i in known]  # an id that is not in the log is dropped
            if not ids:
                continue
            shown.update(ids)
            status = "corrected" if all(known[i].status == "corrected" for i in ids) else (
                "not corrected" if all(known[i].status == "open" for i in ids) else "partly corrected")
            lines += [f"## {en.title.strip()}  ({status}; {', '.join(ids)})", "", en.what_happened.strip(), ""]
            if en.why_it_happened.strip():
                lines += [f"*Why:* {en.why_it_happened.strip()}", ""]
            # an explanation of a fix is printed only if code says every one of these events was corrected
            if status == "corrected" and en.how_it_was_fixed.strip():
                lines += [f"*How it was fixed:* {en.how_it_was_fixed.strip()}",
                          f"*Corrected by (recorded by code):* {'; '.join(sorted({known[i].how for i in ids}))}", ""]
        missed = [e for e in events if e.id not in shown]
        if missed:
            lines += ["## Not explained above", ""] + _table(missed) + [""]
    lines += ["## Every event (from the log)", ""] + _table(events) + [""]
    return "\n".join(lines)


def _group_text(groups: dict[str, list[Event]]) -> str:
    out = []
    for es in list(groups.values())[:MAX_GROUPS]:
        how = sorted({e.how for e in es if e.how})
        out.append(f"- ids {', '.join(e.id for e in es)} | stage {es[0].stage} | {es[0].severity} | status "
                   f"{', '.join(sorted({e.status for e in es}))} | {es[0].message[:300]}"
                   + (f" | corrected by: {'; '.join(how)}" if how else ""))
    return "\n".join(out)


def write_run_history(llm: LLM, run_dir: Path) -> Path | None:
    """<run_dir>/history.md from <run_dir>/events.jsonl. None when the run met no errors."""
    events = EventLog(run_dir / FILE, run_dir.name).events()
    if not events:
        return None
    story = None
    with contextlib.suppress(Exception):  # no usable answer: the plain table is still written
        story, _, _ = complete_validated(llm, ROLE, RUN_PROMPT.format(groups=_group_text(_by_problem(events))), SYSTEM,
                                         RunStory, tag="historian-run", rescue=False)
    path = run_dir / "history.md"
    path.write_text(run_report(events, story, run_dir.name), encoding="utf-8")
    return path


def meta_report(groups: list[Group], story: MetaStory | None, total: dict) -> str:
    known = {g.fingerprint: g for g in groups}
    lines = ["# Meta analysis of the event log", "",
             f"{total['events']} event(s) in {total['runs']} run(s): {total['corrected']} corrected, {total['open']} "
             f"not corrected, {len(groups)} distinct problem(s).", ""]
    if story:
        lines += [story.summary.strip(), ""]
        for p in story.patterns:
            fps = [f for f in p.fingerprints if f in known]  # a fingerprint that is not in the log is dropped
            if fps:
                lines += [f"- **{p.finding.strip()}** ({', '.join(fps)})"
                          + (f" Suggestion: {p.suggestion.strip()}" if p.suggestion.strip() else "")]
        lines.append("")
    returned = [g for g in groups if g.returned_after_fix]
    if returned:
        lines += ["## Corrected once, then came back", ""] + [
            f"- `{g.fingerprint}` {g.stage}: {g.message[:140]} ({g.count} times, {len(g.runs)} runs)" for g in returned] + [""]
    lines += ["## Every problem, most frequent first", "",
              "| fingerprint | source | stage | times | runs | corrected | open | corrected by | message |",
              "|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g.fingerprint} | {g.source} | {g.stage} | {g.count} | {len(g.runs)} | {g.corrected} | {g.open} "
                     f"| {'; '.join(g.how)[:80].replace('|', '/')} | {g.message[:120].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def write_meta_history(llm: LLM, runs_base: Path, out: Path) -> Path | None:
    events = run_events(runs_base)
    if not events:
        return None
    groups = meta_analysis(events)
    story = None
    text = "\n".join(
        f"- {g.fingerprint} | {g.source}/{g.stage} | {g.count} times in {len(g.runs)} runs | {g.corrected} corrected, "
        f"{g.open} open{' | RETURNED AFTER FIX' if g.returned_after_fix else ''} | {g.message[:200]}"
        + (f" | corrected by: {'; '.join(g.how)[:200]}" if g.how else "") for g in groups[:MAX_GROUPS])
    with contextlib.suppress(Exception):
        story, _, _ = complete_validated(llm, ROLE, META_PROMPT.format(runs=len({e.run for e in events}), groups=text),
                                         SYSTEM, MetaStory, tag="historian-meta", rescue=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(meta_report(groups, story, summary(events)), encoding="utf-8")
    return out


ASK_PROMPT = """Question from a person: {question}

{scope}. Every event in the log, as code recorded it (status and 'corrected by' come from code):

{events}

Answer from this log only. If the log does not say, say that it does not. Reply with JSON:
{{"answer": "plain words, a few sentences", "cited": ["E0001"]}}
'cited' lists the event ids your answer rests on, from the ids above."""

ASK_LIMIT = 80  # events one question carries; newest first, and the answer says when older ones were left out


class Answer(BaseModel):
    answer: str
    cited: list[str] = []


def _words(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(w) > 3}


def ask(llm: LLM, question: str, events: list[Event], *, scope: str = "This run") -> tuple[str, list[Event]]:
    """Answer a question about the event log: (answer text, the events it cites).

    Grounded the same way as the reports: the model sees only the log, an id it cites that is not in the log is dropped,
    the cited events are returned from the log (status and fix come from code), and an answer that cites nothing real
    is replaced by the events that best match the question's words. A log with no events needs no model."""
    if not events:
        return "No events were recorded, so there is nothing to answer from.", []
    known = {e.id: e for e in events}
    shown = sorted(events, key=lambda e: e.at, reverse=True)[:ASK_LIMIT]
    lines = "\n".join(f"- {e.id} | run {e.run} | {e.at[:19]} | {e.source}/{e.stage} | {e.severity} | **{e.status}** | "
                      f"{e.message[:300]}" + (f" | corrected by: {e.how}" if e.how else "") for e in shown)
    note = f" ({len(events) - len(shown)} older event(s) were left out)" if len(events) > len(shown) else ""
    text = None
    cited: list[str] = []
    with contextlib.suppress(Exception):  # no usable answer: the matching events are still shown
        done, _, _ = complete_validated(llm, ROLE, ASK_PROMPT.format(question=question, scope=scope, events=lines),
                                        SYSTEM, Answer, tag="historian-ask", rescue=False)
        text, cited = done.answer.strip(), [i for i in done.cited if i in {e.id for e in shown}]
    if text and cited:
        return text + note, [known[i] for i in dict.fromkeys(cited)]
    want = _words(question)
    hits = sorted(shown, key=lambda e: -len(want & _words(f"{e.stage} {e.message} {e.how}")))
    hits = [e for e in hits if want & _words(f"{e.stage} {e.message} {e.how}")][:5]
    why = "The historian gave no answer the log backs up" if not text else (
        "The historian's answer cited no event in the log, so it is not shown")
    return (f"{why}. These are the events that best match your question." if hits else
            f"{why}, and no event matches your question."), hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.historian", description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", help="a run folder: writes its history.md")
    ap.add_argument("--all", action="store_true", help="every run: writes the meta analysis")
    ap.add_argument("--config", default="configs/factory.yaml")
    args = ap.parse_args(argv)
    if not args.run and not args.all:
        ap.error("give a run folder or --all")
    cfg = load_config(args.config)
    llm = LLM(cfg, new_run_dir(cfg, "historian"))
    if args.all:
        path = write_meta_history(llm, cfg.root / cfg.runs_dir, cfg.work_dir / "history" / "meta.md")
    else:
        path = write_run_history(llm, Path(args.run))
    print(path or "No events recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
