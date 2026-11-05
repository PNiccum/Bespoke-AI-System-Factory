"""Precedent: match a new request to projects ASPIRE already delivered, in plain code (no model call).

A request is reduced to a fingerprint: its content keywords (lower-cased, common words dropped, plurals and endings
trimmed) and a hash of the sorted set. Two requests with the same keywords get the same hash (an exact match); two
that share most of them score high on keyword overlap (a close match). Before planning, code scans the workspace for
projects whose final QA approved and whose deliverable exists, scores each against the new request, and offers the best
few to the planner as a starting template: what was asked, how it was designed and laid out, and where the
deliverable is. The planner may follow it, adapt it or ignore it; nothing here changes a plan.

Only the request is fingerprinted, on both sides, because it is the one text that exists at every stage (the Interviewer
has nothing else to match on). Whether a project was delivered is read from its blackboard (a ``final`` entry that
code set to approved, see factory/pipeline/final.py), the same fact the event log records as ``delivered``.

A match from a project holding ARC-only entries (lab data) is marked ARC-only on the new project's blackboard, so
``Blackboard.for_paid`` keeps it away from Claude and OpenAI.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from pydantic import BaseModel

from factory.pipeline.blackboard import License, Project

MIN_SCORE = 0.4  # below this a project is not offered
MIN_SHARED = 2  # keywords two requests must share, whatever the score
MAX_OFFERED = 3

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either else few for from further get give got had has
have having he her here hers him his how i if in into is it its just like make makes may me might more most must my need
needs no nor not now of off on once one only or other our out over own please program programs same script scripts she
should so some such than that the their them then there these they this those through to too tool tools under until up
use used using very want wants was we were what when where which while who whom why will with would write writes you your
build builds create creates python code file files able something thing things new simple small""".split())


class Fingerprint(BaseModel):
    keywords: list[str]  # sorted, deduplicated
    hash: str  # of the sorted keywords: equal requests (in content words) have equal hashes

    @property
    def words(self) -> frozenset[str]:
        return frozenset(self.keywords)


def stem(word: str) -> str:
    """Trim a plural or an ending so "spikes", "spiking" and "spiked" meet. Crude on purpose: it only has to be the
    same on both sides."""
    for end, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(end) and len(word) - len(end) + len(repl) >= 3:
            if end == "es" and not re.search(r"(ch|sh|ss|x|z|o)es$", word):
                continue  # "databases" -> "database", not "databas"
            if end == "s" and word.endswith(("ss", "us", "is")):
                continue  # "class", "status", "analysis" keep their s
            return word[: len(word) - len(end)] + repl
    return word


def fingerprint(text: str) -> Fingerprint:
    words = {stem(w) for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()) if w not in STOPWORDS}
    ordered = sorted(words)
    return Fingerprint(keywords=ordered, hash=hashlib.sha256("|".join(ordered).encode()).hexdigest()[:16])


def similarity(a: Fingerprint, b: Fingerprint) -> tuple[float, list[str]]:
    """(score, shared keywords). The mean of Jaccard (how much of everything the two share) and the overlap
    coefficient (how much of the shorter one the longer one has), so a long request that contains a short one still
    scores, and two requests that merely share a common word do not."""
    shared = sorted(a.words & b.words)
    if a.hash == b.hash and a.keywords:
        return 1.0, shared
    if len(shared) < MIN_SHARED:
        return 0.0, shared
    jaccard = len(shared) / len(a.words | b.words)
    overlap = len(shared) / min(len(a.words), len(b.words))
    return round((jaccard + overlap) / 2, 3), shared


class Precedent(BaseModel):
    project: str
    score: float
    exact: bool
    shared: list[str]
    request: str
    title: str = ""
    goal: str = ""
    language: str = ""
    entry_point: str = ""
    run_command: str = ""
    layout: list[str] = []
    deliverable: str = ""  # path of the zip
    files: list[str] = []  # what the zip holds under output (not the aspire/ provenance)
    arc_only: bool = False  # the project holds ARC-only entries


def request_of(project: Project) -> str:
    first = project.board.entries("request")
    return str(first[0].data.get("text", "")) if first else ""


def delivered(project: Project) -> Path | None:
    """The deliverable zip when final QA approved this project and the file still exists."""
    final = project.board.latest("final")
    if final is None or not final.data.get("approved") or not final.data.get("deliverable"):
        return None
    path = Path(final.data["deliverable"])
    if not path.is_absolute():
        path = project.root / path
    return path if path.is_file() else None


def _summary(project: Project, zip_path: Path, score: float, shared: list[str], request: str) -> Precedent:
    entries = project.board.entries()
    plan = next((e for e in reversed(entries) if e.kind == "plan"), None)
    plan_data = (plan.data.get("plan") if plan else None) or {}
    design = project.board.latest("design")
    d = design.data if design else {}
    try:
        with zipfile.ZipFile(zip_path) as z:
            files = [n for n in z.namelist() if not n.startswith("aspire/")][:40]
    except (zipfile.BadZipFile, OSError):
        files = []
    return Precedent(project=project.id, score=score, exact=score == 1.0, shared=shared, request=request[:500],
                     title=str(plan_data.get("title", "")), goal=str(plan_data.get("goal", "")),
                     language=str(d.get("language", "")), entry_point=str(d.get("entry_point", "")),
                     run_command=str(d.get("run_command", "")), layout=[str(x) for x in d.get("layout", [])][:20],
                     deliverable=str(zip_path), files=files,
                     arc_only=any(e.license == "arc_only" for e in entries))


def find(base: Path, request: str, *, exclude: str = "", limit: int = MAX_OFFERED,
         min_score: float = MIN_SCORE) -> list[Precedent]:
    """The delivered projects under ``base`` closest to ``request``, best first (ties go to the newest). Pure code."""
    mine = fingerprint(request)
    if not mine.keywords or not base.exists():
        return []
    found = []
    for root in sorted(base.iterdir(), reverse=True):  # newest first, so a tie keeps the newest
        if root.name == exclude or not (root / "blackboard.jsonl").exists():
            continue
        project = Project(root)
        zip_path = delivered(project)
        text = request_of(project)
        if zip_path is None or not text:
            continue
        score, shared = similarity(mine, fingerprint(text))
        if score >= min_score:
            found.append(_summary(project, zip_path, score, shared, text))
    return sorted(found, key=lambda p: -p.score)[:limit]


def render(found: list[Precedent]) -> str:
    """The offer as a block for a prompt. Empty when nothing matched."""
    if not found:
        return ""
    lines = ["## Earlier delivered projects with a similar request (matched by code on shared keywords)",
             "Each one passed final QA. Use one as a starting template where it fits: reuse its design and layout, "
             "change what this request differs in, and say so. Ignore it where it does not fit."]
    for i, p in enumerate(found, 1):
        lines.append(f"\n{i}. {p.title or p.project} ({'same keywords' if p.exact else f'match {p.score:.2f}'}; "
                     f"shared: {', '.join(p.shared[:12])})")
        lines.append(f"   asked: {' '.join(p.request.split())[:300]}")
        if p.goal and not p.arc_only:
            lines.append(f"   goal: {p.goal[:300]}")
        if p.language or p.run_command:
            lines.append(f"   built in {p.language or '?'}; entry {p.entry_point or '?'}; run: {p.run_command or '?'}")
        if p.layout:
            lines.append("   layout: " + "; ".join(x[:100] for x in p.layout[:10]))
        lines.append(f"   deliverable: {p.deliverable}")
    return "\n".join(lines)


def offer(project: Project, request: str) -> str:
    """Look once for precedents and record the result on ``project``'s blackboard (kind ``precedent``, written by
    code, an empty list when nothing matched). Returns the prompt block. A second call returns the recorded offer."""
    done = project.board.latest("precedent")
    if done is None:
        request = request_of(project) or request  # the request is the text every stage can match on
        mine = fingerprint(request)
        found = find(project.root.parent, request, exclude=project.id)
        lic: License = "arc_only" if any(p.arc_only for p in found) else "public"
        done = project.board.post("precedent", "code", {"fingerprint": mine.model_dump(),
                                                        "matches": [p.model_dump() for p in found]}, license=lic)
    return render([Precedent.model_validate(m) for m in done.data["matches"]])
