"""Read-only views of what ASPIRE keeps on disk, for the Streamlit inspection pages: the wiki, the projects'
blackboards, and the runs with their model calls and spend. Plain code: nothing here calls a model or writes a file
(the wiki's index is rewritten only by ``python -m factory wiki`` and by the librarian).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from factory.pipeline.blackboard import Project
from factory.pipeline.wiki import Wiki, WikiPage, enough_sources

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
FOOTNOTE_DEF = re.compile(r"^\[\^(\d+)\]:\s*(.+)$", re.M)
CITE_REF = re.compile(r"\[([^\]\[]+)\]\[\^(\d+)\]")  # "[doc id][^3]", as cite.render writes a statement
PAGE_LINK = re.compile(r"\[([^\]]+)\]\(([a-z0-9-]+)\.md\)")  # "[topic](slug.md)", a link to another wiki page


def md_safe(text: str) -> str:
    """Text for st.markdown, where a bare $ starts a formula."""
    return text.replace("$", "\\$")


# --------------------------------------------------------------------------
# The wiki
# --------------------------------------------------------------------------

def status(page: WikiPage) -> str:
    if not page.verified:
        return "not verified"
    return "verified" if enough_sources(page.sources) else "verified, too few sources"


def wiki_rows(pages: list[WikiPage]) -> list[dict]:
    return [{"topic": p.topic, "status": status(p), "sources": len(p.sources), "license": p.license,
             "projects": len(p.projects), "updated": p.updated[:16].replace("T", " "), "slug": p.slug}
            for p in pages]


def matches(page: WikiPage, query: str) -> bool:
    """Every word of the query appears in the page's topic, summary or body (ignoring case)."""
    text = f"{page.topic} {page.summary} {page.body}".lower()
    return all(w in text for w in query.lower().split())


@dataclass
class PageView:
    body: str  # markdown for st.markdown: citations as [doc id]ⁿ, wiki links as ?page=slug links
    quotes: list[tuple[int, str]] = field(default_factory=list)  # footnote number, the quote and where it is
    links: list[str] = field(default_factory=list)  # slugs this page links to that exist
    missing: list[str] = field(default_factory=list)  # slugs it links to that do not
    backlinks: list[str] = field(default_factory=list)  # slugs of pages that link here


def view(page: WikiPage, pages: list[WikiPage]) -> PageView:
    """The page body as Streamlit can show it. cite.render writes footnotes ([^n]) and links to other pages as
    ``slug.md``; the footnotes come out as a list of quotes, and the links point at this viewer instead."""
    slugs = {p.slug for p in pages}
    quotes = [(int(n), text.strip()) for n, text in FOOTNOTE_DEF.findall(page.body)]
    body = FOOTNOTE_DEF.sub("", page.body).rstrip()
    body = CITE_REF.sub(lambda m: f"\\[{m.group(1)}\\]{m.group(2).translate(SUPERSCRIPT)}", body)
    body = PAGE_LINK.sub(lambda m: f"[{m.group(1)}](?page={m.group(2)})" if m.group(2) in slugs else m.group(1),
                         body)
    return PageView(body=md_safe(body), quotes=quotes,
                    links=[s for s in page.links if s in slugs], missing=[s for s in page.links if s not in slugs],
                    backlinks=sorted(p.slug for p in pages if page.slug in p.links and p.slug != page.slug))


def wiki_log(wiki: Wiki, last: int = 50) -> list[str]:
    """The newest saves first, from log.md."""
    path = wiki.root / "log.md"
    if not path.exists():
        return []
    lines = [line.removeprefix("- ") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [PAGE_LINK.sub(r"[\1](?page=\2)", line) for line in lines[::-1][:last]]


# --------------------------------------------------------------------------
# Projects and their blackboards
# --------------------------------------------------------------------------

STAGES = [  # the entry kind that marks each stage done, in pipeline order
    ("approval", "Plan A approved"), ("decision", "plan decided"), ("knowledge", "knowledge gathered"),
    ("final", "built and delivered"), ("redesign", "re-designed"),
]


def projects(base: Path) -> list[Project]:
    """Every project folder, newest first (their names start with a UTC timestamp)."""
    if not base.exists():
        return []
    return [Project(p) for p in sorted(base.iterdir(), reverse=True) if (p / "blackboard.jsonl").exists()]


def stage(project: Project) -> str:
    kinds = {e.kind for e in project.board.entries()}
    done = [label for kind, label in STAGES if kind in kinds]
    return done[-1] if done else "interviewing"


def project_rows(items: list[Project]) -> list[dict]:
    rows = []
    for p in items:
        entries = p.board.entries()
        request = next((e.data.get("text", "") for e in entries if e.kind == "request"), "")
        rows.append({"project": p.id, "stage": stage(p), "entries": len(entries),
                     "arc_only": sum(e.license == "arc_only" for e in entries), "request": str(request)[:120]})
    return rows


def entry_summary(data: dict, width: int = 160) -> str:
    """One line for an entry: its most telling field, else its JSON."""
    for key in ("text", "topic", "title", "summary", "verdict", "reason", "path"):
        if isinstance(data.get(key), str) and data[key].strip():
            return " ".join(data[key].split())[:width]
    return json.dumps(data, ensure_ascii=False)[:width]


# --------------------------------------------------------------------------
# Runs, model calls and spend
# --------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a line cut short by a run that was stopped mid-write
    return out


def runs(base: Path, last: int = 200) -> list[Path]:
    """Run folders, newest first."""
    if not base.exists():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)[:last]


def tokens(call: dict) -> int:
    return sum(call.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"))


def call_totals(calls: list[dict]) -> dict:
    """What a run's llm_calls.jsonl adds up to. Cached replies and failed attempts are counted apart."""
    done = [c for c in calls if "event" not in c and "error" not in c and "fallback_to" not in c]
    return {"calls": sum(not c.get("cached") for c in done), "cached": sum(bool(c.get("cached")) for c in done),
            "errors": sum("error" in c for c in calls), "gates": sum(c.get("event") == "spend_gate" for c in calls),
            "real_usd": round(sum(c.get("real_usd", 0.0) for c in done), 4),
            "priced_usd": round(sum(c.get("priced_usd", 0.0) for c in done), 4),
            "tokens": sum(tokens(c) for c in done)}


def run_rows(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        t = call_totals(read_jsonl(d / "llm_calls.jsonl"))
        rows.append({"run": d.name, "model calls": t["calls"], "cached": t["cached"], "errors": t["errors"],
                     "paid USD": t["real_usd"], "if all were paid USD": t["priced_usd"],
                     "files": sorted(p.name for p in d.iterdir() if p.is_file())[:8]})
    return rows


def by_model(calls: list[dict]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for c in calls:
        if "event" in c or "fallback_to" in c:
            continue
        r = rows.setdefault((c.get("provider", "?"), c.get("model", "?")),
                            {"provider": c.get("provider", "?"), "model": c.get("model", "?"), "calls": 0,
                             "cached": 0, "errors": 0, "paid USD": 0.0})
        if "error" in c:
            r["errors"] += 1
        elif c.get("cached"):
            r["cached"] += 1
        else:
            r["calls"] += 1
            r["paid USD"] = round(r["paid USD"] + c.get("real_usd", 0.0), 4)
    return sorted(rows.values(), key=lambda r: (-r["paid USD"], -r["calls"]))


# --------------------------------------------------------------------------
# Across runs: what each stage does over time, and where the model time went
# --------------------------------------------------------------------------

def stage_tallies(items: list[Project]) -> list[dict]:
    """Per-stage rates across projects, counted from their blackboards by plain code: how often a plan is approved
    on the first proposal, how many documents a search finds and how many are approved, how often a page is
    verified, how many critic rounds a build takes, how often final QA approves at first sight. Together they
    show which stage is improving and which is flat."""
    plans = approvals = queries = found = approved = checks = verified = reviews = accepted = 0
    with_qa = first_pass = repaired = 0
    for p in items:
        entries = p.board.entries()
        kinds = [e.kind for e in entries]
        plans += kinds.count("plan")
        approvals += kinds.count("approval")
        for e in entries:
            if e.kind == "query" and "error" not in e.data:
                queries += 1
                found += len(e.data.get("found", []))
            elif e.kind == "source":
                approved += 1
            elif e.kind == "wiki_check":
                checks += 1
                verified += bool(e.data.get("accurate"))
            elif e.kind == "critic_review":
                reviews += 1
                accepted += bool(e.data.get("acceptable"))
        qa = [e for e in entries if e.kind == "final"]
        if qa:
            with_qa += 1
            first_pass += bool(qa[0].data.get("approved"))
            repaired += any(e.data.get("qa_repair") for e in entries if e.kind == "note")  # set by the QA repair loop

    def rate(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"
    return [
        {"stage": "interview", "measure": "plans approved per plan proposed", "value": rate(approvals, plans)},
        {"stage": "research", "measure": "documents found per search", "value": f"{found / queries:.1f}" if queries else "-"},
        {"stage": "research", "measure": "documents approved per search", "value": f"{approved / queries:.1f}" if queries else "-"},
        {"stage": "wiki", "measure": "pages verified per check", "value": rate(verified, checks)},
        {"stage": "critic", "measure": "reviews accepted per review", "value": rate(accepted, reviews)},
        {"stage": "final QA", "measure": "approved on first QA", "value": rate(first_pass, with_qa)},
        {"stage": "final QA", "measure": "projects that needed a QA repair", "value": rate(repaired, with_qa)},
    ]


def time_lost(calls: list[dict]) -> dict:
    """Where a run's model time went that produced nothing: replies cut off at max_tokens, refusals, and calls that
    failed (their seconds are not logged, so they are counted). From llm_calls.jsonl alone, no model."""
    done = [c for c in calls if "event" not in c and "error" not in c and "fallback_to" not in c and not c.get("cached")]
    total = sum(c.get("seconds", 0.0) for c in done)
    cut = [c for c in done if c.get("stop_reason") == "max_tokens"]
    refused = [c for c in done if c.get("stop_reason") == "refusal"]
    lost = sum(c.get("seconds", 0.0) for c in cut + refused)
    return {"model_seconds": round(total, 1), "lost_seconds": round(lost, 1),
            "lost_share": round(lost / total, 3) if total else 0.0, "cut_off": len(cut), "refused": len(refused),
            # a retried attempt and a fallback each have their own row, so they are counted apart from failed calls
            "failed_calls": sum("error" in c and "event" not in c and "fallback_to" not in c and "attempt" not in c
                                for c in calls),
            "retries": sum("attempt" in c for c in calls), "fallbacks": sum("fallback_to" in c for c in calls)}
