"""Prior-art scout: before the team builds from scratch, look for what already exists.

For a task (most often a coding task) this finds standard-library modules, packages on PyPI and public GitHub
repositories that already do part of the job, so the design can reuse or follow them. Code does nearly all of it:

1. One ARC call proposes search words, package names and standard-library modules. If it fails, code takes the
   task's most frequent words instead, so the scout never depends on the model.
2. Code checks every proposal against the source of truth: ``sys.stdlib_module_names`` for modules, the PyPI JSON
   for packages (an invented name returns nothing and is dropped), the GitHub search API for repositories.
3. Code applies fixed rules to each candidate: an open licence, a release or push within ``MAX_AGE_DAYS``, and
   whether the sandbox can import it. The sandbox runs generated tests with no network and only the packages in
   ``sandbox/Dockerfile`` (pytest), so a third-party package is ``reference`` (read it, follow its approach), not
   ``import``, unless it is listed in ``ASPIRE_SANDBOX_PACKAGES``. The standard library is always ``import``.

Nothing here is written by a model except the proposals, and each candidate carries its URL and the date it was
fetched, so a reader can check every line. Findings go to ``reuse.json`` and the blackboard (kind ``prior_art``);
the knowledge evaluator sees them, and the head orchestrator must say which it adopts.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from factory.llm import complete_validated
from factory.pipeline.sources import GitHubRepos, PyPI

MAX_AGE_DAYS = 3 * 365  # a release or push older than this is stale
MAX_QUERIES = 3
MAX_PACKAGES = 6
MAX_REPOS_PER_QUERY = 4
MAX_CANDIDATES = 12  # what the later stages are shown
OPEN_LICENSES = ("mit", "bsd", "apache", "isc", "psf", "python software foundation", "mpl", "unlicense", "cc0",
                 "zlib", "lgpl")
# a package's own licence field is free text; these are copyleft licences whose code must not be copied in
COPYLEFT = ("gpl", "agpl")
STOPWORDS = frozenset("""the and for with that this from into using use data file files code python write build make
create tool script program should will can then each when where which output input run runs also than are not any all
its has have been your you our one two new get set per via""".split())


class Proposal(BaseModel):
    queries: list[str] = []  # GitHub repository searches, a few plain words each
    packages: list[str] = []  # PyPI names the model believes already do part of the job
    stdlib: list[str] = []  # standard-library modules that do part of the job


class Candidate(BaseModel):
    kind: Literal["stdlib", "pypi", "github"]
    name: str
    url: str
    summary: str
    license: str = ""
    updated: str = ""  # date of the latest release or push
    stars: int | None = None
    use: Literal["import", "reference", "skip"]
    why: str  # the rule that decided `use`, in words
    fetched: str  # when the facts were fetched


class ReuseResult(BaseModel):
    candidates: list[Candidate] = []
    queries: list[str] = []
    errors: list[str] = []  # lookups that failed; a failure is recorded, never fatal

    def usable(self) -> list[Candidate]:
        return [c for c in self.candidates if c.use != "skip"]

    def render(self) -> str:
        """The candidates as a block for a prompt: every line is a fact fetched by code."""
        lines = []
        for c in self.usable():
            lines.append(f"- [{c.use}] {c.kind} {c.name}: {c.summary[:200]} (licence {c.license or 'unknown'}, "
                         f"updated {c.updated or '?'}; {c.why}) {c.url}")
        return "\n".join(lines)


PROPOSE_PROMPT = """A team of agents will build this. Before they build from scratch, name what already exists that \
does part of the job: standard-library modules, packages on PyPI, and a few words to search GitHub for similar tools \
or reference implementations. Name only things you are confident exist; code looks every name up and drops the ones \
that do not.

## Task
{task}

## Plan
{plan}

Reply with JSON only: {{"queries": ["<2 to 5 plain words>"], "packages": ["<PyPI name>"], "stdlib": ["<module>"]}}, \
at most {max_queries} queries and {max_packages} packages."""


def keywords(text: str, n: int = 6) -> str:
    """The most frequent content words of a text: the search the scout falls back on when the model gives none."""
    counts = Counter(w for w in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if len(w) > 3 and w not in STOPWORDS)
    return " ".join(w for w, _ in counts.most_common(n))


def sandbox_packages() -> set[str]:
    """Third-party packages the sandbox image can import (comma-separated, normalised). pytest is always there."""
    raw = os.environ.get("ASPIRE_SANDBOX_PACKAGES", "")
    return {"pytest", *(p.strip().lower().replace("_", "-") for p in raw.split(",") if p.strip())}


def judge_license(text: str) -> tuple[bool, str]:
    """(open, label). Unknown or copyleft is not open: its code is read, not copied."""
    low = (text or "").lower()
    if not low or low in ("noassertion", "other", "no licence", "none"):
        return False, "licence unknown"
    if any(c in low for c in COPYLEFT) and "lgpl" not in low:
        return False, f"copyleft licence ({text[:30]})"
    return any(o in low for o in OPEN_LICENSES), text[:40]


def is_recent(date: str, now: datetime) -> bool:
    try:
        when = datetime.fromisoformat(date.replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when > now - timedelta(days=MAX_AGE_DAYS)


class Scout:
    def __init__(self, llm, agent, pypi: PyPI, github: GitHubRepos, now=lambda: datetime.now(UTC)):
        self.llm, self.agent, self.pypi, self.github, self.now = llm, agent, pypi, github, now

    def propose(self, task: str, plan: str) -> Proposal:
        try:
            prompt = PROPOSE_PROMPT.format(task=task[:4000], plan=plan[:4000], max_queries=MAX_QUERIES,
                                           max_packages=MAX_PACKAGES)
            out, _, _ = complete_validated(self.llm, self.agent.role, prompt, self.agent.system, Proposal,
                                           tag="scout", data="public" if self.llm.cfg.data_public else "private")
        except Exception:  # noqa: BLE001 (the scout is an extra: a model failure falls back to plain keywords)
            out = Proposal()
        if not out.queries:
            out.queries = [keywords(task)] if keywords(task) else []
        return out

    def run(self, task: str, plan: str) -> ReuseResult:
        result = ReuseResult()
        prop = self.propose(task, plan)
        stamp = self.now()
        fetched = stamp.date().isoformat()
        seen: set[str] = set()
        for mod in dict.fromkeys(m.strip().split(".")[0] for m in prop.stdlib):
            if mod in sys.stdlib_module_names and not mod.startswith("_") and f"stdlib:{mod}" not in seen:
                seen.add(f"stdlib:{mod}")
                result.candidates.append(Candidate(
                    kind="stdlib", name=mod, url=f"https://docs.python.org/3/library/{mod}.html",
                    summary=f"Python standard library module {mod}.", license="PSF", use="import",
                    why="in the standard library, so the sandbox has it", fetched=fetched))
        have = sandbox_packages()
        for name in prop.packages[:MAX_PACKAGES]:
            try:
                data = self.pypi.lookup(name)
            except Exception as e:  # noqa: BLE001
                result.errors.append(f"pypi {name}: {type(e).__name__}: {str(e)[:120]}")
                continue
            info = (data or {}).get("info")
            if not info or f"pypi:{info['name'].lower()}" in seen:
                continue  # not on PyPI: an invented name is dropped here
            seen.add(f"pypi:{info['name'].lower()}")
            result.candidates.append(self._pypi_candidate(info, data, have, stamp, fetched))
        for q in prop.queries[:MAX_QUERIES]:
            result.queries.append(q)
            try:
                repos = self.github.repos(q, MAX_REPOS_PER_QUERY)
            except Exception as e:  # noqa: BLE001
                result.errors.append(f"github {q!r}: {type(e).__name__}: {str(e)[:120]}")
                continue
            for r in repos:
                if f"github:{r['full_name'].lower()}" in seen:
                    continue
                seen.add(f"github:{r['full_name'].lower()}")
                result.candidates.append(self._repo_candidate(r, stamp, fetched))
        order = {"import": 0, "reference": 1, "skip": 2}
        result.candidates = sorted(result.candidates, key=lambda c: (order[c.use], -(c.stars or 0)))[:MAX_CANDIDATES]
        return result

    def _pypi_candidate(self, info: dict, data: dict, have: set[str], now: datetime, fetched: str) -> Candidate:
        name = info["name"]
        files = data.get("urls") or []
        updated = (files[0].get("upload_time_iso_8601") or "")[:10] if files else ""
        lic_text = info.get("license_expression") or next(
            (c.split("::")[-1].strip() for c in info.get("classifiers", []) if c.startswith("License ::")), "") \
            or (info.get("license") or "")[:60]
        is_open, label = judge_license(lic_text)
        if not is_open:
            use, why = "skip", label
        elif not is_recent(updated, now):
            use, why = "skip", f"stale (last release {updated or 'unknown'})"
        elif name.lower().replace("_", "-") in have:
            use, why = "import", "installed in the sandbox"
        else:
            use, why = "reference", "not installed in the sandbox: follow its approach, do not import it"
        return Candidate(kind="pypi", name=name, url=f"https://pypi.org/project/{name}/",
                         summary=info.get("summary") or "", license=label, updated=updated, use=use, why=why,
                         fetched=fetched)

    def _repo_candidate(self, r: dict, now: datetime, fetched: str) -> Candidate:
        is_open, label = judge_license((r.get("license") or {}).get("spdx_id") or "")
        pushed = (r.get("pushed_at") or "")[:10]
        if not is_open:
            use, why = "skip", label
        elif not is_recent(pushed, now):
            use, why = "skip", f"stale (last push {pushed or 'unknown'})"
        else:
            use, why = "reference", "a public repository: read its approach, do not import it"
        return Candidate(kind="github", name=r["full_name"], url=r["html_url"], summary=r.get("description") or "",
                         license=label, updated=pushed, stars=r.get("stargazers_count"), use=use, why=why,
                         fetched=fetched)
