"""The knowledge loop: make sure the wiki holds what the plan needs before any work on the solution starts.

1. The knowledge evaluator (gpt-oss) reads the task and the refined plan and lists the topics the work needs.
2. The librarian checks the wiki in plain code: a topic is covered by a verified page with enough sources.
3. For each missing topic, the research orchestrator (GLM) writes searches; the researchers are plain-code
   connectors (sources.py), since searching an API needs no model. The orchestrator then approves or rejects each
   document it found, and runs a second search round when too few pass.
4. The librarian writes a wiki page from the approved documents, citing them. Code checks every citation names an
   approved document; the wiki verifier (DeepSeek, another family) checks every claim against the documents, and the
   librarian gets one rewrite when it finds errors. Only a verified page covers its topic.
5. The evaluator looks again, now with the new pages, and either approves or names what is still missing, so a broad
   first pass narrows to the specifics the work needs.

Stop rules: at most MAX_ROUNDS evaluations (the last only checks), MAX_TOPICS topics a round, RESEARCH_ROUNDS search rounds a topic,
MAX_QUERIES searches a round, and the connectors' shared call budget. The result is ``knowledge.json``: which topics
are covered, by which pages, and what is still missing.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.sources import CONNECTORS, Doc, Searcher
from factory.pipeline.wiki import MIN_SOURCES, Wiki, WikiPage
from factory.registry import Registry

EVALUATOR, ORCHESTRATOR, LIBRARIAN, VERIFIER = ("knowledge_evaluator", "research_orchestrator", "librarian",
                                                "wiki_verifier")
# MAX_ROUNDS evaluations, so at most MAX_ROUNDS - 1 research passes, each followed by a fresh evaluation
MAX_ROUNDS, MAX_TOPICS, RESEARCH_ROUNDS, MAX_QUERIES, PER_QUERY = 3, 6, 2, 3, 5


class Topic(BaseModel):
    name: str
    why: str
    needed_for: list[str] = []  # task ids in the plan
    hints: list[str] = []  # search terms to start from


class Needs(BaseModel):
    enough: bool  # the evaluator approves what the wiki holds for this task
    topics: list[Topic] = []  # what is still needed; empty when enough


class Query(BaseModel):
    connector: str
    query: str


class Queries(BaseModel):
    queries: list[Query]


class Verdict(BaseModel):
    id: str
    approve: bool
    why: str


class Review(BaseModel):
    verdicts: list[Verdict]


class Draft(BaseModel):
    summary: str
    body: str


class Check(BaseModel):
    accurate: bool
    issues: list[str] = []


class KnowledgeResult(BaseModel):
    ok: bool
    covered: dict[str, str]  # topic -> wiki page slug
    missing: list[str]
    rounds: int


EVALUATE_PROMPT = """A team of agents is about to carry out this plan. Decide what knowledge it needs that a capable \
engineer would not already know: facts, methods, data sources, file formats, project templates.

## Task
{task}

## Plan
{plan}

## Wiki pages already available
{pages}

{again}Reply with JSON only: {{"enough": true or false, "topics": [{{"name": "<short topic name>", "why": "<one \
sentence>", "needed_for": ["T1"], "hints": ["<search terms>"]}}]}}
List at most {max_topics} topics, most important first. Set "enough" to true, with no topics, only when the pages \
above cover everything the plan needs."""

QUERIES_PROMPT = """Write searches that will find sources for this topic.

## Topic
{topic}: {why}
Search hints: {hints}

## Search engines
- openalex: scholarly papers in every field
- europepmc: biomedical and life-science papers (PubMed, PubMed Central)
- ncbi_gene: genes by symbol or name, with organism, e.g. "BRCA1[sym] AND human[orgn]"
{previous}
Reply with JSON only: {{"queries": [{{"connector": one of {connectors}, "query": "<search>"}}]}}, at most \
{max_queries}."""

REVIEW_PROMPT = """Approve the documents that are reliable and relevant sources for this topic; reject the rest.

## Topic
{topic}: {why}

## Documents
{docs}

Reply with JSON only: {{"verdicts": [{{"id": "<document id>", "approve": true or false, "why": "<one sentence>"}}]}}"""

WRITE_PROMPT = """Write a wiki page on this topic using only the documents below. Cite the document behind each \
claim as [document id], e.g. [{example}]. Say what the documents do not settle instead of guessing.

## Topic
{topic}: {why}

## Documents
{docs}
{issues}
Reply with JSON only: {{"summary": "<one paragraph>", "body": "<markdown with citations>"}}"""

VERIFY_PROMPT = """Check this wiki page against its source documents. A claim is accurate only when the document it \
cites supports it.

## Page
{page}

## Documents
{docs}

Reply with JSON only: {{"accurate": true or false, "issues": ["<claim>: <what is wrong>"]}}"""


def show_docs(docs: list[Doc]) -> str:
    return "\n\n".join(f"### {d.id}\n{d.title} ({d.year})\n{d.text}" for d in docs)


def cited(body: str, ids: list[str]) -> list[str]:
    return [i for i in ids if f"[{i}]" in body]


class KnowledgeLoop:
    def __init__(self, llm: LLM, registry: Registry, project: Project, wiki: Wiki, searchers: dict[str, Searcher]):
        self.llm, self.project, self.board, self.wiki = llm, project, project.board, wiki
        self.agents = {n: registry.agents[n] for n in (EVALUATOR, ORCHESTRATOR, LIBRARIAN, VERIFIER)}
        self.searchers = searchers

    def _call(self, agent: str, prompt: str, model_cls, tag: str, docs: list[Doc] = ()):
        """A model call; any ARC-only document in the prompt keeps the call on ARC."""
        public = self.llm.cfg.data_public and all(d.license == "public" for d in docs)
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag,
                                       data="public" if public else "private")
        return out

    def run(self) -> KnowledgeResult:
        if self.board.latest("decision") is None:
            raise RuntimeError("the knowledge loop starts from the planning loop's plan; run that first")
        if self.board.latest("knowledge"):
            raise RuntimeError("this project's knowledge loop already ran")
        task = (self.project.root / "task.txt").read_text()
        plan = (self.project.root / "plan.json").read_text()
        covered: dict[str, str] = {}
        missing: list[str] = []
        for rnd in range(1, MAX_ROUNDS + 1):
            if rnd == 1:
                pages = self.wiki.pages()[:40]
                again = ""
            else:
                pages = [p for p in (self.wiki.find(t) for t in covered) if p]
                again = ("Research has run on the topics you asked for; the pages now available are listed. Name only "
                         "what is still missing, including specifics the new pages show the work depends on.\n\n")
            listing = "\n".join(f"- {p.topic}: {p.summary[:300]}" for p in pages) or "(none)"
            needs = self._call(EVALUATOR, EVALUATE_PROMPT.format(task=task, plan=plan, pages=listing, again=again,
                                                                 max_topics=MAX_TOPICS), Needs, f"knowledge-{rnd}")
            topics = needs.topics[:MAX_TOPICS]
            ev = self.board.post("topics", EVALUATOR, {"round": rnd, "enough": needs.enough,
                                                       "topics": [t.model_dump() for t in topics]})
            have = {t.name: self.wiki.covers(t.name) for t in topics}
            missing = [t.name for t in topics if have[t.name] is None]
            covered |= {name: page.slug for name, page in have.items() if page}
            self.board.post("coverage", LIBRARIAN, {"covered": {n: p.slug for n, p in have.items() if p},
                                                    "missing": missing}, refs=[ev.id])
            if not missing or rnd == MAX_ROUNDS:  # the last round only checks; it starts no new research
                break
            for topic in topics:
                if topic.name in missing and (page := self.research(topic)) and page.verified:
                    covered[topic.name] = page.slug
        ok = not missing
        result = KnowledgeResult(ok=ok, covered=covered, missing=missing, rounds=rnd)
        k = self.board.post("knowledge", EVALUATOR if ok else "code", result.model_dump())
        self.project.write_file("knowledge.json", json.dumps(result.model_dump(), indent=1), "code", refs=[k.id])
        return result

    # --- one topic: search, approve, write, verify ---

    def research(self, topic: Topic) -> WikiPage | None:
        approved: dict[str, Doc] = {}
        rejected: list[str] = []
        for rnd in range(1, RESEARCH_ROUNDS + 1):
            previous = (f"\nEarlier searches found too few usable documents (rejected: {'; '.join(rejected[-6:])}). "
                        "Try different terms or engines.\n") if rnd > 1 else ""
            queries = self._call(ORCHESTRATOR, QUERIES_PROMPT.format(
                topic=topic.name, why=topic.why, hints=", ".join(topic.hints) or "none", previous=previous,
                connectors=json.dumps(list(self.searchers)), max_queries=MAX_QUERIES), Queries,
                f"queries-{topic.name}-{rnd}").queries[:MAX_QUERIES]
            found: dict[str, Doc] = {}
            for q in queries:
                search = self.searchers.get(q.connector)
                if search is None:
                    self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "error": "unknown engine"})
                    continue
                try:
                    docs = search(q.query, PER_QUERY)
                except Exception as e:  # noqa: BLE001 (one engine failing is recorded, not fatal)
                    self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "error": f"{type(e).__name__}: {e}"})
                    continue
                self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "found": [d.id for d in docs]})
                found |= {d.id: d for d in docs if d.text.strip() and d.id not in approved}
            if not found:
                continue
            docs = list(found.values())
            review = self._call(ORCHESTRATOR, REVIEW_PROMPT.format(topic=topic.name, why=topic.why,
                                                                   docs=show_docs(docs)), Review,
                                f"review-{topic.name}-{rnd}", docs)
            verdicts = {v.id: v for v in review.verdicts if v.id in found}  # verdicts on unknown ids are ignored
            for d in docs:
                v = verdicts.get(d.id)
                if v and v.approve:
                    approved[d.id] = d
                    self.project.write_file(f"sources/{d.id.replace(':', '_')}.json", d.model_dump_json(indent=1),
                                            ORCHESTRATOR, license=d.license)
                    self.board.post("source", ORCHESTRATOR, {"id": d.id, "title": d.title, "url": d.url,
                                                             "topic": topic.name, "why": v.why}, license=d.license)
                else:
                    rejected.append(f"{d.title[:60]}: {v.why if v else 'no verdict'}")
            if len(approved) >= MIN_SOURCES:
                break
        if len(approved) < MIN_SOURCES:
            self.board.post("note", ORCHESTRATOR, {"topic": topic.name, "text": f"only {len(approved)} approved "
                                                   f"sources after {RESEARCH_ROUNDS} rounds"})
            return None
        return self.write_page(topic, list(approved.values()))

    def write_page(self, topic: Topic, docs: list[Doc]) -> WikiPage:
        ids = [d.id for d in docs]
        license = "arc_only" if any(d.license == "arc_only" for d in docs) else "public"
        issues: list[str] = []
        page = None
        for attempt in (1, 2):  # one rewrite when the checks find problems
            fix = f"\n## Fix these problems from the last draft\n{chr(10).join(issues)}\n" if issues else ""
            draft = self._call(LIBRARIAN, WRITE_PROMPT.format(topic=topic.name, why=topic.why, docs=show_docs(docs),
                                                              example=ids[0], issues=fix), Draft,
                               f"wiki-{topic.name}-{attempt}", docs)
            sources = cited(draft.body, ids)
            page = WikiPage(topic=topic.name, summary=draft.summary, body=draft.body, sources=sources,
                            citations={d.id: d.citation for d in docs if d.id in sources}, license=license,
                            projects=[self.project.id])
            issues = [] if len(sources) >= MIN_SOURCES else [f"cite at least {MIN_SOURCES} documents; cited "
                                                             f"{len(sources)}"]
            if not issues:
                check = self._call(VERIFIER, VERIFY_PROMPT.format(page=page.markdown(),
                                                                  docs=show_docs([d for d in docs
                                                                                  if d.id in sources])),
                                   Check, f"verify-{topic.name}-{attempt}", docs)
                self.board.post("wiki_check", VERIFIER, {"topic": topic.name, **check.model_dump()},
                                license=license)
                issues = [] if check.accurate else (check.issues or ["the verifier found the page inaccurate"])
            if not issues:
                break
        page.verified, page.issues = not issues, issues
        path = self.wiki.save(page)
        self.board.post("wiki_page", LIBRARIAN, {"topic": topic.name, "slug": page.slug, "path": str(path),
                                                 "verified": page.verified, "sources": page.sources},
                        license=license)
        return page


def default_searchers(cache_dir: Path) -> dict[str, Searcher]:
    from factory.pipeline.sources import connectors
    found = connectors(cache_dir)
    return {name: found[name] for name in CONNECTORS}
