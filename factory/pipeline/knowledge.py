"""The knowledge loop: make sure the wiki holds what the plan needs before any work on the solution starts.

1. The knowledge evaluator (gpt-oss) reads the task and the refined plan and lists the topics the work needs.
2. The librarian checks the wiki in plain code: a topic is covered by a verified page with enough sources.
3. For each missing topic, the research orchestrator (GLM) writes searches; the researchers are plain-code
   connectors (sources.py), since searching an API needs no model. The orchestrator then approves or rejects each
   document it found, and runs a second search round when too few pass.
4. The librarian writes a wiki page from the approved documents, tagging every sentence with a verbatim quote, an
   inference or a limit (cite.py, from Bespoke_LLM). Code drops every sentence whose quote is not in its document;
   the wiki verifier (DeepSeek, another family) then checks that each quote supports its sentence, and the librarian
   gets one rewrite when the checks fail. Only a verified page covers its topic.
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
from factory.pipeline import cite
from factory.pipeline.blackboard import Project
from factory.pipeline.sources import CONNECTORS, Doc, Searcher
from factory.pipeline.wiki import MIN_SOURCES, Wiki, WikiPage, enough_sources
from factory.pipeline.wiki import words as wiki_words
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


class Dropped(BaseModel):
    name: str
    why: str  # why the work can go ahead without it: an explicit risk decision, recorded


class Needs(BaseModel):
    enough: bool  # the evaluator approves what the wiki holds for this task
    topics: list[Topic] = []  # what is still needed; empty when enough
    dropped: list[Dropped] = []  # earlier topics it no longer needs, each with a reason


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
    deferred: list[str] = []  # first asked for in the last round, which researches nothing: left to the critic loop
    dropped: dict[str, str] = {}  # topic -> why the evaluator let it go without a page


EVALUATE_PROMPT = """A team of agents is about to carry out this plan. Decide what knowledge it needs that a capable \
engineer would not already know: facts, methods, data sources, file formats, project templates. Leave out general \
software engineering (packaging, command-line parsing, HTTP clients, caching and retries, testing tools, mocking) and \
textbook facts any practitioner knows (a reverse complement, a well-known identifier such as human taxonomy ID 9606, \
scoring and tie-breaking rules the plan can simply state), and rules the task or plan sets for itself (an output \
format, how to tokenise or validate input, a stopword list, escaping, error handling, window or bucket sizes): the \
builders already know or are told them, and research cannot add to them. Ask for the domain knowledge the task turns \
on: facts about the outside world, its data sources and its file formats.

## Task
{task}

## Plan
{plan}

## Wiki pages already available
{pages}

{again}Reply with JSON only: {{"enough": true or false, "topics": [{{"name": "<short topic name>", "why": "<one \
sentence>", "needed_for": ["T1"], "hints": ["<search terms>"]}}], "dropped": [{{"name": "<earlier topic>", "why": \
"<why the work can go ahead without it>"}}]}}
List at most {max_topics} topics, most important first. When a topic is the same as one asked for earlier, use the \
earlier name exactly, so the work already done on it counts. Set "enough" to true, with no topics, only when the pages \
above cover everything the plan needs."""

QUERIES_PROMPT = """Write searches that will find sources for this topic.

## Topic
{topic}: {why}
Search hints: {hints}

## Search engines
- openalex: scholarly papers in every field
- europepmc: biomedical and life-science papers (PubMed, PubMed Central)
- ncbi_gene: genes by symbol or name, with organism, e.g. "BRCA1[sym] AND human[orgn]"
- docs: reference manuals and API documentation (NCBI E-utilities, RefSeq, Primer3, Biopython, UniProt, Europe \
PMC, Human Protein Atlas, Gene Ontology, ChEMBL, OpenAlex, Apache logs, Python csv/codecs/datetime/json, Markdown \
tables, pytest and pyproject packaging); use it for how a \
tool, API or format works, and keyword queries such as "efetch rettype fasta nucleotide"
{previous}
Reply with JSON only: {{"queries": [{{"connector": one of {connectors}, "query": "<search>"}}]}}, at most \
{max_queries}."""

REVIEW_PROMPT = """Approve the documents that are reliable and relevant sources for this topic; reject the rest.

## Topic
{topic}: {why}

## Documents
{docs}

Reply with JSON only: {{"verdicts": [{{"id": "<document id>", "approve": true or false, "why": "<one sentence>"}}]}}"""

WRITE_PROMPT = """Write a wiki page on this topic using only the documents below. End every sentence with exactly \
one tag:
- a fact from a document: {{{{q:<document id>|<a short verbatim quote from that document that states it>}}}}, e.g. \
{{{{q:{example}|...}}}}
- a conclusion you draw from the facts just before it in the same paragraph: {{{{inference}}}}
- something the documents do not settle: {{{{limit}}}}
Code checks every quote against the document word for word and drops any sentence whose quote is not there. An \
inference may not add numbers or names that its facts and their quotes lack. Use headings and bullet lists where they \
help; a short list label such as "Inputs:" needs no tag. To point to another wiki page, write [[<slug>]]:
{pages}

## Topic
{topic}: {why}

## Documents
{docs}
{issues}
Reply with JSON only: {{"summary": "<one paragraph, no tags>", "body": "<markdown, every sentence tagged>"}}"""

VERIFY_PROMPT = """Check this wiki page against its source documents. Code has already checked that every quoted \
passage is really in its document; check that each quote supports the sentence resting on it, and that every \
inference follows from the facts before it. A sentence is accurate only when its evidence supports it.

## Page
{page}

## Each fact and the quote it rests on
{quotes}

## Documents
{docs}

Reply with JSON only: {{"accurate": true or false, "issues": ["<claim>: <what is wrong>"]}}"""


def same_topic(name: str, earlier: list[str]) -> str:
    """The earlier name of a topic the evaluator has reworded (most of the words shared), else the name as given. The
    first live run renamed its topics between rounds, so the pages written for the first names never counted."""
    new = wiki_words(name)
    best = max(earlier, key=lambda e: len(new & wiki_words(e)) / max(len(new | wiki_words(e)), 1), default=None)
    if best and new and len(new & wiki_words(best)) / len(new | wiki_words(best)) >= 0.5:
        return best
    return name


def show_docs(docs: list[Doc]) -> str:
    return "\n\n".join(f"### {d.id}\n{d.title} ({d.year})\n{d.text}" for d in docs)


def page_list(pages: dict[str, str]) -> str:
    return "\n".join(f"- [[{slug}]] {topic}" for slug, topic in sorted(pages.items())[:40]) or "(no other pages yet)"


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

    def run(self, again: bool = False) -> KnowledgeResult:
        """``again`` reruns a project whose knowledge is already approved; the new result is the one build reads."""
        if self.board.latest("decision") is None:
            raise RuntimeError("the knowledge loop starts from the planning loop's plan; run that first")
        if (k := self.board.latest("knowledge")) and k.data.get("ok") and not again:
            raise RuntimeError("this project's knowledge is already approved (rerun with --again)")
        task = (self.project.root / "task.txt").read_text(encoding="utf-8")
        plan = (self.project.root / "plan.json").read_text(encoding="utf-8")
        covered: dict[str, str] = {}
        missing: list[str] = []
        asked: list[str] = []  # every topic name asked for so far, so a reworded topic keeps its first name
        deferred: list[str] = []
        dropped: dict[str, str] = {}
        for rnd in range(1, MAX_ROUNDS + 1):
            if rnd == 1:  # only verified pages: one the verifier rejected must not count toward "enough"
                pages = [p for p in self.wiki.pages() if p.verified][:40]
                again = ""
            else:
                pages = [p for p in (self.wiki.find(t) for t in covered) if p]
                still = [n for n in asked if n not in covered and n not in dropped]
                again = ("Research has run on the topics you asked for; the pages now available are listed. Name only "
                         "what is still missing, including specifics the new pages show the work depends on.\n"
                         f"Topics asked for earlier: {'; '.join(asked)}\n"
                         + (f"Still without a verified page: {'; '.join(still)}. Each stays missing until a page covers "
                            "it or you list it under \"dropped\" with why the work can go ahead without it.\n"
                            if still else "") + "\n")
            listing = "\n".join(f"- {p.topic}: {p.summary[:300]}" for p in pages) or "(none)"
            needs = self._call(EVALUATOR, EVALUATE_PROMPT.format(task=task, plan=plan, pages=listing, again=again,
                                                                 max_topics=MAX_TOPICS), Needs, f"knowledge-{rnd}")
            topics = [t.model_copy(update={"name": same_topic(t.name, asked)}) for t in needs.topics[:MAX_TOPICS]]
            if rnd == MAX_ROUNDS and rnd > 1:  # the last round researches nothing, so a topic new here cannot be met
                new = [t.name for t in topics if t.name not in asked]
                deferred = [n for n in new if self.wiki.covers(n) is None]
                topics = [t for t in topics if t.name not in deferred]
            asked += [t.name for t in topics if t.name not in asked]
            for d in needs.dropped:
                name = same_topic(d.name, asked)
                if name in asked and name not in {t.name for t in topics}:
                    dropped[name] = d.why
            ev = self.board.post("topics", EVALUATOR, {"round": rnd, "enough": needs.enough,
                                                       "topics": [t.model_dump() for t in topics],
                                                       "deferred": deferred, "dropped": dropped})
            have = {t.name: self.wiki.covers(t.name) for t in topics}
            # an earlier topic stays missing until a page covers it or the evaluator drops it with a reason: the
            # 2026-11-03 primer run lost an unresearched topic by simply not naming it again
            missing = [t.name for t in topics if have[t.name] is None]
            missing += [n for n in asked if n not in missing and n not in have and n not in dropped
                        and n not in covered and self.wiki.covers(n) is None]
            covered |= {name: page.slug for name, page in have.items() if page}
            if needs.enough and not topics:  # approved on the pages shown, so those are what the builders get
                covered |= {p.topic: p.slug for p in pages}
            self.board.post("coverage", LIBRARIAN, {"covered": {n: p.slug for n, p in have.items() if p},
                                                    "missing": missing}, refs=[ev.id])
            if not missing or rnd == MAX_ROUNDS:  # the last round only checks; it starts no new research
                break
            if not any(t.name in missing for t in topics):  # only earlier topics, already researched: no new work
                break
            for topic in topics:
                if topic.name in missing and (page := self.research(topic)) and page.verified:
                    covered[topic.name] = page.slug
        ok = not missing
        result = KnowledgeResult(ok=ok, covered=covered, missing=missing, rounds=rnd, deferred=deferred,
                                 dropped=dropped)
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
            if enough_sources(approved):
                break
        if not enough_sources(approved):
            self.board.post("note", ORCHESTRATOR, {"topic": topic.name, "text": f"only {len(approved)} approved "
                                                   f"sources after {RESEARCH_ROUNDS} rounds"})
            return None
        return self.write_page(topic, list(approved.values()))

    def write_page(self, topic: Topic, docs: list[Doc]) -> WikiPage:
        ids = [d.id for d in docs]
        license = "arc_only" if any(d.license == "arc_only" for d in docs) else "public"
        issues: list[str] = []
        page = None
        # pages it may link to; an ARC-only page is listed only for an ARC-only page, so no link leaks its topic
        others = {p.slug: p.topic for p in self.wiki.pages() if p.verified and p.topic != topic.name
                  and (license == "arc_only" or p.license == "public")}
        for attempt in (1, 2):  # one rewrite when the checks find problems
            fix = f"\n## Fix these problems from the last draft\n{chr(10).join(issues)}\n" if issues else ""
            draft = self._call(LIBRARIAN, WRITE_PROMPT.format(topic=topic.name, why=topic.why, docs=show_docs(docs),
                                                              example=ids[0], issues=fix, pages=page_list(others)),
                               Draft, f"wiki-{topic.name}-{attempt}", docs)
            paragraphs = cite.parse(draft.body)
            cite.check(paragraphs, {d.id: d.text for d in docs})
            body = cite.render(paragraphs, others)
            sources = cited(body, ids)
            page = WikiPage(topic=topic.name, summary=draft.summary, body=body, sources=sources,
                            citations={d.id: d.citation for d in docs if d.id in sources}, license=license,
                            projects=[self.project.id], links=cite.links(paragraphs, others))
            lost = cite.dropped(paragraphs)
            self.board.post("note", "code", {"topic": topic.name, "attempt": attempt, "quote_check": lost or "all kept"},
                            license=license)
            issues = [] if enough_sources(sources) else [
                f"after the quote check the page cites {len(sources)} documents; it needs {MIN_SOURCES} (or one "
                "reference manual)",
                *[f"dropped {x}" for x in lost]]
            if not issues:
                check = self._call(VERIFIER, VERIFY_PROMPT.format(page=page.markdown(), quotes=cite.quotes(paragraphs),
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
