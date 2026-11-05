"""The knowledge loop: make sure the wiki holds what the plan needs before any work on the solution starts.

1. The knowledge evaluator (gpt-oss) reads the task and the refined plan, and the SHOWN_PAGES wiki pages most relevant to
   them, and lists the topics the work needs. The wiki is shared by every run and only grows.
2. The librarian checks the wiki in plain code: a topic is covered by a verified page with enough sources.
3. For each missing topic, the research orchestrator (GLM) writes searches; the researchers are plain-code
   connectors (sources.py), since searching an API needs no model. The orchestrator then approves or rejects each
   document it found, and runs more search rounds when too few pass. A later round is told the searches already run
   and the engines not tried yet; code skips a repeated search and never reviews the same document twice for a topic.
4. The librarian writes a wiki page from the approved documents, tagging every sentence with a verbatim quote, an
   inference or a limit (cite.py, from Bespoke_LLM). Code drops every sentence whose quote is not in its document;
   the wiki verifier (DeepSeek, another family) then checks that each quote supports its sentence, and the librarian
   gets one rewrite when the checks fail, told to cut or narrow each flagged sentence rather than reword it. A page
   that still fails sends the orchestrator back to search once for sources that state what the verifier found
   unsupported, and the librarian writes again from the larger set. Only a verified page covers its topic.
5. The evaluator looks again, now with the new pages, and either approves or names what is still missing, so a broad
   first pass narrows to the specifics the work needs. A topic named again keeps the documents already approved for
   it, and its page is not rewritten unless research found something new.
6. The last evaluation still researches what it names, and plain code (the librarian's check) decides coverage, so a
   specific topic first named there gets a page instead of being left to the critic loop. One that still has no page
   is ``deferred``, not missing, as before.

Stop rules: at most MAX_ROUNDS evaluations, MAX_TOPICS topics a round, RESEARCH_ROUNDS search rounds a topic plus
GAP_SEARCHES after a rejected page,
MAX_RESEARCHED topics researched in a run (extras are deferred to the critic loop), MAX_QUERIES searches a round, and the connectors' shared call budget. The result is ``knowledge.json``: which topics
are covered, by which pages, and what is still missing.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.pipeline import cite
from factory.pipeline.blackboard import Project
from factory.pipeline.scout import Scout
from factory.pipeline.sources import CONNECTORS, Doc, Searcher
from factory.pipeline.wiki import MIN_SOURCES, Wiki, WikiPage, enough_sources
from factory.pipeline.wiki import words as wiki_words
from factory.registry import Registry

EVALUATOR, ORCHESTRATOR, LIBRARIAN, VERIFIER = ("knowledge_evaluator", "research_orchestrator", "librarian",
                                                "wiki_verifier")
# MAX_ROUNDS evaluations, so at most MAX_ROUNDS - 1 research passes, each followed by a fresh evaluation
MAX_ROUNDS, MAX_TOPICS, RESEARCH_ROUNDS, MAX_QUERIES, PER_QUERY = 4, 6, 3, 3, 5
MAX_RESEARCHED = 12  # topics researched in one run; the rest are left to the critic loop, so more rounds stay bounded
MAX_OPEN_GAPS = 2  # outside topics researched without a verified page that a run may carry into the build
SHOWN_PAGES = 40  # wiki pages the evaluator sees, the most relevant to the task
GAP_SEARCHES = 1  # search rounds after a page fails its checks, aimed at what the verifier found unsupported
RESEARCH_WORKERS = 4  # topics researched at once; ARC allows 20 requests in flight per user


class Topic(BaseModel):
    name: str
    why: str
    needed_for: list[str] = []  # task ids in the plan
    hints: list[str] = []  # search terms to start from
    # "outside": a fact about the world, its data sources or file formats, which research can find. "project": a
    # choice the task or plan makes for itself (an output schema, a cache layout, exit codes, a stopword list), which
    # no source settles. The 2026-11-04 runs blocked on such choices although the prompt told the evaluator to leave
    # them out, so the evaluator now labels each topic and code keeps project choices out of research and the gate.
    kind: Literal["outside", "project"] = "outside"


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
    deferred: list[str] = []  # first asked for in the last round and still without a page: left to the critic loop
    dropped: dict[str, str] = {}  # topic -> why the evaluator let it go without a page
    decided: dict[str, str] = {}  # topic -> why: a choice the plan or the builders make, so research skips it
    # topic -> what research found: at most MAX_OPEN_GAPS outside topics researched in every round that still have
    # no verified page. The build goes ahead with the gap recorded (the builders are told, the final report shows
    # it) and the critic loop can research it again if the work turns out to need it.
    gaps: dict[str, str] = {}
    skipped: str = ""  # why research did not run at all (a self-contained plan, factory/pipeline/knowledge.py)


# Words that point outside the project: a network service, a named data source, a file format only a specification
# settles, or a field with its own methods. The check is plain code, so it errs toward research: one hit keeps the loop.
OUTSIDE_SIGNALS = re.compile(r"""
    https?://|www\.|\bapis?\b|\bendpoints?\b|\brest(?:ful)?\b|\bgraphql\b|\bhttp\b|\bwebhooks?\b|\bsdk\b|
    \bdatabases?\b|\bsql\b|\bdownload|\bscrap(?:e|ing)|\bcrawl|\bfetch|\bweb\s?(?:site|page|service)s?\b|
    \bonline\b|\bpubmed|\bncbi\b|\buniprot|\bensembl|\bopenalex|\bchembl|\bgenbank|\brefseq|\bwikipedia|
    \bgithub\b|\bpypi\b|\bnpm\b|\bcrossref|\bdoi\b|\bpubchem|\bkegg\b|\bpdb\b|\bgo\s+(?:term|ontology)|
    \bfasta|\bfastq|\bvcf\b|\bbam\b|\bsam\b|\bgff|\bbed\s+file|\bparquet|\bxlsx|\bpdf\b|\bhdf5|\bnetcdf|
    \bgenom|\bgene\b|\bgenes\b|\bprotein|\bpeptide|\bproteom|\bprimer|\bpcr\b|\bsequenc|\brna\b|\bdna\b|
    \bmass\s+spec|\bclinical|\bpatient|\bdrug|\bmolecul|\bchemi|\bphysic|\bastronom|\bgeospatial|\bgis\b|
    \btax(?:es|ation)?\b|\bregulat|\bcompliance|\bhipaa|\bgdpr|\bfinanc|\bstock|\bforex|\bcurrency|
    \bexchange\s+rate|\bweather|\bmachine\s+learning|\bneural|\bembedding|\btransformer|\bllm\b|
    \bcryptograph|\boauth|\bencrypt|\bprotocol\b|\bstandard\s+(?:iso|rfc)|\brfc\s?\d|\biso\s?\d
""", re.I | re.X)


def self_contained(task: str, plan: str) -> str:
    """Plain code, no model call: "" when the task or plan names an outside source or a domain method (research may
    be needed), else why research can be skipped. The 2026-11-05 project runs spent 50-70 calls a project asking an
    evaluator for topics on tools such as word counters and log parsers, where every topic came back "project"."""
    hit = OUTSIDE_SIGNALS.search(f"{task}\n{plan}")
    return "" if hit else "no external API, data source, file format or domain method in the task or plan"


EVALUATE_PROMPT = """A team of agents is about to carry out this plan. Decide what knowledge it needs that a capable \
engineer would not already know: facts, methods, data sources, file formats, project templates. Leave out general \
software engineering (packaging, command-line parsing, HTTP clients, caching and retries, testing tools, mocking) and \
textbook facts any practitioner knows (a reverse complement, a well-known identifier such as human taxonomy ID 9606, \
scoring and tie-breaking rules the plan can simply state), and rules the task or plan sets for itself (an output \
format, how to tokenise or validate input, a stopword list, escaping, error handling, window or bucket sizes): the \
builders already know or are told them, and research cannot add to them. Ask for the domain knowledge the task turns \
on: facts about the outside world, its data sources and its file formats. Where an existing tool below already does part of the job, ask for how to use it (for example "<package> API for <step>", with the package name as a hint) rather than for the method it implements.

## Task
{task}

## Plan
{plan}

## Existing tools found by code (verified to exist; [import] runs in the sandbox, [reference] is to read and follow)
{reuse}

## Wiki pages already available
{pages}

{again}Reply with JSON only: {{"enough": true or false, "topics": [{{"name": "<short topic name>", "why": "<one \
sentence>", "kind": "outside" or "project", "needed_for": ["T1"], "hints": ["<search terms>"]}}], "dropped": [{{"name": "<earlier topic>", "why": \
"<why the work can go ahead without it>"}}]}}
Label each topic: "outside" when it is a fact a source outside this project settles (an API, a file format, a \
published method or constant), "project" when this task or plan decides it or already states it (its own output schema, \
cache layout, exit codes, word lists, parsing rules); project topics are left to the builders and are not researched. \
A task that needs no outside knowledge is a good outcome: set "enough" to true with no topics. \
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
PMC, Human Protein Atlas, Gene Ontology, ChEMBL, Open Targets GraphQL, DAVID, Cellosaurus, Europe PMC search syntax, OpenAlex, Apache logs, Python csv/codecs/datetime/json, Markdown \
tables, pytest and pyproject packaging); use it for how a \
tool, API or format works, and keyword queries such as "efetch rettype fasta nucleotide"
- pypi: exact Python package names, space-separated (e.g. "rapidfuzz pyahocorasick"); returns each package's README
- github: public Python repositories by plain words (e.g. "apache log anomaly detection"); returns descriptions
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

FIX_PROMPT = """
## Fix these problems from {when}
{issues}
For each problem, delete the sentence or narrow it to exactly what its quote states; do not reword a claim the \
documents do not make. Where the documents leave a point open, say so in a {{{{limit}}}} sentence. A shorter page \
that passes the checks is better than a complete one that fails.
"""

GAP_PROMPT = """
A wiki page written from the {n} documents approved so far failed its checks: {issues}
Search for sources that state these points directly (for a table of constants or parameters, the paper that \
published it or a manual that prints it).
"""

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


@dataclass
class _TopicState:
    """What research on one topic has done so far, kept so a later attempt builds on it instead of starting over."""
    approved: dict[str, Doc] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)  # documents reviewed and rejected for this topic
    rejected: list[str] = field(default_factory=list)  # their titles and the reasons
    queries: set[tuple[str, str]] = field(default_factory=set)  # (engine, normalised query) already run
    failed_with: set[str] = field(default_factory=set)  # approved ids behind the last rejected page
    issues: list[str] = field(default_factory=list)  # the verifier's issues with that page


class KnowledgeLoop:
    def __init__(self, llm: LLM, registry: Registry, project: Project, wiki: Wiki, searchers: dict[str, Searcher],
                 scout: Scout | None = None):
        """``scout``: look for existing tools before research (default_scout). None skips it, as in the tests."""
        self.llm, self.project, self.board, self.wiki = llm, project, project.board, wiki
        self.scout = scout
        self.agents = {n: registry.agents[n] for n in (EVALUATOR, ORCHESTRATOR, LIBRARIAN, VERIFIER)}
        self.searchers = searchers
        self._topics: dict[str, _TopicState] = {}
        self._lock = threading.Lock()

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
        reuse = self.find_prior_art(task, plan)
        if not again and (why := self_contained(task, plan)) and os.environ.get("ASPIRE_SKIP_RESEARCH", "1") != "0":
            return self._skipped(why)
        covered: dict[str, str] = {}
        missing: list[str] = []
        asked: list[str] = []  # every topic name asked for so far, so a reworded topic keeps its first name
        deferred: list[str] = []
        researched = 0
        over_budget: list[str] = []  # topics left unresearched because the run's budget was spent
        dropped: dict[str, str] = {}
        decided: dict[str, str] = {}
        for rnd in range(1, MAX_ROUNDS + 1):
            if rnd == 1:  # only verified pages: one the verifier rejected must not count toward "enough"
                pages = self.wiki.relevant(f"{task}\n{plan}", SHOWN_PAGES)
                again = ""
            else:
                pages = [p for p in (self.wiki.find(t) for t in covered) if p]
                still = [n for n in asked if n not in covered and n not in dropped and n not in decided]
                again = ("Research has run on the topics you asked for; the pages now available are listed. Name only "
                         "what is still missing, including specifics the new pages show the work depends on.\n"
                         f"Topics asked for earlier: {'; '.join(asked)}\n"
                         + (f"Still without a verified page: {'; '.join(still)}. Each stays missing until a page covers "
                            "it or you list it under \"dropped\" with why the work can go ahead without it.\n"
                            if still else "") + "\n")
            listing = "\n".join(f"- {p.topic}: {p.summary[:300]}" for p in pages) or "(none)"
            needs = self._call(EVALUATOR, EVALUATE_PROMPT.format(task=task, plan=plan, pages=listing, again=again, reuse=reuse,
                                                                 max_topics=MAX_TOPICS), Needs, f"knowledge-{rnd}")
            topics = [t.model_copy(update={"name": same_topic(t.name, asked)}) for t in needs.topics[:MAX_TOPICS]]
            decided |= {t.name: t.why for t in topics if t.kind == "project"}
            topics = [t for t in topics if t.name not in decided]
            last = rnd == MAX_ROUNDS
            # a topic first named in the last round is researched, but no evaluation follows, so if research fails
            # it is deferred to the critic loop rather than blocking (the 2026-11-04 runs lost such topics unresearched)
            fresh = [t.name for t in topics if t.name not in asked] if last and rnd > 1 else []
            asked += [t.name for t in topics if t.name not in asked]
            for d in needs.dropped:
                name = same_topic(d.name, asked)
                if name in asked and name not in {t.name for t in topics}:
                    dropped[name] = d.why
            ev = self.board.post("topics", EVALUATOR, {"round": rnd, "enough": needs.enough,
                                                       "topics": [t.model_dump() for t in topics],
                                                       "first_in_last_round": fresh, "dropped": dropped,
                                                       "decided": decided})
            have = {t.name: self.wiki.covers(t.name) for t in topics}
            # an earlier topic stays missing until a page covers it or the evaluator drops it with a reason: the
            # 2026-11-03 primer run lost an unresearched topic by simply not naming it again
            missing = [t.name for t in topics if have[t.name] is None and t.name not in fresh]
            missing += [n for n in asked if n not in missing and n not in have and n not in dropped
                        and n not in decided and n not in covered and n not in over_budget
                        and self.wiki.covers(n) is None]
            missing = [n for n in missing if n not in over_budget]
            deferred = [*over_budget, *(n for n in fresh if have[n] is None and n not in over_budget)]
            covered |= {name: page.slug for name, page in have.items() if page}
            if needs.enough and not topics:  # approved on the pages shown, so those are what the builders get
                covered |= {p.topic: p.slug for p in pages}
            self.board.post("coverage", LIBRARIAN, {"covered": {n: p.slug for n, p in have.items() if p},
                                                    "missing": missing, "deferred": deferred}, refs=[ev.id])
            todo = [t for t in topics if t.name in missing or t.name in deferred]
            room = MAX_RESEARCHED - researched
            if len(todo) > room:  # over the run's budget: the extra topics are left to the critic loop, not blocking
                skipped = {t.name for t in todo[max(room, 0):]}
                todo = todo[:max(room, 0)]
                over_budget += sorted(skipped)
                missing = [n for n in missing if n not in skipped]
                deferred = [*deferred, *(n for n in sorted(skipped) if n not in deferred)]
                self.board.post("note", "code", {"text": f"research budget of {MAX_RESEARCHED} topics reached; "
                                                 f"deferred: {'; '.join(sorted(skipped))}"})
            researched += len(todo)
            if not todo:  # nothing missing, or only earlier topics not named again: already researched, no new work
                break
            for topic, page in zip(todo, self.research_many(todo), strict=True):
                if page and page.verified:
                    covered[topic.name] = page.slug
            if last:  # no evaluation follows, so the librarian's plain-code check is the last word
                missing = [n for n in missing if n not in covered and self.wiki.covers(n) is None]
                deferred = [n for n in deferred if n not in covered]
                self.board.post("coverage", LIBRARIAN, {"covered": {t.name: covered[t.name] for t in todo
                                                                    if t.name in covered},
                                                        "missing": missing, "deferred": deferred, "final": True},
                                refs=[ev.id])
                break
        gaps = self._open_gaps(missing, len(covered))
        if gaps:
            missing = []
            self.board.post("note", "code", {"text": "the build goes ahead with a recorded gap: "
                                                 + "; ".join(f"{n} ({why})" for n, why in gaps.items())})
        ok = not missing
        result = KnowledgeResult(ok=ok, covered=covered, missing=missing, rounds=rnd, deferred=deferred,
                                 dropped=dropped, decided=decided, gaps=gaps)
        k = self.board.post("knowledge", EVALUATOR if ok else "code", result.model_dump())
        self.project.write_file("knowledge.json", json.dumps(result.model_dump(), indent=1), "code", refs=[k.id])
        return result

    def _skipped(self, why: str) -> KnowledgeResult:
        """Research skipped by the plain-code check: the build goes ahead with no pages. The prior-art scout has
        already run, because stdlib modules it finds are what a self-contained build most needs."""
        self.board.post("note", "code", {"text": f"research skipped: {why}"})
        result = KnowledgeResult(ok=True, covered={}, missing=[], rounds=0, skipped=why)
        k = self.board.post("knowledge", "code", result.model_dump())
        self.project.write_file("knowledge.json", json.dumps(result.model_dump(), indent=1), "code", refs=[k.id])
        return result

    def find_prior_art(self, task: str, plan: str) -> str:
        """Existing tools for the task, found and checked by code (factory/pipeline/scout.py), written to reuse.json
        for the head orchestrator; returns the block the evaluator sees."""
        if self.scout is None:
            return "(not looked up)"
        found = self.scout.run(task, plan)
        e = self.board.post("prior_art", "code", found.model_dump())
        self.project.write_file("reuse.json", found.model_dump_json(indent=1), "code", refs=[e.id])
        return found.render() or "(none found)"

    def _open_gaps(self, missing: list[str], covered: int = 0) -> dict[str, str]:
        """The one or two topics that stopped a run which otherwise has its knowledge, when each was really researched
        (searched, not just named): the proteomics runs of 2026-11-05 each stopped on one API schema or export format
        and the MCF7 run on two (4 topics covered), none of which a source in reach settled. More open topics than
        MAX_OPEN_GAPS still block."""
        if not missing or len(missing) > MAX_OPEN_GAPS or (len(missing) > 1 and covered < len(missing)):
            return {}  # a second gap also needs at least as many covered topics as gaps
        out = {}
        for name in missing:
            st = self._topics.get(name)
            if st is None or not (st.queries or st.approved):
                return {}  # never searched: that is a failure to research, not a gap research could not close
            out[name] = (f"{len(st.approved)} approved source(s), no verified page"
                         + (f"; last check: {st.issues[0][:160]}" if st.issues else ""))
        return out

    def research_many(self, topics: list[Topic]) -> list[WikiPage | None]:
        """Research several topics at once, results in the order given. Topics are independent (each searches,
        reviews, writes and verifies its own page), and the live logs show every call ran one after another: the
        primer's research took 2h12m, about 20 minutes a topic. Threads share the provider's concurrency gate."""
        if len(topics) <= 1:
            return [self.research(t) for t in topics]
        with ThreadPoolExecutor(max_workers=min(RESEARCH_WORKERS, len(topics)), thread_name_prefix="research") as pool:
            return list(pool.map(self.research, topics))

    # --- one topic: search, approve, write, verify ---

    def research(self, topic: Topic) -> WikiPage | None:
        """One topic's research; a model step that never gives valid output leaves the topic without a page (it
        stays missing) instead of ending the whole loop. The 2026-11-05 MCF7 run lost all its research when one
        page's writer returned malformed JSON three times."""
        try:
            return self._research(topic)
        except RuntimeError as e:
            self.board.post("note", ORCHESTRATOR, {"topic": topic.name, "text": f"research failed: {str(e)[:300]}"})
            return None

    def _research(self, topic: Topic) -> WikiPage | None:
        """Search until enough approved documents, write and verify a page; on a rejected page, search once more for
        what the verifier found unsupported and write again. State carries over when the topic is researched again."""
        st = self._state_for(topic.name)
        if not st.approved:  # documents approved for this topic by an earlier process (a rerun) still count
            st.approved = self._approved_on_board(topic.name)
        for rnd in range(1, RESEARCH_ROUNDS + 1):
            if enough_sources(st.approved) and set(st.approved) != st.failed_with:
                break
            self._search(topic, st, f"{rnd}")
        if not enough_sources(st.approved):
            self.board.post("note", ORCHESTRATOR, {"topic": topic.name, "text": f"only {len(st.approved)} approved "
                                                   f"sources after {RESEARCH_ROUNDS} rounds"})
            return None
        if set(st.approved) == st.failed_with:  # the same documents already made a rejected page
            self.board.post("note", ORCHESTRATOR, {"topic": topic.name, "text": "no new sources since the rejected "
                                                   "page; not rewritten"})
            return None
        page = self.write_page(topic, list(st.approved.values()), st.issues)
        for g in range(1, GAP_SEARCHES + 1):
            if page.verified:
                break
            st.failed_with, st.issues = set(st.approved), page.issues
            if not self._search(topic, st, f"gap{g}"):
                break
            page = self.write_page(topic, list(st.approved.values()), st.issues)
        if page.verified:
            st.failed_with, st.issues = set(), []
        else:
            st.failed_with, st.issues = set(st.approved), page.issues
        return page

    def _state_for(self, name: str) -> _TopicState:
        with self._lock:
            return self._topics.setdefault(name, _TopicState())

    def _approved_on_board(self, name: str) -> dict[str, Doc]:
        out: dict[str, Doc] = {}
        for e in self.board.entries("source"):
            if e.data.get("topic") != name:
                continue
            path = self.project.root / "sources" / f"{e.data['id'].replace(':', '_')}.json"
            if path.exists():
                out[e.data["id"]] = Doc.model_validate_json(path.read_text(encoding="utf-8"))
        return out

    def _search(self, topic: Topic, st: _TopicState, tag: str) -> list[str]:
        """One search round: write queries, run the new ones, review the documents not seen yet. Returns the ids it
        approved."""
        notes = []
        if st.queries:
            untried = [c for c in self.searchers if c not in {c for c, _ in st.queries}]
            notes.append("Searches already run (do not repeat them): "
                         + "; ".join(f"{c}: {q}" for c, q in sorted(st.queries)[-12:]) + ".")
            if untried:
                notes.append(f"Engines not tried yet for this topic: {', '.join(untried)}.")
            if not enough_sources(st.approved):
                notes.append("Earlier searches found too few usable documents"
                             + (f" (rejected: {'; '.join(st.rejected[-6:])})" if st.rejected else "")
                             + ". Try different terms or engines.")
        if st.issues:
            notes.append(GAP_PROMPT.format(n=len(st.approved), issues="; ".join(st.issues[:6])).strip())
        previous = ("\n" + "\n".join(notes) + "\n") if notes else ""
        queries = self._call(ORCHESTRATOR, QUERIES_PROMPT.format(
            topic=topic.name, why=topic.why, hints=", ".join(topic.hints) or "none", previous=previous,
            connectors=json.dumps(list(self.searchers)), max_queries=MAX_QUERIES), Queries,
            f"queries-{topic.name}-{tag}").queries[:MAX_QUERIES]
        found: dict[str, Doc] = {}
        for q in queries:
            key = (q.connector, " ".join(q.query.lower().split()))
            if key in st.queries:  # the same search returns the same documents, already reviewed
                self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "error": "repeat of an earlier search"})
                continue
            search = self.searchers.get(q.connector)
            if search is None:
                self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "error": "unknown engine"})
                continue
            try:
                docs = search(q.query, PER_QUERY)
            except Exception as e:  # noqa: BLE001 (one engine failing is recorded, not fatal; it may work next time)
                self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "error": f"{type(e).__name__}: {e}"})
                continue
            st.queries.add(key)
            self.board.post("query", ORCHESTRATOR, {**q.model_dump(), "found": [d.id for d in docs]})
            found |= {d.id: d for d in docs if d.text.strip() and d.id not in st.approved and d.id not in st.seen}
        if not found:
            return []
        docs = list(found.values())
        review = self._call(ORCHESTRATOR, REVIEW_PROMPT.format(topic=topic.name, why=topic.why,
                                                               docs=show_docs(docs)), Review,
                            f"review-{topic.name}-{tag}", docs)
        verdicts = {v.id: v for v in review.verdicts if v.id in found}  # verdicts on unknown ids are ignored
        new: list[str] = []
        for d in docs:
            v = verdicts.get(d.id)
            if v and v.approve:
                st.approved[d.id] = d
                new.append(d.id)
                self.project.write_file(f"sources/{d.id.replace(':', '_')}.json", d.model_dump_json(indent=1),
                                        ORCHESTRATOR, license=d.license)
                self.board.post("source", ORCHESTRATOR, {"id": d.id, "title": d.title, "url": d.url,
                                                         "topic": topic.name, "why": v.why}, license=d.license)
            else:
                st.seen.add(d.id)
                st.rejected.append(f"{d.title[:60]}: {v.why if v else 'no verdict'}")
        return new

    def write_page(self, topic: Topic, docs: list[Doc], earlier: list[str] = ()) -> WikiPage:
        """``earlier``: the verifier's issues with a rejected page on this topic, which the first draft must avoid."""
        ids = [d.id for d in docs]
        license = "arc_only" if any(d.license == "arc_only" for d in docs) else "public"
        issues: list[str] = list(earlier)
        page = None
        # pages it may link to; an ARC-only page is listed only for an ARC-only page, so no link leaks its topic
        others = {p.slug: p.topic for p in self.wiki.pages() if p.verified and p.topic != topic.name
                  and (license == "arc_only" or p.license == "public")}
        for attempt in (1, 2):  # one rewrite when the checks find problems
            when = "the last draft" if attempt > 1 else "an earlier page on this topic"
            fix = FIX_PROMPT.format(when=when, issues="\n".join(issues)) if issues else ""
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


def default_scout(llm: LLM, registry: Registry, cache_dir: Path) -> Scout | None:
    """The live scout: PyPI and GitHub lookups with their own small call budget, proposals from the research
    orchestrator's ARC role. ``ASPIRE_SCOUT=0`` turns it off."""
    import os
    if os.environ.get("ASPIRE_SCOUT", "1") == "0":
        return None

    from factory.net.http import CallBudget
    from factory.pipeline.sources import GitHubRepos, PyPI
    budget = CallBudget("scout", 30)
    token = os.environ.get("GITHUB_TOKEN")
    return Scout(llm, registry.agents[ORCHESTRATOR], PyPI(cache_dir / "pypi", budget),
                 GitHubRepos(cache_dir / "github", budget, headers={"Authorization": f"Bearer {token}"} if token else {}))


def default_searchers(cache_dir: Path) -> dict[str, Searcher]:
    from factory.pipeline.sources import connectors
    found = connectors(cache_dir)
    return {name: found[name] for name in CONNECTORS}
