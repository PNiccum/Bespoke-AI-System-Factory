"""Production: the head orchestrator designs the solution, sub-orchestrators break each task into work orders, and
builders write the files.

1. The head orchestrator (GLM) reads the task, the refined plan and the wiki pages the knowledge loop approved
   (methods, data sources, project templates), and fixes the design: language, file layout, entry point, the command
   that runs the result, and a note per task.
2. For each task, in plan order, a sub-orchestrator (GLM) turns the task and its subtasks into work orders: which files
   to write or change, which existing files to read first, and what each must do.
3. A builder (the coder role) carries out one work order and returns whole files. Code writes them under ``output/``,
   refusing paths outside it and oversized files.
4. Code organizes the result: ``README.md`` from the design, ``run.json`` with the command, and the plain-code checks
   (every Python file compiles; the tests under ``output/`` run in the sandbox). The critic loop starts from here.

Stop rules: MAX_FILES files, MAX_ORDERS work orders a task, MAX_FILE_CHARS characters a file.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, model_validator

from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import ProjectPlan
from factory.pipeline.scout import ReuseResult, check_imports, unused_reuse
from factory.pipeline.wiki import Wiki, WikiPage
from factory.registry import Registry
from factory.tools.sandbox import RunTestsIn, run_tests

HEAD, SUB, BUILDER = "head_orchestrator", "sub_orchestrator", "builder"
# 60 files: the 2026-11-03 primer build hit 30 with core modules unwritten (part of the 30 was a stale tree)
MAX_FILES, MAX_ORDERS, MAX_FILE_CHARS, MAX_READ_CHARS = 60, 6, 60_000, 12_000
# Files one builder reply may write. In the Nov 2-4 runs 12 builder replies on GLM hit max_tokens and threw away
# 180 min; each had been asked for a whole task's worth of files at once. A larger order is split in code.
MAX_ORDER_FILES = 3
BUILD_WORKERS = 3  # independent work orders of one task built at once
OUTPUT = "output"
SETUP_FAILURE = re.compile(r"(ModuleNotFoundError|ImportError)[^\n]{0,160}")


class Design(BaseModel):
    language: str
    entry_point: str  # path under output/
    run_command: str  # run from inside output/
    layout: list[str]  # files the solution will have, with one line on each
    task_notes: dict[str, str] = {}  # task id -> how it fits the design
    template: str = ""  # the wiki page or pattern it follows, if any
    reuse: list[str] = []  # existing tools from reuse.json it adopts: "<name>: how" (the scout's candidates)


class WorkOrder(BaseModel):
    id: str  # e.g. T2.1
    write: list[str]  # files to create or replace, under output/
    read: list[str] = []  # existing files the builder needs to see
    instructions: str


class Orders(BaseModel):
    orders: list[WorkOrder]


FENCE_OPEN, FENCE_CLOSE = re.compile(r"\A\s*```[\w.+-]*[ \t]*\n"), re.compile(r"\n[ \t]*```\s*\Z")


def unfence(path: str, content: str) -> str:
    """A file body without the Markdown code fence a model wrapped it in. The 2026-11-04 primer build saved
    design.py with a fence on line 1, a SyntaxError that failed every test importing it. Markdown files keep theirs."""
    if path.lower().endswith((".md", ".markdown")) or not FENCE_OPEN.match(content):
        return content
    body = FENCE_OPEN.sub("", content, count=1)
    return FENCE_CLOSE.sub("\n", body, count=1)


class NewFile(BaseModel):
    path: str
    content: str

    @model_validator(mode="after")
    def _unfence(self):
        self.content = unfence(self.path, self.content)
        return self


class Built(BaseModel):
    files: list[NewFile]
    notes: str = ""


class Checks(BaseModel):
    compiled: list[str]
    syntax_errors: dict[str, str]
    tests_ran: bool
    tests_ok: bool | None
    passed: list[str] = []
    failed: list[str] = []
    output_tail: str = ""
    reasons: dict[str, str] = {}  # failing test -> why (first error line)
    setup_error: str = ""  # the tests could not even be imported (a package or path problem, not a failing test)
    # module -> files: imports the sandbox cannot satisfy (neither stdlib, nor sandbox/requirements.txt, nor local).
    # On the laptop's process sandbox such code may pass its tests and still fail anywhere else, so it blocks.
    unavailable_imports: dict[str, list[str]] = {}
    reuse_unused: list[str] = []  # [import] tools the design said it adopts that no file imports (recorded only)

    @property
    def ok(self) -> bool:
        return not self.syntax_errors and self.tests_ok is not False and not self.unavailable_imports


DESIGN_PROMPT = """Design the solution a team of agents will build for this project. The result must run again \
without any model: plain code, using a model only where the task truly needs language understanding.

## Task
{task}

## Plan
{plan}

## Knowledge the team gathered (wiki pages)
{wiki}

## Existing tools found by code
{reuse}
Prefer these over writing the same thing again. Import only [import] tools and the standard library (the sandbox has \
no network and no other packages; code checks every import); list a third-party [import] package in \
requirements.txt. For a [reference] tool follow its approach in your own code and credit it in the README.

Reply with JSON only: {{"language": "...", "entry_point": "<file under output/>", "run_command": "<command run inside \
output/>", "layout": ["<path>: <what it holds>", ...], "task_notes": {{"T1": "<how this task fits>"}}, "template": \
"<the wiki page or known pattern this follows, or empty>", "reuse": ["<tool name>: <what the solution takes from it>"]}}
Include tests under tests/ that check each task's acceptance criterion."""

ORDERS_PROMPT = """Break this task into work orders for builders. Each order writes whole files.

## Design
{design}

## Task
{task_json}

## Files written so far
{files}

Reply with JSON only: {{"orders": [{{"id": "{task_id}.1", "write": ["<path>"], "read": ["<existing path>"], \
"instructions": "<what the files must do, precisely>"}}]}}, at most {max_orders} orders, each writing at most \
{max_files} files (a long file gets an order of its own).
Keep each order small enough to write in one reply: at most two files, each under about 300 lines. Split a large \
test suite into several test files, one order each."""

BUILD_PROMPT = """Carry out this work order. Write every file it names in full.

## Project
{task}

## Design
{design}

## Work order
{order}

## Knowledge (wiki excerpts)
{wiki}

## Existing files you need
{reads}

Reply with the files as plain text blocks, not JSON, so code needs no escaping:

=== FILE: <path under output/> ===
<the whole file, exactly as it should be saved>
=== END FILE ===

One block per file, then a last line: NOTES: <one line>
Do not write test results, test counts or run logs into any file: the checks run the tests themselves, and a recorded \
count goes stale as soon as a test is added."""

FILE_BLOCK = re.compile(r"^=== FILE: *(.+?) *===[ \t]*\n(.*?)^=== END FILE ===[ \t]*$", re.M | re.S)


def parse_files(completion) -> dict:
    """The builder's reply as Built data: ``=== FILE: path ===`` blocks (file bodies are written raw, since bodies
    embedded in JSON strings broke on escaping in the 2026-11-03 primer build: T1.6, T4.2, T4.4 and T5.5 failed three
    times each). A JSON reply in the old shape is still accepted."""
    text = completion.text
    blocks = FILE_BLOCK.findall(text)
    if not blocks:
        if "=== FILE:" in text:
            raise ValueError("a FILE block has no closing '=== END FILE ===' line")
        return completion.json()
    notes = re.search(r"^NOTES: *(.*)$", text, re.M)
    return {"files": [{"path": p.strip().strip("`"), "content": body} for p, body in blocks],
            "notes": notes.group(1).strip() if notes else ""}


class Production:
    def __init__(self, llm: LLM, registry: Registry, project: Project, wiki: Wiki):
        self.llm, self.project, self.board, self.wiki = llm, project, project.board, wiki
        self.agents = {n: registry.agents[n] for n in (HEAD, SUB, BUILDER)}
        self.out = project.root / OUTPUT
        self._write_lock = threading.Lock()

    def _call(self, agent: str, prompt: str, model_cls, tag: str, pages: list[WikiPage] = (), **parse):
        public = self.llm.cfg.data_public and all(p.license == "public" for p in pages)
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag,
                                       data="public" if public else "private", **parse)
        return out

    def pages(self) -> list[WikiPage]:
        k = self.board.latest("knowledge")
        slugs = k.data["covered"].values() if k else []
        return [p for p in (self.wiki.get(s) or self.wiki.find(s) for s in slugs) if p]

    def files(self) -> list[str]:
        if not self.out.exists():
            return []
        return sorted(p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file())

    def start_clean(self) -> Path | None:
        """Move what an earlier build left in output/ to superseded/<time>/, so a new build starts from an empty
        folder. The 2026-11-03 primer build inherited a stale package tree from two earlier builds, and its tests
        imported the wrong one. Returns where the old files went, or None when output/ was empty."""
        if not self.files():
            return None
        dest = self.project.root / "superseded" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.out), str(dest))
        self.board.post("note", "code", {"text": f"moved the earlier build's {OUTPUT}/ to "
                                                 f"{dest.relative_to(self.project.root).as_posix()}"})
        return dest

    def inside(self, rel: str) -> bool:
        """Whether a path a model gave stays inside output/ (no "..", no absolute path)."""
        return bool(rel) and (self.out / rel).resolve().is_relative_to(self.out.resolve())

    def run(self, feedback: str = "") -> Checks:
        """Build the solution. ``feedback`` (from the critic or the final evaluator) starts a rebuild round."""
        if self.board.latest("knowledge") is None:
            raise RuntimeError("production starts after the knowledge loop; run that first")
        task = (self.project.root / "task.txt").read_text(encoding="utf-8")
        plan = ProjectPlan.model_validate_json((self.project.root / "plan.json").read_text(encoding="utf-8"))
        pages = self.pages()
        wiki = "\n\n".join(f"### {p.topic}\n{p.summary}\n{p.body[:3000]}" for p in pages) or "(none)"
        gaps = (self.board.latest("knowledge").data.get("gaps") or {})
        if gaps:
            wiki += ("\n\n## Known gaps (research found no verified source)\n"
                     + "\n".join(f"- {n}: {why}" for n, why in gaps.items())
                     + "\nDo not invent exact values for these. Check the live service or input, make the code "
                     "tolerant, and state the assumption in the README.")
        if feedback:
            task += f"\n\n## Problems to fix in this round\n{feedback}"

        reuse_path = self.project.root / "reuse.json"
        reuse = (ReuseResult.model_validate_json(reuse_path.read_text(encoding="utf-8")).render(detail=True)
                 if reuse_path.exists() else "") or "(none looked up)"
        design = self._call(HEAD, DESIGN_PROMPT.format(task=task, plan=plan.model_dump_json(indent=1), wiki=wiki,
                                                       reuse=reuse),
                            Design, "design", pages)
        d = self.board.post("design", HEAD, design.model_dump(), license=_license(pages))
        for t in plan.tasks:  # plan order respects depends_on (check_plan rules out forward references)
            orders = self._call(SUB, ORDERS_PROMPT.format(design=design.model_dump_json(indent=1),
                                                          task_json=t.model_dump_json(indent=1), task_id=t.id,
                                                          files="\n".join(self.files()) or "(none yet)",
                                                          max_orders=MAX_ORDERS, max_files=MAX_ORDER_FILES), Orders,
                                f"orders-{t.id}")
            orders.orders = split_orders(orders.orders)
            o = self.board.post("orders", SUB, {"task": t.id, "orders": [x.model_dump() for x in orders.orders]},
                                refs=[d.id])
            for wave in order_waves(orders.orders[:MAX_ORDERS]):
                self.build_wave(wave, task, design, wiki, pages, o.id)
        self.organize(design, plan)
        checks = self.check()
        self.board.post("checks", "code", checks.model_dump(), refs=[d.id])
        return checks

    def build_wave(self, wave: list[WorkOrder], task: str, design: Design, wiki: str, pages: list[WikiPage],
                   ref: str) -> None:
        """Work orders that neither read nor write each other's files, built at once."""
        if len(wave) == 1:
            return self.build(wave[0], task, design, wiki, pages, ref)
        with ThreadPoolExecutor(max_workers=min(BUILD_WORKERS, len(wave)), thread_name_prefix="builder") as pool:
            for f in [pool.submit(self.build, o, task, design, wiki, pages, ref) for o in wave]:
                f.result()

    def build(self, order: WorkOrder, task: str, design: Design, wiki: str, pages: list[WikiPage],
              ref: str) -> None:
        reads = []
        for rel in order.read[:4]:
            path = (self.out / rel).resolve()
            if path.is_relative_to(self.out.resolve()) and path.is_file():
                reads.append(f"### {rel}\n```\n{path.read_text(errors='replace', encoding='utf-8')[:MAX_READ_CHARS]}\n```")
        try:
            built = self._call(BUILDER, BUILD_PROMPT.format(task=task, design=design.model_dump_json(indent=1),
                                                            order=order.model_dump_json(indent=1), wiki=wiki,
                                                            reads="\n\n".join(reads) or "(none)"), Built,
                               f"build-{order.id}", pages, parse=parse_files, fmt="FILE blocks")
        except RuntimeError as e:  # no usable reply: the files stay unwritten, and the critic and QA see the gap
            self.board.post("build", BUILDER, {"order": order.id, "written": [], "refused": [],
                                               "failed": str(e)[:1000], "notes": ""}, refs=[ref])
            return
        with self._write_lock:  # parallel builders: the file cap is counted and files written one order at a time
            self._write(built, order, ref)

    def _write(self, built: Built, order: WorkOrder, ref: str) -> None:
        written, refused = [], []
        for f in built.files:
            rel = f.path.removeprefix(f"{OUTPUT}/").lstrip("/")
            if len(self.files()) >= MAX_FILES and rel not in self.files():
                refused.append(f"{rel}: more than {MAX_FILES} files")
            elif len(f.content) > MAX_FILE_CHARS:
                refused.append(f"{rel}: longer than {MAX_FILE_CHARS} characters")
            elif not self.inside(rel):
                refused.append(f"{rel}: outside the output folder")
            else:
                self.project.write_file(f"{OUTPUT}/{rel}", f.content, BUILDER, refs=[ref])
                written.append(rel)
        self.board.post("build", BUILDER, {"order": order.id, "written": written, "refused": refused,
                                           "notes": built.notes}, refs=[ref])

    def organize(self, design: Design, plan: ProjectPlan) -> None:
        """README.md and run.json from the design, so the result explains and runs itself."""
        lines = [f"# {plan.title}", "", plan.goal, "", "## Run it", "", "```", design.run_command, "```", "",
                 "## Files", "", *[f"- {x}" for x in design.layout], "", "## What each task delivered", ""]
        lines += [f"- {t.id} {t.title}: done when {t.acceptance}" for t in plan.tasks]
        if not (self.out / "README.md").exists():
            self.project.write_file(f"{OUTPUT}/README.md", "\n".join(lines) + "\n", "code")
        self.project.write_file(f"{OUTPUT}/run.json", json.dumps({"command": design.run_command,
                                                                  "entry_point": design.entry_point}, indent=1), "code")

    def check(self) -> Checks:
        """Plain-code checks: Python files compile (without running them); tests run in the sandbox."""
        compiled, errors = [], {}
        for rel in self.files():
            if rel.endswith(".py"):
                try:
                    compile((self.out / rel).read_text(encoding="utf-8"), rel, "exec")
                    compiled.append(rel)
                except SyntaxError as e:
                    errors[rel] = f"line {e.lineno}: {e.msg}"
        found = self.reuse_checks()
        has_tests = any(r.startswith("tests/") and r.endswith(".py") for r in self.files())
        if not has_tests:
            return Checks(compiled=compiled, syntax_errors=errors, tests_ran=False, tests_ok=None, **found)
        # a src layout imports without an install: pytest gets src/ (and the root) on its path
        paths = ["src", "."] if (self.out / "src").is_dir() else ["."]
        report = run_tests(RunTestsIn(repo_path=str(self.out), timeout_s=300, pythonpath=paths))  # throwaway copy
        tail = report.output_tail[-2000:]
        setup = ""
        m = next((x for x in (re.search(r"ModuleNotFoundError[^\n]{0,160}", tail), SETUP_FAILURE.search(tail)) if x),
                 None)
        if not report.passed and not report.failed and m:
            setup = (f"the tests could not be imported ({m.group(0)}): fix the package layout or imports so that "
                     f"`pytest` run inside {OUTPUT}/ with {' and '.join(paths)} on the path can import them")
        return Checks(compiled=compiled, syntax_errors=errors, tests_ran=True, tests_ok=report.ok,
                      passed=report.passed, failed=report.failed + report.errors, output_tail=tail,
                      reasons=report.reasons,
                      setup_error=setup, **found)

    def reuse_checks(self) -> dict:
        """Plain-code checks of reuse: imports the sandbox lacks, and adopted tools that are never imported."""
        files = self.files()
        design = self.board.latest("design")
        reuse_path = self.project.root / "reuse.json"
        result = (ReuseResult.model_validate_json(reuse_path.read_text(encoding="utf-8"))
                  if reuse_path.exists() else None)
        return {"unavailable_imports": check_imports(self.out, files),
                "reuse_unused": unused_reuse(design.data.get("reuse", []) if design else [], result, self.out,
                                             files)}


def split_orders(orders: list[WorkOrder], limit: int = MAX_ORDER_FILES) -> list[WorkOrder]:
    """Orders that write more than ``limit`` files, split into parts (T2.1a, T2.1b, ...) with the same reads and
    instructions; a later part reads what the earlier parts write, so the waves keep them in order."""
    out = []
    for o in orders:
        if len(o.write) <= limit:
            out.append(o)
            continue
        chunks = [o.write[i:i + limit] for i in range(0, len(o.write), limit)]
        for n, chunk in enumerate(chunks):
            earlier = [w for c in chunks[:n] for w in c]
            out.append(o.model_copy(update={"id": f"{o.id}{chr(ord('a') + n)}", "write": chunk,
                                            "read": [*o.read, *(w for w in earlier if w not in o.read)]}))
    return out


def order_waves(orders: list[WorkOrder]) -> list[list[WorkOrder]]:
    """Group a task's work orders into waves that can be built at once. An order waits for every earlier order that
    writes a file it reads or writes, or reads a file it writes, so each builder sees exactly the files it would have
    seen one order at a time; otherwise it joins the earliest wave after them. Order within a wave follows the
    sub-orchestrator's list. (The 2026-11-04 primer build ran 34 orders one after another for 4h17m.)"""
    level: list[int] = []
    for j, o in enumerate(orders):
        writes = {_rel(p) for p in o.write}
        touches = writes | {_rel(p) for p in o.read}
        level.append(1 + max((level[i] for i in range(j) if touches & {_rel(p) for p in orders[i].write}
                              or writes & {_rel(p) for p in orders[i].read}), default=-1))
    return [[o for o, lv in zip(orders, level, strict=True) if lv == n] for n in range(max(level, default=-1) + 1)]


def _rel(path: str) -> str:
    return path.strip().removeprefix(f"{OUTPUT}/").lstrip("/")


def _license(pages: list[WikiPage]) -> str:
    return "arc_only" if any(p.license == "arc_only" for p in pages) else "public"
