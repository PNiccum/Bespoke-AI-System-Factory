"""Self-extension: ASPIRE proposes a better version of one of its own model agents and judges it on tasks the
proposer never saw. By default it stops for a person; with ``adopt`` a proposal that passes every gate replaces the
agent's prompt in the registry on its own, because Parker decided (2026-10-29) that a prompt-only change needs no
approval. Structural changes (code, tools, roles, loops) are never made here: they stay pull requests a person reviews.

A proposal is a new agent spec only: a prompt and a description for an existing role and capability. It cannot add
code, tools, roles or capabilities, so it cannot change what ASPIRE is able to touch, only how a model is asked.

    python -m factory.extend --agent coder                  # free models only
    python -m factory.extend --agent coder --paid-panel     # also asks a paid reviewer, after your approval
    python -m factory.extend --agent coder --adopt          # apply it if it passes every gate
    python -m factory.extend --rollback coder               # put back the prompt it had before the last adoption
    python -m factory.extend --agent builder --case <id>    # also replay a failure case from the archive
    python -m factory.extend --agent coder --tasks bench/tasks.yaml   # judge on one tier only
    python -m factory.extend --from-events                  # re-design from errors open in 3 of the last 10 runs

The proposer (a GLM model) writes ``CANDIDATES`` prompts, each told to take a different approach from the ones before
it. Every candidate that passes the scope gate is scored, and only the best goes on: on the held-out benchmark (most
solved, then fewest tokens) when the benchmark exercises the agent, otherwise by a free reviewer of another family.

The gates, in order; a proposal stops at the first it fails:

1. **Scope** (plain code): the spec validates, is a model agent on the same role (so the same model and price), keeps
   or narrows the incumbent's capabilities and tools, has a new name, a prompt under 3,000 characters, and no secret.
2. **Held-out benchmark** (plain code judging free model runs): the candidate replaces the incumbent on the held-out
   tasks of ``bench/tasks.yaml`` and ``bench/tasks_hard.yaml`` (``--tasks`` picks other files). It must solve at
   least as many, without using more than 1.5 times the tokens. The hard tier is where ASPIRE and a single call
   differ most, so it is the gate with signal; the easy tier is where they tie. The
   benchmark only exercises the coding workflow's agents; for any other agent (the pipeline's critics, auditors and
   so on) this gate says so and the panel below gets a second free family instead.
3. **Review panel** (free model of another family; optionally a paid one): reads the old and new spec and must accept.
4. **Adoption**: the proposal, its evidence and the exact file are written to ``workspace/<name>/proposals/<id>/``.
   Without ``adopt`` it waits for a person. With ``adopt``, ``adopt()`` keeps the agent's name, role, tools and
   capabilities, writes only the new prompt and description over its spec, and saves the old spec under
   ``registry/history/<agent>/`` so ``rollback()`` can put it back. Every adoption and rollback is a line in
   ``registry/history/changelog.md``. Judging agents (``JUDGES``) are never adopted this way, and at most
   ``MAX_ADOPTIONS_PER_WEEK`` adoptions happen in seven days; past either limit the proposal waits for a person.

Between 3 and 4 runs **failure replay** (free models, factory/failures.py), when there is a case to replay: the
failure that asked for this change (``case``) and the agent's fixed failures, rebuilt from their snapshots with the
candidate in place. Every one must be delivered. A fixed failure is a regression test: a later prompt for the same
agent cannot bring it back. Once a candidate is adopted and its own failure delivers, that case is marked fixed.

**Recurring errors** (``--from-events``, factory/events.py ``recurring``): an error fingerprint still *open* in 3 of the
newest 10 runs opens a case with no project snapshot (so nothing is replayed for it) and a proposal for the agent whose
stage it is, under every gate and limit above. The same error is taken up again only after it recurs in 3 later runs.
The benchmark's candidate choice breaks a tie in solved tasks by the fewest events the runs logged, then by tokens.

Every proposal, passed or not, is appended to ``proposals/archive.jsonl``: the archive of variants and their scores
that later proposals start from. The proposer also sees the fixes that worked on archived failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from factory.bench import BenchTask, Outcome, load_tasks, run_one
from factory.config import FactoryConfig, load_config
from factory.events import MIN_RUNS, WINDOW, Recurrence, event_count, recurring
from factory.failures import MAX_REPLAYS, FailureArchive
from factory.failures import replay as replay_case
from factory.llm import LLM, BudgetExceeded, complete_validated, new_run_dir
from factory.llm.gate import SpendDenied, SpendRequest, deny_all
from factory.orchestrator import TEMPLATES
from factory.registry import Registry
from factory.registry.spec import AgentSpec
from factory.secrets import scan_text

MAX_PROMPT = 3000
TOKEN_RATIO = 1.5
BENCH_CAPABILITIES = frozenset(s.capability for s in TEMPLATES["bug_fix"])  # what the held-out tasks exercise
SECOND_PANELIST = "plan_analyst"  # gpt-oss: joins the panel when the benchmark cannot judge the agent
# Agents that judge, vote or audit. Changing how they judge is a structural change: ASPIRE never adopts a new prompt
# for them on its own, so it cannot soften the checks its own work must pass.
JUDGES = frozenset({"reviewer", "pr_reviewer", "plan_critic", "plan_analyst", "project_critic", "final_evaluator",
                    "wiki_verifier", "knowledge_evaluator", "auditor_1", "auditor_2"})
MAX_ADOPTIONS_PER_WEEK = 3
MAX_CHANGES = 2  # prompt changes one recurring-error check may try (the same limit as a re-design)
CANDIDATES = 3  # prompts the proposer writes per proposal; only the best goes through the panel

PROPOSE_PROMPT = """Propose an improved version of one of ASPIRE's model agents. Only its prompt and description may
change; the model, tools and capabilities stay as they are.

## Current agent ({name})
```yaml
{spec}
```

## Where it fell short (development tasks only)
{evidence}

## Earlier proposals and how they scored
{archive}

## Fixes to this agent that worked on earlier failures
{fixes}
{guidance}{others}
Reply with JSON only: {{"name": "<new snake_case name, not {name}>", "description": "<one sentence>",
"prompt": "<the new system prompt>", "why": "<what you changed and what it should fix>"}}"""

PANEL_PROMPT = """You review a proposed change to an AI agent's instructions. You did not write it.

## Current
```yaml
{old}
```

## Proposed
```yaml
{new}
```

## Benchmark on held-out tasks
{bench}

Accept only if the new instructions are clearer or better targeted and add nothing unsafe: no request to skip,
edit or weaken tests, to read or send credentials, to approve without checking, or to act outside the task.
Reply with JSON only: {{"verdict": "accept" or "reject", "concerns": ["<concern>", ...]}}"""


class Proposal(BaseModel):
    name: str
    description: str
    prompt: str
    why: str = ""


RANK_PROMPT = """Several rewrites of an AI agent's instructions were proposed. You did not write them. Pick the one
most likely to fix what is described below without making the agent worse at its job.

## Current
```yaml
{old}
```

## What should change
{guidance}

## Candidates
{candidates}

Reply with JSON only: {{"best": "<candidate name>", "why": "<one sentence>"}}"""


class Pick(BaseModel):
    best: str
    why: str = ""


class PanelVerdict(BaseModel):
    verdict: Literal["accept", "reject"]
    concerns: list[str] = []


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class Result:
    id: str
    incumbent: str
    candidate: AgentSpec | None
    why: str
    gates: list[Gate] = field(default_factory=list)
    folder: Path | None = None
    adopted: Path | None = None  # the registry file it was written over, once adopted

    @property
    def status(self) -> str:
        if not self.gates or not all(g.passed for g in self.gates):
            failed = next((g.name for g in self.gates if not g.passed), "proposal")
            return f"rejected at {failed}"
        return "adopted" if self.adopted else "awaiting a person"


def check_scope(proposal: Proposal, incumbent: AgentSpec, registry: Registry,
                taken: set[str] = frozenset()) -> tuple[AgentSpec | None, Gate]:
    """Gate 1: what a proposal may change. Plain code, no model. ``taken``: names of the other candidates."""
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", proposal.name):
        problems.append(f"name {proposal.name!r} is not snake_case")
    if proposal.name in registry.agents or proposal.name in taken:
        problems.append(f"name {proposal.name!r} is taken")
    if len(proposal.prompt) > MAX_PROMPT:
        problems.append(f"prompt is {len(proposal.prompt)} characters (limit {MAX_PROMPT})")
    if hits := scan_text(proposal.model_dump_json()):
        problems.append(f"holds a secret: {', '.join(hits)}")
    spec = None
    try:  # everything but the name, description and prompt comes from the incumbent
        spec = incumbent.model_copy(update={"name": proposal.name, "description": proposal.description,
                                            "prompt": proposal.prompt})
        AgentSpec.model_validate(spec.model_dump())
    except ValidationError as e:
        problems.append(f"invalid spec: {e.errors()[0]['msg']}")
    if incumbent.kind != "llm":
        problems.append("only model agents can be proposed")
    detail = "; ".join(problems) or f"same role ({incumbent.role}), capabilities and tools; prompt only"
    return (None if problems else spec), Gate("scope", not problems, detail)


def evaluate(cfg: FactoryConfig, tasks: list[BenchTask], agents: list[AgentSpec], *,
             providers: dict | None = None, registry_dir: Path | None = None) -> list[Outcome]:
    return [run_one(cfg, t, "aspire", agents=agents, providers=providers, registry_dir=registry_dir) for t in tasks]


def strict(o: Outcome) -> bool:
    """Solved, and not failed by the task's hidden tests (when it has them): a prompt that teaches the agent to fit
    the visible tests only does not count."""
    return o.solved and getattr(o, "hidden_solved", None) is not False


def errors_in(runs: list[Outcome]) -> int:
    """Events the benchmark runs wrote (factory/events.py): retries, fallbacks, unusable outputs. A candidate that
    solves as many tasks with fewer of them is the steadier prompt."""
    return sum(event_count(o.run_dir) for o in runs)


def benchmark_gate(incumbent_runs: list[Outcome], candidate_runs: list[Outcome]) -> Gate:
    """Gate 2: at least as many held-out tasks solved (hidden tests included), within 1.5x the tokens."""
    solved = (sum(map(strict, incumbent_runs)), sum(map(strict, candidate_runs)))
    tokens = (sum(o.tokens for o in incumbent_runs), sum(o.tokens for o in candidate_runs))
    ok = solved[1] >= solved[0] and tokens[1] <= TOKEN_RATIO * max(tokens[0], 1)
    return Gate("held-out benchmark", ok, f"solved {solved[1]}/{len(candidate_runs)} vs {solved[0]}/"
                f"{len(incumbent_runs)}; tokens {tokens[1]:,} vs {tokens[0]:,} (limit {TOKEN_RATIO}x)")


def spec_yaml(spec: AgentSpec) -> str:
    return yaml.safe_dump(spec.model_dump(exclude_defaults=True), sort_keys=False, width=110)


def dev_evidence(outcomes: list[Outcome]) -> str:
    misses = [o for o in outcomes if not o.solved]
    return "\n".join(f"- {o.task}: {o.why}" for o in misses) or "- it solved every development task; aim for fewer tokens"


def archive_text(path: Path, incumbent: str) -> str:
    if not path.exists():
        return "- none yet"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["incumbent"] == incumbent][-5:]
    return "\n".join(f"- {r['name']}: {r['status']} ({r['benchmark']})" for r in rows) or "- none yet"


DEFAULT_TASK_FILES = ("bench/tasks.yaml", "bench/tasks_hard.yaml")


def gate_tasks(root: Path, tasks_file: Path | list[Path] | None = None) -> list[BenchTask]:
    """The benchmark tasks self-extension learns from (development) and is judged on (held out): both tiers unless
    ``tasks_file`` names one or more files. Each file's ``heldout`` flags are kept as written."""
    files = [root / f for f in DEFAULT_TASK_FILES] if tasks_file is None else \
        [tasks_file] if isinstance(tasks_file, Path) else list(tasks_file)
    if missing := [str(f) for f in files if not f.is_file()]:
        raise FileNotFoundError(f"no benchmark file at {', '.join(missing)}")
    tasks = [t for f in files for t in load_tasks(f, root)]
    if len({t.name for t in tasks}) != len(tasks):
        raise ValueError("the benchmark files share a task name")
    return tasks


def propose(cfg: FactoryConfig, agent: str, *, approver: Callable[[SpendRequest], bool] = deny_all,
            paid_panel: bool = False, providers: dict | None = None, registry_dir: Path | None = None,
            tasks_file: Path | list[Path] | None = None, dev_runs: list[Outcome] | None = None, guidance: str = "",
            adopt_if_passed: bool = False, by_person: bool = False, candidates: int = CANDIDATES, case: str = "",
            replay: bool = True) -> Result:
    """Run one proposal through every gate it reaches and write it up. ``guidance`` is what a re-design audit asked
    for, and ``case`` the archived failure it came from. The registry is written only with ``adopt_if_passed`` and
    only after every gate passes. ``by_person`` means a person asked for the adoption (the CLI's ``--adopt``), so
    the judge and weekly limits, which guard adoptions nobody asked for, do not apply. ``replay=False`` skips the
    failure replays (each is a whole build on free models)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir(cfg, f"extend-{agent}")
    llm = LLM(cfg, run_dir, use_cache=False, approver=approver, **({"providers": providers} if providers else {}))
    registry = Registry.load(cfg, registry_dir)
    incumbent = registry.agents[agent]
    tasks = gate_tasks(cfg.root, tasks_file)
    dev, heldout = [t for t in tasks if not t.heldout], [t for t in tasks if t.heldout]
    out = cfg.work_dir / "proposals"
    archive = out / "archive.jsonl"
    failures = FailureArchive.of(cfg)
    data = "public" if cfg.data_public else "private"

    benched = bool(BENCH_CAPABILITIES & set(incumbent.capabilities))
    if dev_runs is None:  # the proposer learns from the development tasks only
        dev_runs = (evaluate(cfg, dev, list(registry.agents.values()), providers=providers, registry_dir=registry_dir)
                    if benched else [])
    proposals = write_candidates(llm, registry, incumbent, max(1, candidates), data, guidance=guidance,
                                 evidence=dev_evidence(dev_runs) if benched else "- not exercised by the development "
                                 "tasks", archive=archive_text(archive, agent), fixes=failures.past_fixes(agent))
    scoped, taken = [], set()
    for p in proposals:
        spec, gate = check_scope(p, incumbent, registry, taken)
        taken.add(p.name)
        scoped.append((p, spec, gate))
    valid = [(p, spec) for p, spec, gate in scoped if gate.passed]
    if not valid:  # report the first candidate's problems
        proposal, candidate, scope = scoped[0]
        if len(scoped) > 1:
            scope = Gate("scope", False, " | ".join(f"{p.name}: {g.detail}" for p, _, g in scoped))
    else:
        bench_gate, best = choose(cfg, llm, registry, incumbent, valid, benched, heldout, guidance, data,
                                  providers=providers, registry_dir=registry_dir)
        proposal, candidate = valid[best]
        scope = next(g for p, _, g in scoped if p is proposal)
    result = Result(id=f"{stamp}-{proposal.name}", incumbent=agent, candidate=candidate, why=proposal.why,
                    gates=[scope])
    bench = "not run"
    replayed: dict[str, Outcome] = {}
    if scope.passed:
        bench = bench_gate.detail
        result.gates.append(bench_gate)
        if bench_gate.passed:
            result.gates.append(panel(llm, registry, incumbent, candidate, bench, data, paid_panel,
                                      second_family=not benched))
        cases = ([failures.get(case)] if case and failures.replayable(case) else []) + \
            failures.regressions(agent, exclude=case)
        if replay and cases and result.gates[-1].passed:
            others = [a for a in registry.agents.values() if a.name != agent]
            for c in cases[:MAX_REPLAYS]:
                replayed[c.id] = replay_case(cfg, failures, c, [*others, candidate], providers=providers)
            result.gates.append(replay_gate(replayed, target=case))
    if adopt_if_passed and result.status == "awaiting a person":
        if not by_person and (why := may_adopt(cfg, agent, registry_dir)):
            result.gates.append(Gate("automatic adoption", True, f"not applied: {why}; waits for a person"))
        else:
            result.adopted = adopt(cfg, incumbent, candidate, registry_dir, stamp, proposal.why)
            if case and case in replayed and replayed[case].solved:
                failures.record_fix(case, agent, result.id, guidance or proposal.why)
            elif case and not failures.replayable(case):  # a recurrence case: nothing to replay, the gates passed
                failures.record_fix(case, agent, result.id, guidance or proposal.why)

    result.folder = out / result.id
    result.folder.mkdir(parents=True, exist_ok=True)
    for _, spec in valid:  # every candidate that passed scope, for a person comparing them
        (result.folder / f"{spec.name}.yaml").write_text(spec_yaml(spec), encoding="utf-8")
    (result.folder / "report.md").write_text(report(result, incumbent), encoding="utf-8")
    with archive.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": stamp, "incumbent": agent, "name": proposal.name, "status": result.status,
                            "benchmark": bench, "candidates": [p.name for p in proposals], "case": case,
                            "gates": [g.__dict__ for g in result.gates]}) + "\n")
    llm.write_summary()
    return result


def write_candidates(llm: LLM, registry: Registry, incumbent: AgentSpec, n: int, data: str, *, guidance: str,
                     evidence: str, archive: str, fixes: str) -> list[Proposal]:
    """Ask the proposer for ``n`` candidates, each after seeing the ones before it. A candidate that never parses is
    skipped; at least one must."""
    proposer = registry.agents["planner"]  # an author-family model; the panel is another family
    out: list[Proposal] = []
    for i in range(n):
        others = ("\n## Candidates already written (take a clearly different approach)\n" +
                  "\n".join(f"- {p.name}: {p.why}" for p in out) + "\n") if out else ""
        prompt = PROPOSE_PROMPT.format(name=incumbent.name, spec=spec_yaml(incumbent), evidence=evidence,
                                       archive=archive, fixes=fixes, others=others,
                                       guidance=f"\n## What a re-design audit asked for\n{guidance}\n" if guidance
                                       else "")
        try:
            p, _, _ = complete_validated(llm, proposer.role, prompt, proposer.system, Proposal,
                                         tag=f"propose-{i + 1}", data=data)
        except RuntimeError:
            if not out and i == n - 1:
                raise
            continue
        out.append(p)
    return out


def choose(cfg: FactoryConfig, llm: LLM, registry: Registry, incumbent: AgentSpec,
           valid: list[tuple[Proposal, AgentSpec]], benched: bool, heldout: list[BenchTask], guidance: str, data: str,
           *, providers: dict | None, registry_dir: Path | None) -> tuple[Gate, int]:
    """Score the candidates and return gate 2 for the best one, with its index in ``valid``."""
    if benched:
        others = [a for a in registry.agents.values() if a.name != incumbent.name]
        before = evaluate(cfg, heldout, list(registry.agents.values()), providers=providers,
                          registry_dir=registry_dir)
        runs = [evaluate(cfg, heldout, [*others, spec], providers=providers, registry_dir=registry_dir)
                for _, spec in valid]
        # most solved; then the fewest errors the runs logged (the event-rate tiebreaker); then the fewest tokens
        best = max(range(len(runs)), key=lambda i: (sum(map(strict, runs[i])), -errors_in(runs[i]),
                                                    -sum(o.tokens for o in runs[i])))
        gate = benchmark_gate(before, runs[best])
        if any(errors_in(r) for r in [*runs, before]):
            gate.detail += f"; logged errors {errors_in(runs[best])} vs {errors_in(before)}"
        if len(valid) > 1:
            gate.detail = (f"best of {len(valid)} ({valid[best][1].name}): {gate.detail}; others: " +
                           ", ".join(f"{spec.name} {sum(map(strict, r))}/{len(r)}"
                                     for (_, spec), r in zip(valid, runs, strict=True) if spec is not valid[best][1]))
        return gate, best
    best, why = 0, ""
    if len(valid) > 1:  # no benchmark to score them: a free reviewer of another family picks
        reviewer = registry.agents["reviewer"]
        shown = "\n\n".join(f"### {spec.name}\n```yaml\n{spec_yaml(spec)}```" for _, spec in valid)
        try:
            pick, _, _ = complete_validated(llm, reviewer.role, RANK_PROMPT.format(
                old=spec_yaml(incumbent), guidance=guidance or "- make it better at its job", candidates=shown),
                reviewer.system, Pick, tag="rank", data=data)
            names = [spec.name for _, spec in valid]
            best, why = (names.index(pick.best), pick.why) if pick.best in names else (0, "reviewer named none")
        except RuntimeError as e:
            why = f"no usable pick ({str(e)[:80]}); kept the first"
    detail = "not exercised by the benchmark; two panel families judge it"
    if len(valid) > 1:
        detail += f"; the reviewer picked {valid[best][1].name} of {len(valid)}" + (f" ({why})" if why else "")
    return Gate("held-out benchmark", True, detail), best


def replay_gate(replayed: dict[str, Outcome], target: str = "") -> Gate:
    """Gate 3b: every replayed failure is delivered with the candidate in place."""
    ok = all(o.solved for o in replayed.values())
    parts = [f"{cid}{' (this failure)' if cid == target else ' (fixed before)'}: "
             f"{'delivered' if o.solved else o.why}" for cid, o in replayed.items()]
    return Gate("failure replay", ok, "; ".join(parts))


def may_adopt(cfg: FactoryConfig, agent: str, registry_dir: Path | None = None) -> str:
    """Why a passed proposal may not be adopted without a person, or "" when it may."""
    if agent in JUDGES:
        return f"{agent} judges other agents' work, so changing it is structural"
    directory = registry_dir or cfg.root / cfg.registry_dir
    week_ago = datetime.now(UTC).timestamp() - 7 * 86400
    recent = [p for p in (directory / "history").glob("*/*.yaml") if p.stat().st_mtime > week_ago]
    if len(recent) >= MAX_ADOPTIONS_PER_WEEK:
        return f"{len(recent)} prompts were already adopted this week (limit {MAX_ADOPTIONS_PER_WEEK})"
    return ""


def log_change(directory: Path, line: str) -> None:
    log = directory / "history" / "changelog.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text("# Prompt changes ASPIRE made to itself\n\nUndo one with "
                       "`python -m factory.extend --rollback <agent>`.\n\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now(UTC):%Y-%m-%d %H:%M}Z {line}\n")


def adopt(cfg: FactoryConfig, incumbent: AgentSpec, candidate: AgentSpec, registry_dir: Path | None,
          stamp: str, why: str = "") -> Path:
    """Write a passed candidate's prompt and description over the incumbent's spec, keeping its name and everything
    else, and keep the old spec for rollback. Plain code; called only after every gate has passed."""
    directory = registry_dir or cfg.root / cfg.registry_dir
    target = directory / "agents" / f"{incumbent.name}.yaml"
    history = directory / "history" / incumbent.name
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{stamp}.yaml").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    new = incumbent.model_copy(update={"description": candidate.description, "prompt": candidate.prompt})
    AgentSpec.model_validate(new.model_dump())
    target.write_text(spec_yaml(new), encoding="utf-8")
    log_change(directory, f"adopted a new prompt for `{incumbent.name}`: {why or candidate.description}")
    return target


def rollback(cfg: FactoryConfig, agent: str, registry_dir: Path | None = None) -> Path:
    """Put back the spec an agent had before its last adoption."""
    directory = registry_dir or cfg.root / cfg.registry_dir
    saved = sorted((directory / "history" / agent).glob("*.yaml"))
    if not saved:
        raise FileNotFoundError(f"no earlier version of {agent!r} to roll back to")
    target = directory / "agents" / f"{agent}.yaml"
    target.write_text(saved[-1].read_text(encoding="utf-8"), encoding="utf-8")
    saved[-1].unlink()
    reopened = FailureArchive.of(cfg).revert_fixes(agent)
    log_change(directory, f"rolled back `{agent}` to its prompt of {saved[-1].stem}" +
               (f"; reopened {', '.join(reopened)}" if reopened else ""))
    return target


def panel(llm: LLM, registry: Registry, incumbent: AgentSpec, candidate: AgentSpec, bench: str, data: str,
          paid: bool, second_family: bool = False) -> Gate:
    """Gate 3: a reviewer of another family reads the change; with ``second_family``, a gpt-oss reviewer too (for
    agents the benchmark cannot judge); with ``paid``, the paid reviewer too, after the spend gate's approval. Every
    member must accept."""
    reviewer = registry.agents["reviewer"]
    prompt = PANEL_PROMPT.format(old=spec_yaml(incumbent), new=spec_yaml(candidate), bench=bench)
    notes = []
    members = [(reviewer.role, False)] + ([(SECOND_PANELIST, False)] if second_family else []) + \
              ([(reviewer.role, True)] if paid else [])
    for role, escalate in members:
        try:
            c = llm.complete(role, prompt, system=reviewer.system,
                             tag=f"panel-{'paid' if escalate else role}", data=data, escalate=escalate)
        except (SpendDenied, BudgetExceeded) as e:
            return Gate("review panel", False, "; ".join([*notes, f"paid reviewer not run: {e}"]))
        verdict = PanelVerdict.model_validate(c.json())
        notes.append(f"{c.model}: {verdict.verdict}" + (f" ({'; '.join(verdict.concerns)})" if verdict.concerns else ""))
        if verdict.verdict != "accept":
            return Gate("review panel", False, "; ".join(notes))
    return Gate("review panel", True, "; ".join(notes))


def report(result: Result, incumbent: AgentSpec) -> str:
    lines = [f"# Proposal {result.id}", "", f"Replaces `{result.incumbent}` for its capabilities. Status: "
             f"**{result.status}**.", "", f"Proposer's reasoning: {result.why}", "", "## Gates", ""]
    lines += [f"- {'passed' if g.passed else 'FAILED'}: **{g.name}**: {g.detail}" for g in result.gates]
    if result.candidate:
        lines += ["", "## Current", "", "```yaml", spec_yaml(incumbent).rstrip(), "```", "", "## Proposed", "",
                  "```yaml", spec_yaml(result.candidate).rstrip(), "```"]
    if result.status == "adopted":
        lines += ["", "## Adopted", "", f"Every gate passed, so the new prompt replaced `{result.incumbent}`'s in "
                  f"`{result.adopted}` (a prompt-only change needs no approval). To undo it: "
                  f"`python -m factory.extend --rollback {result.incumbent}`."]
    elif result.status == "awaiting a person":
        lines += ["", "## To adopt it", "", f"Copy `{result.candidate.name}.yaml` into `registry/agents/` in a pull "
                  "request, or rerun with `--adopt`. As a new agent it bids against the current one, and the "
                  "contract net awards the step to whichever has the better record."]
    return "\n".join(lines) + "\n"


def agent_for(rec: Recurrence, registry: Registry) -> str:
    """The model agent a recurring error belongs to, or "": its stage is the agent's name, or its role when only one
    agent has it. Anything else (a QA verdict, a stage name that is not an agent) needs a person to read it."""
    llm_agents = [a for a in registry.agents.values() if a.kind == "llm"]
    named = [a.name for a in llm_agents if a.name == rec.stage]
    by_role = [a.name for a in llm_agents if a.role == rec.stage]
    return named[0] if named else by_role[0] if len(by_role) == 1 else ""


def redesign_from_events(cfg: FactoryConfig, *, approver: Callable[[SpendRequest], bool] = deny_all,
                         paid_panel: bool = False, adopt: bool = False, window: int = WINDOW, min_runs: int = MIN_RUNS,
                         max_changes: int = MAX_CHANGES, registry_dir: Path | None = None, providers: dict | None = None,
                         **kw) -> list[tuple[Recurrence, str, Result | None]]:
    """Open a re-design case for each error that stays open in ``min_runs`` of the newest ``window`` runs and try one
    prompt change for the agent it belongs to. Every limit of a normal re-design holds: at most ``max_changes`` changes
    (MAX_CHANGES, as in a re-design), all the gates, ``may_adopt`` (judges and the weekly limit) and the
    spend gate for paid calls. A problem is taken up again only after it recurs in ``min_runs`` later runs."""
    failures = FailureArchive.of(cfg)
    registry = Registry.load(cfg, registry_dir)
    found = recurring(cfg.root / cfg.runs_dir, window=window, min_runs=min_runs, since=failures.last_case_stamps())
    out, tried = [], 0
    for rec in found:
        agent = agent_for(rec, registry)
        if not agent:
            out.append((rec, "no single model agent owns this stage; a person should read it", None))
        elif tried >= max_changes:
            out.append((rec, f"not tried: limit of {max_changes} changes per check", None))
        else:
            tried += 1
            case = failures.open_recurrence_case(rec.fingerprint, rec.message, rec.stage, rec.runs, agent)
            guidance = (f"This error stayed open in {len(rec.runs)} of the last {rec.window} runs ({rec.source}, stage "
                        f"'{rec.stage}'): {rec.message}. Change the prompt so it stops happening.")
            out.append((rec, f"case {case.id}", propose(cfg, agent, approver=approver, paid_panel=paid_panel,
                                                        adopt_if_passed=adopt, case=case.id, guidance=guidance,
                                                        registry_dir=registry_dir, providers=providers, **kw)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.extend")
    ap.add_argument("--agent", help="the model agent to improve, e.g. coder")
    ap.add_argument("--guidance", default="", help="what the change should fix, e.g. from a re-design report")
    ap.add_argument("--adopt", action="store_true", help="apply the new prompt if it passes every gate")
    ap.add_argument("--rollback", metavar="AGENT", help="restore the agent's spec from before its last adoption")
    ap.add_argument("--config", default="configs/factory.yaml")
    ap.add_argument("--paid-panel", action="store_true", help="also ask the paid reviewer (asks before spending)")
    ap.add_argument("--case", default="", help="an archived failure (workspace/failures/<id>) the change should fix")
    ap.add_argument("--candidates", type=int, default=CANDIDATES, help="candidate prompts to write and compare")
    ap.add_argument("--from-events", action="store_true",
                    help="re-design from errors that stay open across runs (adopts only with --adopt)")
    ap.add_argument("--window", type=int, default=WINDOW, help="--from-events: newest runs to look at")
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS, help="--from-events: runs an error must be open in")
    ap.add_argument("--no-replay", action="store_true", help="skip replaying archived failures (each is a full build)")
    ap.add_argument("--tasks", action="append", default=[], metavar="FILE",
                    help="benchmark file(s) for the gate (default: bench/tasks.yaml and bench/tasks_hard.yaml, "
                         "nine held-out tasks; one run of each per candidate, free models)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.rollback:
        print(f"restored {rollback(cfg, args.rollback)}")
        return 0
    if not args.agent and not args.from_events:
        ap.error("--agent, --from-events or --rollback is required")
    approver = deny_all
    if args.paid_panel:
        from factory.__main__ import ask_on_terminal
        approver = ask_on_terminal
    if args.from_events:
        rows = redesign_from_events(cfg, approver=approver, paid_panel=args.paid_panel, adopt=args.adopt,
                                    window=args.window, min_runs=args.min_runs, candidates=args.candidates,
                                    replay=not args.no_replay)
        for rec, note, res in rows:
            print(f"{len(rec.runs)}/{rec.window} runs  {rec.source}/{rec.stage}: {rec.message[:100]}\n    -> {note}"
                  + (f"; {res.status}: {res.folder / 'report.md'}" if res else ""))
        print(f"\n{len(rows)} recurring error(s)" if rows else "no error is open in enough runs")
        return 0
    result = propose(cfg, args.agent, approver=approver, paid_panel=args.paid_panel, guidance=args.guidance,
                     adopt_if_passed=args.adopt, by_person=args.adopt, case=args.case, candidates=args.candidates,
                     replay=not args.no_replay, tasks_file=[cfg.root / f for f in args.tasks] or None)
    for g in result.gates:
        print(f"{'passed' if g.passed else 'FAILED':<7} {g.name}: {g.detail}")
    print(f"\n{result.status}: {result.folder / 'report.md'}")
    return 0 if result.status in ("awaiting a person", "adopted") else 1


if __name__ == "__main__":
    sys.exit(main())
