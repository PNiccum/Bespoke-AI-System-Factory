"""Self-extension, dry run: ASPIRE proposes a better version of one of its own model agents, judges it on tasks the
proposer never saw, and stops for a person. It never writes to the registry.

A proposal is a new agent spec only: a prompt and a description for an existing role and capability. It cannot add
code, tools, roles or capabilities, so it cannot change what ASPIRE is able to touch, only how a model is asked.

    python -m factory.extend --agent coder                  # free models only
    python -m factory.extend --agent coder --paid-panel     # also asks a paid reviewer, after your approval

The gates, in order; a proposal stops at the first it fails:

1. **Scope** (plain code): the spec validates, is a model agent on the same role (so the same model and price), keeps
   or narrows the incumbent's capabilities and tools, has a new name, a prompt under 3,000 characters, and no secret.
2. **Held-out benchmark** (plain code judging free model runs): the candidate replaces the incumbent on the held-out
   tasks of ``bench/tasks.yaml``. It must solve at least as many, without using more than 1.5 times the tokens.
3. **Review panel** (free model of another family; optionally a paid one): reads the old and new spec and must accept.
4. **A person**: the proposal, its evidence and the exact file to add are written to
   ``workspace/<name>/proposals/<id>/``. Adopting it is a pull request a person opens; this module stops here.

Every proposal, passed or not, is appended to ``proposals/archive.jsonl``: the archive of variants and their scores
that later proposals start from.
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
from factory.llm import LLM, BudgetExceeded, complete_validated, new_run_dir
from factory.llm.gate import SpendDenied, SpendRequest, deny_all
from factory.registry import Registry
from factory.registry.spec import AgentSpec
from factory.secrets import scan_text

MAX_PROMPT = 3000
TOKEN_RATIO = 1.5

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

    @property
    def status(self) -> str:
        if not self.gates or not all(g.passed for g in self.gates):
            failed = next((g.name for g in self.gates if not g.passed), "proposal")
            return f"rejected at {failed}"
        return "awaiting a person"


def check_scope(proposal: Proposal, incumbent: AgentSpec, registry: Registry) -> tuple[AgentSpec | None, Gate]:
    """Gate 1: what a proposal may change. Plain code, no model."""
    problems = []
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", proposal.name):
        problems.append(f"name {proposal.name!r} is not snake_case")
    if proposal.name in registry.agents:
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


def benchmark_gate(incumbent_runs: list[Outcome], candidate_runs: list[Outcome]) -> Gate:
    """Gate 2: at least as many held-out tasks solved, within 1.5x the tokens."""
    solved = (sum(o.solved for o in incumbent_runs), sum(o.solved for o in candidate_runs))
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
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["incumbent"] == incumbent][-5:]
    return "\n".join(f"- {r['name']}: {r['status']} ({r['benchmark']})" for r in rows) or "- none yet"


def propose(cfg: FactoryConfig, agent: str, *, approver: Callable[[SpendRequest], bool] = deny_all,
            paid_panel: bool = False, providers: dict | None = None, registry_dir: Path | None = None,
            tasks_file: Path | None = None, dev_runs: list[Outcome] | None = None) -> Result:
    """Run one proposal through every gate it reaches and write it up. Never touches the registry."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_dir(cfg, f"extend-{agent}")
    llm = LLM(cfg, run_dir, use_cache=False, approver=approver, **({"providers": providers} if providers else {}))
    registry = Registry.load(cfg, registry_dir)
    incumbent = registry.agents[agent]
    tasks = load_tasks(tasks_file or cfg.root / "bench" / "tasks.yaml", cfg.root)
    dev, heldout = [t for t in tasks if not t.heldout], [t for t in tasks if t.heldout]
    out = cfg.work_dir / "proposals"
    archive = out / "archive.jsonl"
    data = "public" if cfg.data_public else "private"

    if dev_runs is None:  # the proposer learns from the development tasks only
        dev_runs = evaluate(cfg, dev, list(registry.agents.values()), providers=providers, registry_dir=registry_dir)
    proposer = registry.agents["planner"]  # an author-family model; the panel below is another family
    prompt = PROPOSE_PROMPT.format(name=agent, spec=spec_yaml(incumbent), evidence=dev_evidence(dev_runs),
                                   archive=archive_text(archive, agent))
    proposal, _, _ = complete_validated(llm, proposer.role, prompt, proposer.system, Proposal, tag="propose",
                                        data=data)
    candidate, scope = check_scope(proposal, incumbent, registry)
    result = Result(id=f"{stamp}-{proposal.name}", incumbent=agent, candidate=candidate, why=proposal.why,
                    gates=[scope])
    bench = "not run"
    if scope.passed:
        others = [a for a in registry.agents.values() if a.name != agent]
        before = evaluate(cfg, heldout, list(registry.agents.values()), providers=providers, registry_dir=registry_dir)
        after = evaluate(cfg, heldout, [*others, candidate], providers=providers, registry_dir=registry_dir)
        gate = benchmark_gate(before, after)
        bench = gate.detail
        result.gates.append(gate)
        if gate.passed:
            result.gates.append(panel(llm, registry, incumbent, candidate, bench, data, paid_panel))

    result.folder = out / result.id
    result.folder.mkdir(parents=True, exist_ok=True)
    if candidate:
        (result.folder / f"{candidate.name}.yaml").write_text(spec_yaml(candidate))
    (result.folder / "report.md").write_text(report(result, incumbent))
    with archive.open("a") as f:
        f.write(json.dumps({"at": stamp, "incumbent": agent, "name": proposal.name, "status": result.status,
                            "benchmark": bench, "gates": [g.__dict__ for g in result.gates]}) + "\n")
    llm.write_summary()
    return result


def panel(llm: LLM, registry: Registry, incumbent: AgentSpec, candidate: AgentSpec, bench: str, data: str,
          paid: bool) -> Gate:
    """Gate 3: a reviewer of another family reads the change; with ``paid``, the paid reviewer too, after the
    spend gate's approval. Every member must accept."""
    reviewer = registry.agents["reviewer"]
    prompt = PANEL_PROMPT.format(old=spec_yaml(incumbent), new=spec_yaml(candidate), bench=bench)
    notes = []
    for escalate in ([False, True] if paid else [False]):
        try:
            c = llm.complete(reviewer.role, prompt, system=reviewer.system,
                             tag=f"panel-{'paid' if escalate else 'free'}", data=data, escalate=escalate)
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
    if result.status == "awaiting a person":
        lines += ["", "## To adopt it", "", f"Copy `{result.candidate.name}.yaml` into `registry/agents/` in a pull "
                  "request. It will then bid against the current agent, and the contract net awards the step to "
                  "whichever has the better record. ASPIRE never does this step itself."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.extend")
    ap.add_argument("--agent", required=True, help="the model agent to improve, e.g. coder")
    ap.add_argument("--config", default="configs/factory.yaml")
    ap.add_argument("--paid-panel", action="store_true", help="also ask the paid reviewer (asks before spending)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    approver = deny_all
    if args.paid_panel:
        from factory.__main__ import ask_on_terminal
        approver = ask_on_terminal
    result = propose(cfg, args.agent, approver=approver, paid_panel=args.paid_panel)
    for g in result.gates:
        print(f"{'passed' if g.passed else 'FAILED':<7} {g.name}: {g.detail}")
    print(f"\n{result.status}: {result.folder / 'report.md'}")
    return 0 if result.status == "awaiting a person" else 1


if __name__ == "__main__":
    sys.exit(main())
