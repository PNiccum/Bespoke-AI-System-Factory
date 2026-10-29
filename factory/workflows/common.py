"""What every workflow does first and last: award its steps, log the intake, and close the run."""

from __future__ import annotations

from factory.orchestrator import Orchestrator, Plan, RunResult, Step, Task, sha12
from factory.records import ProvenanceEvent
from factory.registry import Bid


def begin(orch: Orchestrator, task: Task, steps: list[Step]) -> tuple[RunResult, dict[str, Bid]]:
    plan = Plan(task_type=task.task_type, steps=steps)
    result = RunResult(status="failed", plan=plan)
    bids = {s.id: orch.award(task, s) for s in steps}
    orch.provenance.record(ProvenanceEvent(kind="source", actor="intake", outputs=[sha12(task.description)],
                                           detail={"task": task.model_dump(), "plan": plan.model_dump(),
                                                   "awards": {k: b.agent.name for k, b in bids.items()}}))
    for i, (sid, bid) in enumerate(bids.items()):
        orch.emit(result, sid, bid.agent.name, "started" if i == 0 else "waiting", f"awarded with bid {bid.score}",
                  parts=bid.parts)
    return result, bids


def finish(orch: Orchestrator, task: Task, result: RunResult, bids: dict[str, Bid], passed: bool) -> RunResult:
    for bid in bids.values():
        orch.registry.scores.record(bid.agent.name, task.task_type, passed=passed)
    orch.llm.write_summary()
    return result
