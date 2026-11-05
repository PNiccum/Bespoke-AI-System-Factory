"""The pipeline after the knowledge loop: production, the critic loop, final QA, and re-design when all else fails.

    production -> critic loop -> final evaluator -> deliverable
                     |                  | not approved: the solver repairs QA's findings (2 tries a round), QA looks again;
     |                  | still not approved: back to the head orchestrator with the issues, once
                     | declined after research, or QA fails twice
                     v
                  re-design (report to the person)
"""

from __future__ import annotations

from pydantic import BaseModel

from factory.events import log_of
from factory.llm import LLM, BudgetExceeded
from factory.llm.gate import SpendDenied
from factory.pipeline.blackboard import Project
from factory.pipeline.critique import CriticLoop
from factory.pipeline.domain import project_domain
from factory.pipeline.final import FinalEvaluator, FinalResult
from factory.pipeline.knowledge import KnowledgeLoop
from factory.pipeline.production import Production
from factory.pipeline.redesign import Redesign, RedesignResult
from factory.pipeline.sources import Searcher
from factory.pipeline.wiki import Wiki
from factory.registry import Registry

REPAIRS = 2  # repair attempts on final QA's findings, before a whole rebuild
BUILD_ROUNDS = 2  # the first build, plus one rebuild with the final evaluator's issues


class BuildResult(BaseModel):
    status: str  # delivered | redesign | stopped
    final: FinalResult | None = None
    redesign: RedesignResult | None = None
    rounds: int
    repairs: int = 0  # repairs made on final QA's findings (a delivered run with repairs was first rejected)


def build(llm: LLM, registry: Registry, project: Project, wiki: Wiki, searchers: dict[str, Searcher], **kw) -> BuildResult:
    """Run the build (see ``_build``); whatever escapes it is written to the run's event log before it propagates."""
    with log_of(llm).capture("pipeline", "build"):
        return _build(llm, registry, project, wiki, searchers, **kw)


def _build(llm: LLM, registry: Registry, project: Project, wiki: Wiki, searchers: dict[str, Searcher],
          on_stage=lambda stage: None, redesign: bool = True, allow_unapproved: bool = False) -> BuildResult:
    """``redesign=False`` (a replay of an archived failure, factory/failures.py) stops where re-design would start.
    A project whose knowledge loop did not approve its knowledge is refused, unless ``allow_unapproved`` (to exercise
    the later stages on purpose; such a run is not a fair end-to-end result)."""
    k = project.board.latest("knowledge")
    if k is not None and not k.data.get("ok") and not allow_unapproved:
        raise RuntimeError(f"the knowledge loop did not approve this project's knowledge (missing: "
                           f"{', '.join(k.data.get('missing', [])) or 'unknown'}); rerun research first")
    events = log_of(llm)
    events.domain = project_domain(project)
    production = Production(llm, registry, project, wiki)
    knowledge = KnowledgeLoop(llm, registry, project, wiki, searchers)
    critic = CriticLoop(llm, registry, project, wiki, production, knowledge)
    final = FinalEvaluator(llm, registry, project, production)
    feedback, result, stage, repairs = "", None, "", 0
    production.start_clean()  # a new build never inherits an earlier build's files
    for rnd in range(1, BUILD_ROUNDS + 1):
        try:
            on_stage(stage := f"production, round {rnd}")
            checks = production.run(feedback)
            on_stage(stage := "critic loop")
            reviewed = critic.run(checks)
            # final QA runs even when the critic escalates: its approval is gated in code on the checks passing,
            # and a stalled critic is no reason to withhold a working result (2026-11-04: log_spikes passed 5/5
            # acceptance tests but stopped after critic round 3 without QA)
            on_stage(stage := "final QA")
            result = final.run(critic.shown_files())
            if not result.approved:
                events.record("qa", "final QA did not approve: " + "; ".join(result.issues)[:400], stage="final QA",
                              severity="warning", refs=[f"round {rnd}"])
            # QA rejects only after it has tried to repair: its findings go to the solver, then QA looks again
            while not result.approved and repairs < REPAIRS * rnd:
                on_stage(stage := "final QA repair")
                if not critic.repair(result, rnd):
                    break
                repairs += 1
                on_stage(stage := "final QA")
                result = final.run(critic.shown_files())
            if reviewed.redesign and not result.approved:
                break
        except (SpendDenied, BudgetExceeded):
            raise  # a person said no to spending, or the cap was reached: stop, do not re-design
        except RuntimeError as e:  # a stage that crashed is a failure re-design should study, not a lost run
            events.record("pipeline", f"{stage} failed: {e}", stage=stage, refs=[f"round {rnd}"])
            project.board.post("note", "code", {"stage": stage, "error": str(e)[:2000]})
            break
        if result.approved:
            how = (f"delivered in round {rnd}" + (f" after {repairs} QA repair(s)" if repairs else "")
                   + (" (an earlier round failed)" if rnd > 1 else ""))
            for src in ("qa", "pipeline"):
                events.resolve_open(how, [project.id], source=src)
            return BuildResult(status="delivered", final=result, rounds=rnd, repairs=repairs)
        feedback = "\n".join(f"- {i}" for i in result.issues)
    if not redesign:
        return BuildResult(status="stopped", final=result, rounds=rnd, repairs=repairs)
    on_stage("re-design")
    rd = Redesign(llm, registry, project, research=knowledge.research).run()
    adopted = [a for a in rd.prompt_changes if a.status == "adopted"]
    if adopted:  # an adopted change passed the scope, benchmark, panel and failure-replay gates for this very case
        for src in ("qa", "pipeline"):
            log_of(llm).resolve_open(f"re-design adopted a change to {', '.join(a.agent for a in adopted)} that "
                                     f"delivered the archived case", [rd.case], source=src)
    return BuildResult(status="redesign", final=result, rounds=rnd, repairs=repairs, redesign=rd)
