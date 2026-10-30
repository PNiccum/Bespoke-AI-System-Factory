"""The pipeline after the knowledge loop: production, the critic loop, final QA, and re-design when all else fails.

    production -> critic loop -> final evaluator -> deliverable
                     |                  | not approved: back to the head orchestrator with the issues, once
                     | declined after research, or QA fails twice
                     v
                  re-design (report to the person)
"""

from __future__ import annotations

from pydantic import BaseModel

from factory.llm import LLM
from factory.pipeline.blackboard import Project
from factory.pipeline.critique import CriticLoop
from factory.pipeline.final import FinalEvaluator, FinalResult
from factory.pipeline.knowledge import KnowledgeLoop
from factory.pipeline.production import Production
from factory.pipeline.redesign import Redesign, RedesignResult
from factory.pipeline.sources import Searcher
from factory.pipeline.wiki import Wiki
from factory.registry import Registry

BUILD_ROUNDS = 2  # the first build, plus one rebuild with the final evaluator's issues


class BuildResult(BaseModel):
    status: str  # delivered | redesign | stopped
    final: FinalResult | None = None
    redesign: RedesignResult | None = None
    rounds: int


def build(llm: LLM, registry: Registry, project: Project, wiki: Wiki, searchers: dict[str, Searcher],
          on_stage=lambda stage: None, redesign: bool = True) -> BuildResult:
    """``redesign=False`` (a replay of an archived failure, factory/failures.py) stops where re-design would start."""
    production = Production(llm, registry, project, wiki)
    knowledge = KnowledgeLoop(llm, registry, project, wiki, searchers)
    critic = CriticLoop(llm, registry, project, wiki, production, knowledge)
    final = FinalEvaluator(llm, registry, project, production)
    feedback, result = "", None
    for rnd in range(1, BUILD_ROUNDS + 1):
        on_stage(f"production, round {rnd}")
        checks = production.run(feedback)
        on_stage("critic loop")
        reviewed = critic.run(checks)
        if reviewed.redesign:
            break
        on_stage("final QA")
        result = final.run(critic.shown_files())
        if result.approved:
            return BuildResult(status="delivered", final=result, rounds=rnd)
        feedback = "\n".join(f"- {i}" for i in result.issues)
    if not redesign:
        return BuildResult(status="stopped", final=result, rounds=rnd)
    on_stage("re-design")
    return BuildResult(status="redesign", final=result, rounds=rnd,
                       redesign=Redesign(llm, registry, project, research=knowledge.research).run())
