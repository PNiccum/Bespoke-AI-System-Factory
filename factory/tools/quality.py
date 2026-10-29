"""Typed entrypoints for the two plain-code judges, so the registry and the MCP server can call them like the others.

``mutate.py`` and ``diff_inspect.py`` use only the standard library because bundles ship copies of them; these
wrappers add the pydantic types the orchestrator and the MCP server pass around.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from factory.tools import diff_inspect, mutate


class ScoreTestsIn(BaseModel):
    repo_path: str = Field(description="Root of the repository")
    files: list[str] = Field(description="Files to plant bugs in, relative to the root")
    tests: list[str] = Field(description="pytest paths or node ids run against each mutant")
    functions: list[str] = Field([], description="Plant bugs only inside these functions; empty means anywhere")
    patch: str | None = Field(None, description="Unified diff applied first, e.g. the tests being scored")
    limit: int = Field(12, ge=1, le=100)
    timeout_s: float = Field(60, gt=0, le=600)


class MutationReport(BaseModel):
    total: int
    killed: int
    score: float  # killed / total
    survivors: list[dict] = []  # {"path", "line", "change"} for each planted bug the tests missed
    killed_list: list[dict] = []
    error: str = ""  # set when the tests could not be run at all


class InspectDiffIn(BaseModel):
    patch: str = Field(description="Unified diff")
    repo_path: str | None = Field(None, description="Given, the diff is applied to a copy and changed files parsed")


class DiffFacts(BaseModel):
    files: list[dict]  # {"path", "added", "removed", "test"}
    applied: bool | None  # None when no repository was given
    findings: list[dict]  # {"rule", "severity", "path", "line", "message", "text"}


def score_tests(inp: ScoreTestsIn) -> MutationReport:
    """Plant small bugs in the target code, one at a time, and count how many the tests catch. No model."""
    return MutationReport(**mutate.score_tests(mutate.ScoreIn(**inp.model_dump())).to_dict())


def inspect_diff(inp: InspectDiffIn) -> DiffFacts:
    """Parse a diff and run the fixed rule checks over the lines it adds. No model."""
    return DiffFacts(**diff_inspect.inspect(inp.patch, inp.repo_path))
