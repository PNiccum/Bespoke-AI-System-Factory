"""The factory's plain-code agents as an MCP server.

The same functions the orchestrator awards steps to (find the code an issue is about, run the tests on a throwaway
copy, prove a fix) are served here as MCP tools. Two things come of that:

- The orchestrator can call them in a separate process (``tool_transport: mcp`` in the config). That process starts
  with every key and token removed from its environment, so code under test never sees a credential, and the tool
  side can be swapped or moved to another machine without touching the orchestrator.
- Any MCP client can use them without the factory: Claude Desktop, Claude Code, or a professor's own agent can find
  code and prove a patch with the same checks the factory uses, at no model cost.

Every tool works only inside the allowed roots (``--root``, else FACTORY_MCP_ROOTS, else the current directory).
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from factory.tools import quality, repo, sandbox
from factory.tools.repo import LocateIn, Locations
from factory.tools.quality import DiffFacts, InspectDiffIn, MutationReport, ScoreTestsIn
from factory.tools.sandbox import FixProof, RunTestsIn, TestReport

INSTRUCTIONS = """Plain-code tools from the Bespoke AI System Factory. None of them calls a model.
locate finds the files an issue is about; read_file shows part of one; run_tests runs pytest on a throwaway copy,
optionally with a unified diff applied; prove_fix checks that tests fail without a patch and pass with it;
score_tests plants small bugs in code and counts how many a set of tests catches; inspect_diff parses a diff and runs
rule checks (debug calls, credentials, syntax) over the lines it adds. The original repository is never written to."""


class FileSlice(BaseModel):
    path: str
    start: int  # 1-based first line shown
    lines: list[str]
    total_lines: int


def allowed_roots(roots: list[str] | None = None) -> list[Path]:
    if roots is None:
        roots = [r for r in os.environ.get("FACTORY_MCP_ROOTS", "").split(os.pathsep) if r] or [os.getcwd()]
    return [Path(r).resolve() for r in roots]


def build_server(roots: list[str] | None = None) -> MCPServer:
    allowed = allowed_roots(roots)

    def inside(path: str) -> Path:
        p = Path(path).resolve()
        if not any(p == r or r in p.parents for r in allowed):
            raise ToolError(f"{path} is outside the folders this server may read")
        return p

    server = MCPServer("factory-tools", instructions=INSTRUCTIONS)

    @server.tool()
    def locate(inp: LocateIn) -> Locations:
        """Find the files, functions and tests an issue is about, ranked by evidence. No model."""
        inside(inp.repo_path)
        return repo.locate(inp)

    @server.tool()
    def read_file(repo_path: str = Field(description="Root of the repository"),
                  path: str = Field(description="File path relative to the root"),
                  start: int = Field(1, ge=1), count: int = Field(200, ge=1, le=2000)) -> FileSlice:
        """Read part of a file in the repository."""
        f = inside(str(Path(repo_path) / path))
        if not f.is_file():
            raise ToolError(f"no file {path}")
        text = f.read_text(errors="replace").splitlines()
        return FileSlice(path=path, start=start, lines=text[start - 1:start - 1 + count], total_lines=len(text))

    @server.tool()
    def run_tests(inp: RunTestsIn) -> TestReport:
        """Run pytest on a throwaway copy of the repository, optionally with a unified diff applied first."""
        inside(inp.repo_path)
        return sandbox.run_tests(inp)

    @server.tool()
    def prove_fix(repo_path: str = Field(description="Root of the repository"),
                  patch: str = Field(description="Unified diff"),
                  tests: list[str] = Field(description="pytest node ids that should go from failing to passing"),
                  timeout_s: float = Field(300, gt=0, le=3600)) -> FixProof:
        """Check a fix: the tests must fail without the patch and pass with it."""
        inside(repo_path)
        return sandbox.prove_fix(repo_path, patch, tests, timeout_s)

    @server.tool()
    def score_tests(inp: ScoreTestsIn) -> MutationReport:
        """Plant small bugs in the target code one at a time and count how many the tests catch (mutation testing)."""
        inside(inp.repo_path)
        return quality.score_tests(inp)

    @server.tool()
    def inspect_diff(inp: InspectDiffIn) -> DiffFacts:
        """Parse a unified diff and run rule checks over the lines it adds; with repo_path, also check it applies."""
        if inp.repo_path:
            inside(inp.repo_path)
        return quality.inspect_diff(inp)

    return server
