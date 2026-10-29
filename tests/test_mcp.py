"""The factory's tools over MCP: the server's tools, its folder fence, and a bug fix routed through it."""

import asyncio
import json
from pathlib import Path

import pytest
from mcp import Client

from factory.mcp import McpTools, build_server, mcp_function
from factory.mcp.client import McpToolError
from factory.tools.repo import LocateIn, locate
from factory.tools.sandbox import RunTestsIn, run_tests
from tests.test_orchestrator import ACCEPT, GOOD, REPO, cfg, diff, run  # noqa: F401 (cfg is a fixture)

FIXTURES = Path(__file__).parent / "fixtures"


def call(name, arguments, roots=(str(FIXTURES),)):
    async def go():
        async with Client(build_server(list(roots))) as client:
            return await client.call_tool(name, arguments)
    return asyncio.run(go())


def test_the_server_offers_the_plain_code_tools():
    async def go():
        async with Client(build_server([str(FIXTURES)])) as client:
            return [t.name for t in (await client.list_tools()).tools]
    assert sorted(asyncio.run(go())) == ["inspect_diff", "locate", "prove_fix", "read_file", "run_tests",
                                         "score_tests"]


def test_locate_over_mcp_matches_the_in_process_answer():
    inp = LocateIn(repo_path=str(REPO), issue_text="parse_date swaps month and day")
    result = call("locate", {"inp": inp.model_dump()})
    assert not result.is_error
    assert result.structured_content == json.loads(locate(inp).model_dump_json())


def test_prove_fix_over_mcp():
    result = call("prove_fix", {"repo_path": str(REPO), "patch": GOOD,
                                "tests": ["tests/test_dates.py::test_parse_date_order"]})
    proof = result.structured_content
    assert proof["before"]["failed"] and not proof["after"]["failed"]


def test_tools_refuse_paths_outside_the_allowed_roots(tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    result = call("read_file", {"repo_path": str(tmp_path), "path": "secret.txt"})
    assert result.is_error and "outside the folders" in result.content[0].text
    result = call("read_file", {"repo_path": str(REPO), "path": "../../../conftest.py"})
    assert result.is_error


def test_read_file_returns_the_lines_asked_for():
    content = call("read_file", {"repo_path": str(REPO), "path": "calc/dates.py", "start": 3, "count": 2})
    assert content.structured_content["lines"][0].startswith("    year, month, day")


@pytest.fixture(scope="module")
def tools():
    return McpTools([str(FIXTURES)])


def test_the_stdio_client_runs_a_tool_agent_in_a_separate_process(tools):
    report = mcp_function(run_tests, tools)(RunTestsIn(repo_path=str(REPO)))
    assert report.failed == ["tests/test_dates.py::test_parse_date_order"] and report.passed


def test_the_server_process_gets_no_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert "ANTHROPIC_API_KEY" not in McpTools([str(FIXTURES)]).server.env


def test_a_tool_error_is_raised_in_the_caller(tmp_path, tools):
    with pytest.raises(McpToolError, match="outside the folders"):
        mcp_function(locate, tools)(LocateIn(repo_path=str(tmp_path), issue_text="x"))


def test_a_bug_fix_can_route_its_tool_agents_through_mcp(cfg, tmp_path):
    cfg.tool_transport = "mcp"
    result, _, events, _ = run(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    assert result.status == "fixed"
    assert any(e.step == "baseline" and e.status == "done" for e in events)
