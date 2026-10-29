# MCP in the factory

The factory runs one MCP server of its own, `factory-tools` (`python -m factory.mcp`). It serves the plain-code
agents, the steps that need no model, as MCP tools.

| Tool | What it does | Used by |
|---|---|---|
| `locate` | Ranks the files, functions and tests an issue is about (raised-at frames, definitions, mentions) | `code_navigator` |
| `read_file` | Returns a slice of one file in the repository | MCP clients browsing a repo |
| `run_tests` | Runs pytest on a throwaway copy, optionally with a unified diff applied | `test_runner` (baseline and check) |
| `prove_fix` | Checks that the named tests fail without a patch and pass with it | MCP clients, exported-bundle checks |
| `score_tests` | Plants small bugs in target code one at a time and counts how many the tests catch | `mutation_tester` (test_gen) |
| `inspect_diff` | Parses a diff and rule-checks the lines it adds (debug calls, credentials, syntax); checks it applies | `diff_inspector` (pr_review) |

## Why MCP here

1. **Keys stay out of the code under test.** With `tool_transport: mcp` (the default in `configs/factory.yaml`),
   the orchestrator starts the server as a separate process whose environment has every key, token, secret and
   git auth setting removed. The repository's tests, which may contain generated code, run in that process,
   never in the one holding the API keys.
2. **One tool, many callers.** The orchestrator, Claude Desktop, Claude Code, or a professor's own agent call the
   same `locate` and `prove_fix` with the same checks, at no model cost. A fix the factory proves can be re-proved
   by anyone with an MCP client.
3. **Swappable tool side.** The orchestrator sees only a function with the same signature and return type
   (`factory/mcp/client.py`). The server can move to ARC or a container later without changing the orchestrator.
4. **A fence on what tools may read.** Every tool works only inside the allowed folders (`--root`, else
   `FACTORY_MCP_ROOTS`, else the current directory); the orchestrator allows only the task's repository.

Model calls do not go through MCP: they go through `factory/llm`, which owns the spend gate, the ledger and the
cache, so every paid call still waits for a person's approval.

## Using it from Claude Desktop or Claude Code

```json
{
  "mcpServers": {
    "factory-tools": {
      "command": "C:/Users/you/.venvs/factory-311/Scripts/python.exe",
      "args": ["-m", "factory.mcp", "--root", "C:/path/to/repos"],
      "cwd": "C:/path/to/Bespoke-AI-System-Factory"
    }
  }
}
```

Set `tool_transport: direct` in `configs/factory.yaml` to call the same functions in-process instead.
