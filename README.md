# SIMPLE: Solution In Minutes from Plain Language Executable

SIMPLE (the Bespoke AI System Factory) takes a task written in plain language and returns a reproducible workflow,
exported as a code bundle that runs on its own. Plain code does every step it can, and a language model appears
only at the steps that need one. Open models on Virginia Tech's ARC do the model work by default. Claude and OpenAI
models are used only when an ARC model's output fails its check, or to review work, and never without a person
approving the spend.

Built for ADS 5984 (Agentic AI in Practice), fall 2026. Planning doc:
[Bespoke AI System Factory: Plan](https://claude.ai/artifact/WgDo3pezJxTQNv284UKpXg).

## Status

Phase 1 (bug fixing) works end to end. On 2026-10-28 SIMPLE fixed the example bug against ARC on its first attempt
using free models only (GLM wrote the patch, DeepSeek reviewed it; 815 tokens, $0), and the exported bundle's
`rerun.py` re-proved the fix without any model. In place: the model adapter, spend gate and credit ledger, agent
registry with contract-net selection, the plain-code agents and their MCP server, the bug-fix orchestrator (plan,
award, patch, check, review, repair, escalate), the bundle exporter, the Streamlit app, and UML diagrams. Two more task types run on the same agents and checks:
**writing tests** (judged by the planted bugs they catch) and **reviewing a pull request** (rule checks and the test
delta first, then a model's comments, kept only when they point at changed lines). Next: a model-written planner.

## Setup

Python 3.11 or newer, git, and pytest.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,ui]"     # Windows: .venv\Scripts\pip install -e ".[dev,ui]"
.venv/bin/pytest -q                      # no network, no keys needed
.venv/bin/ruff check . && .venv/bin/bandit -q -c pyproject.toml -r factory app.py   # lint and security scan
```

GitHub Actions runs the same three checks on every pull request (`.github/workflows/ci.yml`), with tests on
Python 3.11 and 3.12.

On Windows, keep the virtual environment outside a synced folder such as Dropbox (for example
`C:\Users\you\.venvs\simple`); the Microsoft Store build of Python redirects `%LOCALAPPDATA%` paths.

### Keys (`.env`)

Copy `.env.example` to `.env` in the project root and fill it in. `.env` is git-ignored; never commit a key or paste
one into a chat. The submission zip lists these names only, never their values.

| Variable | Needed for | When it is used |
|---|---|---|
| `ARC_API_KEY` | Every run | The free default models on ARC (`https://llm-api.arc.vt.edu/api`); may need the VT network or VPN |
| `ANTHROPIC_API_KEY` | Paid reviewer step | Only when the reviewer escalates to Claude, after a person approves |
| `OPENAI_API_KEY` | Paid coder step | Only when the coder escalates to OpenAI, after a person approves |

A run with only `ARC_API_KEY` works as long as no paid step is approved; approving a paid step whose key is missing
ends the run with an error that names the missing variable.

## Run it

**Web app** (task form, live agent activity, approval cards for paid calls, budget bars, bundle download):

```bash
streamlit run app.py
```

**Terminal** (asks before any paid call; the bundle lands in `exports/`):

```bash
python -m factory fix --repo tests/fixtures/buggy_repo \
  --issue "parse_date swaps month and day: parse_date('2026-10-28') returns (2026, 28, 10) instead of (2026, 10, 28)"
```

Write tests for existing code, or review a diff:

```bash
python -m factory tests --repo tests/fixtures/buggy_repo --request "Write tests for days_in_month in calc/dates.py"
python -m factory review --repo tests/fixtures/buggy_repo --patch-file tests/fixtures/buggy_repo.patch \
  --description "Fix parse_date swapping month and day"
```

**Check a bundle without SIMPLE**: unzip it and run `python rerun.py path/to/the/repo`. No model is called.

| Task type | Bundle holds | `rerun.py` checks |
|---|---|---|
| Fix a bug (`fix`) | `fix.patch` | The patch applies, the target tests pass, then the whole suite |
| Write tests (`tests`) | `tests.patch`, `mutate.py` | The new tests pass and catch as many planted bugs as recorded |
| Review a PR (`review`) | `review.md`, `review.json`, `change.patch`, `diff_inspect.py` | The rule findings and the test delta match the review |

**MCP server** (the plain-code tools for any MCP client, such as Claude Desktop or Claude Code):

```bash
python -m factory.mcp --root path/to/repos
```

See [docs/mcp.md](docs/mcp.md) for the tools, why SIMPLE uses MCP, and a client config.

## Demo walkthrough

1. Start the web app and enter the issue above, with `tests/fixtures/buggy_repo` as the repository.
2. The plan table shows five steps; only `patch` and `review` need a model. The contract net awards each step:
   `code_navigator` and `test_runner` (plain code, through the MCP server) win the steps that need no model.
3. Agent activity shows the steps live: locate, baseline (1 failing test), patch (GLM), check (tests pass),
   review (DeepSeek, a different model family from the coder).
4. If the review rejects or the tests still fail, the coder gets the feedback and tries again. A third attempt moves
   the coder to the paid model, and an approval card shows the model and the most it can cost; declining stops the
   run with nothing spent.
5. Download the bundle and run its `rerun.py`.

## How it fits together

Diagrams: [docs/uml/](docs/uml/) (class and sequence diagrams; `SIMPLE-UML.pdf` has both).

| Path | What it does |
|---|---|
| `app.py` | The Streamlit page |
| `factory/__main__.py` | The terminal command (`python -m factory fix`, `tests`, `review`) |
| `factory/ui/session.py` | Runs a task on a background thread; turns spend-gate questions into approval cards |
| `factory/orchestrator.py` | Runs a task: template plan, contract-net awards, patch, check, cross-family review, repair |
| `factory/workflows/` | The other task types: `test_gen.py` (write, check, mutation score, review), `pr_review.py` (inspect, test delta, review, grounding), and their bundles |
| `factory/registry/` | Agent specs and selection by contract net (bids computed from specs and history, no model calls) |
| `registry/agents/` | One YAML spec per agent |
| `factory/tools/` | Plain-code agents: locate code from an issue (`repo.py`), run tests and prove a fix (`sandbox.py`), plant bugs to score tests (`mutate.py`), rule-check a diff (`diff_inspect.py`) |
| `factory/mcp/` | The plain-code agents as an MCP server, and the client the orchestrator calls them through |
| `factory/llm/` | The one way to call a model: cache, retries, pricing, logging, spend gate (`gate.py`) |
| `factory/agents/` | Typed tools, per-agent allow-lists, the capped tool-calling loop, replay |
| `factory/records.py` | Provenance: sources by content hash, append-only `provenance.jsonl` per run |
| `factory/export.py` | Writes the bundle: patch, workflow, evidence, provenance, and a model-free `rerun.py` |
| `factory/net/http.py` | Cached, throttled, budgeted HTTP client for outside APIs |
| `configs/factory.yaml` | Providers, which model plays each role, the paid escalation step per role, budgets, credit |
| `configs/models.yaml` | Model catalog: family (for separation) and price per million tokens |

Each run writes `runs/<time>-fix/` (model calls, provenance, cost) and, when it succeeds, `exports/bundle-*.zip`.
Both folders are git-ignored.

## Rules the code enforces

- **No model reviews its own family's work.** Authors use GLM, then OpenAI; reviewers use DeepSeek, then Claude.
  `load_config` refuses a config that breaks this in either tier.
- **Paid calls need approval.** The first paid call per role in a run, and any call that would pass the run's $1
  soft limit, is put to a person; with no one to ask, it is refused. A run stops at $3 of real spend.
- **Credit is tracked across runs.** Spend per provider is kept in `workspace/<name>/spend.json`. Paid calls stop
  when a provider's credit is used up, or when only the $10 demo reserve is left.
- **Public data only goes to paid providers.** Claude and OpenAI calls must be marked public, and the config must
  declare its inputs public.
- **Tests are judged by what they catch, not by a model's opinion.** Generated tests must pass, break nothing, and
  catch at least 60% of small bugs planted in the target code before a reviewer sees them.
- **Facts outrank a reviewer.** A pull request that does not apply, breaks a test, or adds a credential is never
  approved, whatever the model says, and a review comment on a line the diff does not change is dropped.
- **A fix may not change the tests.** A bug-fix patch that touches a test file or pytest's configuration
  (`conftest.py`, `pytest.ini`, ...) is refused before it runs, and the check fails if a target test did not run
  (deleted, renamed or skipped). `--allow-test-edits` lifts the first rule; the second always holds.
- **No secrets leave in a bundle.** Before a bundle is zipped, every file is scanned for key and token shapes and
  for the exact values of the keys in `.env`; on a hit the bundle is deleted and the run says which file held what.
- **SIMPLE cannot loosen its own checks.** A fix to SIMPLE's own repository may not touch the spend gate, config,
  sandbox, MCP server, orchestrator, registry or workflows; no fix anywhere may touch a `.env` file. A check also
  fails when fewer tests run, or more are skipped, than before the patch.
- **Every bundle names its registry.** `registry.json` in each run and bundle holds a hash of every agent spec and
  the track record as it stood when the steps were awarded, so a run elsewhere can be checked against it.
- **Keys never reach the code under test.** Tests run on a throwaway copy, in the MCP server's process, with every
  key, token and git auth setting removed from the environment.

## Origin

The model adapter, agent loop, caps, HTTP client, and provenance records are lifted from Bespoke_LLM (the
developer's earlier project, design v6.7.1); its corpus, expansion, and training code stayed behind.
