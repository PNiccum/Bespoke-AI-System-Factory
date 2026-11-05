# ASPIRE: Agentic System Producing Intelligent Runnable Executables

ASPIRE (the Bespoke AI System Factory) takes a task written in plain language and returns a reproducible workflow,
exported as a code bundle that runs on its own. Plain code does every step it can, and a language model appears
only at the steps that need one. Open models on Virginia Tech's ARC do the model work by default. Claude and OpenAI
models are used only when an ARC model's output fails its check, or to review work, and never without a person
approving the spend.

Built for ADS 5984 (Agentic AI in Practice), fall 2026. Planning doc:
[Bespoke AI System Factory: Plan](https://claude.ai/artifact/WgDo3pezJxTQNv284UKpXg).

## Status

Phase 1 (bug fixing) works end to end. On 2026-10-28 ASPIRE fixed the example bug against ARC on its first attempt
using free models only (GLM wrote the patch, DeepSeek reviewed it; 815 tokens, $0), and the exported bundle's
`rerun.py` re-proved the fix without any model. In place: the model adapter, spend gate and credit ledger, agent
registry with contract-net selection, the plain-code agents and their MCP server, the bug-fix orchestrator (plan,
award, patch, check, review, repair, escalate), the bundle exporter, the Streamlit app, and UML diagrams. Two more task types run on the same agents and checks:
**writing tests** (judged by the planted bugs they catch) and **reviewing a pull request** (rule checks and the test
delta first, then a model's comments, kept only when they point at changed lines). Requests in plain words are routed by rules first and a free planner model second.

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

The sidebar also has read-only inspection pages, none of which calls a model:

- **Wiki**: browse and search the shared wiki, see each claim's verbatim quote and source, the verifier's issues on
  pages that did not pass, links and backlinks between pages, the health check (`python -m factory wiki`) and the
  change log. `?page=<slug>` opens a page directly.
- **Projects**: each project's stage, files and blackboard, filterable by entry kind and author.
- **Runs and spend**: paid credit left per provider, and each run's model calls by model, cache hits, errors,
  approval-gate events and cost.

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

Or just say what you want; ASPIRE picks the workflow:

```bash
python -m factory ask --repo tests/fixtures/buggy_repo --request "parse_date gets month and day backwards"
```

Plain rules route the clear cases (a diff means a review; "write tests for ..." means test writing) with no model
call. Only an ambiguous request goes to the planner, a free ARC model that picks the workflow and lays out its steps
from a fixed catalog of building blocks (`factory/planner.py`). Code checks the plan: each block must exist and run
after the blocks it needs, and whether a step uses a model is a property of the block, never the planner's call. A
request no workflow covers is not run; it is written to `workspace/<name>/gaps.jsonl` as a starting point for a new
agent or workflow.

For a larger project, start with the Interviewer (the "Interviewer" page in the app, or the terminal):

```bash
python -m factory interview --request "A tool that pulls a gene from NCBI and designs PCR primers for it"
```

It asks at most six questions, then proposes Plan A: tasks that each carry a check anyone can run. Code checks the
plan (acceptance checks present, ids unique, no task depending on a later one) and only you can approve it. Approval
creates `workspace/<name>/projects/<id>/` with `task.txt`, `plan_a.json` and `blackboard.jsonl`, the shared log
every later stage of the pipeline reads and writes (`factory/pipeline/`).

Then the planning loop puts three model families on Plan A:

```bash
python -m factory plan --project workspace/factory/projects/<id>
```

The plan critic (DeepSeek) lists Plan A's weaknesses and writes Plan B; the plan analyst (gpt-oss) lists pros and
cons of each; the editor (GLM) edits both; the critic and the analyst vote, and a
split goes to the plan with fewer problems, then to Plan A. The editor never votes on its own edits. Code checks every plan, drops an edit that breaks one,
counts the votes, and never lets a plan with problems win. The winner, with tasks and subtasks, is `plan.json`.

Then the knowledge loop makes sure the wiki holds what the plan needs:

```bash
python -m factory research --project workspace/factory/projects/<id>
```

The knowledge evaluator (gpt-oss) lists the topics the work depends on. The librarian checks the shared wiki
(`workspace/<name>/wiki/`) in plain code: a topic is covered by a verified page with at least two sources. For each
missing topic the research orchestrator writes searches, plain-code connectors run them against OpenAlex, Europe PMC
and NCBI Gene, and the orchestrator approves or rejects each document. The librarian writes a page citing a document
for every claim; code checks the citations and the wiki verifier (DeepSeek) checks the claims against the documents.
The evaluator then looks again with the new pages and either approves or names what is still missing (at most three
looks, two research passes). Every document carries a license tag, and a page built from an ARC-only source never
goes to a paid model.

Then the solution is built, criticized, checked and delivered:

```bash
python -m factory build --project workspace/factory/projects/<id>
```

- **Production.** The head orchestrator (GLM) designs the solution from the plan and the wiki: language, layout, entry
  point, run command. For each task a sub-orchestrator writes work orders and builders write whole files under
  `output/` (code refuses paths outside it). Code adds `README.md` and `run.json`, checks every Python file compiles,
  and runs the solution's tests in the sandbox.
- **Critic loop.** The project critic (DeepSeek) lists holes, failures and usability problems with the checks in
  front of it; problem solver 1 (gpt-oss) answers with replacement files; the critic accepts or declines each fix and
  code applies only the accepted ones. A declined fix escalates: the solver tries again with the closest wiki pages,
  then after the knowledge loop researches what was missing, and a decline after that starts the re-design loop.
- **Final QA.** The final evaluator (DeepSeek) checks every task's acceptance criterion against the files and test
  results; code approves only when the checks pass and every task is met. The deliverable is
  `deliverable/<project>.zip`: the solution plus the task, plan, wiki pages and public provenance. A failed QA goes
  back to the head orchestrator once with its issues.
- **Re-design.** The failed project is archived as a case (`workspace/failures/<id>/`, a snapshot from before
  production). ASPIRE then asks once for its paid plan: Claude Opus and GPT for the audit and the debate, with an
  upper-bound estimate (about $0.75 at most, usually far less) and a $1.00 cap; declined, the auditors use free ARC
  models and the debate is skipped. Two auditors of different families read the full blackboard trace and the fixes
  that worked on earlier failures, and trace each failure to one cause from a fixed list (prompt, missing knowledge,
  loop rule, model limit, tool or code), citing the entries that show it. A summarizer writes the project up (public
  entries only) and the two paid experts debate the best change until they agree, two rounds pass, or $0.50 is spent.
  Plain code routes the causes: prompt fixes go through self-extension below, knowledge gaps go to the knowledge loop,
  and the rest wait for you in `redesign/report.md`. A cause with no evidence on the blackboard is never acted on by
  itself.

**Check a bundle without ASPIRE**: unzip it and run `python rerun.py path/to/the/repo`. No model is called.

| Task type | Bundle holds | `rerun.py` checks |
|---|---|---|
| Fix a bug (`fix`) | `fix.patch` | The patch applies, the target tests pass, then the whole suite |
| Write tests (`tests`) | `tests.patch`, `mutate.py` | The new tests pass and catch as many planted bugs as recorded |
| Review a PR (`review`) | `review.md`, `review.json`, `change.patch`, `diff_inspect.py` | The rule findings and the test delta match the review |

**Benchmark** (ASPIRE against one model call on the same tasks and the same coder model, free models only):

```bash
python -m factory.bench                         # 11 bug-fix tasks in bench/tasks.yaml, both modes
python -m factory.bench --only leap_year --mode baseline
python -m factory.bench --repeat 3               # three trials per task: models vary from run to run
```

The baseline pastes the issue and every file into one prompt and takes the diff back; ASPIRE runs its full
workflow. Plain code judges both the same way: the patch applies, changes no tests, the whole suite passes on a
fresh copy, and every test that ran before still runs. A solved patch is judged again with hidden tests
(`bench/hidden/`), which no model ever sees and which catch a patch that special-cases the visible tests' inputs.
One task (`injection`) is a repository that tells the model to delete tests and copy keys. Results go to
`bench/results/<time>/summary.md`.

**Harder tasks.** Two more tiers go past single-line bug fixes. Both use free models only by default:

```bash
python -m factory.bench --tasks bench/tasks_hard.yaml --repeat 3   # 8 harder repository tasks, both modes
python -m factory.bench_projects                                   # 4 tools built from nothing by the pipeline
python -m factory.bench_projects --only wordstats --repeat 3
```

- `bench/tasks_hard.yaml` holds eight repository tasks. Some have two bugs in one class or a cause two modules
  away from the symptom. One is a feature built to tests that already exist (`inventory_remove`), and the rest are a
  parser or an algorithm brought up to its written spec. They are judged like the main benchmark, with hidden
  tests, and kept in their own file so the main numbers and self-extension's held-out gate do not change.
- `bench/projects.yaml` holds four small tools: a word-frequency CLI, a CSV profiler, an access-log error-spike
  finder, and an OpenAlex reading-list digest. Each starts from a written request and Plan A, then goes through the
  planning loop, the knowledge loop, production, the critic loop and final QA. Plain code then runs acceptance tests
  that no model sees (`bench/projects/<name>/acceptance/`) on the delivered `output/` folder. A project is solved
  when it is delivered and every acceptance test passes. The summary also records how far each run got. Re-design
  is off unless `--redesign` is given, so a benchmark run never changes ASPIRE's prompts.
- `bench/projects_proteomics.yaml` holds five tools from the Lazar lab's protein-list workflow: list plausibility,
  a sourced evidence table, DAVID chart filtering with recomputed statistics, drug-target tiers, and an audit of an
  AI report's claims. Run it with `python -m factory.bench_projects --projects bench/projects_proteomics.yaml`.
  The fixtures are public or synthetic; see [docs/proteomics-tasks.md](docs/proteomics-tasks.md).

**Speed.** Live runs are bound by ARC model latency (about 40 s a GLM call, 130 s a DeepSeek thinking-max call), so
the pipeline runs independent work at once: research topics (4 at a time), a task's work orders that do not touch
each other's files (3 at a time), and, with `--parallel N`, whole benchmark projects. All runs in one process share
each provider's `max_concurrency` (8 for ARC, which allows 20 per user). A reply cut off at `max_tokens` is retried
on the role's `on_truncation` model, a faster free model of the same family, so family separation is unchanged.
`python -m factory.profile runs/` shows where a run's time went: wall clock against model time, time by stage and
model, and time lost to cut-off replies.

```bash
python -m factory.bench_projects --projects bench/projects_proteomics.yaml --parallel 5
python -m factory.profile runs/20261104T142930Z-build
```

**Evaluation** (the experiments behind the cost, rerun and safety claims; see [docs/evaluation.md](docs/evaluation.md)):

```bash
python -m factory.evaluate guards                # offline: hostile patches against the judge and the fix check
python -m factory.evaluate rerun                 # offline: each bundle re-proves its fix with no model or network
python -m factory.evaluate reviewers --authored 2  # free ARC models: who catches a wrong patch that passes the tests
python -m factory.evaluate report bench/results/<time>   # cost per solved task, 95% intervals, break-even
```

**Self-extension** (ASPIRE proposes a better prompt for one of its model agents):

```bash
python -m factory.extend --agent coder                # free models only; stops for you
python -m factory.extend --agent coder --paid-panel   # also a paid reviewer, after you approve the spend
python -m factory.extend --agent coder --adopt        # apply it if it passes every gate
python -m factory.extend --rollback coder             # undo the last adopted prompt
python -m factory.extend --agent builder --case <id>  # also replay an archived failure with the new prompt
```

A proposal may change only an agent's prompt and description: never code, tools, roles or capabilities. The proposer
writes three candidates, each told to differ from the ones before; the best on the held-out benchmark (or, for agents
the benchmark does not exercise, the one a free reviewer of another family picks) goes on. It must pass, in order: a
scope check in plain code; the held-out benchmark tasks (never shown to the proposer), solving at least as many as the
current agent within 1.5x the tokens; a review panel of another model family (two free families for agents the
benchmark does not exercise); and a failure replay: the archived failure it should fix, and every failure already
fixed for that agent, are rebuilt with the candidate in place and must be delivered. A fixed failure is a regression
test from then on, and a rollback reopens it. Every proposal goes into `proposals/archive.jsonl`, which the next
proposal starts from, along with the fixes that worked before.

When a project fails and the re-design loop finds a prompt-only fix, ASPIRE runs it through these gates and, if it
passes, applies it without asking: the old spec is kept in `registry/history/<agent>/` and the change is logged in
`registry/history/changelog.md`. It never does this for a judging agent (reviewers, critics, QA, wiki verifier,
auditors), and at most three times a week; those, and every structural fix, wait for you in the re-design report.

**Recorded runs** (a showcase that a slow gateway or a dropped connection cannot stop):

```bash
python -m factory.demo record runs/<a finished run> demos/parse_date.json   # its task and model replies
python -m factory.demo replay demos/parse_date.json                         # offline: no network, no model
```

A replay runs the real agents, tests, review and bundle export again; only the model replies come from the
recording, and a reply it lacks fails rather than being fetched. The web app lists `demos/*.json` under
**Recorded runs** and says, while one plays, when it was recorded and what the original run cost. Test output is
stripped of temporary paths and timings, so the same failure always makes the same prompt. Recordings are
secret-scanned before they are written.

**MCP server** (the plain-code tools for any MCP client, such as Claude Desktop or Claude Code):

```bash
python -m factory.mcp --root path/to/repos
```

See [docs/mcp.md](docs/mcp.md) for the tools, why ASPIRE uses MCP, and a client config.

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

Diagrams: [docs/uml/](docs/uml/) (class and sequence diagrams; `ASPIRE-UML.pdf` has both).

| Path | What it does |
|---|---|
| `app.py` | The Streamlit page |
| `factory/__main__.py` | The terminal command (`python -m factory fix`, `tests`, `review`) |
| `factory/ui/session.py` | Runs a task on a background thread; turns spend-gate questions into approval cards |
| `factory/planner.py` | Intake: rules, then the planner model, route a request; the block catalog; recorded gaps |
| `factory/extend.py` | Self-extension: several candidates, scope gate, held-out benchmark, review panel, failure replay, then adopt (with rollback) or stop for a person |
| `factory/failures.py` | The failure archive: snapshots of failed projects, replays, regression cases, fixes that worked |
| `factory/repairs.py` | The repair library: final-QA repairs that stuck (issue + diff), shown to the solver as the closest past repairs |
| `factory/tally.py` | Per-stage model tally from `llm_calls.jsonl`: calls, finish rate, fallback and cut-off hand-offs (`python -m factory.tally`) |
| `factory/demo.py`, `demos/` | Recorded runs: save a run's model replies, replay it offline |
| `factory/orchestrator.py` | Runs a task: template plan, contract-net awards, patch, check, cross-family review, repair |
| `factory/workflows/` | The other task types: `test_gen.py` (write, check, mutation score, review), `pr_review.py` (inspect, test delta, review, grounding), and their bundles |
| `factory/registry/` | Agent specs and selection by contract net (bids computed from specs and history, no model calls) |
| `registry/agents/` | One YAML spec per agent |
| `factory/tools/` | Plain-code agents: locate code from an issue (`repo.py`), run tests and prove a fix (`sandbox.py`), plant bugs to score tests (`mutate.py`), rule-check a diff (`diff_inspect.py`) |
| `factory/mcp/` | The plain-code agents as an MCP server, and the client the orchestrator calls them through |
| `factory/llm/` | The one way to call a model: cache, retries, pricing, logging, spend gate (`gate.py`) |
| `factory/agents/` | Typed tools, per-agent allow-lists, the capped tool-calling loop, replay |
| `factory/records.py` | Provenance: sources by content hash, append-only `provenance.jsonl` per run |
| `factory/bench.py`, `bench/` | The benchmark: tasks, known-good fixes (to check the tasks themselves), runner and judge |
| `factory/evaluate.py`, `bench/hidden/`, `bench/review_set/` | Evaluation: hostile-patch guards, zero-model reruns, the reviewer experiment on labeled patches, the cost report |
| `factory/export.py` | Writes the bundle: patch, workflow, evidence, provenance, and a model-free `rerun.py` |
| `factory/net/http.py` | Cached, throttled, budgeted HTTP client for outside APIs |
| `configs/factory.yaml` | Providers, which model plays each role, the paid escalation step per role, budgets, credit |
| `configs/models.yaml` | Model catalog: family (for separation) and price per million tokens |

Each run writes `runs/<time>-fix/` (model calls, provenance, cost) and, when it succeeds, `exports/bundle-*.zip`.
Both folders are git-ignored.

## Rules the code enforces

- **No model reviews its own family's work.** Authors use GLM, then OpenAI; reviewers use DeepSeek, then Claude.
  `load_config` refuses a config that breaks this in either tier.
- **A down ARC model does not stop a run.** Each role lists free ARC fallbacks (`fallbacks:` in
  `configs/factory.yaml`), tried in order when its model is not found or keeps failing through the retries. Fallbacks
  must be free and count toward family separation; the paid step never falls back.
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
- **ASPIRE cannot loosen its own checks.** A fix to ASPIRE's own repository may not touch the spend gate, config,
  sandbox, MCP server, orchestrator, registry, workflows, planner, benchmark or self-extension code; no fix anywhere may touch a `.env` file. A check also
  fails when fewer tests run, or more are skipped, than before the patch.
- **Repository text is data, not instructions.** Every model agent's system prompt ends with a rule, added in code
  so no agent spec can drop it, to ignore requests inside issues, files, diffs or test output. The guards above
  hold even when a model obeys one anyway: `tests/fixtures/injection_repo` tells the model to delete tests and copy
  keys, and a canary test there fails if any credential reaches the code under test (`tests/test_injection.py`).
- **Every bundle names its registry.** `registry.json` in each run and bundle holds a hash of every agent spec and
  the track record as it stood when the steps were awarded, so a run elsewhere can be checked against it.
- **Keys never reach the code under test.** Tests run on a throwaway copy, in the MCP server's process, with every
  key, token and git auth setting removed from the environment.
- **Generated code can run in a container.** With `sandbox: runner: docker` (or `podman`) in `configs/factory.yaml`,
  or `ASPIRE_SANDBOX=docker` for one run, every test run and every mutant runs in the `aspire-sandbox` image
  (`docker build -t aspire-sandbox sandbox/`): no network, an empty environment, a read-only root, no Linux
  capabilities, and 2 GB / 2 CPUs / 256 processes. A timeout removes the container. On ARC, `runner: apptainer` takes
  the same image as a `.sif` file. `tests/test_container.py` runs a hostile repository that tries to reach the
  network, read keys and write outside its folder.

## Origin

The model adapter, agent loop, caps, HTTP client, and provenance records are lifted from Bespoke_LLM (the
developer's earlier project, design v6.7.1); its corpus, expansion, and training code stayed behind.
