# Bespoke AI System Factory

Give the factory a task; it returns a reproducible workflow, exported as a code bundle that runs on its own. Plain
code does every step it can, and a language model appears only at the steps that need one. Open models on Virginia
Tech's ARC do the model work by default; Claude and OpenAI models are used only when an ARC model's output fails its
check, or to review work, and never without a person approving the spend.

Built for ADS 5984 (Agentic AI in Practice), fall 2026. Planning doc:
[Bespoke AI System Factory: Plan](https://claude.ai/artifact/WgDo3pezJxTQNv284UKpXg).

## Status

Phase 0 (foundations). In place: the model adapter, spend gate and credit ledger, agent loop, provenance log, and
the agent registry. Not yet built: the orchestrator, the phase 1 tools (`code_navigator`, `test_runner`), the bundle
compiler, and the Streamlit app.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip install -e ".[dev]"
.venv/bin/pytest -q                    # no network, no keys needed
```

Keys go in a `.env` file at the project root (copy `.env.example`). `.env` is git-ignored; never commit a key or
paste one into a chat. The ARC gateway (`https://llm-api.arc.vt.edu/api`) may need the VT network or VPN.

## How it fits together

| Path | What it does |
|---|---|
| `configs/factory.yaml` | Providers, which model plays each role, the paid escalation step per role, budgets and credit |
| `configs/models.yaml` | Model catalog: family (for separation) and price per million tokens |
| `factory/llm/` | The one way to call a model: cache, retries, pricing, logging, spend gate (`gate.py`) |
| `factory/agents/` | Typed tools, per-agent allow-lists, the capped tool-calling loop, replay |
| `factory/registry/` | Agent specs and selection by contract net (bids computed from specs and history, no model calls) |
| `factory/records.py` | Provenance: sources by content hash, append-only `provenance.jsonl` per run |
| `factory/net/http.py` | Cached, throttled, budgeted HTTP client for outside APIs |
| `registry/agents/` | One YAML spec per agent |

## Rules the code enforces

- **No model reviews its own family's work.** Authors use GLM, then OpenAI; reviewers use DeepSeek, then Claude.
  `load_config` refuses a config that breaks this in either tier.
- **Paid calls need approval.** The first paid call per role in a run, and any call that would pass the run's $1
  soft limit, is put to an approver; with none, it is refused. A run stops at $3.
- **Credit is tracked across runs.** Spend per provider is kept in `workspace/<name>/spend.json`. Paid calls stop
  when a provider's credit is used up, or when only the $10 demo reserve is left.
- **Public data only goes to paid providers.** Claude and OpenAI calls must be marked public, and the config must
  declare its inputs public.

## Origin

The model adapter, agent loop, caps, HTTP client, and provenance records are lifted from Bespoke_LLM (the
developer's earlier project, design v6.7.1); its corpus, expansion, and training code stayed behind.
