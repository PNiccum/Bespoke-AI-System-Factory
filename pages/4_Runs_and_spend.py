"""Runs and spend: what each run called, on which model, at what cost, and what is left on each paid account.

Read-only and model-free.
"""

from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.llm.gate import SpendLedger
from factory.ui import inspect

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="ASPIRE: Runs and spend", layout="wide")


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


def usd(x: float) -> str:
    return f"\\${x:.2f}"


cfg = config()
st.title("Runs and spend")
st.caption("Free ARC models do the work; a paid call needs your approval. Every call a run makes is logged in its "
           "folder's llm_calls.jsonl, and paid spend is tracked across runs in the workspace's spend ledger.")

st.subheader("Paid credit")
spent = SpendLedger(cfg.work_dir / "spend.json", cfg.credits).spent
cols = st.columns(max(1, len(cfg.credits.by_provider)))
for col, (provider, credit) in zip(cols, cfg.credits.by_provider.items(), strict=False):
    used = spent.get(provider, 0.0)
    col.metric(f"{provider} left", usd(credit - used))
    col.progress(min(used / credit, 1.0) if credit else 0.0, text=f"{usd(used)} spent of {usd(credit)}")
st.caption(f"{usd(cfg.credits.reserve_usd)} is held back for the demo. Paid calls stop when the credit minus that "
           "reserve is used up.")

base = cfg.root / cfg.runs_dir
dirs = inspect.runs(base)
st.subheader("Runs")
st.caption(f"Folder: `{base}` (newest 200)")
if not dirs:
    st.info("No runs yet.")
    st.stop()
rows = inspect.run_rows(dirs)
paid_only = st.checkbox("Only runs with paid calls")
st.dataframe([r for r in rows if not paid_only or r["paid USD"] > 0], hide_index=True, use_container_width=True)

names = [d.name for d in dirs]
wanted = st.query_params.get("run")
name = st.selectbox("Open a run", names, index=names.index(wanted) if wanted in names else 0)
if name != wanted:
    st.query_params["run"] = name
run = base / name
calls = inspect.read_jsonl(run / "llm_calls.jsonl")
t = inspect.call_totals(calls)
a, b, c, d = st.columns(4)
a.metric("Model calls", t["calls"])
b.metric("From cache", t["cached"])
c.metric("Paid", usd(t["real_usd"]))
d.metric("If every call were paid", usd(t["priced_usd"]))
if t["gates"]:
    st.caption(f"{t['gates']} paid call(s) went to the approval gate.")

if calls:
    st.markdown("**By model**")
    st.dataframe(inspect.by_model(calls), hide_index=True, use_container_width=True)
    st.markdown("**Every call**")
    st.dataframe([{"at": x.get("at", "")[11:19], "tag": x.get("tag", x.get("event", "")), "role": x.get("role", ""),
                   "model": x.get("model", ""), "cached": bool(x.get("cached")), "tokens": inspect.tokens(x),
                   "seconds": x.get("seconds"), "paid USD": x.get("real_usd", 0.0),
                   "note": x.get("error") or x.get("reason") or (f"fell back to {x['fallback_to']}"
                                                                   if "fallback_to" in x else "")}
                  for x in calls], hide_index=True, use_container_width=True)
for fname, title in (("run_info.json", "Models per role (run_info.json)"),
                     ("cost_summary.json", "Cost summary (cost_summary.json)")):
    if (run / fname).exists():
        with st.expander(title):
            st.json(inspect.read_json(run / fname))
