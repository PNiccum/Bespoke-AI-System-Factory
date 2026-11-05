"""Runs and spend: what each run called, on which model, at what cost, and what is left on each paid account.

Read-only and model-free.
"""

from pathlib import Path

import streamlit as st

from factory import events as ev
from factory import historian
from factory.config import load_config
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendLedger, deny_all
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

elog = ev.EventLog(run / ev.FILE, run.name).events()
st.subheader("Events: errors this run met")
if not elog:
    st.caption("No errors were recorded for this run.")
else:
    s = ev.summary(elog)
    a, b = st.columns(2)
    a.metric("Corrected", s["corrected"])
    b.metric("Not corrected", s["open"])
    st.caption("Code records each event and decides when it is corrected; the historian only explains.")
    st.dataframe([{"id": e.id, "status": e.status, "stage": e.stage, "severity": e.severity, "message": e.message[:160],
                   "corrected by": e.how} for e in elog], hide_index=True, use_container_width=True)
    if (run / "history.md").exists():
        with st.expander("Historian's account (history.md)", expanded=True):
            st.markdown((run / "history.md").read_text(encoding="utf-8"))

st.subheader("Ask the historian")
st.caption("A free ARC model answers from the event log only and names the events it used; their status and fix are "
           "read from the log, not from the model.")
scope_all = st.radio("Look at", ["This run", "All runs"], horizontal=True) == "All runs"
with st.form("ask-historian"):
    question = st.text_input("Question", placeholder="Why did the build stage fail? Did the same error happen before?")
    asked = st.form_submit_button("Ask")
if asked and question.strip():
    pool = ev.run_events(base) if scope_all else elog
    if "historian_llm" not in st.session_state:
        st.session_state.historian_llm = LLM(cfg, new_run_dir(cfg, "historian-ask"), approver=deny_all)
    with st.spinner("The historian is reading the log..."):
        reply, cited = historian.ask(st.session_state.historian_llm, question.strip(), pool,
                                     scope="All runs" if scope_all else f"Run {name}")
    st.markdown(reply)
    if cited:
        st.dataframe([{"id": e.id, "run": e.run, "status": e.status, "stage": e.stage, "message": e.message[:160],
                       "corrected by": e.how} for e in cited], hide_index=True, use_container_width=True)

st.subheader("Events across all runs (meta analysis)")
every = ev.run_events(base)
if not every:
    st.caption("No events recorded in any run yet.")
else:
    t = ev.summary(every)
    st.caption(f"{t['events']} events in {t['runs']} runs: {t['corrected']} corrected, {t['open']} not corrected.")
    st.dataframe([{"problem": g.fingerprint, "stage": g.stage, "times": g.count, "runs": len(g.runs),
                   "corrected": g.corrected, "open": g.open, "came back after a fix": g.returned_after_fix,
                   "corrected by": "; ".join(g.how)[:100], "message": g.message[:140]}
                  for g in ev.meta_analysis(every)], hide_index=True, use_container_width=True)
    meta = cfg.work_dir / "history" / "meta.md"
    if meta.exists():
        with st.expander("Historian's meta analysis (run `python -m factory.historian --all` to refresh)"):
            st.markdown(meta.read_text(encoding="utf-8"))
