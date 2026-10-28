"""Bespoke AI System Factory, web front end: streamlit run app.py

Submit a bug fix, watch each agent work, approve or refuse paid model calls, and download the bundle.
"""

from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.llm.gate import SpendLedger
from factory.orchestrator import Task, plan_for
from factory.ui.session import RunSession

ROOT = Path(__file__).parent
STATUS_ICON = {"started": "🔄", "done": "✅", "failed": "❌", "waiting": "⏳"}

st.set_page_config(page_title="Bespoke AI System Factory", layout="wide")


def usd(x: float) -> str:
    """Dollars for Streamlit markdown, where a bare $ starts a formula."""
    return f"\\${x:.2f}"


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


if "session" not in st.session_state:
    st.session_state.session = RunSession(config())
session: RunSession = st.session_state.session

st.title("Bespoke AI System Factory")
st.caption("Give it a bug; it returns a tested fix as a bundle you can rerun without any model. "
           "Free ARC models do the work; paid models wait for your approval.")

with st.sidebar:
    st.header("Task")
    with st.form("task"):
        repo = st.text_input("Repository path", value=str(ROOT / "tests" / "fixtures" / "buggy_repo"))
        issue = st.text_area("Bug report", height=160,
                             value='parse_date("2026-10-28") returns (2026, 28, 10): month and day are swapped.')
        tests = st.text_input("Tests the fix must turn green (optional, comma separated)")
        submitted = st.form_submit_button("Run", disabled=session.running, type="primary")
    if submitted:
        session.start(Task(description=issue, repo_path=repo,
                           failing_tests=[t.strip() for t in tests.split(",") if t.strip()]))
    cfg = config()
    st.header("Budget")
    st.write(f"Soft limit per run: {usd(cfg.budget.approve_above_usd)}, hard stop {usd(cfg.budget.max_real_usd)}")
    spent = SpendLedger(cfg.work_dir / "spend.json", cfg.credits).spent
    for provider, credit in cfg.credits.by_provider.items():
        st.progress(min(spent.get(provider, 0.0) / credit, 1.0),
                    text=f"{provider}: {usd(spent.get(provider, 0.0))} of {usd(credit)}")
    st.caption(f"{usd(cfg.credits.reserve_usd)} is held back for the demo.")

st.subheader("Plan")
st.table([{"step": s.id, "capability": s.capability, "model?": "yes" if s.needs_llm else "no", "why": s.reason}
          for s in plan_for(Task(description="", repo_path=".")).steps])


@st.fragment(run_every=1)
def live():
    pending = session.pending
    if pending:
        r = pending.request
        with st.container(border=True):
            st.markdown(f"**Approve a paid call?** {r.reason}")
            st.write(f"{r.role} on **{r.model}** via {r.provider}, up to **{usd(r.estimate_usd)}**. "
                     f"This run has spent {usd(r.run_spent_usd)}.")
            yes, no = st.columns(2)
            if yes.button("Approve", type="primary", use_container_width=True):
                session.answer(True)
            if no.button("Reject", use_container_width=True):
                session.answer(False)

    st.subheader("Agent activity")
    if session.events:
        st.dataframe([{"": STATUS_ICON.get(e.status, ""), "step": e.step, "agent": e.agent, "status": e.status,
                       "detail": e.detail} for e in session.events], hide_index=True, use_container_width=True)
    elif not session.running:
        st.info("Submit a task to start.")

    if session.llm:
        cost = session.llm.ledger.summary()
        a, b, c = st.columns(3)
        a.metric("Model calls", cost["calls"])
        b.metric("Real spend", f"${cost['real_usd']:.2f}")
        c.metric("Shadow cost (if ARC were paid)", f"${cost['shadow_usd']:.2f}")

    if session.error:
        st.error(session.error)
    result = session.result
    if result and not session.running:
        (st.success if result.status == "fixed" else st.warning)(
            f"{result.status} after {result.attempts} attempt(s)" + (f": {result.stop_reason}" if result.stop_reason
                                                                     else ""))
        if result.patch:
            st.code(result.patch, language="diff")
        if result.review:
            st.write(f"Review: **{result.review.verdict}**", [d.problem for d in result.review.defects] or "")
        if session.bundle:
            st.download_button("Download bundle", session.bundle.read_bytes(), file_name=session.bundle.name,
                               mime="application/zip", type="primary")


live()
