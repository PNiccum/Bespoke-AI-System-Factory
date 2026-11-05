"""ASPIRE (the Bespoke AI System Factory), web front end: streamlit run app.py

Pick a task (fix a bug, write tests, review a pull request), watch each agent work, approve or refuse paid model
calls, and download the bundle.
"""

import json
from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.demo import recorded_task
from factory.llm.cost import BudgetExceeded
from factory.llm.gate import SpendLedger
from factory.orchestrator import Task, plan_for
from factory.ui.session import RunSession
from factory.workflows import TASK_TYPES

ROOT = Path(__file__).parent
STATUS_ICON = {"started": "🔄", "done": "✅", "failed": "❌", "waiting": "⏳"}

st.set_page_config(page_title="ASPIRE", layout="wide")


def usd(x: float) -> str:
    """Dollars for Streamlit markdown, where a bare $ starts a formula."""
    return f"\\${x:.2f}"


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


if "session" not in st.session_state:
    st.session_state.session = RunSession(config())
session: RunSession = st.session_state.session

st.title("ASPIRE")
st.caption("Agentic System Producing Intelligent Runnable Executables. "
           "Give it a bug, code to test, or a pull request; it returns the result as a bundle you can recheck "
           "without any model. Free ARC models do the work; paid models wait for your approval.")
FIXTURE = ROOT / "tests" / "fixtures"

with st.sidebar:
    st.header("Task")
    task_type = st.selectbox("Task type", list(TASK_TYPES), format_func=TASK_TYPES.get)
    with st.form("task"):
        repo = st.text_input("Repository path", value=str(FIXTURE / "buggy_repo"))
        tests, patch = "", ""
        if task_type == "bug_fix":
            text = st.text_area("Bug report", height=160,
                                value='parse_date("2026-10-28") returns (2026, 28, 10): month and day are swapped.')
            tests = st.text_input("Tests the fix must turn green (optional, comma separated)")
        elif task_type == "test_gen":
            text = st.text_area("What to test", height=120, value="Write tests for days_in_month in calc/dates.py")
        else:
            text = st.text_area("Pull request description", height=100,
                                value="Fix parse_date swapping month and day")
            patch = st.text_area("Diff (unified, as git diff prints it)", height=200,
                                 value=(FIXTURE / "buggy_repo.patch").read_text(encoding="utf-8"))
        submitted = st.form_submit_button("Run", disabled=session.running, type="primary")
    if submitted:
        session.start(Task(description=text, repo_path=repo, task_type=task_type, patch=patch,
                           failing_tests=[t.strip() for t in tests.split(",") if t.strip()]))
    recordings = sorted((ROOT / "demos").glob("*.json"))
    if recordings:
        st.header("Recorded runs")
        st.caption("Replays a real run offline: tests, checks and the bundle run again; the model replies come "
                   "from the recording.")
        chosen = st.selectbox("Recording", recordings, format_func=lambda p: p.stem)
        if st.button("Replay", disabled=session.running):
            recording = json.loads(chosen.read_text(encoding="utf-8"))
            st.session_state.replayed = recording
            session.start(recorded_task(recording), recording=recording)
    cfg = config()
    st.header("Budget")
    st.write(f"Soft limit per run: {usd(cfg.budget.approve_above_usd)}, hard stop {usd(cfg.budget.max_real_usd)}")
    try:
        spent = SpendLedger(cfg.work_dir / "spend.json", cfg.credits).spent
    except BudgetExceeded as e:
        spent = {}
        st.error(str(e))
    for provider, credit in cfg.credits.by_provider.items():
        st.progress(min(spent.get(provider, 0.0) / credit, 1.0),
                    text=f"{provider}: {usd(spent.get(provider, 0.0))} of {usd(credit)}")
    st.caption(f"{usd(cfg.credits.reserve_usd)} is held back for the demo.")

st.subheader("Plan")
st.table([{"step": s.id, "capability": s.capability, "model?": "yes" if s.needs_llm else "no", "why": s.reason}
          for s in plan_for(Task(description="", repo_path=".", task_type=task_type)).steps])


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

    if (recording := st.session_state.get("replayed")) and session.llm and "replay" in session.llm.run_dir.name:
        original = recording.get("original_cost", {})
        st.info(f"Replay of a run recorded {recording['recorded_at'][:10]} with {', '.join(recording['models'])}. "
                f"That run made {original.get('calls', '?')} model calls and spent "
                f"{usd(original.get('real_usd', 0.0))}; this replay calls no model.")
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
        (st.success if result.status in ("fixed", "tested", "reviewed") else st.warning)(
            f"{result.status} after {result.attempts} attempt(s)" + (f": {result.stop_reason}" if result.stop_reason
                                                                     else ""))
        if result.plan.task_type == "pr_review" and "review" in result.report:
            show_pr_review(result.report)
        elif result.patch:
            st.code(result.patch, language="diff")
        if mutation := result.report.get("mutation"):
            st.write(f"The new tests caught **{mutation['killed']} of {mutation['total']}** small bugs planted in "
                     f"{', '.join(result.report['targets'])}.")
            if mutation["survivors"]:
                st.caption("Missed: " + "; ".join(f"line {m['line']}: {m['change']}" for m in mutation["survivors"]))
        if result.review:
            st.write(f"Review: **{result.review.verdict}**", [d.problem for d in result.review.defects] or "")
        if session.bundle:
            st.download_button("Download bundle", session.bundle.read_bytes(), file_name=session.bundle.name,
                               mime="application/zip", type="primary")


def show_pr_review(report: dict) -> None:
    review, tests = report["review"], report["tests"]
    st.markdown(f"**Verdict: {review['verdict']}**. {review['summary']}")
    if report["overridden"]:
        st.warning(report["overridden"])
    st.write("The patch does not apply." if not tests["applied"] else
             f"Tests with the change: fixes {len(tests['newly_passing'])}, breaks {len(tests['newly_failing'])}.")
    rows = [{"from": "rules", "where": f"{f['path']}:{f['line']}", "severity": f["severity"], "note": f["message"]}
            for f in report["facts"]["findings"]]
    rows += [{"from": "reviewer", "where": f"{c['file']}:{c['line']}", "severity": c["severity"],
              "note": c["problem"] + (f" Suggestion: {c['suggestion']}" if c["suggestion"] else "")}
             for c in review["comments"]]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    if report["dropped_comments"]:
        st.caption(f"{len(report['dropped_comments'])} reviewer comments were dropped: not on a line the diff changes.")


live()
