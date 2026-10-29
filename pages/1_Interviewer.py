"""The Interviewer: describe what you want, answer its questions, and approve Plan A to start a project.

Free ARC models only: this page refuses every paid call.
"""

import json
from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.llm import LLM, new_run_dir
from factory.llm.gate import deny_all
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import MAX_QUESTIONS, Interview
from factory.pipeline.planning import PlanningLoop
from factory.registry import Registry

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="SIMPLE: Interviewer", layout="wide")


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


def show_plan(entry) -> None:
    plan = entry.data["plan"]
    st.markdown(f"**{plan['title']}**: {plan['goal']}  \nDeliverable: {plan['deliverable']}")
    st.dataframe([{"task": t["id"], "what": t["title"], "done when": t["acceptance"],
                   "after": ", ".join(t.get("depends_on", []))} for t in plan["tasks"]],
                 hide_index=True, use_container_width=True)
    for heading, key in (("Assumptions", "assumptions"), ("Out of scope", "out_of_scope")):
        if plan.get(key):
            st.caption(f"{heading}: " + "; ".join(plan[key]))
    for problem in entry.data["problems"]:
        st.error(f"Plan check: {problem}")


st.title("Interviewer")
st.caption(f"Say what you want built. The Interviewer asks up to {MAX_QUESTIONS} questions, then proposes Plan A. "
           "Approving it creates the project folder and its task.txt.")

iv: Interview | None = st.session_state.get("interview")
if iv is None:
    with st.form("request"):
        request = st.text_area("What do you want built?", height=120)
        go = st.form_submit_button("Start", type="primary")
    if go and request.strip():
        cfg = config()
        project = Project.create(cfg.work_dir / "projects", request)
        iv = Interview(LLM(cfg, new_run_dir(cfg, "interview"), approver=deny_all), Registry.load(cfg), project)
        st.session_state.interview = iv
        with st.spinner("The Interviewer is reading your request..."):
            iv.start(request)
        st.rerun()
    st.stop()

for e in iv.board.entries():
    if e.kind == "request":
        st.chat_message("user").write(e.data["text"])
    elif e.kind == "answer":
        st.chat_message("user").write(e.data["text"])
    elif e.kind == "question":
        with st.chat_message("assistant"):
            if e.data.get("note"):
                st.write(e.data["note"])
            st.markdown("\n".join(f"- {q}" for q in e.data["questions"]))
    elif e.kind == "plan":
        with st.chat_message("assistant"):
            if e.data.get("note"):
                st.write(e.data["note"])
            show_plan(e)

if iv.board.latest("approval"):
    st.success(f"Plan A approved. Project folder: {iv.project.root}")
    st.code((iv.project.root / "task.txt").read_text(), language="markdown")
    decision = iv.board.latest("decision")
    if decision is None and st.button("Run the planning loop", type="primary"):
        with st.spinner("Three model families are critiquing, weighing, editing and voting..."):
            PlanningLoop(iv.llm, Registry.load(config()), iv.project).run()
        st.rerun()
    if decision is not None:
        st.subheader(f"Planning loop: Plan {decision.data['winner']} won")
        st.dataframe([{"voter": e.author, "choice": e.data["choice"], "counted": e.data["counted"],
                       "why": e.data["why"]} for e in iv.board.entries("vote")], hide_index=True)
        critique, analysis = iv.board.latest("critique"), iv.board.latest("analysis")
        st.caption("Weaknesses found in Plan A: " + "; ".join(critique.data["weaknesses"]))
        a, b = st.columns(2)
        for col, key, side in ((a, "a", "Plan A"), (b, "b", "Plan B")):
            col.markdown(f"**{side}**  \nPros: {'; '.join(analysis.data[key]['pros'])}  \n"
                         f"Cons: {'; '.join(analysis.data[key]['cons'])}")
        for problem in decision.data["problems"]:
            st.error(f"Left in the winning plan: {problem}")
        st.code((iv.project.root / "plan.json").read_text(), language="json")
    with st.expander("Blackboard"):
        st.code("\n".join(json.dumps(e.model_dump()) for e in iv.board.entries()), language="json")
    if st.button("New request"):
        del st.session_state.interview
        st.rerun()
    st.stop()

last = iv.board.entries()[-1]
if last.kind == "plan" and not last.data["problems"] and st.button("Approve Plan A", type="primary"):
    iv.approve()
    st.rerun()
prompt = "Your answers" if last.kind == "question" else "Or say what to change"
if text := st.chat_input(prompt):
    with st.spinner("The Interviewer is thinking..."):
        iv.reply(text)
    st.rerun()
