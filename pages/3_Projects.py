"""The project browser: each project's stage, its files, and its blackboard (every entry the agents wrote, in order).

Read-only and model-free. ``?project=<id>`` opens a project directly.
"""

import json
from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.ui import inspect

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="ASPIRE: Projects", layout="wide")


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


base = config().work_dir / "projects"
items = inspect.projects(base)

st.title("Projects")
st.caption("Each project the Interviewer started. Its blackboard is the append-only log every agent reads and "
           "writes, so it is also the project's provenance: who wrote what, and what it built on.")
st.caption(f"Folder: `{base}`")
if not items:
    st.info("No projects yet. Start one on the Interviewer page.")
    st.stop()

st.dataframe(inspect.project_rows(items), hide_index=True, use_container_width=True)
ids = [p.id for p in items]
wanted = st.query_params.get("project")
pid = st.selectbox("Open a project", ids, index=ids.index(wanted) if wanted in ids else 0)
if pid != wanted:
    st.query_params["project"] = pid
project = next(p for p in items if p.id == pid)
entries = project.board.entries()

st.subheader(f"{pid}: {inspect.stage(project)}")
task = project.root / "task.txt"
if task.exists():
    with st.expander("task.txt (what was asked, once Plan A was approved)"):
        st.text(task.read_text(encoding="utf-8"))

files = sorted(p for p in project.root.rglob("*") if p.is_file() and p.name != "blackboard.jsonl")
if files:
    with st.expander(f"Files ({len(files)})"):
        rel = st.selectbox("File", files, format_func=lambda p: str(p.relative_to(project.root)))
        try:
            st.code(rel.read_text(encoding="utf-8")[:20000], language=None)
        except UnicodeDecodeError:
            st.caption("Binary file.")
        st.download_button("Download", rel.read_bytes(), file_name=rel.name)

st.subheader(f"Blackboard ({len(entries)} entries)")
kinds = sorted({e.kind for e in entries})
left, mid, right = st.columns(3)
chosen = left.multiselect("Kinds", kinds, default=kinds)
authors = sorted({e.author for e in entries})
who = mid.multiselect("Authors", authors, default=authors)
text = right.text_input("Contains")
shown = [e for e in entries if e.kind in chosen and e.author in who
         and (not text or text.lower() in json.dumps(e.data, ensure_ascii=False).lower())]
st.dataframe([{"id": e.id, "at": e.at[:19].replace("T", " "), "kind": e.kind, "author": e.author,
               "license": e.license, "builds on": ", ".join(e.refs), "summary": inspect.entry_summary(e.data)}
              for e in shown], hide_index=True, use_container_width=True)
if shown:
    eid = st.selectbox("Entry", [e.id for e in shown], index=len(shown) - 1)
    entry = next(e for e in shown if e.id == eid)
    if entry.license == "arc_only":
        st.warning("ARC-only entry: do not copy it into anything sent to Claude or OpenAI.")
    st.json(entry.data, expanded=2)
st.download_button("Download blackboard.jsonl", project.board.path.read_bytes(), file_name=f"{pid}-blackboard.jsonl")
