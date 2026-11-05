"""Lessons and known-bad sources: what earlier projects taught, and what research now skips. Read-only, model-free."""

from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.pipeline.lessons import BadSources, lessons_dir, read_lessons

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="ASPIRE: Lessons", layout="wide")


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


directory = lessons_dir(config().work_dir)
st.title("Lessons")
st.caption("After a project is delivered or re-designed, one free model call writes lessons, and code keeps only those "
           "that cite blackboard entries that exist. Research skips a source that failed for the same topic before.")

rows = read_lessons(directory)
st.subheader("Lessons by domain")
if not rows:
    st.info("No lessons yet. They appear after a project is delivered or re-designed.")
for domain in sorted({r["domain"] for r in rows}):
    with st.expander(domain, expanded=True):
        mine = [r for r in rows if r["domain"] == domain]
        st.dataframe([{"kind": r["kind"], "lesson": r["lesson"], "evidence": ", ".join(r["evidence"]),
                       "project": r["project"], "outcome": r["outcome"], "date": r["at"][:10]} for r in mine],
                     use_container_width=True, hide_index=True)

st.subheader("Known-bad sources")
bad = BadSources(directory / "bad_sources.json").rows()
if bad:
    st.dataframe([{k: r.get(k, "") for k in ("domain", "topic", "id", "why", "count", "last", "url")} for r in bad],
                 use_container_width=True, hide_index=True)
else:
    st.info("None yet. A source lands here when research rejects it for a topic; entries expire after 30 days.")
