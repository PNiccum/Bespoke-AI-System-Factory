"""The wiki viewer: browse what ASPIRE has learned, one page per topic, with each claim's quote and source.

Read-only and model-free. ``?page=<slug>`` opens a page directly, so one wiki page can link to another.
"""

from pathlib import Path

import streamlit as st

from factory.config import load_config
from factory.pipeline.wiki import MIN_SOURCES, Wiki
from factory.ui import inspect

ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="ASPIRE: Wiki", layout="wide")


@st.cache_resource
def config():
    return load_config(ROOT / "configs" / "factory.yaml")


wiki = Wiki(config().work_dir / "wiki")
pages = wiki.pages()
by_slug = {p.slug: p for p in pages}

st.title("Wiki")
st.caption("What ASPIRE has learned, shared by every project. The librarian writes a page from approved sources, "
           "every sentence must quote its source word for word, and a verifier from another model family checks it. "
           f"Only a verified page with {MIN_SOURCES} sources (or one reference manual) lets a later project skip "
           "the research.")
st.caption(f"Folder: `{wiki.root}`")

if not pages:
    st.info("The wiki is empty. Pages appear here after the knowledge loop runs "
            "(`python -m factory research <project>`, or the Interviewer page).")
    st.stop()

verified = sum(inspect.status(p) == "verified" for p in pages)
a, b, c, d = st.columns(4)
a.metric("Pages", len(pages))
b.metric("Verified", verified)
c.metric("Not verified", sum(not p.verified for p in pages))
d.metric("ARC-only", sum(p.license == "arc_only" for p in pages))

browse, health, log = st.tabs(["Pages", "Health check", "Change log"])

with browse:
    left, right = st.columns([1, 2], gap="large")
    with left:
        query = st.text_input("Search", placeholder="words in the topic or text")
        shown = st.multiselect("Status", ["verified", "verified, too few sources", "not verified"],
                               default=["verified", "verified, too few sources", "not verified"])
        found = [p for p in pages if inspect.status(p) in shown and (not query or inspect.matches(p, query))]
        st.caption(f"{len(found)} of {len(pages)} pages")
        if not found:
            st.warning("No page matches.")
            st.stop()
        wanted = st.query_params.get("page")
        slugs = [p.slug for p in found]
        index = slugs.index(wanted) if wanted in slugs else 0
        slug = st.radio("Open", slugs, index=index, format_func=lambda s: f"{by_slug[s].topic}  ({inspect.status(by_slug[s])})",
                        label_visibility="collapsed")
        if slug != wanted:
            st.query_params["page"] = slug

    page = by_slug[slug]
    v = inspect.view(page, pages)
    with right:
        st.header(page.topic)
        status = inspect.status(page)
        color = {"verified": "green", "not verified": "red"}.get(status, "orange")
        st.markdown(f":{color}[**{status}**] · license {page.license} · {len(page.sources)} sources · "
                    f"updated {page.updated[:10] or '?'}")
        if page.license == "arc_only":
            st.warning("ARC-only: built from text that may not leave ARC. Do not copy it into anything sent to "
                       "Claude or OpenAI.")
        if page.issues:
            st.error("The verifier did not pass this page:\n\n" + "\n".join(f"- {inspect.md_safe(i)}"
                                                                         for i in page.issues))
        st.markdown(f"**{inspect.md_safe(page.summary)}**")
        st.markdown(v.body)
        if v.quotes:
            with st.expander(f"Quotes behind each claim ({len(v.quotes)})"):
                for n, text in v.quotes:
                    st.markdown(f"{str(n).translate(inspect.SUPERSCRIPT)} {inspect.md_safe(text)}")
        st.subheader("Sources")
        for i in page.sources:
            st.markdown(f"- `{i}` {inspect.md_safe(page.citations.get(i, ''))}")
        cols = st.columns(3)
        for col, title, items in ((cols[0], "Links to", v.links), (cols[1], "Linked from", v.backlinks),
                                  (cols[2], "Projects that used it", page.projects)):
            with col:
                st.markdown(f"**{title}**")
                if not items:
                    st.caption("none")
                for s in items:
                    if s in by_slug:
                        st.markdown(f"- [{by_slug[s].topic}](?page={s})")
                    else:
                        st.markdown(f"- `{s}`")
        if v.missing:
            st.caption("Links to pages that do not exist: " + ", ".join(v.missing))
        md = wiki.root / f"{page.slug}.md"
        if md.exists():
            st.download_button("Download this page (.md)", md.read_bytes(), file_name=md.name)

with health:
    problems = wiki.lint()
    st.caption("Plain-code checks, no model: broken links, orphan pages, pages that did not pass and verified "
               "pages with too few sources. The same list as `python -m factory wiki`.")
    if problems:
        st.dataframe([{"page": p.split(":", 1)[0], "problem": p.split(":", 1)[1].strip()} for p in problems],
                     hide_index=True, use_container_width=True)
    else:
        st.success("No problems found.")

with log:
    entries = inspect.wiki_log(wiki)
    if entries:
        st.caption("Each save, newest first (from log.md).")
        st.markdown("\n".join(f"- {inspect.md_safe(e)}" for e in entries))
    else:
        st.info("No log.md yet.")
