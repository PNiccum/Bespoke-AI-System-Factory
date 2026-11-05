"""A project's subject area, decided once by code from the request text.

The tag lets the event log (factory/events.py) tell a problem that happens everywhere from one that happens only in,
say, proteomics. It is set at the interview on the ``request`` entry and never by a model: a model choosing a label
per event would be inconsistent, and the meta analysis would then group on noise. The list is short on purpose and
grows only when runs in a new area arrive.
"""

from __future__ import annotations

import re

from factory.pipeline.blackboard import Project

DOMAINS = ("bio", "data", "text", "code", "other")

# A hint matches a whole word (``gene`` never matches "general"); a hint ending in ``*`` matches a word that starts
# with it (``proteom*`` matches proteomics and proteome). A word belongs to one domain only.
HINTS: dict[str, tuple[str, ...]] = {
    "bio": ("protein*", "proteom*", "gene", "genes", "genom*", "primer*", "pcr", "rna", "mrna", "dna", "sequence*",
            "uniprot", "ncbi", "pubmed", "pmc", "cell line*", "mcf7", "assay*", "enzyme*", "antibod*", "clinical",
            "drug*", "chembl", "open targets", "pathway*", "ontology", "biolog*", "peptide*", "serum", "refseq"),
    "data": ("csv", "dataframe*", "spreadsheet*", "table", "tables", "column", "columns", "dataset*", "statistic*",
             "profile", "aggregate*", "pivot", "dedup*", "log file*", "logs", "metric*", "anomal*", "time series",
             "parquet", "null*"),
    "text": ("summar*", "article*", "document*", "report*", "citation*", "literature", "openalex", "abstract*",
             "essay*", "transcript*", "translat*", "classif*", "sentiment", "claims", "wiki"),
    "code": ("bug", "bugs", "patch", "pull request*", "diff", "refactor*", "unit test*", "pytest", "repositor*",
             "repo", "function*", "module*", "library", "api client", "cli", "failing test*", "stack trace",
             "exception*"),
}
_PATTERNS = {d: re.compile("|".join(rf"\b{re.escape(h[:-1])}\w*" if h.endswith("*") else rf"\b{re.escape(h)}\b"
                                    for h in hints)) for d, hints in HINTS.items()}


def classify(text: str) -> str:
    """The domain whose hints the text mentions most, by count; ``other`` when none matches. Bio wins a tie with
    data (a protein table is a bio task), code loses a tie (every task mentions a function somewhere)."""
    low = re.sub(r"\s+", " ", text.lower())
    scores = {d: len(p.findall(low)) for d, p in _PATTERNS.items()}
    best = max(scores.values())
    if best == 0:
        return "other"
    for d in ("bio", "data", "text", "code"):  # tie order
        if scores[d] == best:
            return d
    return "other"


def project_domain(project: Project) -> str:
    """The tag on the project's request entry, or, for a project made before tags existed, one from its text."""
    req = next((e for e in project.board.entries("request")), None)
    if req is None:
        return ""
    if req.data.get("domain") in DOMAINS:
        return req.data["domain"]
    task = project.root / "task.txt"
    return classify(task.read_text(encoding="utf-8") if task.is_file() else str(req.data.get("text", "")))
