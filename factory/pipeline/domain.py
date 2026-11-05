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

# Longest, most specific hints first within each domain; a word may belong to one domain only.
HINTS: dict[str, tuple[str, ...]] = {
    "bio": ("protein", "proteom", "gene", "genom", "primer", "pcr", "rna", "dna", "sequence", "uniprot", "ncbi",
            "pubmed", "europe pmc", "pmc", "cell line", "mcf7", "assay", "enzyme", "antibod", "clinical",
            "drug", "chembl", "open targets", "pathway", "ontology", "proteins", "biolog", "peptide", "serum"),
    "data": ("csv", "dataframe", "spreadsheet", "table", "column", "dataset", "statistic", "profile", "aggregate",
             "pivot", "dedup", "log file", "logs", "metric", "anomal", "time series", "json record", "parquet"),
    "text": ("summar", "article", "document", "report", "citation", "literature", "openalex", "abstract", "essay",
             "transcript", "translate", "classif", "sentiment", "extract claims", "claims", "wiki"),
    "code": ("bug", "patch", "pull request", "diff", "refactor", "unit test", "pytest", "repository", "repo",
             "function", "module", "library", "api client", "cli", "fix the", "failing test"),
}


def classify(text: str) -> str:
    """The domain whose hints the text mentions most, by count; ``other`` when none matches. Bio wins a tie with
    data (a protein table is a bio task), code loses a tie (every task mentions a function somewhere)."""
    low = re.sub(r"\s+", " ", text.lower())
    scores = {d: sum(low.count(h) for h in hints) for d, hints in HINTS.items()}
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
