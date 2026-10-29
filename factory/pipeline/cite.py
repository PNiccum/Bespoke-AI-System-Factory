"""Quote-level citations for wiki pages, adapted from Bespoke_LLM's wiki builder (bespoke/wiki/cite.py).

The librarian ends every sentence of a draft with one tag:

    Primers should melt at 55-65 C. {{q:europepmc:PMC1|primers with a Tm of 55 to 65 C}}
    So both primers of a pair should be designed together. {{inference}}
    The sources do not compare design tools. {{limit}}

A ``q`` tag names a document and gives a verbatim quote from it. Plain code then checks every claim before any model
sees the page: a statement stays only if its document was approved for this page and its quote is really in that
document's text (ignoring case, dashes, quote marks and spacing; "..." may join pieces that appear in order). An
inference stays only if a kept statement before it in the same paragraph supports it. A limit stays as written.
Untagged sentences and everything else are dropped, each with a recorded reason. The kept page cites documents as
``[doc id]``, and the quotes go to the verifier, which then checks only that each quote supports its sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TAG = re.compile(r"\{\{\s*(?:q:([^|}]+?)\s*\|\s*(.+?)|(inference|limit))\s*\}\}", re.S)
MAX_QUOTE_WORDS = 60
DASHES = {chr(c) for c in range(0x2010, 0x2016)} | {"−"}


def normalise(text: str) -> str:
    out = []
    for ch in text:
        if ch in "*_`":
            continue
        out.append("-" if ch in DASHES else "'" if ch in "‘’" else '"' if ch in "“”" else ch)
    return re.sub(r"\s+", " ", "".join(out)).strip().lower()


def quote_parts(quote: str) -> list[str]:
    parts = [normalise(p).strip(" .,;:") for p in re.split(r"\.{3}|…", quote)]
    return [p for p in parts if p]


def quote_in(quote: str, text: str) -> bool:
    """Every piece of the quote appears in the text, in order."""
    norm, pos = normalise(text), 0
    for part in quote_parts(quote):
        found = norm.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return bool(quote_parts(quote))


@dataclass
class Sentence:
    kind: str  # statement | inference | limit | untagged
    text: str
    doc_id: str | None = None
    quote: str | None = None
    reason: str | None = None  # None: kept


def parse(body: str) -> list[list[Sentence]]:
    """Paragraphs of tagged sentences. Headings and blank lines separate paragraphs and are not claims."""
    paragraphs = []
    for block in re.split(r"\n\s*\n|\n(?=#)", body.strip()):
        lines = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
        text = " ".join(ln.strip() for ln in lines).strip()
        if not text:
            continue
        sentences, pos = [], 0
        for m in TAG.finditer(text):
            seg = text[pos:m.start()].strip()
            pos = m.end()
            if m.group(1):
                sentences.append(Sentence("statement", seg, m.group(1).strip(), m.group(2).strip().strip('"“”')))
            else:
                sentences.append(Sentence(m.group(3), seg))
        if re.search(r"[A-Za-z]{2}", text[pos:]):
            sentences.append(Sentence("untagged", text[pos:].strip()))
        paragraphs.append(sentences)
    return paragraphs


def check(paragraphs: list[list[Sentence]], texts: dict[str, str]) -> None:
    """Set ``reason`` on every sentence that fails; ``texts`` maps approved document ids to the text models read."""
    for sentences in paragraphs:
        kept_statement = False
        for s in sentences:
            if not s.text:
                s.reason = "empty"
            elif s.kind == "untagged":
                s.reason = "no tag"
            elif s.kind == "statement":
                if s.doc_id not in texts:
                    s.reason = f"{s.doc_id} is not an approved document"
                elif len(s.quote.split()) > MAX_QUOTE_WORDS:
                    s.reason = "quote too long"
                elif not quote_in(s.quote, texts[s.doc_id]):
                    s.reason = f"quote not found in {s.doc_id}"
                else:
                    kept_statement = True
            elif s.kind == "inference" and not kept_statement:
                s.reason = "inference with no kept statement before it"


def render(paragraphs: list[list[Sentence]]) -> str:
    """The checked page: kept sentences only, statements cited as [doc id], inferences and limits marked."""
    out = []
    for sentences in paragraphs:
        words = []
        for s in sentences:
            if s.reason is not None:
                continue
            words.append(f"{s.text} [{s.doc_id}]" if s.kind == "statement" else
                         f"{s.text} _(inference)_" if s.kind == "inference" else f"{s.text} _(limit)_")
        if words:
            out.append(" ".join(words))
    return "\n\n".join(out)


def quotes(paragraphs: list[list[Sentence]]) -> str:
    """Each kept statement with its quote, for the verifier."""
    return "\n".join(f'- "{s.text}" rests on {s.doc_id}: "{s.quote}"' for p in paragraphs for s in p
                     if s.kind == "statement" and s.reason is None)


def dropped(paragraphs: list[list[Sentence]]) -> list[str]:
    return [f"{s.text[:80]!r}: {s.reason}" for p in paragraphs for s in p if s.reason is not None]
