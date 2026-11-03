"""Quote-level citations for wiki pages, adapted from Bespoke_LLM's wiki builder (bespoke/wiki/cite.py, papers.py).

The librarian ends every sentence of a draft with one tag:

    Primers should melt at 55-65 C. {{q:europepmc:PMC1|primers with a Tm of 55 to 65 C}}
    So both primers of a pair should be designed together. {{inference}}
    The sources do not compare design tools. {{limit}}

A ``q`` tag names a document and gives a verbatim quote from it. Plain code then checks every claim before any model
sees the page: a statement stays only if its document was approved for this page and its quote is really in that
document's text (ignoring case, dashes, quote marks and spacing; "..." may join pieces that appear in order). An
inference stays only if a kept statement before it in the same block supports it and it adds no numbers or symbols
those statements and their quotes lack (Bespoke_LLM's D9 rule). A limit stays as written. Untagged sentences are
dropped, each with a recorded reason; a short fragment with no sentence ending ("Inputs:") is a list label and stays.

Ported from Bespoke_LLM along with the checks: the page keeps its structure (headings, bullets, link-only lines), a
heading whose section lost everything goes, each kept quote is located in its document (the tightest span, so an
ellipsis quote does not land on an earlier mention, and the nearest section above it), and every statement gets a
footnote with its quote. ``[[page]]`` links to other wiki pages stay links when the page exists and become plain
words when it does not. The kept page cites documents as ``[doc id]``; the quotes also go to the verifier, which then
checks only that each quote supports its sentence.

Left in Bespoke_LLM: its D1/D2/D10/D16 cue rules (their module, bespoke/verify, was not in the handoff) and the two
semantic checkers per sentence (ASPIRE uses one verifier of another family per page, to keep the token cost down).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TAG = re.compile(r"\{\{\s*(?:q:([^|}]+?)\s*\|\s*(.+?)|(inference|limit))\s*\}\}", re.S)
WIKILINK = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_BULLET = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
FULL_TEXT = "From the full text:"  # the marker sources.py puts between an abstract and its full-text passage
MAX_QUOTE_WORDS = 60
DASHES = {chr(c) for c in range(0x2010, 0x2016)} | {"−"}


# --------------------------------------------------------------------------
# Matching quotes (bespoke/wiki/papers.py)
# --------------------------------------------------------------------------

def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """The text lowercased, without emphasis marks, with dashes, quote marks and runs of space made plain; plus, for
    each output character, the index of the input character it came from (so a match can be traced back)."""
    out: list[str] = []
    idx: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch in "*_`":
            continue
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            idx.append(i - 1)
            pending_space = False
        ch = "-" if ch in DASHES else "'" if ch in "‘’" else '"' if ch in "“”" else ch
        for c in ch.lower():  # lower() can expand a character (e.g. 'İ')
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def normalise(text: str) -> str:
    return normalise_with_map(text)[0]


def quote_parts(quote: str) -> list[str]:
    """The pieces of a quote between ellipses."""
    parts = [normalise(p).strip(" .,;:") for p in re.split(r"\.{3}|…", quote)]
    return [p for p in parts if p]


def best_span(norm: str, parts: list[str]) -> tuple[int, int] | None:
    """Start and end, in normalised text, of the tightest place where the parts occur in order. A quote with an
    ellipsis can start with a common word; the first occurrence of the first part may be an earlier mention."""
    if not parts:
        return None
    best = None
    i = norm.find(parts[0])
    while i >= 0:
        pos = end = i + len(parts[0])
        for p in parts[1:]:
            j = norm.find(p, pos)
            if j < 0:
                return best  # later starts cannot fit either
            pos = end = j + len(p)
        if best is None or end - i < best[1] - best[0]:
            best = (i, end)
        i = norm.find(parts[0], i + 1)
    return best


def quote_in(quote: str, text: str) -> bool:
    """Every piece of the quote appears in the text, in order."""
    return best_span(normalise(text), quote_parts(quote)) is not None


@dataclass
class Location:
    section: str | None  # nearest heading above the quote ("full text" after the full-text marker)
    start: int  # character span in the document text
    end: int


def locate(quote: str, text: str) -> Location | None:
    norm, idx = normalise_with_map(text)
    span = best_span(norm, quote_parts(quote))
    if span is None:
        return None
    start, end = idx[span[0]], idx[span[1] - 1] + 1
    full = text.find(FULL_TEXT)
    marks = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    if full >= 0:
        marks.append((full, "full text"))
    section = "abstract" if full >= 0 else None
    for at, title in sorted(marks):
        if at > start:
            break
        section = title
    return Location(section, start, end)


# --------------------------------------------------------------------------
# Parsing (bespoke/wiki/cite.py)
# --------------------------------------------------------------------------

@dataclass
class Sentence:
    kind: str  # statement | inference | limit | untagged | label
    text: str  # markdown as written, without the tag
    doc_id: str | None = None
    quote: str | None = None
    reason: str | None = None  # None: kept
    location: Location | None = None
    basis: list[int] = field(default_factory=list)  # indexes of kept statements in the same block

    @property
    def plain(self) -> str:
        return plain_text(self.text)


@dataclass
class Block:
    kind: str  # heading | nav | prose | table | blank
    raw: str
    prefix: str = ""  # a bullet or number, for list items
    sentences: list[Sentence] = field(default_factory=list)


def plain_text(md: str) -> str:
    """Text for the checks: wikilinks become their words, emphasis and stray tags go."""
    def link(m):
        return m.group(3) or m.group(1).split("/")[-1].replace("-", " ")
    t = WIKILINK.sub(link, md)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)  # markdown links keep their text
    t = TAG.sub("", t)
    return re.sub(r"\s+", " ", t.replace("**", "").replace("__", "")).strip()


def _is_nav(text: str) -> bool:
    """Only links and separators, optionally after a short label: 'Related pages: [[a]], [[b]]'."""
    if TAG.search(text):
        return False
    rest = re.sub(r"\[[^\]]*\]\([^)]*\)", "", WIKILINK.sub("", text))
    if not re.search(r"[A-Za-z]{2}", rest):
        return bool(text.strip())
    label = re.fullmatch(r"\s*\*{0,2}[^*:]{1,40}:\*{0,2}[\s,;·/()&and-]*", rest)
    return bool(label) and bool(WIKILINK.search(text))


def _prose(text: str, prefix: str) -> Block:
    raw = prefix + text
    if _is_nav(text):
        return Block("nav", raw, prefix)
    sentences, pos = [], 0
    for m in TAG.finditer(text):
        seg = text[pos:m.start()].strip()
        pos = m.end()
        if m.group(1):
            sentences.append(Sentence("statement", seg, m.group(1).strip(), m.group(2).strip().strip('"“”')))
        else:
            sentences.append(Sentence(m.group(3), seg))
    tail = text[pos:].strip()
    if re.search(r"[A-Za-z]{2}", WIKILINK.sub("", tail)):
        # A short fragment with no sentence ending is a list label ("Inputs:", "Soluble factors"), not a claim.
        is_label = not re.search(r"[.!?]\)?\**$", tail) and len(plain_text(tail).split()) <= 12
        sentences.append(Sentence("label" if is_label else "untagged", tail))
    elif tail and sentences:
        sentences[-1].text += tail  # stray punctuation after the last tag
    return Block("prose", raw, prefix, sentences)


def parse(body: str) -> list[Block]:
    """The page as blocks: headings, link-only lines, tables, blank lines, and prose (paragraphs and list items) split
    into tagged sentences."""
    blocks: list[Block] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append(_prose(" ".join(ln.strip() for ln in para), ""))
            para.clear()

    for line in body.strip().split("\n"):
        s = line.strip()
        if not s:
            flush()
            blocks.append(Block("blank", ""))
        elif s.startswith("#"):
            flush()
            blocks.append(Block("heading", s))
        elif s.startswith("|"):
            flush()
            blocks.append(Block("table", line))
        elif m := _BULLET.match(line):
            flush()
            blocks.append(_prose(line[m.end():], m.group(1)))
        elif para or not blocks or blocks[-1].kind != "prose" or not blocks[-1].prefix:
            para.append(line)
        else:  # continuation of a list item
            b = blocks.pop()
            blocks.append(_prose(b.raw[len(b.prefix):] + " " + s, b.prefix))
    flush()
    return blocks


def sentences(blocks: list[Block]) -> list[Sentence]:
    return [s for b in blocks for s in b.sentences]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

SPECIFIC = re.compile(r"\b\d+(?:[.,]\d+)?\b|\b(?=[\w-]*\d)(?=[\w-]*[A-Z])[\w-]{2,}\b")


def specifics(text: str) -> set[str]:
    """Numbers and symbol-like tokens (BRCA1, NM_007294, T7): what an inference may only repeat, never add. A
    stand-in for Bespoke_LLM's rules.specifics, whose module was not in the handoff."""
    return {m.group(0).lower().replace(",", ".") for m in SPECIFIC.finditer(text)}


def check(blocks: list[Block], texts: dict[str, str]) -> None:
    """Set ``reason`` on every sentence that fails, and locate kept quotes; ``texts`` maps approved document ids to
    the text models read."""
    for b in blocks:
        kept: list[int] = []
        for i, s in enumerate(b.sentences):
            if not s.text:
                s.reason = "empty"
            elif s.kind == "untagged":
                s.reason = "no tag"
            elif s.kind == "statement":
                if s.doc_id not in texts:
                    s.reason = f"{s.doc_id} is not an approved document"
                elif len(s.quote.split()) > MAX_QUOTE_WORDS:
                    s.reason = "quote too long"
                elif (loc := locate(s.quote, texts[s.doc_id])) is None:
                    s.reason = f"quote not found in {s.doc_id}"
                else:
                    s.location = loc
                    kept.append(i)
            elif s.kind == "inference":
                s.basis = list(kept)
                if not kept:
                    s.reason = "inference with no kept statement before it"
                else:
                    allowed = set().union(*(specifics(b.sentences[j].plain + " " + b.sentences[j].quote)
                                            for j in kept))
                    if new := specifics(s.plain) - allowed:
                        s.reason = "inference adds specifics its facts lack: " + ", ".join(sorted(new))


def drop_dependents(blocks: list[Block]) -> None:
    """After statements are removed later (by a reviewer), an inference whose every basis went goes too."""
    for b in blocks:
        for s in b.sentences:
            if s.kind == "inference" and s.reason is None and not any(b.sentences[j].reason is None
                                                                      for j in s.basis):
                s.reason = "inference whose facts were removed"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def rewrite_links(md: str, pages: dict[str, str] | None) -> str:
    """``[[slug]]`` or ``[[slug|words]]``: a link to the page's markdown when the wiki has it, else just the words, so
    a page never links to nothing. ``pages`` maps slug to topic."""
    pages = pages or {}

    def sub(m):
        slug = m.group(1).strip().split("/")[-1].removesuffix(".md")
        words = m.group(3) or pages.get(slug) or slug.replace("-", " ")
        return f"[{words}]({slug}.md)" if slug in pages else words
    return WIKILINK.sub(sub, md)


def links(blocks: list[Block], pages: dict[str, str]) -> list[str]:
    """Slugs of existing pages the kept text links to."""
    found = []
    for b in blocks:
        text = b.raw if b.kind in ("heading", "nav") else " ".join(s.text for s in b.sentences if s.reason is None)
        found += [m.group(1).strip().split("/")[-1].removesuffix(".md") for m in WIKILINK.finditer(text)]
    return sorted({s for s in found if s in pages})


def _footnote(s: Sentence) -> str:
    where = f", {s.location.section}" if s.location and s.location.section else ""
    return f"“{s.quote}” ({s.doc_id}{where})"


def _drop_empty_sections(lines: list[str]) -> str:
    """Remove a heading whose section lost all its content (no text before the next heading at its level or above)."""
    def level(line):
        m = re.match(r"^(#{1,6})\s", line)
        return len(m.group(1)) if m else 0

    keep = [True] * len(lines)
    for i, line in enumerate(lines):
        if lv := level(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j == len(lines) or 0 < level(lines[j]) <= lv:
                keep[i] = False
    text = "\n".join(line for line, k in zip(lines, keep, strict=True) if k)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def render(blocks: list[Block], pages: dict[str, str] | None = None) -> str:
    """The checked page: its structure kept, kept sentences only, statements cited as [doc id] with a footnote
    holding the quote, inferences and limits marked. Tables are dropped: their cells carry no tags."""
    notes: dict[tuple[str, str], int] = {}
    defs: list[str] = []
    lines: list[str] = []
    for b in blocks:
        if b.kind == "prose":
            out = []
            for s in b.sentences:
                if s.reason is not None:
                    continue
                if s.kind == "statement":
                    key = (s.doc_id, s.quote)
                    if key not in notes:
                        notes[key] = len(notes) + 1
                        defs.append(f"[^{notes[key]}]: {_footnote(s)}")
                    out.append(f"{s.text} [{s.doc_id}][^{notes[key]}]")
                elif s.kind == "inference":
                    out.append(f"{s.text} _(inference)_")
                elif s.kind == "limit":
                    out.append(f"{s.text} _(limit)_")
                else:  # label
                    out.append(s.text)
            if out:
                lines.append(b.prefix + rewrite_links(" ".join(out), pages))
        elif b.kind in ("heading", "nav", "blank"):
            lines.append(rewrite_links(b.raw, pages))
    body = _drop_empty_sections(lines)
    return body + ("\n\n" + "\n".join(defs) if defs else "")


def quotes(blocks: list[Block]) -> str:
    """Each kept statement with its quote, for the verifier."""
    return "\n".join(f'- "{s.plain}" rests on {s.doc_id}: "{s.quote}"' for s in sentences(blocks)
                     if s.kind == "statement" and s.reason is None)


def dropped(blocks: list[Block]) -> list[str]:
    return [f"{s.text[:80]!r}: {s.reason}" for s in sentences(blocks) if s.reason is not None]
