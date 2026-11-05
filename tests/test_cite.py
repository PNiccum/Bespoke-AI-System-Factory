"""Quote-level citations: code keeps a claim only when its quote is really in its document."""

from factory.pipeline.cite import (
    check,
    drop_dependents,
    dropped,
    links,
    locate,
    parse,
    quote_in,
    quotes,
    render,
    specifics,
)

TEXTS = {"a:1": "PCR primers \u2013 18 to 24 bases \u2013 work best. Their Tm should be 55\u201365 \u00b0C.",
         "b:2": "GC content of 40-60% gives stable binding."}


def test_quotes_match_across_case_dashes_spacing_and_ellipses():
    assert quote_in("primers - 18 to 24 bases", TEXTS["a:1"])
    assert quote_in("PCR primers ... Tm should be 55-65", TEXTS["a:1"])
    assert not quote_in("Tm should be 55-65 ... PCR primers", TEXTS["a:1"])  # out of order
    assert not quote_in("Tm should be 70", TEXTS["a:1"])
    assert not quote_in("...", TEXTS["a:1"])


def test_only_supported_sentences_survive():
    body = """# Primer design

Primers are 18-24 bases. {{q:a:1|18 to 24 bases}} GC should be 40-60%. {{q:b:2|GC content of 40-60%}}
So short GC-rich primers work. {{inference}} Nothing on cost. {{limit}}

Tm must be 70 C. {{q:a:1|Tm must be 70}} An unsupported leap. {{inference}} No tag here.

From an unapproved paper. {{q:c:3|anything}}"""
    paragraphs = parse(body)
    check(paragraphs, TEXTS)
    out = render(paragraphs)
    assert out == ("# Primer design\n\nPrimers are 18-24 bases. [a:1][^1] GC should be 40-60%. [b:2][^2] So short "
                   "GC-rich primers work. _(inference)_ Nothing on cost. _(limit)_\n\n"
                   "[^1]: \u201c18 to 24 bases\u201d (a:1)\n[^2]: \u201cGC content of 40-60%\u201d (b:2)")
    reasons = " ".join(dropped(paragraphs))
    assert "quote not found in a:1" in reasons and "inference with no kept statement" in reasons
    assert "no tag" in reasons and "c:3 is not an approved document" in reasons
    assert '"Primers are 18-24 bases." rests on a:1: "18 to 24 bases"' in quotes(paragraphs)


def test_structure_labels_and_empty_sections():
    body = """# Design

## Rules
Inputs:
- Length is 18-24 bases. {{q:a:1|18 to 24 bases}}
- A claim with no source. {{q:a:1|not in the text}}

## Dropped section
Nothing here survives. {{q:b:2|absent}}

## See also
Related: [[pcr-basics]], [[missing-page|Missing page]]
"""
    blocks = parse(body)
    check(blocks, TEXTS)
    out = render(blocks, {"pcr-basics": "PCR basics"})
    assert "## Rules\nInputs:\n- Length is 18-24 bases. [a:1][^1]" in out
    assert "Dropped section" not in out and "claim with no source" not in out
    assert "## See also\nRelated: [PCR basics](pcr-basics.md), Missing page" in out
    assert links(blocks, {"pcr-basics": "PCR basics"}) == ["pcr-basics"]


def test_inference_may_not_add_specifics():
    body = ("Tm should be 55-65 C. {{q:a:1|Tm should be 55\u201365}} So aim for 60 C. {{inference}} "
            "So aim for 72 C with BRCA1. {{inference}}")
    blocks = parse(body)
    check(blocks, TEXTS)
    kinds = [(s.kind, s.reason) for s in blocks[0].sentences]
    assert kinds[1][1] is not None and "60" in kinds[1][1]  # 60 is not in the fact either
    assert "72" in kinds[2][1] and "brca1" in kinds[2][1]
    assert specifics("BRCA1 at 55-65 C, NM_007294") == {"brca1", "55", "65", "nm_007294"}


def test_locate_takes_tightest_span_and_section():
    text = "Abstract text. Primers bind.\n\nFrom the full text:\nPrimers are short. Later, primers bind well."
    loc = locate("primers ... bind well", text)
    assert text[loc.start:loc.end] == "primers bind well" and loc.section == "full text"
    assert locate("Primers bind", text).section == "abstract"


def test_inference_goes_when_its_facts_are_removed():
    blocks = parse("Length is 18-24 bases. {{q:a:1|18 to 24 bases}} So primers are short. {{inference}}")
    check(blocks, TEXTS)
    blocks[0].sentences[0].reason = "rejected by reviewer"
    drop_dependents(blocks)
    assert blocks[0].sentences[1].reason == "inference whose facts were removed"


def test_wiki_index_log_and_lint(tmp_path):
    from factory.pipeline.wiki import Wiki, WikiPage
    w = Wiki(tmp_path)
    w.save(WikiPage(topic="PCR basics", summary="What PCR is.", body="b", sources=["a", "b"], verified=True,
                    links=["primer-tm"]))
    w.save(WikiPage(topic="Primer Tm", summary="Melting.", body="b", sources=["a"], verified=False,
                    issues=["too few sources"]))
    w.save(WikiPage(topic="Lonely", summary="x", body="b", sources=["docs:x"], verified=True, links=["gone"]))
    index = (tmp_path / "index.md").read_text()
    assert "| [PCR basics](pcr-basics.md) | verified | 2 | public | What PCR is. |" in index
    assert len((tmp_path / "log.md").read_text().splitlines()) == 3
    problems = w.lint()
    assert "lonely: links to missing page gone" in problems
    assert any(p.startswith("primer-tm: not verified") for p in problems)
    assert not any(p.startswith("pcr-basics: orphan") for p in problems)


def test_a_sentence_that_restates_its_quote_needs_no_model_check():
    from factory.pipeline.cite import needs_model_check, restates

    texts = {"a:1": "Primers should have a Tm of 55-65 C. GC content of 40-60% gives stable binding."}
    same = parse("Primers should have a Tm of 55-65 C. {{q:a:1|Primers should have a Tm of 55-65 C}}")
    check(same, texts)
    assert restates(same[0].sentences[0]) and not needs_model_check(same)
    # a number, a negation or a real paraphrase is for the verifier
    for body in ("Primers should have a Tm of 55-70 C. {{q:a:1|Primers should have a Tm of 55-65 C}}",
                 "Primers should not have a Tm of 55-65 C. {{q:a:1|Primers should have a Tm of 55-65 C}}",
                 "Melting temperature matters for primers. {{q:a:1|Primers should have a Tm of 55-65 C}}",
                 "Primers should have a Tm of 55-65 C. {{q:a:1|Primers should have a Tm of 55-65 C}} "
                 "So GC matters. {{inference}}"):
        blocks = parse(body)
        check(blocks, texts)
        assert needs_model_check(blocks), body
    assert quotes(same, only_doubtful=True) == ""
