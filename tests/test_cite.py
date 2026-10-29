"""Quote-level citations: code keeps a claim only when its quote is really in its document."""

from factory.pipeline.cite import check, dropped, parse, quote_in, quotes, render

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
    assert out == ("Primers are 18-24 bases. [a:1] GC should be 40-60%. [b:2] So short GC-rich primers work. "
                   "_(inference)_ Nothing on cost. _(limit)_")
    reasons = " ".join(dropped(paragraphs))
    assert "quote not found in a:1" in reasons and "inference with no kept statement" in reasons
    assert "no tag" in reasons and "c:3 is not an approved document" in reasons
    assert '"Primers are 18-24 bases." rests on a:1: "18 to 24 bases"' in quotes(paragraphs)
