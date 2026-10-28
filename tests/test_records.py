"""Provenance: sources are identified by content, and every event lands in an append-only log."""

from factory.records import ProvenanceEvent, ProvenanceLog, Source


def test_source_id_comes_from_content_not_name(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "renamed.csv"
    a.write_text("x,y\n1,2\n")
    b.write_text("x,y\n1,2\n")
    assert Source.from_file(a).source_id == Source.from_file(b).source_id
    assert Source.from_file(a).public is False  # unknown counts as not public


def test_provenance_log_appends_and_reads_back(tmp_path):
    log = ProvenanceLog(tmp_path)
    f = tmp_path / "in.txt"
    f.write_text("hello")
    src = log.add_source(Source.from_file(f, public=True))
    log.record(ProvenanceEvent(kind="tool_call", actor="test_runner", inputs=[src.source_id], outputs=["report.json"]))
    events = log.events()
    assert [e.kind for e in events] == ["source", "tool_call"]
    assert events[1].inputs == [src.source_id]
