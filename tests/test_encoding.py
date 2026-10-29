"""Every text file ASPIRE reads or writes says UTF-8. Without it, Windows uses cp1252 and the first model reply with a
character outside it (a non-breaking hyphen, say) crashes the run; Linux CI defaults to UTF-8 and would never notice."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [*(ROOT / "factory").rglob("*.py"), *(ROOT / "pages").glob("*.py"), ROOT / "app.py"]


def text_io_without_encoding(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("read_text", "write_text", "open")):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if node.func.attr == "open":
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ("zipfile", "webbrowser", "os"):
                continue
            mode = node.args[0] if node.args else None
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
                continue
            if mode is not None and not isinstance(mode, ast.Constant):
                continue
        out.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return out


def test_every_text_file_is_read_and_written_as_utf8():
    missing = [hit for p in SOURCES if p.exists() for hit in text_io_without_encoding(p)]
    assert not missing, f"add encoding='utf-8': {missing}"


def test_the_blackboard_keeps_characters_outside_cp1252(tmp_path):
    from factory.pipeline.blackboard import Project
    project = Project.create(tmp_path, "unicode")
    project.board.post("note", "analyst", {"text": "non‑breaking hyphen, 5′-ATG-3′, ΔTm"})
    assert project.board.entries("note")[0].data["text"] == "non‑breaking hyphen, 5′-ATG-3′, ΔTm"
