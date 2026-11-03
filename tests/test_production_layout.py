"""Production starts from an empty output folder, and its tests import a src layout without an install."""

from pathlib import Path

from factory.tools.sandbox import RunTestsIn, run_tests


def _src_repo(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("VALUE = 3\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_pkg.py").write_text("from pkg import VALUE\n\ndef test_value():\n    assert VALUE == 3\n")


def test_src_layout_imports_with_pythonpath(tmp_path):
    _src_repo(tmp_path)
    without = run_tests(RunTestsIn(repo_path=str(tmp_path)))
    assert not without.passed and "ModuleNotFoundError" in without.output_tail
    with_path = run_tests(RunTestsIn(repo_path=str(tmp_path), pythonpath=["src", "."]))
    assert with_path.ok and with_path.passed


def test_pythonpath_outside_the_repo_is_ignored(tmp_path):
    _src_repo(tmp_path)
    report = run_tests(RunTestsIn(repo_path=str(tmp_path), pythonpath=["../..", "/"]))
    assert not report.passed
