"""add_decoys.py writes the planted rows in the list's own columns, delimiter and fold-change scale."""

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("add_decoys", ROOT / "bench/research/mcf7_release/add_decoys.py")
add_decoys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_decoys)


def rows(path, delim):
    return list(csv.DictReader(path.open(encoding="utf-8"), delimiter=delim))


def test_decoys_follow_the_lists_headers_and_log2_scale(tmp_path):
    lst, out = tmp_path / "list.tsv", tmp_path / "out.tsv"
    lst.write_text("Protein IDs\tGene names\tLog2 Fold Change (24h/0h)\tNotes\nP06493\tCDK1\t2.1\tx\n"
                   "P12004\tPCNA\t-0.3\t\n", encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 0
    got = rows(out, "\t")
    assert len(got) == 6 and got[0]["Gene names"] == "CDK1"
    vim = next(r for r in got if r["Protein IDs"] == "P08670")
    assert vim["Gene names"] == "VIM" and vim["Log2 Fold Change (24h/0h)"] == "1.60" and vim["Notes"] == ""


def test_decoys_use_a_plain_ratio_when_the_list_has_one(tmp_path):
    lst, out = tmp_path / "list.csv", tmp_path / "out.csv"
    lst.write_text("Accession,Symbol,FC\nP06493,CDK1,2.4\nP12004,PCNA,1.3\n", encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 0
    fake = next(r for r in rows(out, ",") if r["Accession"] == "Q9ZZZ9")
    assert fake["FC"] == f"{2 ** 2.2:.2f}"


def test_a_list_without_id_columns_writes_nothing(tmp_path):
    lst, out = tmp_path / "list.csv", tmp_path / "out.csv"
    lst.write_text("a,b\n1,2\n", encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 1 and not out.exists()


def test_an_oddly_named_numeric_column_is_taken_as_the_fold_change(tmp_path, capsys):
    lst, out = tmp_path / "list.csv", tmp_path / "out.csv"
    lst.write_text("Protein ID,Gene name,Up in release (24h vs 0h)\nP06493,CDK1,2.1\nP12004,PCNA,0.4\n",
                   encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 0
    fake = next(r for r in rows(out, ",") if r["Protein ID"] == "Q9ZZZ9")
    assert fake["Up in release (24h vs 0h)"] == "2.20" and "WARNING" not in capsys.readouterr().out


def test_fold_col_names_a_column_outright_and_a_missing_one_warns(tmp_path, capsys):
    lst, out = tmp_path / "list.csv", tmp_path / "out.csv"
    lst.write_text("Protein ID,Gene name,Score,Notes\nP06493,CDK1,5,a\nP12004,PCNA,3,b\n", encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 0  # Score is the one numeric column
    assert add_decoys.main([str(lst), str(out), "--fold-col", "Nope"]) == 1
    lst.write_text("Protein ID,Gene name,Notes\nP06493,CDK1,a\n", encoding="utf-8")
    assert add_decoys.main([str(lst), str(out)]) == 0
    assert "WARNING: no fold-change column" in capsys.readouterr().out
