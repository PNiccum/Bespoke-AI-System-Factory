import json
import subprocess
import sys

from conftest import ROOT
from wordstats import top_words

TEXT = "The cat and the hat. The CAT sat; a cat's hat sat on the mat. 'Mat' is a mat!"


def run(*args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "wordstats.py"), *args], cwd=cwd, capture_output=True,
                          text=True, timeout=60)


def test_counts_case_folded_without_stopwords_ties_alphabetical():
    assert top_words(TEXT, 3) == [("mat", 3), ("cat", 2), ("hat", 2)]


def test_apostrophes_kept_inside_words_only():
    assert dict(top_words("don't 'quote' don't", 5)) == {"don't": 2, "quote": 1}


def test_default_stopwords_and_custom_list():
    words = dict(top_words("the of and to in is a cat", 10))
    assert words == {"cat": 1}
    assert dict(top_words("the cat the", 10, stopwords=set())) == {"the": 2, "cat": 1}


def test_cli_plain(tmp_path):
    (tmp_path / "t.txt").write_text(TEXT, encoding="utf-8")
    r = run("t.txt", "--top", "2", cwd=tmp_path)
    assert r.returncode == 0
    assert r.stdout.strip().splitlines() == ["mat\t3", "cat\t2"]


def test_cli_json_and_keep_stopwords(tmp_path):
    (tmp_path / "t.txt").write_text(TEXT, encoding="utf-8")
    r = run("t.txt", "--top", "1", "--json", "--keep-stopwords", cwd=tmp_path)
    assert r.returncode == 0
    assert json.loads(r.stdout) == [{"word": "the", "count": 4}]


def test_cli_missing_file(tmp_path):
    r = run("nope.txt", cwd=tmp_path)
    assert r.returncode != 0 and r.stderr.strip()
