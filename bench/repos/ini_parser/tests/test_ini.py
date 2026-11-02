from conf.ini import parse

TEXT = """
; who maintains this
[db]
hosts = a
  b
port = 5432
"""


def test_semicolon_comments_and_continuation_lines():
    assert parse(TEXT) == {"db": {"hosts": "a\nb", "port": "5432"}}


def test_hash_comment_and_default_section():
    assert parse("# hi\nname = x\n") == {"DEFAULT": {"name": "x"}}
