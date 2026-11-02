from conf.ini import parse
from conf.settings import get_list


def test_colon_separator_splits_at_first_separator():
    text = "[t]\nstart: 12:30\nurl = http://x/?a=b\nKey=V\n"
    assert parse(text) == {"t": {"start": "12:30", "url": "http://x/?a=b", "key": "V"}}


def test_indented_comment_is_still_a_comment():
    text = "[s]\nk = v\n    ; note\n"
    assert parse(text) == {"s": {"k": "v"}}


def test_get_list():
    text = "[mail]\nto =\n  a@x.org\n  b@x.org\nfrom = me\n"
    assert get_list(text, "mail", "to") == ["a@x.org", "b@x.org"]
    assert parse(text)["mail"]["from"] == "me"


def test_empty_section_kept():
    assert parse("[empty]\n") == {"empty": {}}
