from tabular.rows import parse_line
from tabular.table import column


def test_plain():
    assert parse_line("a,b,c") == ["a", "b", "c"]


def test_quoted_comma():
    assert parse_line('1,"Smith, Jane",3') == ["1", "Smith, Jane", "3"]


def test_escaped_quote():
    assert parse_line('"say ""hi""",x') == ['say "hi"', "x"]


def test_column_after_a_quoted_comma():
    text = 'id,name,city\n1,"Smith, Jane",Blacksburg\n2,Lee,Roanoke\n'
    assert column(text, "city") == ["Blacksburg", "Roanoke"]
