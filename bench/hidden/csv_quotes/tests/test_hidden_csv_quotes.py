from tabular.rows import parse_line
from tabular.table import column


def test_empty_fields():
    assert parse_line("a,,b") == ["a", "", "b"]
    assert parse_line(",x") == ["", "x"]


def test_other_quoted_fields():
    assert parse_line('"Doe, John",42,"a ""b"" c"') == ["Doe, John", "42", 'a "b" c']


def test_column_with_two_quoted_names():
    text = 'name,city\n"Lee, A",Roanoke\n"Kim, B",Radford\n'
    assert column(text, "city") == ["Roanoke", "Radford"]


def test_an_empty_line_is_not_an_error():
    # The original returned [""] here; a fix may return [] or [""], but must not raise.
    assert parse_line("") in ([], [""])
    assert parse_line("\n") in ([], [""])
