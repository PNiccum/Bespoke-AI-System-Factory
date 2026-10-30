import copy

from conf.defaults import DEFAULTS, load
from conf.merge import merge


def test_flat_override():
    assert load({"debug": True})["debug"] is True


def test_nested_sections_are_merged():
    assert load({"db": {"port": 6543}})["db"] == {"host": "localhost", "port": 6543}


def test_deeper_nesting():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    assert merge(base, {"a": {"b": {"d": 3}}}) == {"a": {"b": {"c": 1, "d": 3}}}


def test_inputs_are_not_changed():
    before = copy.deepcopy(DEFAULTS)
    load({"db": {"port": 1}})
    assert DEFAULTS == before
