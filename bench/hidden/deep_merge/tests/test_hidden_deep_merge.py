import copy

from conf.merge import merge


def test_any_section_name_merges():
    assert merge({"cache": {"ttl": 5, "size": 10}}, {"cache": {"ttl": 9}}) == {"cache": {"ttl": 9, "size": 10}}


def test_four_levels():
    base = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
    assert merge(base, {"a": {"b": {"c": {"e": 3}}}}) == {"a": {"b": {"c": {"d": 1, "e": 3}}}}


def test_a_value_may_replace_a_section_and_inputs_stay_unchanged():
    base, override = {"db": {"host": "h"}}, {"db": None}
    before = copy.deepcopy((base, override))
    assert merge(base, override) == {"db": None}
    assert (base, override) == before
