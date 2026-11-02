import pytest

from deps.graph import CycleError, build_order


def test_simple_chain():
    assert build_order({"app": ["lib"], "lib": ["cfg"]}) == ["cfg", "lib", "app"]


def test_cycle_raises():
    with pytest.raises(CycleError):
        build_order({"a": ["b"], "b": ["a"], "c": []})


def test_ties_are_alphabetical():
    assert build_order({"b": ["a"], "c": ["a"], "a": [], "d": []}) == ["a", "b", "c", "d"]
