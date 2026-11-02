import pytest

from deps.graph import CycleError, build_order


def check_cycle(deps, cycle):
    assert len(cycle) >= 2 and cycle[0] == cycle[-1]
    for a, b in zip(cycle, cycle[1:]):
        assert b in deps.get(a, []), (a, b)


def test_cycle_names_the_loop():
    deps = {"a": ["b"], "b": ["c"], "c": ["a"], "x": [], "y": ["x"]}
    with pytest.raises(CycleError) as e:
        build_order(deps)
    check_cycle(deps, e.value.cycle)
    assert set(e.value.cycle) == {"a", "b", "c"}


def test_task_downstream_of_a_cycle_is_not_in_it():
    deps = {"a": ["b"], "b": ["a"], "z": ["a"]}
    with pytest.raises(CycleError) as e:
        build_order(deps)
    check_cycle(deps, e.value.cycle)
    assert "z" not in e.value.cycle


def test_self_loop():
    with pytest.raises(CycleError) as e:
        build_order({"a": ["a"]})
    assert e.value.cycle == ["a", "a"]


def test_cycle_error_is_a_value_error():
    with pytest.raises(ValueError):
        build_order({"p": ["q"], "q": ["p"]})


def test_diamond_ties():
    deps = {"top": ["left", "right"], "left": ["base"], "right": ["base"], "aux": ["base"]}
    assert build_order(deps) == ["base", "aux", "left", "right", "top"]
