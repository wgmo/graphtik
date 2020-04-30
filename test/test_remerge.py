# coding: utf-8

# Standard Library
from pprint import pprint

import pytest

# Gitlab Project Configurator Modules
# from gpc.helpers.remerge import remerge
from graphtik.plot import remerge


def test_override_string():
    defaults = {"key_to_override": "value_from_defaults"}

    first_override = {"key_to_override": "value_from_first_override"}

    source_map = {}
    merged = remerge(
        ("defaults", defaults),
        ("first_override", first_override),
        source_map=source_map,
    )

    expected_merged = {"key_to_override": "value_from_first_override"}
    assert merged == expected_merged
    assert source_map == {("key_to_override",): "first_override"}

    merged = remerge(defaults, first_override, source_map=None)
    assert merged == expected_merged


def test_override_subdict():
    defaults = {
        "subdict": {
            "other_subdict": {
                "key_to_override": "value_from_defaults",
                "integer_to_override": 2222,
            }
        }
    }

    first_override = {
        "subdict": {
            "other_subdict": {
                "key_to_override": "value_from_first_override",
                "integer_to_override": 5555,
            }
        }
    }

    expected_merge = {
        "subdict": {
            "other_subdict": {
                "integer_to_override": 5555,
                "key_to_override": "value_from_first_override",
            }
        }
    }

    source_map = {}
    merged = remerge(
        ("defaults", defaults),
        ("first_override", first_override),
        source_map=source_map,
    )
    assert merged == expected_merge
    assert source_map == {
        ("subdict",): "first_override",
        ("subdict", "other_subdict"): "first_override",
        ("subdict", "other_subdict", "integer_to_override"): "first_override",
        ("subdict", "other_subdict", "key_to_override"): "first_override",
    }

    merged = remerge(defaults, first_override, source_map=None)
    assert merged == expected_merge


def test_override_list_append():
    defaults = {"list_to_append": [{"a": 1}]}
    first_override = {"list_to_append": [{"b": 1}]}

    source_map = {}
    merged = remerge(
        ("defaults", defaults),
        ("first_override", first_override),
        source_map=source_map,
    )
    expected_merged = {"list_to_append": [{"a": 1}, {"b": 1}]}

    assert merged == expected_merged
    assert source_map == {("list_to_append",): ["defaults", "first_override"]}

    merged = remerge(defaults, first_override, source_map=None)
    assert merged == expected_merged


def test_complex_dict():
    defaults = {
        "key_to_override": "value_from_defaults",
        "integer_to_override": 1111,
        "list_to_append": [{"a": 1}],
        "subdict": {
            "other_subdict": {
                "key_to_override": "value_from_defaults",
                "integer_to_override": 2222,
            },
            "second_subdict": {
                "key_to_override": "value_from_defaults",
                "integer_to_override": 3333,
            },
        },
    }

    first_override = {
        "key_to_override": "value_from_first_override",
        "integer_to_override": 4444,
        "list_to_append": [{"b": 2}],
        "subdict": {
            "other_subdict": {
                "key_to_override": "value_from_first_override",
                "integer_to_override": 5555,
            }
        },
        "added_in_first_override": "some_string",
    }

    second_override = {
        "subdict": {"second_subdict": {"key_to_override": "value_from_second_override"}}
    }

    source_map = {}
    merged = remerge(
        ("defaults", defaults),
        ("first_override", first_override),
        ("second_override", second_override),
        source_map=source_map,
    )
    print("")
    print("'merged' dictionary:")
    pprint(merged)
    print("")
    pprint(source_map)
    print(len(source_map), "paths")

    assert merged["key_to_override"] == "value_from_first_override"
    assert merged["integer_to_override"] == 4444
    assert (
        merged["subdict"]["other_subdict"]["key_to_override"]
        == "value_from_first_override"
    )
    assert merged["subdict"]["other_subdict"]["integer_to_override"] == 5555
    assert (
        merged["subdict"]["second_subdict"]["key_to_override"]
        == "value_from_second_override"
    )
    assert merged["subdict"]["second_subdict"]["integer_to_override"] == 3333
    assert merged["added_in_first_override"] == "some_string"
    assert merged["list_to_append"] == [{"a": 1}, {"b": 2}]


_xfail = pytest.mark.xfail(reason="Brittle remerge()...")


@pytest.mark.parametrize(
    "inp, reverse, exp",
    [
        pytest.param(({1: None}, {}), 0, {}, marks=_xfail),
        (({1: None}, {1: {}}), 0, {1: {}}),
        (({1: None}, {1: []}), 0, {1: []}),
        (({1: None}, {1: 33}), 0, {1: 33}),
        pytest.param(({1: None}, {}), 1, {}, marks=_xfail),
        pytest.param(({1: None}, {1: {}}), 1, {1: {}}, marks=_xfail),
        pytest.param(({1: None}, {1: []}), 1, {1: []}, marks=_xfail),
        pytest.param(({1: None}, {1: 33}), 1, {1: 33}, marks=_xfail),
    ],
)
def test_none_values(inp, reverse, exp):
    if reverse:
        inp = reversed(inp)
    assert remerge(*inp) == exp

    ## Test one level deeper
    #
    inp = tuple({"a": i} for i in inp)
    exp = {"a": exp}
    assert remerge(*inp) == exp


@pytest.mark.parametrize(
    "inp",
    [
        ## <don't collapse>
        ({1: []}, {1: {}}),
        ({1: {}}, {1: []}),
        pytest.param(({1: []}, {1: "a"}), marks=_xfail),
        pytest.param(({1: {}}, {1: "a"}), marks=_xfail),
    ],
)
def test_incompatible_containers(inp):
    with pytest.raises(TypeError, match="Incompatible types"):
        remerge(*inp)
    with pytest.raises(TypeError, match="Incompatible types"):
        remerge(*reversed(inp))

    ## Test one level deeper
    #
    inp = tuple({"A": i} for i in inp)
    with pytest.raises(TypeError, match="Incompatible types"):
        remerge(*inp)
    with pytest.raises(TypeError, match="Incompatible types"):
        remerge(*reversed(inp))
