# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
Test utilities for :term:`json pointer path` modifier.

Copied from pypi/pandalone.
"""
import pytest

from graphtik.jsonpointer import (
    ResolveError,
    escape_jsonpointer_part,
    jsonp_path,
    pop_path,
    resolve_path,
    set_path_value,
    unescape_jsonpointer_part,
)


def test_jsonpointer_escape_parts():
    def un_esc(part):
        return unescape_jsonpointer_part(escape_jsonpointer_part(part))

    part = "hi/there"
    assert un_esc(part) == part
    part = "hi~there"
    assert un_esc(part) == part
    part = "/hi~there/"
    assert un_esc(part) == part


def test_jsonp_path_empty():
    assert jsonp_path("") == [""]


def test_jsonp_path_root():
    assert jsonp_path("/") == [""]


def test_jsonp_path_regular():
    assert jsonp_path("/a") == ["", "a"]

    assert jsonp_path("/a/b") == ["", "a", "b"]


def test_jsonp_path_folder():
    assert jsonp_path("/a/") == [""]


def test_jsonp_path_None():
    with pytest.raises(TypeError):
        jsonp_path(None)


def test_jsonp_path_with_spaces():
    assert jsonp_path("/ some ") == ["", " some "]
    assert jsonp_path("/ some /  ") == ["", " some ", "  "]

    assert jsonp_path(" some ") == [" some "]
    assert jsonp_path(" some /  ") == [" some ", "  "]


def test_jsonp_path_cached():
    class C(str):
        pass

    p = C("a/b")
    assert jsonp_path(p) == ["a", "b"]
    p.jsonp = False
    assert jsonp_path(p) == [p]
    assert p.jsonp == False
    p.jsonp = None
    assert jsonp_path(p) == ["a", "b"]
    assert p.jsonp == None


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("/a", ["a"]),
        ("/a/", ["a", ""]),
        ("/a/b", ["a", "b"]),
        ("/a/b/", ["a", "b", ""]),
        ("/a//b", ["a", "", "b"]),
        ("/a/../b", ["a", "..", "b"]),
        ("/", [""]),
        ("", []),
        ("/ some ", [" some "]),
        ("/ some /", [" some ", ""]),
        ("/ some /  ", [" some ", "  "]),
        ("/ some /  /", [" some ", "  ", ""]),
        (None, AttributeError()),
        ("a", ValueError()),
    ],
)
def test_jsonp_path_massive(inp, exp):
    if isinstance(exp, Exception):
        with pytest.raises(type(exp), match=str(exp)):
            jsonp_path(inp)
    else:
        assert jsonp_path(inp) == exp


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("/a", ["", "a"]),
        ("/a/", [""]),
        ("/a/b", ["", "a", "b"]),
        ("/a/b/", [""]),
        ("/a//b", ["", "b"]),
        ("/", [""]),
        ("", [""]),
        ("/ some ", ["", " some "]),
        ("/ some /", [""]),
        ("/ some /  ", ["", " some ", "  "]),
        (None, TypeError()),
        ("a", ["a"]),
        ("a/", [""]),
        ("a/b", ["a", "b"]),
        ("a/b/", [""]),
        ("a/../b/.", ["a", "..", "b", "."]),
        ("a/../b/.", ["a", "..", "b", "."]),
        (" some ", [" some "]),
        (" some /  ", [" some ", "  "]),
    ],
)
def test_jsonp_path_massive(inp, exp):
    if isinstance(exp, Exception):
        with pytest.raises(type(exp), match=str(exp)):
            jsonp_path(inp)
    else:
        assert jsonp_path(inp) == exp


@pytest.mark.parametrize(
    "inp, exp", [("/foo", 1), ("/bar/0", 11), ("/bar/1/a", 222), ("/bar/1/a", 222),],
)
def test_resolve_simple(inp, exp):
    doc = {"foo": 1, "bar": [11, {"a": 222}]}
    assert resolve_path(doc, inp) == exp


def test_resolve_path_sequence():
    doc = [1, [22, 33]]

    path = "/0"
    assert resolve_path(doc, path) == 1

    path = "/1"
    assert resolve_path(doc, path) == [22, 33]

    path = "/1/0"
    assert resolve_path(doc, path) == 22
    path = "/1/1"
    assert resolve_path(doc, path) == 33


def test_resolve_path_missing_screams():
    doc = {}

    path = "/foo"
    with pytest.raises(ResolveError):
        resolve_path(doc, path)


def test_resolve_path_empty_path():
    doc = {}
    path = ""
    assert resolve_path(doc, path) == doc

    doc = {"foo": 1}
    assert resolve_path(doc, path) == doc


@pytest.fixture
def std_doc():
    """From https://tools.ietf.org/html/rfc6901#section-5 """
    return {
        r"foo": ["bar", r"baz"],
        r"": 0,
        r"a/b": 1,
        r"c%d": 2,
        r"e^f": 3,
        r"g|h": 4,
        r"i\\j": 5,
        r"k\"l": 6,
        r" ": 7,
        r"m~n": 8,
    }


@pytest.fixture(
    params=[
        (r"", ...),
        (r"/foo", ["bar", "baz"]),
        (r"/foo/0", "bar"),
        # (r"/", 0), #resolve_path() resolves '/' to root (instead of to '' key)"
        (r"/", ...),
        (r"/a~1b", 1),
        (r"/c%d", 2),
        (r"/e^f", 3),
        (r"/g|h", 4),
        (r"/i\\j", 5),
        (r"/k\"l", 6),
        (r"/ ", 7),
        (r"/m~0n", 8),
    ]
)
def std_case(std_doc, request):
    """From https://tools.ietf.org/html/rfc6901#section-5 """
    path, exp = request.param
    if exp is ...:
        exp = std_doc
    return path, exp


def test_resolve_path_examples_from_spec(std_doc, std_case):
    path, exp = std_case
    assert resolve_path(std_doc, path) == exp


def test_resolve_root_path_only():
    doc = {}
    path = "/"
    assert resolve_path(doc, path) == doc

    doc = {"foo": 1}
    assert resolve_path(doc, path) == doc

    doc = {"": 1}
    assert resolve_path(doc, path) == doc


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("/", ...),
        ("//", ...),
        ("///", ...),
        ("/bar//", ...),
        ("/bar/1/", ...),
        ("/foo//", ...),
        ("/bar/1//foo", 1),
        ("/bar/1//foo/", ...),
        ("/foo//bar/1/a", 222),
    ],
)
def test_resolve_path_re_root(inp, exp):
    doc = {"foo": 1, "bar": [11, {"a": 222}]}
    assert resolve_path(doc, inp) == doc if exp is ... else exp


def test_set_path_empty_doc():
    doc = {}
    path = "/foo"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value

    doc = {}
    path = "/foo/bar"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value


def test_set_path_replace_value():
    doc = {"foo": "bar", 1: 2}
    path = "/foo"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2

    doc = {"foo": 1, 1: 2}
    path = "/foo"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2

    doc = {"foo": {"bar": 1}, 1: 2}
    path = "/foo"
    value = 2
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2


def test_set_path_deepen_map_str_value():
    doc = {"foo": "bar", 1: 2}
    path = "/foo/bar"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2

    doc = {"foo": "bar", 1: 2}
    path = "/foo/bar/some/other"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2


def test_set_path_append_path_preserves_intermediate():
    doc = {"foo": {"bar": 1}, 1: 2}
    path = "/foo/foo2"
    value = "value"
    set_path_value(doc, path, value)
    print(doc)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2
    assert resolve_path(doc, "/foo/bar") == 1


def test_set_path_deepen_map_int_value():
    doc = {"foo": 1, 1: 2}
    path = "/foo/bar"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2

    doc = {"foo": 1, 1: 2}
    path = "/foo/bar/some/other"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert doc[1] == 2


def test_set_path_deepen_sequence_scalar_item():
    doc = [1, 2]
    path = "/1"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value

    doc = [1, 2]
    path = "/1/foo/bar"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value


@pytest.mark.xfail(reason="Use dash(-) instead to append lists")
def test_set_path_sequence_insert_end():
    doc = [0, 1]
    path = "/2"
    value = "value"
    set_path_value(doc, path, value)
    assert resolve_path(doc, path) == value
    assert resolve_path(doc, "/0") == 0
    assert resolve_path(doc, "/1") == 1


def test_set_path_sequence_tail_dash():
    doc = [0, 1]
    path = "/-"
    value = "value"
    set_path_value(doc, path, value)
    assert doc == [0, 1, "value"]


def test_set_path_sequence_out_of_bounds():
    doc = [0, 1]
    path = "/3"
    value = "value"
    with pytest.raises(ValueError):
        set_path_value(doc, path, value)


def test_set_path_sequence_with_str_screams():
    doc = [0, 1]
    path = "/str"
    value = "value"
    with pytest.raises(ValueError):
        set_path_value(doc, path, value)


def test_pop_path_examples_from_spec(std_doc, std_case):
    path, exp = std_case
    orig = dict(std_doc)

    assert pop_path(std_doc, path) == exp

    if exp == std_doc:
        # '/' pops nothing :-(
        assert std_doc == exp
    else:
        exp_doc = {k: v for k, v in orig.items() if v != exp}
        assert std_doc == exp_doc


@pytest.mark.parametrize(
    "inp, pop_item, culled_doc",
    [
        # Empties
        (({}, "/"), {}, {}),
        (({}, ""), {}, {}),
        (({}, "/a", "A"), "A", {}),
        (({}, ""), {}, {}),
        # ResolveErrors
        (({}, "/a"), ResolveError, 0),
        (({"a": 1}, "/b"), ResolveError, 0),
        (({"a": 1}, "b"), ResolveError, 0),
        (({"a": 1}, "/a/b"), ResolveError, 0),
        (({"a": 1}, "a/b"), ResolveError, 0),
        ## Ok but strange!
        (({"a": 1}, "a/"), {"a": 1}, {"a": 1}),  # `/`` pops nothing :-(
        (({"a": 1}, "a/1"), ResolveError, 0),
        ## Ok
        (({"a": 1}, "a"), 1, {}),
        (({"a": 1}, "/a"), 1, {}),
        (({"a": {"b": 1}}, "/a"), {"b": 1}, {}),
        (({"a": {"b": 1}}, "/a/b"), 1, {"a": {}}),
        (({"a": {"b": 1, "c": 2}}, "/a/b"), 1, {"a": {"c": 2}}),
        (({1: 2}, "/1"), 2, {}),
        ## Lists
        (([1, 2], "/1"), 2, [1]),
        (([{1: 22}, 2], "/0"), {1: 22}, [2]),
        (([{1: 22}, 2], "/0/1"), 22, [{}, 2]),
    ],
)
def test_pop_path_cases(inp, pop_item, culled_doc):
    if isinstance(pop_item, type) and issubclass(pop_item, Exception):
        with pytest.raises(pop_item):
            pop_path(*inp)
    else:
        doc, *_ = inp
        assert pop_path(*inp) == pop_item
        assert doc == culled_doc
