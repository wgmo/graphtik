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
    iter_path,
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


def test_iter_path_empty():
    assert list(iter_path("")) == [""]


def test_iter_path_root():
    assert list(iter_path("/")) == ["", ""]


def test_iter_path_regular():
    assert list(iter_path("/a")) == ["", "a"]

    assert list(iter_path("/a/b")) == ["", "a", "b"]


def test_iter_path_folder():
    assert list(iter_path("/a/")) == ["", "a", ""]


def test_iter_path_None():
    with pytest.raises(AttributeError):
        list(iter_path(None))


def test_iter_path_with_spaces():
    assert list(iter_path("/ some ")) == ["", " some "]
    assert list(iter_path("/ some /  ")) == ["", " some ", "  "]

    assert list(iter_path(" some ")) == [" some "]
    assert list(iter_path(" some /  ")) == [" some ", "  "]


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
def test_iter_path_massive(inp, exp):
    if isinstance(exp, Exception):
        with pytest.raises(type(exp), match=str(exp)):
            list(iter_path(inp))
    else:
        assert list(iter_path(inp)) == exp


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("/a", ["", "a"]),
        ("/a/", ["", "a", ""]),
        ("/a/b", ["", "a", "b"]),
        ("/a/b/", ["", "a", "b", ""]),
        ("/a//b", ["", "a", "", "b"]),
        ("/", ["", ""]),
        ("", [""]),
        ("/ some ", ["", " some "]),
        ("/ some /", ["", " some ", ""]),
        ("/ some /  ", ["", " some ", "  "]),
        (None, AttributeError()),
        ("a", ["a"]),
        ("a/", ["a", ""]),
        ("a/b", ["a", "b"]),
        ("a/b/", ["a", "b", ""]),
        ("a/../b/.", ["a", "..", "b", "."]),
        ("a/../b/.", ["a", "..", "b", "."]),
        (" some ", [" some "]),
        (" some /", [" some ", ""]),
        (" some /  ", [" some ", "  "]),
        (" some /  /", [" some ", "  ", ""]),
    ],
)
def test_iter_path_massive(inp, exp):
    if isinstance(exp, Exception):
        with pytest.raises(type(exp), match=str(exp)):
            list(iter_path(inp))
    else:
        assert list(iter_path(inp)) == exp


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
