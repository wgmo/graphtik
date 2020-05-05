import dill
import pytest

from graphtik.modifiers import (
    dep_renamed,
    dep_singularized,
    dep_stripped,
    mapped,
    optional,
    sideffect,
    sideffected,
    sideffected_vararg,
    sideffected_varargs,
    vararg,
    varargs,
)


def test_dill_modifier():
    s = sideffected("foo", "gg")
    s == dill.loads(dill.dumps(s))


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", None), "b"),
        (lambda: mapped("b", ""), "b"),
        (lambda: mapped("b", "bb"), "b"),
        (lambda: optional("b"), "b"),
        (lambda: optional("b", "bb"), "b"),
        (lambda: vararg("c"), "c"),
        (lambda: varargs("d"), "d"),
        (lambda: sideffect("e"), "sideffect: 'e'"),
        (lambda: sideffect("e", optional=1), "sideffect: 'e'"),
        (lambda: sideffected("f", "a", "b"), "sideffected('f'<--'a', 'b')"),
        (lambda: sideffected("f", "ff", fn_kwarg="F"), "sideffected('f'<--'ff')",),
        (
            lambda: sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "sideffected('f'<--'ff')",
        ),
        (lambda: sideffected("f", "ff", optional=1), "sideffected('f'<--'ff')"),
    ],
)
def test_modifs_str(mod, exp):
    got = str(mod())
    print(got)
    assert got == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", None), "mapped('b')"),
        (lambda: mapped("b", ""), "mapped('b')"),
        (lambda: mapped("b", "bb"), "mapped('b', fn_kwarg='bb')"),
        (lambda: optional("b"), "optional('b')"),
        (lambda: optional("b", "bb"), "optional('b', fn_kwarg='bb')"),
        (lambda: vararg("c"), "vararg('c')"),
        (lambda: varargs("d"), "varargs('d')"),
        (lambda: sideffect("e"), "sideffect: 'e'"),
        (lambda: sideffect("e", optional=1), "sideffect?: 'e'"),
        (lambda: sideffected("f", "a", "b"), "sideffected('f'<--'a', 'b')"),
        (
            lambda: sideffected("f", "ff", fn_kwarg="F"),
            "sideffected('f'<--'ff', fn_kwarg='F')",
        ),
        (
            lambda: sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "sideffected?('f'<--'ff', fn_kwarg='F')",
        ),
        (lambda: sideffected("f", "ff", optional=1), "sideffected?('f'<--'ff')"),
        (lambda: sideffected_vararg("f", "a"), "sideffected*('f'<--'a')"),
        (lambda: sideffected_varargs("f", "a", "b"), "sideffected#('f'<--'a', 'b')"),
    ],
)
def test_modifs_repr(mod, exp):
    got = repr(mod())
    print(got)
    assert got == exp


def test_recreation():
    assert optional(optional("a")) == optional("a")
    assert mapped(mapped("a", 1), 1) == mapped("a", 1)
    assert vararg(vararg("a")) == vararg("a")
    assert varargs(varargs("a")) == varargs("a")


def test_recreation_repr():
    assert repr(optional(optional("a"))) == repr(optional("a"))
    assert repr(mapped(mapped("a", 1), 1)) == repr(mapped("a", 1))
    assert repr(vararg(vararg("a"))) == repr(vararg("a"))
    assert repr(varargs(varargs("a"))) == repr(varargs("a"))


@pytest.mark.parametrize(
    "call, exp",
    [
        (lambda: sideffect(sideffect("a")), "^`sideffected` cannot"),
        (lambda: sideffected("a", sideffect("a")), "^`sfx_list` cannot"),
        (lambda: sideffected(sideffect("a"), "a"), "^`sideffected` cannot"),
        (lambda: sideffected_vararg(sideffect("a"), "a"), "^`sideffected` cannot"),
        (lambda: sideffected_varargs(sideffect("a"), "a"), "^`sideffected` cannot"),
    ],
)
def test_sideffected_bad(call, exp):
    with pytest.raises(ValueError, match=exp):
        call()


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: "b", "'p.b'"),
        (lambda: mapped("b", None), "mapped('p.b', fn_kwarg='b')"),
        (lambda: mapped("b", ""), "mapped('p.b', fn_kwarg='b')"),
        (lambda: mapped("b", "bb"), "mapped('p.b', fn_kwarg='bb')"),
        (lambda: optional("b"), "optional('p.b', fn_kwarg='b')"),
        (lambda: optional("b", "bb"), "optional('p.b', fn_kwarg='bb')"),
        (lambda: vararg("c"), "vararg('p.c')"),
        (lambda: varargs("d"), "varargs('p.d')"),
        (lambda: sideffect("e"), "sideffect: 'p.e'"),
        (lambda: sideffect("e", optional=1), "sideffect?: 'p.e'"),
        (lambda: sideffected("f", "a", "b"), "sideffected('p.f'<--'a', 'b')",),
        (
            lambda: sideffected("f", "ff", fn_kwarg="F"),
            "sideffected('p.f'<--'ff', fn_kwarg='F')",
        ),
        (
            lambda: sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "sideffected?('p.f'<--'ff', fn_kwarg='F')",
        ),
        (
            lambda: sideffected("f", "ff", optional=1),
            "sideffected?('p.f'<--'ff', fn_kwarg='f')",
        ),
        (lambda: sideffected_vararg("f", "a", "b"), "sideffected*('p.f'<--'a', 'b')"),
        (lambda: sideffected_varargs("f", "a"), "sideffected#('p.f'<--'a')"),
    ],
)
def test_modifs_rename_fn(mod, exp):
    renamer = lambda n: f"p.{n}"
    got = repr(dep_renamed(mod(), renamer))
    print(got)
    assert got == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: "s", "'D'"),
        (lambda: mapped("b", "bb"), "mapped('D', fn_kwarg='bb')"),
        (lambda: optional("b"), "optional('D', fn_kwarg='b')"),
        (lambda: optional("b", "bb"), "optional('D', fn_kwarg='bb')"),
        (lambda: vararg("c"), "vararg('D')"),
        (lambda: varargs("d"), "varargs('D')"),
        (lambda: sideffect("e"), "sideffect: 'D'"),
        (
            lambda: sideffected("f", "a", "b", optional=1,),
            "sideffected?('D'<--'a', 'b', fn_kwarg='f')",
        ),
        (
            lambda: sideffected("f", "a", "b", optional=1, fn_kwarg="F"),
            "sideffected?('D'<--'a', 'b', fn_kwarg='F')",
        ),
    ],
)
def test_modifs_rename_str(mod, exp):
    got = repr(dep_renamed(mod(), "D"))
    print(got)
    assert got == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (varargs("d"), "varargs('d')"),
        (sideffect("e", optional=1), "sideffect?: 'e'"),
        (sideffected("f", "a", "b"), "'f'"),
        (sideffected("f", "ff", fn_kwarg="F"), "mapped('f', fn_kwarg='F')",),
        (
            sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "optional('f', fn_kwarg='F')",
        ),
        (sideffected("f", "ff", optional=1), "optional('f')"),
        (sideffected_vararg("f", "a"), "vararg('f')"),
        (sideffected_varargs("f", "a", "b"), "varargs('f')"),
    ],
)
def test_sideffected_strip(mod, exp):
    got = dep_stripped(mod)
    assert repr(got) == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        ("a", ["a"]),
        (sideffect("a"), [sideffect("a")]),
        (sideffected("a", "b"), [sideffected("a", "b")]),
        (sideffected("a", "b", "c"), [sideffected("a", "b"), sideffected("a", "c")]),
    ],
)
def test_sideffected_singularized(mod, exp):
    got = list(dep_singularized(mod))
    assert got == exp
