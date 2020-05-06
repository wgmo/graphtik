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
        (lambda: sideffect("e"), "sfx: 'e'"),
        (lambda: sideffect("e", optional=1), "sfx: 'e'"),
        (lambda: sideffected("f", "a", "b"), "sfxed('f', 'a', 'b')"),
        (lambda: sideffected("f", "ff", fn_kwarg="F"), "sfxed('f', 'ff')",),
        (lambda: sideffected("f", "ff", optional=1, fn_kwarg="F"), "sfxed('f', 'ff')",),
        (lambda: sideffected("f", "ff", optional=1), "sfxed('f', 'ff')"),
    ],
)
def test_modifs_str(mod, exp):
    got = str(mod())
    print(got)
    assert got == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", None), "'b'(>)"),
        (lambda: mapped("b", ""), "'b'(>)"),
        (lambda: mapped("b", "bb"), "'b'(>'bb')"),
        (lambda: optional("b"), "'b'(?)"),
        (lambda: optional("b", "bb"), "'b'(?>'bb')"),
        (lambda: vararg("c"), "'c'(*)"),
        (lambda: varargs("d"), "'d'(+)"),
        (lambda: sideffect("e"), "sfx: 'e'"),
        (lambda: sideffect("e", optional=1), "sfx(?): 'e'"),
        (lambda: sideffected("f", "a", "b"), "sfxed('f', 'a', 'b')"),
        (lambda: sideffected("f", "ff", fn_kwarg="F"), "sfxed('f'(>'F'), 'ff')",),
        (
            lambda: sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "sfxed('f'(?>'F'), 'ff')",
        ),
        (lambda: sideffected("f", "ff", optional=1), "sfxed('f'(?), 'ff')"),
        (lambda: sideffected_vararg("f", "a"), "sfxed('f'(*), 'a')"),
        (lambda: sideffected_varargs("f", "a", "b"), "sfxed('f'(+), 'a', 'b')"),
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
        ("b", "'p.b'"),
        (mapped("b", None), "'p.b'(>'b')"),
        (mapped("b", ""), "'p.b'(>'b')"),
        (mapped("b", "bb"), "'p.b'(>'bb')"),
        (optional("b"), "'p.b'(?>'b')"),
        (optional("b", "bb"), "'p.b'(?>'bb')"),
        (vararg("c"), "'p.c'(*)"),
        (varargs("d"), "'p.d'(+)"),
        (sideffect("e"), "sfx: 'p.e'"),
        (sideffect("e", optional=1), "sfx(?): 'p.e'"),
        (sideffected("f", "a", "b"), "sfxed('p.f', 'a', 'b')",),
        (sideffected("f", "ff", fn_kwarg="F"), "sfxed('p.f'(>'F'), 'ff')",),
        (
            sideffected("f", "ff", optional=1, fn_kwarg="F"),
            "sfxed('p.f'(?>'F'), 'ff')",
        ),
        (sideffected("f", "ff", optional=1), "sfxed('p.f'(?>'f'), 'ff')",),
        (sideffected_vararg("f", "a", "b"), "sfxed('p.f'(*), 'a', 'b')"),
        (sideffected_varargs("f", "a"), "sfxed('p.f'(+), 'a')"),
    ],
)
def test_modifs_rename_fn(mod, exp):
    renamer = lambda n: f"p.{n}"
    got = repr(dep_renamed(mod, renamer))
    print(got)
    assert got == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: "s", "'D'"),
        (lambda: mapped("b", "bb"), "'D'(>'bb')"),
        (lambda: optional("b"), "'D'(?>'b')"),
        (lambda: optional("b", "bb"), "'D'(?>'bb')"),
        (lambda: vararg("c"), "'D'(*)"),
        (lambda: varargs("d"), "'D'(+)"),
        (lambda: sideffect("e"), "sfx: 'D'"),
        (
            lambda: sideffected("f", "a", "b", optional=1,),
            "sfxed('D'(?>'f'), 'a', 'b')",
        ),
        (
            lambda: sideffected("f", "a", "b", optional=1, fn_kwarg="F"),
            "sfxed('D'(?>'F'), 'a', 'b')",
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
        (varargs("d"), "'d'(+)"),
        (sideffect("e", optional=1), "sfx(?): 'e'"),
        (sideffected("f", "a", "b"), "'f'"),
        (sideffected("f", "ff", fn_kwarg="F"), "'f'(>'F')",),
        (sideffected("f", "ff", optional=1, fn_kwarg="F"), "'f'(?>'F')",),
        (sideffected("f", "ff", optional=1), "'f'(?)"),
        (sideffected_vararg("f", "a"), "'f'(*)"),
        (sideffected_varargs("f", "a", "b"), "'f'(+)"),
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
