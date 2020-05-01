import pytest

from graphtik.modifiers import (
    Dependency,
    mapped,
    optional,
    sideffect,
    sol_sideffect,
    vararg,
    varargs,
)


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", "bb"), "b"),
        (lambda: optional("b"), "b"),
        (lambda: optional("b", "bb"), "b"),
        (lambda: vararg("c"), "c"),
        (lambda: varargs("d"), "d"),
        (lambda: sideffect("e"), "sideffect: 'e'"),
        (lambda: sideffect("e", optional=1), "sideffect: 'e'"),
        (lambda: sol_sideffect("f", "ff"), "sol_sideffect('f'<--'ff')"),
        (
            lambda: sol_sideffect("f", "ff", fn_kwarg="F"),
            "sol_sideffect('f'<--'ff', fn_kwarg='F')",
        ),
        (
            lambda: sol_sideffect("f", "ff", optional=1, fn_kwarg="F"),
            "sol_sideffect?('f'<--'ff', fn_kwarg='F')",
        ),
        (lambda: sol_sideffect("f", "ff", optional=1), "sol_sideffect?('f'<--'ff')"),
    ],
)
def test_modifs_str(mod, exp):
    mod = mod()
    print(mod)
    assert str(mod) == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", "bb"), "mapped('b'-->'bb')"),
        (lambda: optional("b"), "optional('b')"),
        (lambda: optional("b", "bb"), "optional('b'-->'bb')"),
        (lambda: vararg("c"), "vararg('c')"),
        (lambda: varargs("d"), "varargs('d')"),
        (lambda: sideffect("e"), "sideffect: 'e'"),
        (lambda: sideffect("e", optional=1), "sideffect: 'e'"),
        (lambda: sol_sideffect("f", "ff"), "sol_sideffect('f'<--'ff')"),
        (
            lambda: sol_sideffect("f", "ff", fn_kwarg="F"),
            "sol_sideffect('f'<--'ff', fn_kwarg='F')",
        ),
        (
            lambda: sol_sideffect("f", "ff", optional=1, fn_kwarg="F"),
            "sol_sideffect?('f'<--'ff', fn_kwarg='F')",
        ),
        (lambda: sol_sideffect("f", "ff", optional=1), "sol_sideffect?('f'<--'ff')"),
    ],
)
def test_modifs_repr(mod, exp):
    mod = mod()
    print(repr(mod))
    # Strip outer quotes
    assert repr(mod) == exp


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
    # assert repr(sideffect(sideffect("a"))) == repr(sideffect("a"))
