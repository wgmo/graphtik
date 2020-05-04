import pytest

from graphtik.modifiers import (
    mapped,
    optional,
    sideffect,
    sideffected,
    vararg,
    varargs,
)


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
    mod = mod()
    print(mod)
    assert str(mod) == exp


@pytest.mark.parametrize(
    "mod, exp",
    [
        (lambda: mapped("b", None), "'b'"),
        (lambda: mapped("b", ""), "'b'"),
        (lambda: mapped("b", "bb"), "mapped('b'-->'bb')"),
        (lambda: optional("b"), "optional('b')"),
        (lambda: optional("b", "bb"), "optional('b'-->'bb')"),
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
    ],
)
def test_modifs_repr(mod, exp):
    mod = mod()
    print(repr(mod))
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


@pytest.mark.parametrize(
    "call, exp",
    [
        (lambda: sideffect(sideffect("a")), "^Expecting "),
        (lambda: sideffected("a", sideffect("a")), "^Expecting "),
        (lambda: sideffected(sideffect("a"), "a"), "^Expecting "),
    ],
)
def test_sideffected_bad(call, exp):
    with pytest.raises(ValueError, match=exp):
        call()


def test_withset_bad_kwargs():
    with pytest.raises(ValueError, match="Invalid kwargs:"):
        mapped("a", "b").withset(j=2)
