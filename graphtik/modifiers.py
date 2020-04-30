# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
A :term:`modifier` changes the behavior of specific `needs` or `provides`.

The `needs` and `provides` annotated with *modifiers* designate, for instance,
:term:`optional <optionals>` function arguments, or "ghost" :term:`sideffects`.
"""
import re
import enum
from typing import Tuple, Union


class _Optionals(enum.Enum):
    optional = 1
    vararg = 2
    varargs = 3

    @property
    def varargish(self):
        return self.value > 1


class Dependency(str):
    # avoid __dict__ on instances
    __slots__ = (
        "fn_kwarg",
        "optional",
        "sideffected",
        "sideffects",
        "_repr",
    )

    #: Map my name in `needs` into this kw-argument of the function.
    fn_kwarg: str
    #: required = None, regular optional or varargish?
    optional: _Optionals
    #: An existing `dependency` in `solution` that sustain (must have sustained)
    #: the :term:`sideffects` by(for) the underlying function.
    sideffected: str
    #: At least one name(s) denoting the :term:`sideffects` modification(s) on
    #: the :term:`sideffected`, performed/required by the operation.
    #: If it is an empty tuple`, it is an abstract sideffect.
    sideffects: Tuple[Union[str, None]]
    #: pre-calculcated representation
    _repr: str

    def __new__(
        cls,
        name: str,
        fn_kwarg: str = None,
        optional: _Optionals = None,
        sideffects: Tuple[Union[str, None]] = None,
    ) -> "Dependency":
        sideffected = _repr = None
        ## Sanity checks and decide
        #  - string-name on sideffects and
        #  - repr for all
        #
        if optional and optional.varargish:
            assert not fn_kwarg and not sideffects, (
                "Varargish cannot map `fn_kwargs` or sideffects:",
                name,
                fn_kwarg,
                optional,
                optional,
                sideffects,
            )
            _repr = f"{optional.name}({name!r})"
        else:
            if sideffects is not None:
                if sideffects == ():
                    assert fn_kwarg is None, (
                        "Pure sideffects cannot map `fn_kwarg`:",
                        name,
                        fn_kwarg,
                        optional,
                        sideffects,
                    )

                    # m = re.match(r"sideffect\((.*)\)", name)
                    # if m:
                    #     name = m.group(1)
                    name = f"sideffect: {name}"
                    _repr = name  # avoid quotes around whole repr
                else:  # sol_sideffect
                    sideffected = name
                    sfx_str = ", ".join(str(i) for i in sideffects)
                    qmark = "?" if optional else ""
                    name = f"sol_sideffect{qmark}({name!r}<--{sfx_str!r})"
                    if fn_kwarg:
                        name = f"{name[:-1]}, fn_kwarg={fn_kwarg!r})"
                    _repr = name  # avoid quotes around whole repr
            elif optional or fn_kwarg:
                kwarg_str = f"-->{fn_kwarg!r}" if fn_kwarg else ""
                # TODO: Use qmark for optional
                _repr = f"{'optional' if optional else 'mapped'}({name!r}{kwarg_str})"

        obj = str.__new__(cls, name)

        obj._repr = str(_repr) if _repr is not None else None
        obj.fn_kwarg = fn_kwarg
        obj.optional = optional
        obj.sideffected = sideffected
        obj.sideffects = sideffects

        return obj

    @property
    def fn_arg(self):
        # TODO: DROPPPPP renamed fn_arg
        return self.fn_kwarg

    def __repr__(self):
        return super().__repr__() if self._repr is None else self._repr


def is_mapped(dep) -> bool:
    try:
        return bool(dep.fn_kwarg)
    except Exception:
        return False


def is_optional(dep) -> bool:
    try:
        return bool(dep.optional)
    except Exception:
        return False


def is_vararg(dep) -> bool:
    try:
        return dep.optional is _Optionals.vararg
    except Exception:
        return False


def is_varargs(dep) -> bool:
    try:
        return dep.optional is _Optionals.varargs
    except Exception:
        return False


def is_varargish(dep) -> bool:
    try:
        return dep.optional.varargish
    except Exception:
        return False


def is_sideffect(dep) -> bool:
    try:
        return dep.sideffects is not None
    except Exception:
        return False


def is_pure_sideffect(dep) -> bool:
    try:
        return dep.sideffects == ()
    except Exception:
        return False


def is_sol_sideffect(dep) -> bool:
    try:
        return bool(dep.sideffected)
    except Exception:
        return False


def mapped(name: str, fn_kwarg: str):
    """
    Annotate a :term:`needs` that (optionally) map `inputs` name --> argument-name.

    :param fn_kwarg:
        The argument-name corresponding to this named-input.

        .. Note::
            This extra mapping argument is needed either for `optionals` or
            for functions with keywords-only arguments (like ``def func(*, foo, bar): ...``),
            since `inputs`` are normally fed into functions by-position, not by-name.

    **Example:**

    In case the name of the function arguments is different from the name in the
    `inputs` (or just because the name in the `inputs` is not a valid argument-name),
    you may *map* it with the 2nd argument of :class:`.mapped` (or :class:`.optional`):

        >>> from graphtik import operation, compose, mapped

        >>> @operation(needs=['a', mapped("name-in-inputs", "b")], provides="sum")
        ... def myadd(a, *, b):
        ...    return a + b
        >>> myadd
        FunctionalOperation(name='myadd',
                            needs=['a', mapped('name-in-inputs'-->'b')],
                            provides=['sum'],
                            fn='myadd')

        >>> graph = compose('mygraph', myadd)
        >>> graph
        NetworkOperation('mygraph', needs=['a', 'name-in-inputs'], provides=['sum'], x1 ops: myadd)

        >>> sol = graph.compute({"a": 5, "name-in-inputs": 4})['sum']
        >>> sol
        9

        .. graphtik::
    """
    return Dependency(name, fn_kwarg=fn_kwarg)


def optional(name: str, fn_kwarg: str = None):
    """
    Annotate :term:`optionals` `needs` corresponding to *defaulted* op-function arguments, ...

    received only if present in the `inputs` (when operation is invoked).
    The value of an optional is passed as a keyword argument to the underlying function.


    **Example:**

        >>> from graphtik import operation, compose, optional

        >>> @operation(name='myadd',
        ...            needs=["a", optional("b")],
        ...            provides="sum")
        ... def myadd(a, b=0):
        ...    return a + b

    Notice the default value ``0`` to the ``b`` annotated as optional argument:

        >>> graph = compose('mygraph', myadd)
        >>> graph
        NetworkOperation('mygraph',
                         needs=['a', optional('b')],
                         provides=['sum'],
                         x1 ops: myadd)

    .. graphtik::

    The graph works both with and without ``c`` provided in the inputs:

        >>> graph(a=5, b=4)['sum']
        9
        >>> graph(a=5)
        {'a': 5, 'sum': 5}

    Like :class:`.mapped` you may map input-name to a different function-argument:

        >>> operation(needs=['a', optional("quasi-real", "b")],
        ...           provides="sum"
        ... )(myadd.fn)  # Cannot wrap an operation, its `fn` only.
        FunctionalOperation(name='myadd',
                            needs=['a', optional('quasi-real'-->'b')],
                            provides=['sum'],
                            fn='myadd')

    """
    return Dependency(name, fn_kwarg=fn_kwarg, optional=_Optionals.optional)


def vararg(name: str):
    """
    Annotate :term:`optionals` `needs` to  be fed as op-function's ``*args`` when present in inputs.

    .. seealso::
        Consult also the example test-case in: :file:`test/test_op.py:test_varargs()`,
        in the full sources of the project.

    **Example:**

    We designate ``b`` & ``c`` as `vararg` arguments:

        >>> from graphtik import operation, compose, vararg

        >>> @operation(
        ...     needs=['a', vararg('b'), vararg('c')],
        ...     provides='sum'
        ... )
        ... def addall(a, *b):
        ...    return a + sum(b)
        >>> addall
        FunctionalOperation(name='addall', needs=['a', vararg('b'), vararg('c')], provides=['sum'], fn='addall')


        >>> graph = compose('mygraph', addall)

    .. graphtik::

    The graph works with and without any of ``b`` or ``c`` inputs:

        >>> graph(a=5, b=2, c=4)['sum']
        11
        >>> graph(a=5, b=2)
        {'a': 5, 'b': 2, 'sum': 7}
        >>> graph(a=5)
        {'a': 5, 'sum': 5}

    """
    return Dependency(name, optional=_Optionals.vararg)


def varargs(name: str):
    """
    Like :class:`vararg`, naming an :term:`optional <optionals>` *iterable* value in the inputs.

    .. seealso::
        Consult also the example test-case in: :file:`test/test_op.py:test_varargs()`,
        in the full sources of the project.

    **Example:**

        >>> from graphtik import operation, compose, varargs

        >>> def enlist(a, *b):
        ...    return [a] + list(b)

        >>> graph = compose('mygraph',
        ...     operation(name='enlist', needs=['a', varargs('b')],
        ...     provides='sum')(enlist)
        ... )
        >>> graph
        NetworkOperation('mygraph',
                         needs=['a', optional('b')],
                         provides=['sum'],
                         x1 ops: enlist)

    .. graphtik::

    The graph works with or without `b` in the inputs:

        >>> graph(a=5, b=[2, 20])['sum']
        [5, 2, 20]
        >>> graph(a=5)
        {'a': 5, 'sum': [5]}
        >>> graph(a=5, b=0xBAD)
        Traceback (most recent call last):
        ...
        graphtik.base.MultiValueError: Failed preparing needs:
            1. Expected needs[varargs('b')] to be non-str iterables!
            +++inputs: ['a', 'b']
            +++FunctionalOperation(name='enlist', needs=['a', varargs('b')], provides=['sum'], fn='enlist')

    .. Attention::
        To avoid user mistakes, *varargs* does not accept strings (though iterables):

        >>> graph(a=5, b="mistake")
        Traceback (most recent call last):
        ...
        graphtik.base.MultiValueError: Failed preparing needs:
            1. Expected needs[varargs('b')] to be non-str iterables!
            +++inputs: ['a', 'b']
            +++FunctionalOperation(name='enlist', needs=['a', varargs('b')], provides=['sum'], fn='enlist')

    """

    return Dependency(name, optional=_Optionals.varargs)


def sideffect(name, optional: bool = None):
    """
    Abstract :term:`sideffects` take part in the graph but not when calling functions.

    Both `needs` & `provides` may be designated as *sideffects* using this modifier.
    They work as usual while solving the graph (:term:`compilation`) but
    they do not interact with the `operation`'s function;  specifically:

    - input sideffects must exist in the :term:`inputs` for an operation to kick-in;
    - input sideffects are NOT fed into the function;
    - output sideffects are NOT expected from the function;
    - output sideffects are stored in the :term:`solution`.

    Their purpose is to describe operations that modify some internal state
    not expressed in solution's the inputs/outputs ("side-effects").

    .. hint::
        If modifications involve some input/output, prefer the :class:`.sol_sideffect`
        modifier.

        You may still convey this relationships simply by including the dependency name
        in the string - in the end, it's just a string - but no enforcement of any kind
        will happen from *graphtik*, like:

        >>> sideffect("price[sales_df]")  # doctest: +SKIP
        'sideffect(price[sales_df])'

    **Example:**

    A typical use-case is to signify changes in some "global" context,
    outside `solution`:


        >>> from graphtik import operation, compose, sideffect

        >>> @operation(provides=sideffect("lights off"))  # sideffect names can be anything
        ... def close_the_lights():
        ...    pass

        >>> graph = compose('strip ease',
        ...     close_the_lights,
        ...     operation(
        ...         name='undress',
        ...         needs=[sideffect("lights off")],
        ...         provides="body")(lambda: "TaDa!")
        ... )
        >>> graph
        NetworkOperation('strip ease', needs=['sideffect: lights off'],
                         provides=['sideffect: lights off', 'body'],
                         x2 ops: close_the_lights, undress)

        >>> sol = graph()
        >>> sol
        {'body': 'TaDa!'}

    .. graphtik::
        :name: sideffect

    .. note::
        Something has to provide a sideffect for a function needing it to execute -
        this could be another operation, like above, or the user-inputs;
        just specify some dummy value for the sideffect:

            >>> sol = graph.compute({sideffect("lights off"): True})

        .. graphtik::

    """
    return Dependency(
        name, optional=_Optionals.optional if optional else None, sideffects=()
    )


def sol_sideffect(
    sideffected: str,
    sideffect0: str,
    *sideffects: str,
    optional: bool = None,
    fn_kwarg: str = None,
):
    r"""
    Annotates a :term:`sideffected` dependency in the solution sustaining side-effects.

    Like :class:`.sideffect` but annotating the function's :term:`dependency` it relates to,
    allowing that dependency to be present both in :term:`needs` and :term:`provides`
    of the function.

    .. Note::

        When declaring an `operation` with *sideffected* dependency, it is important
        not to put the actual :attr:`sideffected` in the `needs` & `provides`,
        or else, dependency cycles will form, and network will not :term:`compile`.

    **Example:**

    A typical use-case is to signify columns required to produce new ones in
    pandas dataframes (:gray:`emulated with dictionaries`):


        >>> from graphtik import operation, compose, sol_sideffect

        >>> @operation(needs="order_items",
        ...            provides=sol_sideffect("ORDER", "Items", "Prices"))
        ... def new_order(items: list) -> "pd.DataFrame":
        ...     order = {"items": items}
        ...     # Pretend we get the prices from sales.
        ...     order['prices'] = list(range(1, len(order['items']) + 1))
        ...     return order

        >>> @operation(
        ...     needs=[sol_sideffect("ORDER", "Items"), "vat rate"],
        ...     provides=sol_sideffect("ORDER", "VAT")
        ... )
        ... def fill_in_vat(order: "pd.DataFrame", vat: float):
        ...     order['VAT'] = [i * vat for i in order['prices']]
        ...     return order

        >>> @operation(
        ...     needs=[sol_sideffect("ORDER", "Prices", "VAT")],
        ...     provides=sol_sideffect("ORDER", "Totals")
        ... )
        ... def finalize_prices(order: "pd.DataFrame"):
        ...     order['totals'] = [p + v for p, v in zip(order['prices'], order['VAT'])]
        ...     return order

    To view all internal :term:`dependencies <dependency>`, enable DEBUG
    in :term:`configurations`:

        >>> from graphtik import debug_enabled

        >>> with debug_enabled(True):
        ...     finalize_prices
        FunctionalOperation(name='finalize_prices',
                            needs=["sol_sideffect('ORDER'<--'Prices'",
                                   "sol_sideffect('ORDER'<--'VAT'"],
                            op_needs=["sol_sideffect('ORDER'<--'Prices'",
                                      "sol_sideffect('ORDER'<--'VAT'"],
                            fn_needs=['ORDER'],
                            provides=["sol_sideffect('ORDER'<--'Totals'"],
                            op_provides=["sol_sideffect('ORDER'<--'Totals'"],
                            fn_provides=['ORDER'],
                            fn='finalize_prices')

    - Notice that although the function consumes & produces ``ORDER``
      (check ``fn_needs`` & ``fn_provides``, above), which :orange:`would have created a cycle`,
      the wrapping operation :term:`needs` and :term:`provides` different
      `sol_sideffects`, breaking thus the cycle.

    - Notice also that declaring a single *solution sideffect* with multiple *sideffects*,
      expands into multiple  *"singular"* ``sol_sideffect`` dependencies in the network
      (check ``needs``, above).

        >>> proc_order = compose('process order', new_order, fill_in_vat, finalize_prices)
        >>> sol = proc_order.compute({
        ...      "order_items": ["toilet-paper", "soap"],
        ...      "vat rate": 0.18,
        ... })
        >>> sol
        {'order_items': ['toilet-paper', 'soap'],
         'vat rate': 0.18,
         'ORDER': {'items': ['toilet-paper', 'soap'],
                   'prices': [1, 2],
                   'VAT': [0.18, 0.36],
                   'totals': [1.18, 2.36]}}

    .. graphtik::
        :height: 640
        :width: 100%
        :name: solution-sideffects

    """
    sideffects = (sideffect0,) + sideffects
    return Dependency(
        sideffected,
        sideffects=sideffects,
        optional=_Optionals.optional if optional else None,
        fn_kwarg=fn_kwarg,
    )
