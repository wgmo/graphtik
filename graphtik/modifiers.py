# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
A :term:`modifier` changes the behavior of specific `needs` or `provides`.

The `needs` and `provides` annotated with *modifiers* designate, for instance,
:term:`optional <optionals>` function arguments, or "ghost" :term:`sideffects`.
"""
import re
from typing import Tuple


class mapped(str):
    """
    Annotate a :term:`needs` that (optionally) map `inputs` name --> argument-name.

    :param fn_arg:
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

    __slots__ = ("fn_arg",)  # avoid __dict__ on instances

    fn_arg: str

    def __new__(cls, inp_key: str, fn_arg: str = None) -> "optional":
        obj = super().__new__(cls, inp_key)
        obj.__init__(inp_key, str(fn_arg))
        return obj

    def __init__(self, _inp_key: str, fn_arg: str = None):
        self.fn_arg = fn_arg

    def __repr__(self):
        return (
            str.__repr__(self)
            if self.fn_arg is None
            else f"mapped({str.__repr__(self)}-->{self.fn_arg!r})"
        )


class optional(mapped):
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

    def __repr__(self):
        return (
            f"optional({str.__repr__(self)})"
            if self.fn_arg is None
            else f"optional({str.__repr__(self)}-->{self.fn_arg!r})"
        )


class vararg(str):
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

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return "vararg('%s')" % self


class varargs(str):
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

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return "varargs('%s')" % self


class sideffect(str):
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

    .. note::
        An `operation` with *sideffects* outputs only, have functions that return
        no value at all (like the one above).  Such operation would still be called for
        their side-effects, if requested in `outputs`.


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

    .. graphtik::

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __new__(cls, name):
        m = re.match(r"sideffect\((.*)\)", name)
        if m:
            name = m.group(1)
        return super().__new__(cls, f"sideffect: {name}")


class sol_sideffect(sideffect):
    r"""
    Annotates a :term:`sideffected` dependency in the solution sustaining side-effects.

    Like :class:`.sideffect` but annotating the function's :term:`dependency` it relates to,
    allowing that dependency to be present both in :term:`needs` and :term:`provides`
    of the function.

    .. Note::
        When declaring `operation`s with *sideffected* dependencies, it is important
        not to put the actual :attr:`sideffected` in the `needs` & `provides`,
        or else, "cycles" will form, and network will not :term:`compile`.

    **Example:**

    A typical use-case is to signify columns required to produce new ones in
    pandas dataframes:


        >>> from graphtik import operation, compose, sol_sideffect

        >>> @operation(needs="order_items", provides=sol_sideffect("ORDER", "Items"))
        ... def new_order(items):
        ...     return pd.DataFrame(items, columns=["items"])

        >>> @operation(
        ...     needs=[sol_sideffect("ORDER", "Items"), "vat"],
        ...     provides=sol_sideffect("ORDER", "Prices", "Vat", "Totals")
        ... )
        ... def fill_in_prices(order: "pd.DataFrame", vat: float):
        ...     order['prices'] = ...  # beyond our example
        ...     order['VAT'] = order['prices'] * vat
        ...     order['totals'] = order['prices'] + order['VAT']
        ...     return order

    To view the internal differences, enable DEBUG in :term:`configurations`:

        >>> from graphtik import debug_enabled

        >>> with debug_enabled(True):
        ...     fill_in_prices
        FunctionalOperation(name='fill_in_prices',
                            needs=["sol_sideffect('ORDER'<--'Items'", 'vat'],
                            provides=["sol_sideffect('ORDER'<--'Prices, Vat, Totals'"],
                            needs=("sol_sideffect('ORDER'<--'Items'", 'vat'),
                            op_provides=("sol_sideffect('ORDER'<--'Prices'",
                                         "sol_sideffect('ORDER'<--'Vat'",
                                         "sol_sideffect('ORDER'<--'Totals'"),
                            fn_needs=('ORDER', 'vat'),
                            fn_provides=('ORDER'),
                            fn='fill_in_prices')

        Notice how the `fn_needs` & `fn_provides` differ from `needs` & `op_provides
        on the 2nd operation, due to the `sol_sideffect` dependencies:

        >>> proc_order = compose('process order', new_order, fill_in_prices)

    .. graphtik::
        :height: 360
        :width: 100%

    Notice how declaring a single *solution sideffect* with multiple *sideffects* on it,
    results into multiple  *"singular"* ``sol_sideffect`` dependencies in the network.
    """

    __slots__ = (
        "sideffected",
        "sideffects",
    )  # avoid __dict__ on instances

    #: An existing `dependency` in `solution` that sustain (must have sustained)
    #: the :term:`sideffects` by(for) the underlying function.
    sideffected: str
    #: At least one name(s) denoting the :term:`sideffects` modification(s) on
    #: the :term:`sideffected`, performed/required by the operation.
    sideffects: Tuple[str]

    def __new__(cls, sideffected, sideffect0, *sideffects):
        sideffects = (sideffect0,) + sideffects
        sfx_str = ", ".join(str(i) for i in sideffects)
        obj = str.__new__(cls, f"sol_sideffect({sideffected!r}<--{sfx_str!r}")
        obj.__init__(sideffected, sideffects)
        return obj

    def __init__(self, sideffected, *sideffects):
        self.sideffected = sideffected
        self.sideffects = sideffects
