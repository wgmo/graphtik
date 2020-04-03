# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
:term:`Modifiers` change the behavior of specific `needs` or `provides`.

The `needs` and `provides` annotated with *modifiers* designate, for instance,
:term:`optional <optionals>` function arguments, or "ghost" :term:`sideffects`.
"""
import re


class arg(str):
    """
    Annotate a :term:`needs` to map from its name in the `inputs` to a different argument-name.

    :param fn_arg:
        The argument-name corresponding to this named-input.

        .. Note::
            This extra mapping argument is needed either for `optionals` or
            for functions with keywords-only arguments (like ``def func(*, foo, bar): ...``),
            since `inputs`` are normally fed into functions by-position, not by-name.

    **Example:**

    In case the name of the function arguments is different from the name in the
    `inputs` (or just because the name in the `inputs` is not a valid argument-name),
    you may *map* it with the 2nd argument of :class:`.arg` (or :class:`.optional`):

        >>> from graphtik import operation, compose, arg, debug

        >>> def myadd(a, *, b):
        ...    return a + b

        >>> graph = compose('mygraph',
        ...     operation(name='myadd',
        ...               needs=['a', arg("name-in-inputs", "b")],
        ...               provides="sum")(myadd)
        ... )
        >>> with debug(True):
        ...     graph
        NetworkOperation('mygraph', needs=['a', 'name-in-inputs'], provides=['sum'], x1 ops:
          +--FunctionalOperation(name='myadd',
                                 needs=['a',
                                 arg('name-in-inputs'-->'b')],
                                 provides=['sum'],
                                 fn='myadd'))
        >>> graph.compute({"a": 5, "name-in-inputs": 4})['sum']
        9

    """

    __slots__ = ("fn_arg",)  # avoid __dict__ on instances

    fn_arg: str

    def __new__(cls, inp_key: str, fn_arg: str = None) -> "optional":
        obj = str.__new__(cls, inp_key)
        obj.fn_arg = fn_arg
        return obj

    def __repr__(self):
        inner = self if self.fn_arg is None else f"{self}'-->'{self.fn_arg}"
        return f"arg('{inner}')"


class optional(arg):
    """
    Annotate :term:`optionals` `needs` corresponding to *defaulted* op-function arguments, ...

    received only if present in the `inputs` (when operation is invoked).
    The value of an optional is passed as a keyword argument to the underlying function.


    **Example:**

        >>> from graphtik import operation, compose, optional

        >>> def myadd(a, b=0):
        ...    return a + b

    Annotate ``b`` as optional argument (and notice it's default value ``0``):

        >>> graph = compose('mygraph',
        ...     operation(name='myadd',
        ...               needs=["a", optional("b")],
        ...               provides="sum")(myadd)
        ... )
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

    Like :class:`.arg` you may map input-name to a different function-argument:

        >>> from graphtik import debug

        >>> graph = compose('mygraph',
        ...     operation(name='myadd',
        ...               needs=['a', optional("quasi-real", "b")],
        ...               provides="sum")(myadd)
        ... )
        >>> with debug(True):
        ...     graph
        NetworkOperation('mygraph', needs=['a', optional('quasi-real')], provides=['sum'], x1 ops:
          +--FunctionalOperation(name='myadd', needs=['a', optional('quasi-real'-->'b')], provides=['sum'], fn='myadd'))
        >>> graph.compute({"a": 5, "quasi-real": 4})['sum']
        9
    """

    def __repr__(self):
        inner = self if self.fn_arg is None else f"{self}'-->'{self.fn_arg}"
        return f"optional('{inner}')"


class vararg(str):
    """
    Annotate :term:`optionals` `needs` to  be fed as op-function's ``*args`` when present in inputs.

    .. seealso::
        Consult also the example test-case in: :file:`test/test_op.py:test_varargs()`,
        in the full sources of the project.

    **Example:**

        >>> from graphtik import operation, compose, vararg, debug

        >>> def addall(a, *b):
        ...    return a + sum(b)

    Designate ``b`` & ``c`` as an `vararg` arguments:

        >>> graph = compose(
        ...     'mygraph',
        ...     operation(
        ...               name='addall',
        ...               needs=['a', vararg('b'), vararg('c')],
        ...               provides='sum'
        ...     )(addall)
        ... )
        >>> with debug(True):
        ...     graph
        NetworkOperation('mygraph',
                         needs=['a', optional('b'), optional('c')],
                         provides=['sum'],
                         x1 ops:
          +--FunctionalOperation(name='addall', needs=['a', vararg('b'), vararg('c')], provides=['sum'], fn='addall'))

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
    :term:`sideffects` dependencies participates in the graph but not exchanged with functions.

    Both `needs` & `provides` may be designated as *sideffects* using this modifier.
    They work as usual while solving the graph (:term:`compilation`) but
    they do not interact with the `operation`'s function;  specifically:

    - input sideffects must exist in the :term:`inputs` for an operation to kick-in;
    - input sideffects are NOT fed into the function;
    - output sideffects are NOT expected from the function;
    - output sideffects are stored in the :term:`solution`.

    Their purpose is to describe operations that modify the internal state of
    some of their arguments ("side-effects").

    **Example:**

    A typical use-case is to signify columns required to produce new ones in
    pandas dataframes:


        >>> from graphtik import operation, compose, sideffect

        >>> # Function appending a new dataframe column from two pre-existing ones.
        >>> def addcolumns(df):
        ...    df['sum'] = df['a'] + df['b']

    Designate ``a``, ``b`` & ``sum`` column names as an sideffect arguments:

        >>> graph = compose('mygraph',
        ...     operation(
        ...         name='addcolumns',
        ...         needs=['df', sideffect('df.b')],  # sideffect names can be anything
        ...         provides=[sideffect('df.sum')])(addcolumns)
        ... )
        >>> graph
        NetworkOperation('mygraph', needs=['df', 'sideffect(df.b)'],
                         provides=['sideffect(df.sum)'], x1 ops: addcolumns)

    .. graphtik::

        >>> df = pd.DataFrame({'a': [5, 0], 'b': [2, 1]})   # doctest: +SKIP
        >>> graph({'df': df})['df']                         # doctest: +SKIP
        	a	b
        0	5	2
        1	0	1

    We didn't get the ``sum`` column because the ``b`` sideffect was unsatisfied.
    We have to add its key to the inputs (with *any* value):

        >>> graph({'df': df, sideffect("df.b"): 0})['df']   # doctest: +SKIP
        	a	b	sum
        0	5	2	7
        1	0	1	1

    Note that regular data in `needs` and `provides` do not match same-named `sideffects`.
    That is, in the following operation, the ``prices`` input is different from
    the ``sideffect(prices)`` output:

        >>> def upd_prices(sales_df, prices):
        ...     sales_df["Prices"] = prices

        >>> operation(fn=upd_prices,
        ...           name="upd_prices",
        ...           needs=["sales_df", "price"],
        ...           provides=[sideffect("price")])
        operation(name='upd_prices', needs=['sales_df', 'price'],
                  provides=['sideffect(price)'], fn='upd_prices')

    .. note::
        An `operation` with *sideffects* outputs only, have functions that return
        no value at all (like the one above).  Such operation would still be called for
        their side-effects, if requested in `outputs`.

    .. tip::
        You may associate sideffects with other data to convey their relationships,
        simply by including their names in the string - in the end, it's just a string -
        but no enforcement will happen from *graphtik*, like:

        >>> sideffect("price[sales_df]")
        'sideffect(price[sales_df])'

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __new__(cls, name):
        m = re.match(r"sideffect\((.*)\)", name)
        if m:
            name = m.group(1)
        return super().__new__(cls, f"sideffect({name})")
