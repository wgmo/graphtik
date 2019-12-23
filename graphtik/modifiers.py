# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
This sub-module contains input/output modifiers that can be applied to
arguments to ``needs`` and ``provides`` to let Graphtik know it should treat
them differently.
"""
import re


class optional(str):
    """
    An optional need signifies that the function's argument may not receive a value.

    Only input values in ``needs`` may be designated as optional using this modifier.
    An ``operation`` will receive a value for an optional need only if if it is available
    in the graph at the time of its invocation.
    The ``operation``'s function should have a defaulted parameter with the same name
    as the opetional, and the input value will be passed as a keyword argument,
    if it is available.

    Here is an example of an operation that uses an optional argument::

        >>> from graphtik import operation, compose, optional

        >>> def myadd(a, b, c=0):
        ...    return a + b + c

    Designate c as an optional argument::

        >>> graph = compose('mygraph',
        ...     operation(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        ... )
        >>> graph
        NetworkOperation('mygraph',
                         needs=['a', 'b', optional('c')],
                         provides=['sum'],
                         x1 ops:
        ...

    The graph works with and without `c` provided as input::

        >>> graph(a=5, b=2, c=4)['sum']
        11
        >>> graph(a=5, b=2)
        {'a': 5, 'b': 2, 'sum': 7}

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return "optional('%s')" % self


class vararg(optional):
    """
    Like :class:`optional` but feeds as ONE OF the ``*args`` into the function (instead of ``**kwargs``).

    For instance::

        >>> from graphtik import operation, compose, vararg

        >>> def addall(a, *b):
        ...    return a + sum(b)

    Designate `b` & `c` as an `vararg` arguments::

        >>> graph = compose('mygraph',
        ...     operation(name='addall', needs=['a', vararg('b'), vararg('c')],
        ...     provides='sum')(addall)
        ... )
        >>> graph
        NetworkOperation('mygraph',
                         needs=['a', optional('b'), optional('c')],
                         provides=['sum'],
                         x1 ops:
          +--FunctionalOperation(name='addall', needs=['a', vararg('b'), vararg('c')], provides=['sum'], fn='addall'))

    The graph works with and without any of `b` and `c` inputs::

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


class varargs(optional):
    """
    An optional like :class:`vararg` feeds as MANY ``*args`` into the function (instead of ``**kwargs``).
    """

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return "varargs('%s')" % self


class sideffect(str):
    """
    A sideffect data-dependency participates in the graph but never given/asked in functions.

    Both inputs & outputs in ``needs`` & ``provides`` may be designated as *sideffects*
    using this modifier.  *Sideffects* work as usual while solving the graph but
    they do not interact with the ``operation``'s function;  specifically:

    - input sideffects are NOT fed into the function;
    - output sideffects are NOT expected from the function.

    .. info:
        an ``operation`` with just a single *sideffect* output return no value at all,
        but it would still be called for its side-effect only.

    Their purpose is to describe operations that modify the internal state of
    some of their arguments ("side-effects").
    A typical use case is to signify columns required to produce new ones in
    pandas dataframes::


        >>> from graphtik import operation, compose, sideffect

        >>> # Function appending a new dataframe column from two pre-existing ones.
        >>> def addcolumns(df):
        ...    df['sum'] = df['a'] + df['b']

    Designate `a`, `b` & `sum` column names as an sideffect arguments::

        >>> graph = compose('mygraph',
        ...     operation(
        ...         name='addcolumns',
        ...         needs=['df', sideffect('df.b')],  # sideffect names can be anything
        ...         provides=[sideffect('df.sum')])(addcolumns)
        ... )
        >>> graph
        NetworkOperation('mygraph', needs=['df', 'sideffect(df.b)'],
                         provides=['sideffect(df.sum)'], x1 ops:
          +--FunctionalOperation(name='addcolumns', needs=['df', 'sideffect(df.b)'], provides=['sideffect(df.sum)'], fn='addcolumns'))

        >>> df = pd.DataFrame({'a': [5, 0], 'b': [2, 1]})   # doctest: +SKIP
        >>> graph({'df': df})['df']                         # doctest: +SKIP
        	a	b
        0	5	2
        1	0	1

    We didn't get the ``sum`` column because the `b` sideffect was unsatisfied.
    We have to add its key to the inputs (with _any_ value)::

        >>> graph({'df': df, sideffect("df.b"): 0})['df']   # doctest: +SKIP
        	a	b	sum
        0	5	2	7
        1	0	1	1

    Note that regular data in *needs* and *provides* do not match same-named *sideffects*.
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
        An ``operation`` with *sideffects* outputs only, have functions that return
        no value at all (like the one above).  Such operation would still be called for
        their side-effects.

    .. tip::
        You may associate sideffects with other data to convey their relationships,
        simply by including their names in the string - in the end, it's just a string -
        but no enforcement will happen from *graphtik*.

        >>> sideffect("price[sales_df]")
        'sideffect(price[sales_df])'

    """

    __slots__ = ()  # avoid __dict__ on instances

    def __new__(cls, name):
        m = re.match(r"sideffect\((.*)\)", name)
        if m:
            name = m.group(1)
        return super().__new__(cls, f"sideffect({name})")
