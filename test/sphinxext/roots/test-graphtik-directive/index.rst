1. `graphtik` with :graphvar:
=============================
.. graphtik::
    :graphvar: netop1

    >>> from graphtik import compose, operation

    >>> netop1 = compose(
    ...     "test_netop1",
    ...     operation(name="op1", needs=["a", "b"], provides="aa")(lambda a, b: [a, b]),
    ... )


2. Solved `graphtik` WITHOUT :graphvar:
=======================================
.. graphtik::
    :caption: Solved *netop2* with ``a=1``, ``b=2``

    >>> netop2 = compose(
    ...     "test_netop2",
    ...     operation(name="op1", needs=["a", "b"], provides="aa")(lambda a, b: [a, b]),
    ... )
    >>> sol = netop2(a=1, b=2)


3. `graphtik` inherit from literal-block WITHOUT :graphvar:
===========================================================

>>> netop3 = compose(
...     "test_netop3",
...     operation(name="op1", needs=["a", "b"], provides="aa")(lambda a, b: [a, b]),
... )

.. graphtik::


4. `graphtik` inherit from doctest-block with :graphvar:
========================================================

>>> netop4 = compose(
...     "test_netop4",
...     operation(name="op1", needs=["a", "b"], provides="aa")(lambda a, b: [a, b]),
... )

.. graphtik::
    :graphvar: netop4


5. Image for :hide:
===================
.. graphtik::
    :graphvar: netop1
    :hide:


6. Nothing for :skipif:
=======================
.. graphtik::
    :graphvar: netop1
    :skipif: True
