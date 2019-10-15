Graphtik
========

|python-ver| |pypi-ver| |gh-ver| |travis-status| |rtd-status| |cov-status|
|lic-kind|

|gh-watchers| |gh-stargazers| |gh-forks| |gh-issues|

   It’s a DAG all the way down

Lightweight computation graphs for Python
-----------------------------------------

**Graphtik** is an an understandable and lightweight Python module for
building and running ordered graphs of computations. The API posits a
fair compromise between features and complexity without precluding any.
It might be of use in computer vision, machine learning and other data
science domains, or become the core of a custom ETL pipelne.

.. note:
    *Graphtik* is a temporary fork of `Graphkit`_ to experiment with Python
    3.6+ features.

Quick start
-----------

Here’s how to install:

::

   pip install graphtik

OR with dependencies for plotting support (and you need to install
`Graphviz`_ suite separately, with your OS tools)::

   pip install graphtik[plot]

Here’s a Python script with an example Graphtik computation graph that
produces multiple outputs (``a * b``, ``a - a * b``, and
``abs(a - a * b) ** 3``)::

   >>> from operator import mul, sub
   >>> from graphtik import compose, operation

   >>> # Computes |a|^p.
   >>> def abspow(a, p):
   ...     c = abs(a) ** p
   ...     return c

Compose the ``mul``, ``sub``, and ``abspow`` functions into a computation graph::

   >>> graphop = compose(name="graphop")(
   ...     operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
   ...     operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
   ...     operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"], params={"p": 3})(abspow)
   ... )


Run the graph and request all of the outputs::

   >>> graphop({'a': 2, 'b': 5})
   {'a': 2, 'b': 5, 'ab': 10, 'a_minus_ab': -8, 'abs_a_minus_ab_cubed': 512}

   >>> graphop({'a': 2, 'b': 5}, outputs=["a_minus_ab"])
   {'a_minus_ab': -8}

.. _Graphkit: https://github.com/yahoo/graphkit
.. _`Graphviz`: https://graphviz.org

.. |python-ver| image:: https://img.shields.io/pypi/pyversions/graphtik.svg?label=Python
.. |pypi-ver| image:: https://img.shields.io/pypi/v/graphtik.svg?label=PyPi%20version
.. |gh-ver| image:: https://img.shields.io/github/v/release/pygraphkit/graphtik.svg?label=GitHub%20release&include_prereleases
.. |travis-status| image:: https://travis-ci.org/pygraphkit/graphtik.svg?branch=master
.. |rtd-status| image:: https://img.shields.io/readthedocs/graphtik.svg?branch=master
.. |cov-status| image:: https://cov-status.io/gh/pygraphkit/graphtik/branch/master/graph/badge.svg
.. |lic-kind| image:: https://img.shields.io/pypi/l/graphtik.svg
.. |gh-watchers| image:: https://img.shields.io/github/watchers/pygraphkit/graphtik.svg?style=social
.. |gh-stargazers| image:: https://img.shields.io/github/stars/pygraphkit/graphtik.svg?style=social
.. |gh-forks| image:: https://img.shields.io/github/forks/pygraphkit/graphtik.svg?style=social
.. |gh-issues| image:: http://img.shields.io/github/issues/pygraphkit/graphtik.svg?style=social