
========
Graphtik
========

|python-ver| |dev-status| (|release|, |today|) |gh-version| |pypi-version|
|travis-status| |doc-status| |cover-status| |codestyle| |proj-lic|

|gh-watch| |gh-star| |gh-fork| |gh-issues|

.. epigraph::
   It's a DAG all the way down!

   |sample-plot|

Lightweight computation graphs for Python
-----------------------------------------

**Graphtik** is an an understandable and lightweight Python module for building and running
ordered graphs of computations.
The API posits a fair compromise between features and complexity, without precluding any.
It can be used as is to build machine learning pipelines for data science projects.
It should be extendable to act as the core for a custom ETL engine or
a workflow-processor for interdependent files and processes.

*Graphtik* sprang from `Graphkit`_ to experiment with Python 3.6+ features.

.. toctree::
   :maxdepth: 3
   :numbered: 1

   operations
   composition
   plotting
   arch
   reference
   Changes <changes>
   genindex

.. _quick-start:

Quick start
-----------

Here's how to install::

   pip install graphtik

OR with dependencies for plotting support (and you need to install `Graphviz
<https://graphviz.org>`_ program separately with your OS tools)::

   pip install graphtik[plot]


Let's build a *graphtik* computation graph that produces x3 outputs
out of 2 inputs `a` and `b`:

.. math::

   a \times b

   a - a \times b

   |a - a \times b| ^ 3

..

   >>> from graphtik import compose, operation
   >>> from operator import mul, sub

   >>> @operation(name="abs qubed",
   ...            needs=["a_minus_ab"],
   ...            provides=["abs_a_minus_ab_cubed"])
   ... def abs_qubed(a):
   ...    return abs(a) ** 3

Compose the ``abspow`` function along with ``mul`` & ``sub``  built-ins
into a computation graph:

   >>> graphop = compose("graphop",
   ...    operation(needs=["a", "b"], provides=["ab"])(mul),
   ...    operation(sub, needs=["a", "ab"], provides=["a_minus_ab"])(),
   ...    abs_qubed,
   ... )
   >>> graphop
   NetworkOperation('graphop', needs=['a', 'b', 'ab', 'a_minus_ab'],
                     provides=['ab', 'a_minus_ab', 'abs_a_minus_ab_cubed'],
                     x3 ops: <built-in function mul>, <built-in function sub>, abs qubed)

You may plot the function graph in a file like this
(if in *jupyter*, no need to specify the file):

   >>> graphop.plot('graphop.svg')      # doctest: +SKIP

.. graphtik::

As you can see, any function can be used as an operation in Graphtik,
even ones imported from system modules.

Run the graph-operation and request all of the outputs:

   >>> sol = graphop(**{'a': 2, 'b': 5})
   >>> sol
   {'a': 2, 'b': 5, 'ab': 10, 'a_minus_ab': -8, 'abs_a_minus_ab_cubed': 512}

.. graphtik::

Run the graph-operation and request a subset of the outputs:

   >>> solution = graphop.compute({'a': 2, 'b': 5}, outputs=["a_minus_ab"])
   >>> solution
   {'a_minus_ab': -8}

Solutions are plottable as well:

   >>> solution.plot('solution.svg')      # doctest: +SKIP

.. graphtik::

... where the leged is: |plot-legend|

.. |sample-plot| raw:: html
    :file:  images/barebone_2ops.svg
.. |plot-legend| raw:: html
    :file:  images/GraphtikLegend.svg
.. include:: ../../README.rst
    :start-after: _badges_substs: