
========
Graphtik
========

|pypi-version| |gh-version| (|release|, |today|) |python-ver| |dev-status|
|ci-status| |doc-status| |cover-status| |codestyle| |proj-lic|

|gh-watch| |gh-star| |gh-fork| |gh-issues|

.. default-role:: term
.. epigraph::
   It's a DAG all the way down!

   |sample-plot|

Computation graphs for Python & Pandas
--------------------------------------
**Graphtik** is a library to compose, solve, execute & plot *graphs of python functions*
(a.k.a `pipeline`\s) that consume and populate named data
(a.k.a `dependencies <dependency>`), whose names may be
`nested <hierarchical data>` (such as, *pandas* dataframe columns),
based on whether values for those dependencies exist in the inputs or
have been calculated earlier.

.. math-overview-start

In mathematical terms, given:

- a partially populated `data tree`, and
- `a set <pipeline>` of `functions operating <operation>` (consuming/producing) on
  `branches <dependency>` of the data tree,

*graphtik* `collects <compile>` a `subset of functions in a graph <plan>`
that when `executed <execute>` consume & produce as `values as possible in the data-tree
<solution>`.

|usage-overview|

.. math-overview-end

- Its primary use case is building flexible algorithms for data science/machine learning projects.
- It should be extendable to implement the following:

  - an `IoC dependency resolver <https://en.wikipedia.org/wiki/Dependency_injection>`_
    (e.g. Java Spring, Google Guice);
  - an executor of interdependent tasks based on files (e.g. GNU Make);
  - a custom ETL engine;
  - a spreadsheet calculation engine.

Graph\ **tik** `sprang <https://docs.google.com/spreadsheets/d/1HPgtg2l6v3uDS81hLOcFOZxIBLCnHGrcFOh3pFRIDio/edit#gid=0>`_
from `Graphkit`_ (summer 2019, v1.2.2) to :gh:`experiment <22>` with Python 3.6+ features,
but has diverged significantly with enhancements ever since.

.. raw:: html

   <details>
   <summary><a>Table of Contents</a></summary>

.. toctree::
   :maxdepth: 4
   :numbered: 1

   operations
   pipelines
   plotting
   arch
   reference
   Changes <changes>
   genindex

.. raw:: html

   </details>


.. _features:

.. include:: ../../README.rst
   :start-after:  .. _features:
   :end-before:  Quick start

.. _quick-start:

Quick start
-----------
Here's how to install::

   pip install graphtik

OR with dependencies for plotting support (and you need to install `Graphviz`_ program
separately with your OS tools)::

   pip install graphtik[plot]


Let's build a *graphtik* computation `pipeline` that produces the following
x3 `outputs` out of x2 `inputs` (``α`` and ``β``):

.. math::
   :label: sample-formula

   α \times β

   α - α \times β

   |α - α \times β| ^ 3

..

   >>> from graphtik import compose, operation
   >>> from operator import mul, sub

   >>> @operation(name="abs qubed",
   ...            needs=["α-α×β"],
   ...            provides=["|α-α×β|³"])
   ... def abs_qubed(a):
   ...    return abs(a) ** 3

.. hint::
   Notice that *graphtik* has not problem working in unicode chars
   for `dependency` names.

Compose the ``abspow`` function along with ``mul`` & ``sub``  built-ins
into a computation `graph`:

   >>> graphop = compose("graphop",
   ...    operation(mul, needs=["α", "β"], provides=["α×β"]),
   ...    operation(sub, needs=["α", "α×β"], provides=["α-α×β"]),
   ...    abs_qubed,
   ... )
   >>> graphop
   Pipeline('graphop', needs=['α', 'β', 'α×β', 'α-α×β'],
            provides=['α×β', 'α-α×β', '|α-α×β|³'],
            x3 ops: mul, sub, abs qubed)

You may plot the function graph in a file like this
(if in *jupyter*, no need to specify the file, see :ref:`jupyter_rendering`):

   >>> graphop.plot('graphop.svg')      # doctest: +SKIP

.. graphtik::

As you can see, any function can be used as an operation in Graphtik,
even ones imported from system modules.

Run the graph-operation and request all of the outputs:

   >>> sol = graphop(**{'α': 2, 'β': 5})
   >>> sol
   {'α': 2, 'β': 5, 'α×β': 10, 'α-α×β': -8, '|α-α×β|³': 512}

`Solutions <solution>` are `plottable` as well:

   >>> solution.plot('solution.svg')      # doctest: +SKIP

.. graphtik::

Run the graph-operation and request a subset of the outputs:

   >>> solution = graphop.compute({'α': 2, 'β': 5}, outputs=["α-α×β"])
   >>> solution
   {'α-α×β': -8}

.. graphtik::

... where the (interactive) legend is this:

.. graphtik::
   :width: 65%
   :name: legend

   >>> from graphtik.plot import legend
   >>> l = legend()

.. default-role:: obj
.. |sample-plot| raw:: html
    :file:  images/sample.svg
.. |usage-overview| image:: images/GraphkitUsageOverview.svg
    :alt: Usage overview of graphtik library
    :width: 640px
    :align: middle
.. from https://docs.google.com/document/d/1P73jgcAEzR_Vw491DQR0zogdunJOj3qh0h_lvphdaHk
.. include:: ../../README.rst
    :start-after: _badges_substs: