Graphtik
========

|release|, |today| |gh-version| |pypi-version| |python-ver|
|dev-status| |ci-status| |doc-status| |cover-status|
|codestyle| |proj-lic|

|gh-watch| |gh-star| |gh-fork| |gh-issues|

.. epigraph::

    It's a DAG all the way down!

    |sample-plot|

Computation graphs for Python & Pandas
--------------------------------------

**Graphtik** is a library to compose, solve, execute & plot *graphs of python functions*
(a.k.a pipelines) that consume and populate named data
(a.k.a dependencies), whose names may be nested (such as. *pandas* dataframe columns),
based on whether values for those dependencies exist in the inputs or
have been calculated earlier.

In mathematical terms, given:

- a partially populated data-tree, and
- a set of functions operating on (consuming/producing) branches of the data tree,

*graphtik* collects a subset of functions in a graph that when executed
consume & produce as many values as possible in the data-tree.

|usage-overview|

- Its primary use case is building flexible algorithms for data science/machine learning projects.
- It should be extendable to implement the following:

  - an `IoC dependency resolver <https://en.wikipedia.org/wiki/Dependency_injection>`_
    (e.g. Java Spring, Google Guice);
  - an executor of interdependent tasks based on files (e.g. GNU Make);
  - a custom ETL engine;
  - a spreadsheet calculation engine.

*Graphtik* `sprang <https://docs.google.com/spreadsheets/d/1HPgtg2l6v3uDS81hLOcFOZxIBLCnHGrcFOh3pFRIDio/edit#gid=0>`_
from `Graphkit`_ (summer 2019, v1.2.2) to `experiment
<https://github.com/yahoo/graphkit/issues/>`_ with Python 3.6+ features,
but has diverged significantly with enhancements ever since.

.. _features:

Features
--------

- Deterministic pre-decided `execution plan` (unless *partial-outputs* or
  *endured operations* defined, see below).
- Can assemble existing functions without modifications into `pipeline`\s.
- `dependency` resolution can bypass calculation cycles based on data given and asked.
- Support functions with `optional <optionals>` input args and/or `varargs <varargish>`.
- Support functions with `partial outputs`; keep working even if certain `endured` operations fail.
- Facilitate trivial `conveyor operation`\s and `alias` on `provides`.
- Support cycles, by annotating repeated updates of `dependency` values as `tokens`
  or `sideffected` (e.g. to add columns into :class:`pandas.DataFrame`\s).
- `Hierarchical dependencies <subdoc>` may access data values deep in `solution`
  with `json pointer path` expressions.
- Hierarchical dependencies annotated as `implicit` imply which subdoc dependency
  the function reads or writes in the parent-doc.
- `Merge <operation merging>` or `nest <operation nesting>` sub-pipelines.
- Early `eviction` of intermediate results from `solution`, to optimize memory footprint.
- Solution tracks all intermediate `overwritten <overwrite>` values for the same dependency.
- Elaborate `Graphviz`_ plotting with configurable `plot theme`\s.
- Integration with Sphinx sites with the new :rst:dir:`graphtik` directive.
- Authored with :ref:`debugging <debugging>` in mind.
- Parallel execution (but underdeveloped & DEPRECATED).

Anti-features
^^^^^^^^^^^^^

- It's not meant to follow complex conditional logic based on `dependency` values
  (though it does support that to `a limited degree <partial outputs>`).

- It's not an orchestrator for long-running tasks, nor a calendar scheduler -
  `Apache Airflow <https://airflow.apache.org/>`_, `Dagster
  <https://github.com/dagster-io/dagster>`_ or `Luigi <https://luigi.readthedocs.io/>`_
  may help for that.

- It's not really a parallelizing optimizer, neither a map-reduce framework - look
  additionally at `Dask <https://dask.org/>`_, `IpyParallel
  <https://ipyparallel.readthedocs.io/en/latest/>`_, `Celery
  <https://docs.celeryproject.org/en/stable/getting-started/introduction.html>`_,
  Hive, Pig, Hadoop, etc.

- It's not a stream/batch processor, like Spark, Storm, Fink, Kinesis,
  because it pertains function-call semantics, calling only once each function
  to process data-items.

Differences with *schedula*
%%%%%%%%%%%%%%%%%%%%%%%%%%%

`schedula <https://schedula.readthedocs.io/>`_ is a powerful library written roughly
for the same purpose, and works differently along these lines
(ie features below refer to *schedula*):

- terminology (<graphtik> := <schedula>):

  - pipeline := dispatcher
  - plan := workflow
  - solution := solution

- Dijkstra planning runs while calling operations:

  - Powerful & flexible (ie all operations are dynamic, *domains* are possible, etc).
  - Supports *weights*.
  - Cannot pre-calculate & cache execution plans (slow).

- Calculated values are stored inside a graph (mimicking the structure of the functions):

  - graph visualizations absolutely needed to inspect & debug its solutions.
  - graphs imply complex pre/post processing & traversal algos
      (vs constructing/traversing data-trees).

- Reactive plotted diagrams, web-server runs behind the scenes.
- Operation graphs are stackable:

  - plotted nested-graphs support drill-down.
  - *graphtik* emulates that with data/operation names (`operation nesting`),
    but always a unified graph is solved at once,
    bc it is impossible to dress *nesting-funcs* as a *python-funcs* and pre-solve plan
    (*schedula* does not pre-solve plan, Dijkstra runs all the time).
    See TODO about plotting such nested graphs.

- *Schedula* does not calculate all possible values (ie no `overwrite`\s).
- *Schedula* computes precedence based on weights and lexicographical order of function name.

  - Re-inserting operation does not overrides its current function - must remove it first.
  - *graphtik* precedence based insertion order during `composition`.

- Virtual *start* and *end* data-nodes needed for Dijkstra to solve the dag.
- No domains (execute-time conditionals deciding whether a function must run).
- Probably :ref:`recompute` is more straightforward in *graphtik*.
- TODO: more differences with *schedula* exist.

Quick start
-----------
Here’s how to install:

::

   pip install graphtik

OR with various "extras" dependencies, such as, for plotting::

   pip install graphtik[plot]

. Tip::
    Supported extras:

    **plot**
        for plotting with `Graphviz`_,
    **matplot**
        for plotting in *maplotlib* windows
    **sphinx**
        for embedding plots in *sphinx*\-generated sites,
    **test**
        for running *pytest*\s,
    **dill**
        may help for pickling `parallel` tasks - see `marshalling` term
        and ``set_marshal_tasks()`` configuration.
    **all**
        all of the above, plus development libraries, eg *black* formatter.
    **dev**
        like *all*

Let's build a *graphtik* computation graph that produces x3 outputs
out of 2 inputs `α` and `β`:

- `α x β`
- `α - αxβ`
- `|α - αxβ| ^ 3`

..

>>> from graphtik import compose, operation
>>> from operator import mul, sub

>>> @operation(name="abs qubed",
...            needs=["α-α×β"],
...            provides=["|α-α×β|³"])
... def abs_qubed(a):
...     return abs(a) ** 3

Compose the ``abs_qubed`` function along the ``mul`` & ``sub``  built-ins
into a computation graph:

>>> graphop = compose("graphop",
...     operation(needs=["α", "β"], provides=["α×β"])(mul),
...     operation(needs=["α", "α×β"], provides=["α-α×β"])(sub),
...     abs_qubed,
... )
>>> graphop
Pipeline('graphop', needs=['α', 'β', 'α×β', 'α-α×β'],
                    provides=['α×β', 'α-α×β', '|α-α×β|³'],
                    x3 ops: mul, sub, abs qubed)

Run the graph and request all of the outputs
(notice that unicode characters work also as Python identifiers):

>>> graphop(α=2, β=5)
{'α': 2, 'β': 5, 'α×β': 10, 'α-α×β': -8, '|α-α×β|³': 512}

... or request a subset of outputs:

>>> solution = graphop.compute({'α': 2, 'β': 5}, outputs=["α-α×β"])
>>> solution
{'α-α×β': -8}

... and plot the results (if in *jupyter*, no need to create the file):

>>> solution.plot('executed_3ops.svg')  # doctest: +SKIP

|sample-sol|

|plot-legend|

.. |sample-plot| image:: docs/source/images/sample.svg
    :alt: sample graphtik plot
    :width: 380px
    :align: middle
.. |usage-overview| image:: docs/source/images/GraphkitUsageOverview.svg
    :alt: Usage overview of graphtik library
    :width: 640px
    :align: middle
.. |sample-sol| image:: docs/source/images/executed_3ops.svg
    :alt: sample graphtik plot
    :width: 380px
    :align: middle
.. |plot-legend| image:: docs/source/images/GraphtikLegend.svg
    :alt: graphtik legend
    :align: middle


.. _Graphkit: https://github.com/yahoo/graphkit
.. _Graphviz: https://graphviz.org
.. _badges_substs:

.. |ci-status| image:: https://github.com/pygraphkit/graphtik/actions/workflows/ci.yaml/badge.svg
    :alt: GitHub Actions CI testing ok? (Linux)
    :target: https://github.com/pygraphkit/graphtik/actions

.. |doc-status| image:: https://img.shields.io/readthedocs/graphtik?branch=master&logo=read-the-docs
    :alt: ReadTheDocs ok?
    :target: https://graphtik.readthedocs.org

.. |cover-status| image:: https://codecov.io/gh/pygraphkit/graphtik/branch/master/graph/badge.svg?token=Qr8LHNkSXK
    :target: https://codecov.io/gh/pygraphkit/graphtik

.. |gh-version| image:: https://img.shields.io/github/v/release/pygraphkit/graphtik?label=GitHub%20release&include_prereleases&logo=github
    :target: https://github.com/pygraphkit/graphtik/releases
    :alt: Latest release in GitHub

.. |pypi-version| image::  https://img.shields.io/pypi/v/graphtik?label=PyPi%20release&logo=pypi
    :target: https://pypi.python.org/pypi/graphtik/
    :alt: Latest version in PyPI

.. |python-ver| image:: https://img.shields.io/pypi/pyversions/graphtik?label=Python&logo=pypi
    :target: https://pypi.python.org/pypi/graphtik/
    :alt: Supported Python versions of latest release in PyPi

.. |dev-status| image:: https://img.shields.io/pypi/status/graphtik
    :target: https://pypi.python.org/pypi/graphtik/
    :alt: Development Status

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-black
    :target: https://github.com/ambv/black
    :alt: Code Style

.. |gh-watch| image:: https://img.shields.io/github/watchers/pygraphkit/graphtik?style=social
    :target: https://github.com/pygraphkit/graphtik
    :alt: Github watchers

.. |gh-star| image:: https://img.shields.io/github/stars/pygraphkit/graphtik?style=social
    :target: https://github.com/pygraphkit/graphtik
    :alt: Github stargazers

.. |gh-fork| image:: https://img.shields.io/github/forks/pygraphkit/graphtik?style=social
    :target: https://github.com/pygraphkit/graphtik
    :alt: Github forks

.. |gh-issues| image:: http://img.shields.io/github/issues/pygraphkit/graphtik?style=social
    :target: https://github.com/pygraphkit/graphtik/issues
    :alt: Issues count

.. |proj-lic| image:: https://img.shields.io/pypi/l/graphtik
    :target:  https://www.apache.org/licenses/LICENSE-2.0
    :alt: Apache License, version 2.0
