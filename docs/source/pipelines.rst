.. _graph-composition:

Pipelines
=========

Graphtik's :func:`.operation.compose()` factory handles the work of tying together ``operation``
instances into a runnable computation graph.

The simplest use case is to assemble a collection of individual operations
into a runnable computation graph.
The example script from :ref:`quick-start` illustrates this well:

    >>> from operator import mul, sub
    >>> from functools import partial
    >>> from graphtik import compose, operation

    >>> def abspow(a, p):
    ...    """Computes |a|^p. """
    ...    c = abs(a) ** p
    ...    return c


The call here to ``compose()`` yields a runnable computation graph that looks like this
(where the circles are operations, squares are data, and octagons are parameters):

    >>> # Compose the mul, sub, and abspow operations into a computation graph.
    >>> graphop = compose("graphop",
    ...    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
    ...    operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
    ...    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"])
    ...    (partial(abspow, p=3))
    ... )

This yields a graph which looks like this (see :ref:`plotting`):

.. graphtik::
    :name: graphop

    >>> graphop.plot('calc_power.svg')  # doctest: +SKIP


.. _graph-computations:

Running a pipeline
------------------

The graph composed above can be run by simply calling it with a dictionary of values
with keys corresponding to the named :term:`dependencies <dependency>` (`needs` &
`provides`):

    >>> # Run the graph and request all of the outputs.
    >>> out = graphop(a=2, b=5)
    >>> out
    {'a': 2, 'b': 5, 'ab': 10, 'a_minus_ab': -8, 'abs_a_minus_ab_cubed': 512}

You may plot the solution:

.. graphtik::
    :caption: the solution of the graph
    :graphvar: out

    >>> out.plot('a_solution.svg')  # doctest: +SKIP


Producing a subset of outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, calling a graph-operation on a set of inputs will yield all of
that graph's :term:`outputs`.
You can use the ``outputs`` parameter to request only a subset.
For example, if ``graphop`` is as above:

    >>> # Run the graph-operation and request a subset of the outputs.
    >>> out = graphop.compute({'a': 2, 'b': 5}, outputs="a_minus_ab")
    >>> out
    {'a_minus_ab': -8}

.. graphtik::

When asking a subset of the graph's `outputs`, Graphtik does 2 things:

- it :term:`prune`\s any :term:`operation`\s that are not on the path from
  given :term:`inputs` to the requested `outputs` (e.g. the ``abspow1`` operation, above,
  is not executed);
- it :term:`evicts <evictions>` any intermediate data from :term:`solution` as soon as
  they are not needed.

You may see (2) in action by including the sequence of :term:`execution steps`
into the plot:

.. graphtik::

    >>> dot = out.plot(theme={"include_steps": True})

.. tip:
   Read :ref:`plot-customizations` to understand the trick with the :term:`plotter`.



Short-circuiting a pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can short-circuit a graph computation, making certain inputs unnecessary,
by providing a value in the graph that is further downstream in the graph than those inputs.
For example, in the graph-operation we've been working with, you could provide
the value of ``a_minus_ab`` to make the inputs ``a`` and ``b`` unnecessary:

    >>> # Run the graph-operation and request a subset of the outputs.
    >>> out = graphop(a_minus_ab=-8)
    >>> out
    {'a_minus_ab': -8, 'abs_a_minus_ab_cubed': 512}

.. graphtik::

When you do this, any ``operation`` nodes that are not on a path from the downstream input
to the requested outputs (i.e. predecessors of the downstream input) are not computed.
For example, the ``mul1`` and ``sub1`` operations are not executed here.

This can be useful if you have a graph-operation that accepts alternative forms of the same input.
For example, if your graph-operation requires a ``PIL.Image`` as input, you could
allow your graph to be run in an API server by adding an earlier ``operation``
that accepts as input a string of raw image data and converts that data into the needed ``PIL.Image``.
Then, you can either provide the raw image data string as input, or you can provide
the ``PIL.Image`` if you have it and skip providing the image data string.


Extending pipelines
-------------------
Sometimes we have existing computation graph(s) to which we want to append
operations or other pipelines.

Combining
^^^^^^^^^
This is simple, since ``compose()`` can *combine* whole pipelines along with
individual operations and pipelines.

For example, if we have the above ``graph``, we can add another operation to it
and create a new graph:

    >>> # Add another subtraction operation to the graph.
    >>> bigger_graph = compose("bigger_graph",
    ...    graphop,
    ...    operation(name="sub2", needs=["a_minus_ab", "c"], provides="a_minus_ab_minus_c")(sub)
    ... )

.. graphtik::
    :graphvar: bigger_graph

Notice that the original pipeline is preserved intact in an "isolated" cluster,
and its operations have been prefixed by the name of that pipeline.

Run the graph and print the output:

    >>> sol = bigger_graph.compute({'a': 2, 'b': 5, 'c': 5},
    ...                            outputs=["a_minus_ab_minus_c"])
    >>> sol
    {'a_minus_ab_minus_c': -13}
    >>> dot = sol.plot(clusters=True)

.. graphtik::

.. tip::
    We had to plot with ``clusters=True`` so that we prevent the :term:`plan`
    to insert the "after pruning" cluster (see :attr:`.PlotArgs.clusters`).


Merging
^^^^^^^

Sometimes we have computation graphs -- perhaps ones that share operations --
and we want to *merge* them into one.

This is doable with ``compose(..., merge=True))``.
Any identically-named operations are consolidate into a single node, where
the operation added later in the call (further to the right) wins.

For example, let's say we have ``graphop``, as in the examples above, along with this graph:

    >>> another_graph = compose("another_graph",
    ...    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
    ...    operation(name="mul2", needs=["c", "ab"], provides=["cab"])(mul)
    ... )
    >>> another_graph
    NetworkOperation('another_graph', needs=['a', 'b', 'c', 'ab'], provides=['ab', 'cab'], x2 ops: mul1, mul2)

.. graphtik::
    :name: another_graph

We can merge :graphtik:`graphop` and :graphtik:`another_graph` like so, avoiding a redundant ``mul1`` operation:

.. Note::

    The *names* of the graphs must differ.

..

    >>> merged_graph = compose("merged_graph", graphop, another_graph, merge=True)
    >>> print(merged_graph)
    NetworkOperation('merged_graph',
                      needs=['a', 'b', 'ab', 'a_minus_ab', 'c'],
                      provides=['ab', 'a_minus_ab', 'abs_a_minus_ab_cubed', 'cab'],
                      x4 ops:  mul1, sub1, abspow1, mul2)

.. graphtik::

As always, we can run computations with this graph by simply calling it:

    >>> sol = merged_graph.compute({"a": 2, "b": 5, "c": 5}, outputs=["cab"])
    >>> sol
    {'cab': 50}

.. graphtik::

.. seealso::
    Consult these test-cases from the full sources of the project:

    - :file:`test/test_graphtik.py:test_network_simple_merge()`
    - :file:`test/test_graphtik.py:test_network_deep_merge()`


Advanced pipelines
------------------
.. _endured:

Depending on sideffects
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: graphtik.modifiers.sideffect
   :noindex:

Modifying existing values in solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: graphtik.modifiers.sideffected
   :noindex:

Resilience when operations fail (*endurance*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible for a pipeline to persist executing operations, even if some of them
are raising errors, if they are marked as :term:`endured`:

    >>> @operation(endured=1, provides=["space", "time"])
    ... def get_out():
    ...     raise ValueError("Quarantined!")
    >>> get_out
    FunctionalOperation!(name='get_out', needs=[], provides=['space', 'time'], fn='get_out')

Notice the exclamation(``!``) before the parenthesis in the string representation
of the operation.

    >>> @operation(needs="space", provides="fun")
    ... def exercise(where):
    ...     return "refreshed"
    >>> @operation(endured=1, provides="time")
    ... def stay_home():
    ...     return "1h"
    >>> @operation(needs="time", provides="fun")
    ... def read_book(for_how_long):
    ...     return "relaxed"

    >>> netop = compose("covid19", get_out, stay_home, exercise, read_book)
    >>> netop
    NetworkOperation('covid19',
                     needs=['space', 'time'],
                     provides=['space', 'time', 'fun'],
                     x4 ops: get_out, stay_home, exercise, read_book)

.. graphtik::

Notice the thick outlines of the endured (or :term:`reschedule`\d, see below) operations.

When executed, the pipeline produced :term:`outputs`, although one of its operations
has failed:

    >>> sol = netop()
    >>> sol
    {'time': '1h', 'fun': 'relaxed'}

.. graphtik::

You may still abort on failures, later, by raising an appropriate exception from
:class:`.Solution`:

    >>> sol.scream_if_incomplete()
    Traceback (most recent call last):
    ...
    graphtik.network.IncompleteExecutionError:
    Not completed x2 operations ['exercise', 'get_out'] due to x1 failures and x0 partial-ops:
      +--get_out: ValueError('Quarantined!')


.. _rescheduled:

Operations with partial outputs (*rescheduled*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In case the actually produce `outputs` depend on some condition in the `inputs`,
the `solution` has to :term:`reschedule` the plan amidst execution, and consider the
actual `provides` delivered:


    >>> @operation(rescheduled=1,
    ...            needs="quarantine",
    ...            provides=["space", "time"],
    ...            returns_dict=True)
    ... def get_out_or_stay_home(quarantine):
    ...     if quarantine:
    ...          return {"time": "1h"}
    ...     else:
    ...          return {"space": "around the block"}
    >>> get_out_or_stay_home
    FunctionalOperation?(name='get_out_or_stay_home',
                         needs=['quarantine'],
                         provides=['space', 'time'],
                         fn{}='get_out_or_stay_home')
    >>> @operation(needs="space", provides=["fun", "body"])
    ... def exercise(where):
    ...     return "refreshed", "strong feet"
    >>> @operation(needs="time", provides=["fun", "brain"])
    ... def read_book(for_how_long):
    ...     return "relaxed", "popular physics"

    >>> netop = compose("covid19", get_out_or_stay_home, exercise, read_book)

Depending on "quarantine' state we get to execute different part of the pipeline:

.. graphtik::

    >>> sol = netop(quarantine=True)

.. graphtik::

    >>> sol = netop(quarantine=False)


In both case, a warning gets raised about the missing outputs, but the execution
proceeds regularly to what it is possible to evaluate.  You may collect a report of
what has been canceled using this:

    >>> print(sol.check_if_incomplete())
    Not completed x1 operations ['read_book'] due to x0 failures and x1 partial-ops:
      +--get_out_or_stay_home: ['time']

In case you wish to cancel the output of a single-result operation,
return the special value :data:`graphtik.NO_RESULT`.