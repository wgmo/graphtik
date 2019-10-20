Operations
==========

At a high level, an operation is a node in a computation graph.  Graphtik uses an ``operation`` class to represent these computations.

The ``operation`` class
-----------------------

The ``operation`` class specifies an operation in a computation graph, including its input data dependencies as well as the output data it provides.  It provides a lightweight wrapper around an arbitrary function to make these specifications.

There are many ways to instantiate an ``operation``, and we'll get into more detail on these later.  First off, though, here's the specification for the ``operation`` class:

.. autoclass:: graphtik.operation
   :members: __init__, __call__
   :member-order: bysource
   :special-members:


Operations are just functions
------------------------------

At the heart of each ``operation`` is just a function, any arbitrary function.  Indeed, you can instantiate an ``operation`` with a function and then call it just like the original function, e.g.::

   >>> from operator import add
   >>> from graphtik import operation

   >>> add_op = operation(name='add_op', needs=['a', 'b'], provides=['a_plus_b'])(add)

   >>> add_op(3, 4) == add(3, 4)
   True


Specifying graph structure: ``provides`` and ``needs``
------------------------------------------------------

Of course, each ``operation`` is more than just a function.  It is a node in a computation graph, depending on other nodes in the graph for input data and supplying output data that may be used by other nodes in the graph (or as a graph output).  This graph structure is specified via the ``provides`` and ``needs`` arguments to the ``operation`` constructor.  Specifically:

* ``provides``: this argument names the outputs (i.e. the returned values) of a given ``operation``.  If multiple outputs are specified by ``provides``, then the return value of the function comprising the ``operation`` must return an iterable.

* ``needs``: this argument names data that is needed as input by a given ``operation``.  Each piece of data named in needs may either be provided by another ``operation`` in the same graph (i.e. specified in the ``provides`` argument of that ``operation``), or it may be specified as a named input to a graph computation (more on graph computations :ref:`here <graph-computations>`).

When many operations are composed into a computation graph (see :ref:`graph-composition` for more on that), Graphtik matches up the values in their ``needs`` and ``provides`` to form the edges of that graph.

Let's look again at the operations from the script in :ref:`quick-start`, for example::

   >>> from operator import mul, sub
   >>> from functools import partial
   >>> from graphtik import compose, operation

   >>> # Computes |a|^p.
   >>> def abspow(a, p):
   ...   c = abs(a) ** p
   ...   return c

   >>> # Compose the mul, sub, and abspow operations into a computation graph.
   >>> graphop = compose(name="graphop")(
   ...    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
   ...    operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
   ...    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"])
   ...    (partial(abspow, p=3))
   ... )

.. Tip::
   Notice the use of :func:`functools.partial()` to set parameter ``p`` to a contant value.

The ``needs`` and ``provides`` arguments to the operations in this script define
a computation graph that looks like this (where the oval are *operations*,
squares/houses are *data*):

.. image:: images/barebone_3ops.svg

.. Tip::
  See :ref:`plotting` on how to make diagrams like this.


Instantiating operations
------------------------

There are several ways to instantiate an ``operation``, each of which might be more suitable for different scenarios.

Decorator specification
^^^^^^^^^^^^^^^^^^^^^^^

If you are defining your computation graph and the functions that comprise it all in the same script, the decorator specification of ``operation`` instances might be particularly useful, as it allows you to assign computation graph structure to functions as they are defined.  Here's an example::

   >>> from graphtik import operation, compose

   >>> @operation(name='foo_op', needs=['a', 'b', 'c'], provides='foo')
   ... def foo(a, b, c):
   ...   return c * (a + b)

   >>> graphop = compose(name='foo_graph')(foo)

Functional specification
^^^^^^^^^^^^^^^^^^^^^^^^

If the functions underlying your computation graph operations are defined elsewhere than the script in which your graph itself is defined (e.g. they are defined in another module, or they are system functions), you can use the functional specification of ``operation`` instances::

   >>> from operator import add, mul
   >>> from graphtik import operation, compose

   >>> add_op = operation(name='add_op', needs=['a', 'b'], provides='sum')(add)
   >>> mul_op = operation(name='mul_op', needs=['c', 'sum'], provides='product')(mul)

   >>> graphop = compose(name='add_mul_graph')(add_op, mul_op)

The functional specification is also useful if you want to create multiple ``operation``
instances from the same function, perhaps with different parameter values, e.g.::

   >>> from functools import partial

   >>> def mypow(a, p=2):
   ...    return a ** p

   >>> pow_op1 = operation(name='pow_op1', needs=['a'], provides='a_squared')(mypow)
   >>> pow_op2 = operation(name='pow_op2', needs=['a'], provides='a_cubed')(partial(mypow, p=3))

   >>> graphop = compose(name='two_pows_graph')(pow_op1, pow_op2)

A slightly different approach can be used here to accomplish the same effect
by creating an operation "builder pattern"::

   >>> def mypow(a, p=2):
   ...    return a ** p

   >>> pow_op_factory = operation(mypow, needs=['a'], provides='a_squared')

   >>> pow_op1 = pow_op_factory(name='pow_op1')
   >>> pow_op2 = pow_op_factory.withset(name='pow_op2', provides='a_cubed')(partial(mypow, p=3))
   >>> pow_op3 = pow_op_factory(lambda a: 1, name='pow_op0')

   >>> graphop = compose(name='two_pows_graph')(pow_op1, pow_op2, pow_op3)
   >>> graphop({'a': 2})
   {'a': 2, 'a_cubed': 8, 'a_squared': 4}

.. Note::
   You cannot call again the factory to overwrite the *function*,
   you have to use either the ``fn=`` keyword with ``withset()`` method or
   call once more.


Modifiers on ``operation`` inputs and outputs
---------------------------------------------

Certain modifiers are available to apply to input or output values in ``needs`` and ``provides``, for example to designate an optional input.  These modifiers are available in the ``graphtik.modifiers`` module:


Optionals
^^^^^^^^^
.. autoclass:: graphtik.modifiers.optional

Sideffects
^^^^^^^^^^
.. autoclass:: graphtik.modifiers.sideffect
