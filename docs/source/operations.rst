Operations
==========

At a high level, an :term:`operation` is a function in a computation :term:`pipeline`,
abstractly represented by the :class:`.Operation` class.
This class specifies the :term:`dependencies <dependency>` of the *operation*
in the *pipeline*.

The :class:`.FunctionalOperation` provides a concrete lightweight wrapper
around any arbitrary function to define those *dependencies*.
Instead of constructing it directly, prefer to instantiate it by calling
the :func:`.operation()` factory.


Operations are just functions
------------------------------

At the heart of each `operation` is just a function, any arbitrary function.
Indeed, you can wrap an existing function into an `operation`, and then call it
just like the original function, e.g.:

   >>> from operator import add
   >>> from graphtik import operation

   >>> add_op = operation(add,
   ...                    needs=['a', 'b'],
   ...                    provides=['a_plus_b'])
   >>> add_op
   FunctionalOperation(name='add', needs=['a', 'b'], provides=['a_plus_b'], fn='add')

   >>> add_op(3, 4) == add(3, 4)
   True

   But ``__call__()`` is just to facilitate quick experimentation - it does not
   perform any checks or matching of *needs*/*provides* to function arguments
   & results (which happen when :term:`pipeline`\s :term:`compute`).

   The way Graphtik works is by invoking their :meth:`.Operation.compute()` method,
   which, among others, allow to specify what results you desire to receive back
   (read more on :ref:`graph-computations`).


Specifying graph structure: ``provides`` and ``needs``
------------------------------------------------------

Each :term:`operation` is a node in a computation :term:`graph`,
depending and supplying data from and to other nodes (via the :term:`solution`),
in order to :term:`compute`.

This graph structure is specified (mostly) via the ``provides`` and ``needs`` arguments
to the :func:`.operation` factory, specifically:

``needs``
   this argument names the list of (positionally ordered) :term:`inputs` data the `operation`
   requires to receive from *solution*.
   The list corresponds, roughly, to the arguments of the underlying function
   (plus any :term:`sideffects`).

   It can be a single string, in which case a 1-element iterable is assumed.

   :seealso: :term:`needs`, :term:`modifier`, :attr:`.FunctionalOperation.needs`,
      :attr:`.FunctionalOperation.op_needs`, :attr:`.FunctionalOperation._fn_needs`

``provides``
   this argument names the list of (positionally ordered) :term:`outputs` data
   the operation provides into the *solution*.
   The list corresponds, roughly, to the returned values of the `fn`
   (plus any :term:`sideffects` & :term:`alias`\es).

   It can be a single string, in which case a 1-element iterable is assumed.

   If they are more than one, the underlying function must return an iterable
   with same number of elements (unless it :term:`returns dictionary`).

   :seealso: :term:`provides`, :term:`modifier`, :attr:`.FunctionalOperation.provides`,
      :attr:`.FunctionalOperation.op_provides`, :attr:`.FunctionalOperation._fn_provides`

.. _aliases:

Aliased `provides`
^^^^^^^^^^^^^^^^^^
Sometimes, you need to interface operations where they name some :term:`dependency`
differently.
This is doable without introducing "pipe-through" interface operation, either by
annotating `needs` with :class:`.kw` `modifiers` (see docs) or with :term:`alias`\es
on the `provides` side:

   >>> op = operation(str,
   ...                name="`provides` with `aliases`",
   ...                needs="anything",
   ...                provides="real thing",
   ...                aliases=("real thing", "phony"))

.. graphtik::


Considerations for when building pipelines
------------------------------------------
When many operations are composed into a computation graph, Graphtik matches up
the values in their *needs* and *provides* to form the edges of that graph
(see :ref:`graph-composition` for more on that), like the operations from the script
in :ref:`quick-start`:

   >>> from operator import mul, sub
   >>> from functools import partial
   >>> from graphtik import compose, operation

   >>> def abspow(a, p):
   ...   """Compute |a|^p. """
   ...   c = abs(a) ** p
   ...   return c

   >>> # Compose the mul, sub, and abspow operations into a computation graph.
   >>> graphop = compose("graphop",
   ...    operation(mul, needs=["a", "b"], provides=["ab"]),
   ...    operation(sub, needs=["a", "ab"], provides=["a_minus_ab"]),
   ...    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"])
   ...    (partial(abspow, p=3))
   ... )
   >>> graphop
   NetworkOperation('graphop',
                    needs=['a', 'b', 'ab', 'a_minus_ab'],
                    provides=['ab', 'a_minus_ab', 'abs_a_minus_ab_cubed'],
                    x3 ops: mul, sub, abspow1)


- Notice that if ``name`` is not given, it is deduced from the function name.
- Notice the use of :func:`functools.partial()` to set parameter ``p`` to a constant value.
- And this is done by calling once more the returned "decorator* from :func:`operation()`,
  when called without a functions.

The ``needs`` and ``provides`` arguments to the operations in this script define
a computation graph that looks like this:

.. graphtik::
.. Tip::
  See :ref:`plotting` on how to make diagrams like this.

Builder pattern
^^^^^^^^^^^^^^^
There 2 ways to instantiate an :class:`.FunctionalOperation`\s, each one suitable
for different scenarios, and so far we have only seen the 1st one:

We've seen that calling manually :func:`.operation()` allows putting into a pipeline
functions that are defined elsewhere (e.g. in another module, or are system functions).

But that method is also useful if you want to create multiple operation instances
with similar attributes, e.g. ``needs``:

   >>> op_factory = operation(needs=['a'])

Notice that we specified a `fn`, in order to get back a :class:`.FunctionalOperation`
instance (and not a decorator).

   >>> from functools import partial

   >>> def mypow(a, p=2):
   ...    return a ** p

   >>> pow_op2 = op_factory.withset(fn=mypow, provides="^2")
   >>> pow_op3 = op_factory.withset(fn=partial(mypow, p=3), name='pow_3', provides='^3')
   >>> pow_op0 = op_factory.withset(fn=lambda a: 1, name='pow_0', provides='^0')

   >>> graphop = compose('powers', pow_op2, pow_op3, pow_op0)
   >>> graphop
   NetworkOperation('powers', needs=['a'], provides=['^2', '^3', '^0'], x3 ops:
      mypow, pow_3, pow_0)


   >>> graphop(a=2)
   {'a': 2, '^2': 4, '^3': 8, '^0': 1}

.. graphtik::


Decorator specification
^^^^^^^^^^^^^^^^^^^^^^^

If you are defining your computation graph and the functions that comprise it all in the same script,
the decorator specification of ``operation`` instances might be particularly useful,
as it allows you to assign computation graph structure to functions as they are defined.
Here's an example:

   >>> from graphtik import operation, compose

   >>> @operation(name='foo_op', needs=['a', 'b', 'c'], provides='foo')
   ... def foo(a, b, c):
   ...   return c * (a + b)

   >>> graphop = compose('foo_graph', foo)

.. graphtik::



Modifiers on `operation` `needs` and `provides`
-----------------------------------------------
Annotations on a `dependency` such as `optionals` & `sideffects` modify their behavior,
and eventually the :term:`pipeline`.

Read `mod:`.modifiers` for more.