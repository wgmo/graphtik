# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import networkx as nx
from boltons.setutils import IndexedSet as iset

from .base import Operation, Plotter, aslist, jetsam
from .modifiers import optional, sideffect

try:
    from collections import abc
except ImportError:
    import collections as abc


class FunctionalOperation(Operation):
    """Use operation() to build instances of this class instead"""

    def __init__(self, fn=None, **kwargs):
        self.fn = fn
        Operation.__init__(self, **kwargs)
        self._validate()

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        needs = aslist(self.needs)
        provides = aslist(self.provides)
        fn_name = self.fn and getattr(self.fn, "__name__", str(self.fn))
        return f"FunctionalOperation(name={self.name!r}, needs={needs!r}, provides={provides!r}, fn={fn_name!r})"

    def _validate(self):
        super()._validate()
        if not self.fn or not callable(self.fn):
            raise ValueError(f"Operation was not provided with a callable: {self.fn}")

    def compute(self, named_inputs, outputs=None):
        try:
            args = [
                named_inputs[n]
                for n in self.needs
                if not isinstance(n, optional) and not isinstance(n, sideffect)
            ]

            # Find any optional inputs in named_inputs.  Get only the ones that
            # are present there, no extra `None`s.
            optionals = {
                n: named_inputs[n]
                for n in self.needs
                if isinstance(n, optional) and n in named_inputs
            }

            # Don't expect sideffect outputs.
            provides = [n for n in self.provides if not isinstance(n, sideffect)]

            results = self.fn(*args, **optionals)

            if not provides:
                # All outputs were sideffects.
                return {}

            if len(provides) == 1:
                results = [results]

            results = zip(provides, results)
            if outputs:
                outputs = set(n for n in outputs if not isinstance(n, sideffect))
                results = filter(lambda x: x[0] in outputs, results)

            return dict(results)
        except Exception as ex:
            jetsam(
                ex,
                locals(),
                "outputs",
                "provides",
                "results",
                operation="self",
                args=lambda locs: {
                    "args": locs.get("args"),
                    "kwargs": locs.get("optionals"),
                },
            )

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class operation:
    """
    A builder for graph-operations wrapping functions.

    :param function fn:
        The function used by this operation.  This does not need to be
        specified when the operation object is instantiated and can instead
        be set via ``__call__`` later.
    :param str name:
        The name of the operation in the computation graph.
    :param list needs:
        Names of input data objects this operation requires.  These should
        correspond to the ``args`` of ``fn``.
    :param list provides:
        Names of output data objects this operation provides.

    :return:
        when called, it returns a :class:`FunctionalOperation`

    **Example:**
    
    Here it is an example of it use with the "builder pattern"::

        >>> from graphtik import operation

        >>> opb = operation(name='add_op')
        >>> opb.withset(needs=['a', 'b'])
        operation(name='add_op', needs=['a', 'b'], provides=None, fn=None)
        >>> opb.withset(provides='SUM', fn=sum)
        operation(name='add_op', needs=['a', 'b'], provides='SUM', fn='sum')
    
    You may keep calling ``withset()`` till you invoke ``__call__()`` on the builder;
    then you get te actual :class:`Operation` instance::

        >>> # Create `Operation` and overwrite function at the last moment.
        >>> opb(sum)
        FunctionalOperation(name='add_op', needs=['a', 'b'], provides=['SUM'], fn='sum')
    """

    fn = name = needs = provides = None

    def __init__(self, fn=None, *, name=None, needs=None, provides=None):
        self.withset(fn=fn, name=name, needs=needs, provides=provides)

    def withset(self, *, fn=None, name=None, needs=None, provides=None):
        if fn is not None:
            self.fn = fn
        if name is not None:
            self.name = name
        if needs is not None:
            self.needs = needs
        if provides is not None:
            self.provides = provides

        return self

    def __call__(self, fn=None, *, name=None, needs=None, provides=None):
        """
        This enables ``operation`` to act as a decorator or as a functional
        operation, for example::

            @operator(name='myadd1', needs=['a', 'b'], provides=['c'])
            def myadd(a, b):
                return a + b

        or::

            def myadd(a, b):
                return a + b
            operator(name='myadd1', needs=['a', 'b'], provides=['c'])(myadd)

        :param function fn:
            The function to be used by this ``operation``.

        :return:
            Returns an operation class that can be called as a function or
            composed into a computation graph.
        """

        self.withset(fn=fn, name=name, needs=needs, provides=provides)

        return FunctionalOperation(**vars(self))

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        needs = aslist(self.needs)
        provides = aslist(self.provides)
        fn_name = self.fn and getattr(self.fn, "__name__", str(self.fn))
        return f"operation(name={self.name!r}, needs={needs!r}, provides={provides!r}, fn={fn_name!r})"


class NetworkOperation(Operation, Plotter):
    #: The execution_plan of the last call to compute(), cached as debugging aid.
    last_plan = None
    #: set execution mode to single-threaded sequential by default
    method = "sequential"
    overwrites_collector = None

    def __init__(self, net, method="sequential", overwrites_collector=None, **kwargs):
        """
        :param method:
            if ``"parallel"``, launches multi-threading.
            Set when invoking a composed graph or by
            :meth:`~NetworkOperation.set_execution_method()`.

        :param overwrites_collector:
            (optional) a mutable dict to be fillwed with named values.
            If missing, values are simply discarded.

        """
        self.net = net
        Operation.__init__(self, **kwargs)
        self.set_execution_method(method)
        self.set_overwrites_collector(overwrites_collector)

    def _build_pydot(self, **kws):
        """delegate to network"""
        kws.setdefault("title", self.name)
        plotter = self.last_plan or self.net
        return plotter._build_pydot(**kws)

    def compute(self, named_inputs, outputs=None):
        """
        Solve & execute the graph, sequentially or parallel.

        :param dict named_inputs:
            A maping of names --> values that must contain at least
            the compulsory inputs that were specified when the plan was built
            (but cannot enforce that!).
            Cloned, not modified.

        :param outputs:
            a string or a list of strings with all data asked to compute.
            If you set this variable to ``None``, all data nodes will be kept
            and returned at runtime.

        :returns: a dictionary of output data objects, keyed by name.
        """
        try:
            if isinstance(outputs, str):
                outputs = [outputs]
            elif not isinstance(outputs, (list, tuple)) and outputs is not None:
                raise ValueError(
                    "The outputs argument must be a list or None, was: %s", outputs
                )

            net = self.net

            # Build the execution plan.
            self.last_plan = plan = net.compile(named_inputs.keys(), outputs)

            solution = plan.execute(
                named_inputs, self.overwrites_collector, self.execution_method
            )

            return solution
        except Exception as ex:
            jetsam(ex, locals(), "plan", "solution", "outputs", network="net")

    def __call__(
        self, named_inputs, outputs=None, method=None, overwrites_collector=None
    ):
        return self.compute(named_inputs, outputs=outputs)

    def set_execution_method(self, method):
        """
        Determine how the network will be executed.

        :param str method:
            If "parallel", execute graph operations concurrently
            using a threadpool.
        """
        choices = ["parallel", "sequential"]
        if method not in choices:
            raise ValueError(
                "Invalid computation method %r!  Must be one of %s" % (method, choices)
            )
        self.execution_method = method

    def set_overwrites_collector(self, collector):
        """
        Asks to put all *overwrites* into the `collector` after computing

        An "overwrites" is intermediate value calculated but NOT stored
        into the results, becaues it has been given also as an intemediate
        input value, and the operation that would overwrite it MUST run for
        its other results.

        :param collector:
            a mutable dict to be fillwed with named values
        """
        if collector is not None and not isinstance(collector, abc.MutableMapping):
            raise ValueError(
                "Overwrites collector was not a MutableMapping, but: %r" % collector
            )
        self.overwrites_collector = collector


class compose(object):
    """
    This is a simple class that's used to compose ``operation`` instances into
    a computation graph.

    :param str name:
        A name for the graph being composed by this object.

    :param bool merge:
        If ``True``, this compose object will attempt to merge together
        ``operation`` instances that represent entire computation graphs.
        Specifically, if one of the ``operation`` instances passed to this
        ``compose`` object is itself a graph operation created by an
        earlier use of ``compose`` the sub-operations in that graph are
        compared against other operations passed to this ``compose``
        instance (as well as the sub-operations of other graphs passed to
        this ``compose`` instance).  If any two operations are the same
        (based on name), then that operation is computed only once, instead
        of multiple times (one for each time the operation appears).
    """

    def __init__(self, name=None, merge=False):
        assert name, "compose needs a name"
        self.name = name
        self.merge = merge

    def __call__(self, *operations, **kwargs):
        """
        Composes a collection of operations into a single computation graph,
        obeying the ``merge`` property, if set in the constructor.

        :param operations:
            Each argument should be an operation instance created using
            ``operation``.

        :return:
            Returns a special type of operation class, which represents an
            entire computation graph as a single operation.
        """
        assert len(operations), "no operations provided to compose"

        # If merge is desired, deduplicate operations before building network
        if self.merge:
            merge_set = iset()  # Preseve given node order.
            for op in operations:
                if isinstance(op, NetworkOperation):
                    netop_nodes = nx.topological_sort(op.net.graph)
                    merge_set.update(s for s in netop_nodes if isinstance(s, Operation))
                else:
                    merge_set.add(op)
            operations = merge_set

        provides = iset(p for op in operations for p in op.provides)
        # Mark them all as optional, now that #18 calmly ignores
        # non-fully satisfied operations.
        needs = iset(optional(n) for op in operations for n in op.needs) - provides

        # Build network
        from .network import Network

        net = Network()
        for op in operations:
            net.add_op(op)

        return NetworkOperation(
            net, name=self.name, needs=needs, provides=provides, **kwargs
        )
