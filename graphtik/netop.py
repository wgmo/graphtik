# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""About network-operations (those based on graphs)"""

from collections import abc

import networkx as nx
from boltons.setutils import IndexedSet as iset

from .modifiers import optional
from .network import Network
from .op import Operation, Plotter, jetsam


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

    def compute(self, named_inputs, outputs=None) -> dict:
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

    # TODO: def __call__(self, **kwargs) -> tuple:
    def __call__(self, named_inputs, outputs=None) -> dict:
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

    def __call__(self, *operations, **kwargs) -> NetworkOperation:
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

        ## Build network
        #
        net = Network()
        for op in operations:
            net.add_op(op)

        return NetworkOperation(
            net, name=self.name, needs=needs, provides=provides, **kwargs
        )
