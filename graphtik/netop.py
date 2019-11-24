# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""About network-operations (those based on graphs)"""

import copy
import logging
from collections import abc

import networkx as nx
from boltons.setutils import IndexedSet as iset

from .base import Plotter, aslist, jetsam
from .modifiers import optional
from .network import Network
from .op import Operation, reparse_operation_data

log = logging.getLogger(__name__)


class NetworkOperation(Operation, Plotter):
    """
    An Operation performing a network-graph of other operations.

    Use :func:`compose()` to prepare the `net` and build instances of this class.
    """

    #: set execution mode to single-threaded sequential by default
    method = None
    overwrites_collector = None
    #: The execution_plan of the last call to compute(),
    #: stored as debugging aid.
    last_plan = None

    def __init__(
        self,
        net,
        name,
        *,
        needs=None,
        provides=None,
        method=None,
        overwrites_collector=None,
    ):
        """
        :param needs:
            if not None, network is pruned to consume just these
        :param provides:
            if not None, network is pruned to produce just these
        :param method:
            either `parallel` or None (default);
            if ``"parallel"``, launches multi-threading.
            Set when invoking a composed graph or by
            :meth:`~NetworkOperation.set_execution_method()`.

        :param overwrites_collector:
            (optional) a mutable dict to be fillwed with named values.
            If missing, values are simply discarded.

        """
        self.net = net
        ## Set data asap, for debugging, although `prune()` will reset them.
        super().__init__(name, needs, provides)
        self.set_execution_method(method)
        self.set_overwrites_collector(overwrites_collector)
        self._prune(needs, provides)
        self.name, self.needs, self.provides = reparse_operation_data(
            self.name, self.needs, self.provides
        )

    def copy(self, needs=None, provides=None):
        """
        Makes a copy with a network pruned for the given `needs` & `provides`.

        If both `needs` & `provides` are `None`, they are compiled from the net.
        """
        netop = copy.copy(self)
        netop.net = netop.net.copy()
        netop.last_plan = None

        netop._prune(needs, provides)

        return netop

    def _prune(self, needs=None, provides=None):
        """
        Prunes internal network  for the given `needs` & `provides`.

        - If both `needs` & `provides` are `None`, they are compiled from the net.
        - Private to discourage mutating callables - use :meth:`copy()`.
        """
        operations = [op for op in self.net.graph.nodes if isinstance(op, Operation)]
        all_provides = iset(p for op in operations for p in op.provides)
        all_needs = iset(n for op in operations for n in op.needs) - all_provides

        if needs is None and provides is None:
            self.needs = all_needs
            self.provides = all_provides
        else:
            plan = self.net.compile(needs, provides, skip_cache_update=True)
            self.net.graph.remove_nodes_from(plan.steps)

            if needs is not None:
                needs = aslist(needs, "needs")
                unknown = iset(needs) - all_needs
                if unknown:
                    log.warning("Unused `needs`: %s", unknown)
                self.needs = needs

            if provides is not None:
                provides = aslist(provides, "provides")
                assert set(provides) < all_provides, (
                    "Expected compile() above to have detected unknown outputs!",
                    iset(provides) - all_provides,
                )
                self.provides = provides

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
        choices = ["parallel", None]
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


def compose(
    name,
    op1,
    *operations,
    needs=None,
    provides=None,
    merge=False,
    method=None,
    overwrites_collector=None,
) -> NetworkOperation:
    """
    Composes a collection of operations into a single computation graph,
    obeying the ``merge`` property, if set in the constructor.

    :param str name:
        A optional name for the graph being composed by this object.

    :param op1:
        syntactically force at least 1 operation
    :param operations:
        Each argument should be an operation instance created using
        ``operation``.

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

    :param method:
        either `parallel` or None (default);
        if ``"parallel"``, launches multi-threading.
        Set when invoking a composed graph or by
        :meth:`~NetworkOperation.set_execution_method()`.

    :param overwrites_collector:
        (optional) a mutable dict to be fillwed with named values.
        If missing, values are simply discarded.

    :return:
        Returns a special type of operation class, which represents an
        entire computation graph as a single operation.
    """
    operations = (op1,) + operations
    if not all(isinstance(op, Operation) for op in operations):
        raise ValueError(f"Non-Operation instances given: {operations}")

    # If merge is desired, deduplicate operations before building network
    if merge:
        merge_set = iset()  # Preseve given node order.
        for op in operations:
            if isinstance(op, NetworkOperation):
                netop_nodes = nx.topological_sort(op.net.graph)
                merge_set.update(s for s in netop_nodes if isinstance(s, Operation))
            else:
                merge_set.add(op)
        operations = merge_set

    ## Build network
    #
    net = Network()
    for op in operations:
        net.add_op(op)

    return NetworkOperation(
        net,
        name,
        needs=needs,
        provides=provides,
        method=method,
        overwrites_collector=overwrites_collector,
    )
