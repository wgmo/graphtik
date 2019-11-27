# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""About network-operations (those based on graphs)"""

import copy
import logging
import re
from collections import abc
from typing import Collection

import networkx as nx
from boltons.setutils import IndexedSet as iset

from .base import Plotter, aslist, astuple, jetsam
from .modifiers import optional, sideffect
from .network import Network
from .op import Operation, reparse_operation_data

log = logging.getLogger(__name__)


class NetworkOperation(Operation, Plotter):
    """
    An Operation performing a network-graph of other operations.

    .. Tip::
        Use :func:`compose()` factory to prepare the `net` and build
        instances of this class.
    """

    #: set execution mode to single-threaded sequential by default
    method = None
    overwrites_collector = None
    #: The default plan, enforcing stable `needs` & `provides`
    #: when :meth:`compute()` called with ``recompile=False``
    #: (default is ``recompile=None``, which means, only if `output` given).
    plan = None
    #: The execution_plan of the last call to compute(), stored as debugging aid.
    last_plan = None
    #: The inputs names (possibly `None`)used to compile the :attr:`plan`.
    inputs = None
    #: The outputs names (possibly `None`) used to compile the :attr:`plan`.
    outputs = None

    def __init__(
        self,
        net,
        name,
        *,
        inputs=None,
        outputs=None,
        method=None,
        overwrites_collector=None,
    ):
        """
        :param inputs:
            see :meth:`narrow()`
        :param outputs:
            see :meth:`narrow()`
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
        super().__init__(name, inputs, outputs)
        self.set_execution_method(method)
        self.set_overwrites_collector(overwrites_collector)

        ## Mimic `narrow()` but mutate myself
        #  to setup dependencies & plan.
        #
        orig_inputs = inputs
        if inputs is None:
            all_needs, _all_provides = self.net.dependencies()
            inputs = all_needs
        self.plan = self.net.compile(inputs, outputs)
        ## Use original `None`, to re-fetch dependencies from plan
        self._narrow_dependencies(orig_inputs, outputs)
        self.name, self.needs, self.provides = reparse_operation_data(
            self.name, self.needs, self.provides
        )

    def dependencies(self):
        return self.needs, self.provides

    def narrow(
        self, inputs: Collection = None, outputs: Collection = None
    ) -> "NetworkOperation":
        """
        Return a copy with a network pruned for the given `needs` & `provides`.

        :param inputs:
            a collection of inputs that must be given to :meth:`compute()`;
            a WARNing is issued for any irrelevant arguments.
            If `None`, they are collected from the :attr:`net`.
            They become the `needs` of the returned `netop`.
        :param outputs:
            a collection of outputs that will be asked from :meth:`compute()`;
            RAISES if those cannnot be satisfied.
            If `None`, they are collected from the :attr:`net`.
            They become the `provides` of the returned `netop`.

        :raise ValueError:
            IF `outputs` asked cannot be produced by the narrowed dag.

        If `inputs` or `outputs` are `None`, they are collected from
        the :attr:`net`.

        """
        if inputs is None:
            all_needs, _all_provides = self.net.dependencies()
            inputs = all_needs

        netop = copy.copy(self)
        netop.last_plan = None
        # Will scream on unknown `outputs`.
        netop.plan = self.net.compile(inputs, outputs)
        netop._narrow_dependencies(inputs, outputs)

        return netop

    def _narrow_dependencies(
        self, inputs: Collection = None, outputs: Collection = None
    ):
        """
        Prune dependencies based on :attr:`plan` and the given `needs` & `provides`.

        If both `needs` & `provides` are `None`, they are collected from the plan.
        """
        all_needs, all_provides = self.plan.dependencies()

        if inputs is None:
            inputs = all_needs
        else:
            inputs = astuple(inputs, "needs", allowed_types=abc.Collection)
            unknown = iset(inputs) - all_needs
            if unknown:
                log.warning("Unused needs%s for %s!", list(unknown), self.net)

        if outputs is None:
            outputs = all_provides
        else:
            outputs = aslist(outputs, "provides", allowed_types=abc.Collection)
            unknown = iset(outputs) - all_provides
            if unknown:
                raise ValueError(f"Impossible provides{list(unknown)} for {self.net}!")

        ## Retain optionality of `needs`.
        # TODO: Unify _DataNode + modifiers to avoid ugly hacks on ``netop._narrow()``.
        #
        def optionalized(node):
            dag = self.plan.dag
            all_optionals = all(e[2] for e in dag.out_edges(node, "optional", False))
            sideffector = dag.nodes(data="sideffect")
            return (
                optional(node)
                if all_optionals
                # Nodes are _DataNode instances, not `optional` or `sideffect`
                else sideffect(re.match(r"sideffect\((.*)\)", node).group(1))
                if sideffector[node]
                else str(node)  # un-optionalize
            )

        self.needs = [optionalized(n) for n in inputs]
        self.provides = outputs

    def _build_pydot(self, **kws):
        """delegate to network"""
        kws.setdefault("title", self.name)
        plotter = self.last_plan or self.net
        return plotter._build_pydot(**kws)

    def compute(self, named_inputs, outputs=None, recompile=None) -> dict:
        """
        Solve & execute the graph, sequentially or parallel.

        It see also :meth:`.Operation.compute()`.

        :param dict named_inputs:
            A maping of names --> values that must contain at least
            the compulsory inputs that were specified when the plan was built
            (but cannot enforce that!).
            Cloned, not modified.
        :param outputs:
            a string or a list of strings with all data asked to compute.
            If you set this variable to ``None``, all data nodes will be kept
            and returned at runtime.
        :param recompile:
            - if `False`, uses fixed :attr:`plan`;
            - if true, recompiles a temporary plan from network;
            - if `None`, assumed true if `outputs` given (is not `None`).

            In all cases, the `:attr:`last_plan` is updated.

        :returns: a dictionary of output data objects, keyed by name.
        """
        try:
            net = self.net
            if outputs is not None and recompile is None:
                recompile = True

            # Build the execution plan.
            self.last_plan = plan = (
                net.compile(named_inputs.keys(), outputs) if recompile else self.plan
            )

            solution = plan.execute(
                named_inputs, self.overwrites_collector, self.execution_method
            )

            return solution
        except Exception as ex:
            jetsam(ex, locals(), "plan", "solution", "outputs", network="net")

    def __call__(self, **input_kwargs) -> dict:
        """
        FIXME: doc netop()
        """
        # respect narrowed outputs, even when recompiled
        return self.compute(input_kwargs, outputs=self.plan.provides, recompile=True)

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
        inputs=needs,
        outputs=provides,
        method=method,
        overwrites_collector=overwrites_collector,
    )
