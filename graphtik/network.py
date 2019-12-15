# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
""":term:`Compile` & :term:`execute` network graphs of operations."""
import copy
import logging
import re
import sys
import time
from collections import ChainMap, abc, defaultdict, namedtuple
from contextvars import ContextVar
from itertools import count
from multiprocessing.dummy import Pool
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import networkx as nx
from boltons.setutils import IndexedSet as iset

from .base import Items, Plotter, aslist, astuple, jetsam
from .modifiers import optional, sideffect
from .op import Operation

log = logging.getLogger(__name__)


#: Global configurations for all (nested) networks in a computaion run.
_execution_configs: ContextVar[dict] = ContextVar(
    "execution_configs",
    default={"execution_pool": Pool(7), "abort": False, "skip_evictions": False},
)


class AbortedException(Exception):
    """Raised from the Network code when :func:`abort_run()` is called."""


def abort_run():
    _execution_configs.get()["abort"] = True


def _reset_abort():
    _execution_configs.get()["abort"] = False


def is_abort():
    return _execution_configs.get()["abort"]


def set_evictions_skipped(skipped):
    _execution_configs.get()["skip_evictions"] = skipped


def is_skip_evictions():
    return _execution_configs.get()["skip_evictions"]

def _break_incoming_edges(dag, nodes):
    """Modifies `dag` by removing all incoming edges of `nodes`."""
    for n in nodes:
        # Coalesce to a list, to avoid concurrent modification.
        dag.remove_edges_from(list(dag.in_edges(n)))

class Solution(ChainMap, Plotter):
    """
    Collects outputs from operations, preserving :term:`overwrites`.

    :ivar plan:
        the plan that produced this solution
    :ivar executed:
        A set with operations executed
    :ivar finished:
        a flag denoting that this instance cannot acccept more results
        (after the :meth:`finished` has been invoked)
    :ivar times:
        a dictionary with execution timings for each operation
    """
    def __init__(self, plan, *args, **kw):
        super().__init__(*args, **kw)

        self.plan = plan
        self.executed = iset()
        self.finished = False
        self.times = {}

    def __repr__(self):
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{{{items}}}"

    def operation_executed(self, op, outputs):
        """invoked once per operation, with its results"""
        assert not self.finished, f"Cannot reuse solution: {self}"
        self.maps.append(outputs)
        self.executed.add(op)

    def finish(self):
        """invoked only once, after all ops have been executed"""
        # Invert solution so that last value wins
        if not self.finished:
            self.maps = self.maps[::-1]
            self.finised = True

    def __delitem__(self, key):
        log.debug("removing data '%s' from solution.", key)
        for d in self.maps:
            d.pop(key, None)

    def overwrites(self) -> Mapping[Any, List]:
        """
        Collect items in the maps that exist more than once.

        :return:
            a dictionary with keys only those items that existed in more than one map,
            an values, all those values, in the order of given `maps`
        """
        maps = self.maps
        dd = defaultdict(list)
        for d in maps:
            for k, v in d.items():
                dd[k].append(v)

        return {k: v for k, v in dd.items() if len(v) > 1}

    def _build_pydot(self, **kws):
        """delegate to network"""
        kws.setdefault("solution", self)
        plotter = self.plan
        return plotter._build_pydot(**kws)


class _DataNode(str):
    """
    Dag node naming a data-value produced or required by an operation.
    """

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return f"DataNode('{self}')"


class _EvictInstruction(str):
    """
    A step in the ExecutionPlan to evict a computed value from the `solution`.

    It's a step in :attr:`ExecutionPlan.steps` for the data-node `str` that
    frees its data-value from `solution` after it is no longer needed,
    to reduce memory footprint while computing the graph.
    """

    __slots__ = ()  # avoid __dict__ on instances

    def __repr__(self):
        return f"EvictInstruction('{self}')"


def _yield_datanodes(nodes):
    return (n for n in nodes if isinstance(n, _DataNode))


def yield_ops(nodes):
    return (n for n in nodes if isinstance(n, Operation))


def _optionalized(graph, data):
    """Retain optionality of a `data` node based on all `needs` edges."""
    all_optionals = all(e[2] for e in graph.out_edges(data, "optional", False))
    sideffector = graph.nodes(data="sideffect")
    return (
        optional(data)
        if all_optionals
        # Nodes are _DataNode instances, not `optional` or `sideffect`
        # TODO: Unify _DataNode + modifiers to avoid ugly hack `net.collect_requirements()`.
        else sideffect(re.match(r"sideffect\((.*)\)", data).group(1))
        if sideffector[data]
        else str(data)  # un-optionalize
    )


def collect_requirements(graph) -> Tuple[iset, iset]:
    """Collect & split datanodes in (possibly overlapping) `needs`/`provides`."""
    operations = list(yield_ops(graph))
    provides = iset(p for op in operations for p in op.provides)
    needs = iset(_optionalized(graph, n) for op in operations for n in op.needs)
    # TODO: Unify _DataNode + modifiers to avoid ugly hack `net.collect_requirements()`.
    provides = iset(str(n) if not isinstance(n, sideffect) else n for n in provides)
    return needs, provides


class ExecutionPlan(
    namedtuple("ExecPlan", "net needs provides dag steps evict"), Plotter
):
    """
    A pre-compiled list of operation steps that can :term:`execute` for the given inputs/outputs.

    It is the result of the network's :term:`compilation` phase.

    Note the execution plan's attributes are on purpose immutable tuples.

    :ivar net:
        The parent :class:`Network`
    :ivar needs:
        An :class:`iset` with the input names needed to exist in order to produce all `provides`.
    :ivar provides:
        An :class:`iset` with the outputs names produces when all `inputs` are given.
    :ivar dag:
        The regular (not broken) *pruned* subgraph of net-graph.
    :ivar steps:
        The tuple of operation-nodes & *instructions* needed to evaluate
        the given inputs & asked outputs, free memory and avoid overwritting
        any given intermediate inputs.
    :ivar evict:
        when false, keep all inputs & outputs, and skip prefect-evictions check.
    """

    def _build_pydot(self, **kws):
        from .plot import build_pydot

        clusters = None
        if self.dag.nodes != self.net.graph.nodes:
            clusters = {n: "after prunning" for n in self.dag.nodes}
        mykws = {
            "graph": self.net.graph,
            "steps": self.steps,
            "inputs": self.needs,
            "outputs": self.provides,
            "clusters": clusters,
        }
        mykws.update(kws)

        return build_pydot(**mykws)

    def __repr__(self):
        needs = aslist(self.needs, "needs")
        provides = aslist(self.provides, "provides")
        steps = "".join(f"\n  +--{s}" for s in self.steps)
        return f"ExecutionPlan(needs={needs}, provides={provides}, steps:{steps})"

    def validate(self, inputs: Items, outputs: Items):
        """
        Scream on invalid inputs, outputs or no operations in graph.

        :raises ValueError:
            - If cannot produce any `outputs` from the given `inputs`, with msg:

                *Unsolvable graph: ...*

            - If given `inputs` mismatched plan's :attr:`needs`, with msg:

                *Plan needs more inputs...*

            - If `outputs` asked cannot be produced by the :attr:`dag`, with msg:

                *Impossible outputs...*

        """
        if not self.dag:
            raise ValueError(f"Unsolvable graph:\n  {self}")

        # Check plan<-->inputs mismatch.
        #
        missing = iset(self.needs) - set(inputs)
        if missing:
            raise ValueError(
                f"Plan needs more inputs: {list(missing)}"
                f"\n  given inputs: {list(inputs)}\n  {self}"
            )

        if outputs:
            unknown = (
                iset(astuple(outputs, "outputs", allowed_types=abc.Sequence))
                - self.provides
            )
            if unknown:
                raise ValueError(
                    f"Impossible outputs: {list(unknown)}\n for graph: {self}"
                    f"\n  {self}"
                )

    def _check_if_aborted(self, executed):
        if is_abort():
            # Restore `abort` flag for next run.
            _reset_abort()
            raise AbortedException({s: s in executed for s in self.steps})

    def _call_operation(self, op, solution):
        # Although `plan` have added to jetsam in `compute()``,
        # add it again, in case compile()/execute is called separately.
        t0 = time.time()
        try:
            return op.compute(solution)
        except Exception as ex:
            jetsam(ex, locals(), "solution", plan="self")
        finally:
            # record execution time
            t_complete = round(time.time() - t0, 5)
            solution.times[op.name] = t_complete
            log.debug("...step completion time: %s", t_complete)

    def _execute_thread_pool_barrier_method(self, solution: Solution):
        """
        This method runs the graph using a parallel pool of thread executors.
        You may achieve lower total latency if your graph is sufficiently
        sub divided into operations using this method.

        :param solution:
            must contain the input values only, gets modified
        """
        pool = _execution_configs.get()["execution_pool"]

        # with each loop iteration, we determine a set of operations that can be
        # scheduled, then schedule them onto a thread pool, then collect their
        # results onto a memory solution for use upon the next iteration.
        while True:
            self._check_if_aborted(solution.executed)

            # the upnext list contains a list of operations for scheduling
            # in the current round of scheduling
            upnext = []
            for node in self.steps:
                ## Determines if a Operation is ready to be scheduled for execution
                #  based on what has already been executed.
                if (
                    isinstance(node, Operation)
                    and node not in solution.executed
                    and set(
                        n
                        for n in nx.ancestors(self.dag, node)
                        if isinstance(n, Operation)
                    ).issubset(solution.executed)
                ):
                    upnext.append(node)
                elif isinstance(node, _EvictInstruction):
                    # Only evict if all successors for the data node
                    # have been executed.
                    if (
                        # An optional need may not have a value in the solution.
                        node in solution
                        and (
                            # The 2nd eviction branch for unused provides.
                            node not in self.dag.nodes
                            # Scan node's successors in `broken_dag`, not to block
                            # an op waiting for calced data already given as input.
                            or set(self.dag.successors(node)).issubset(
                                solution.executed
                            )
                        )
                    ):
                        del solution[node]

            # stop if no nodes left to schedule, exit out of the loop
            if not upnext:
                break

            done_iterator = pool.imap_unordered(
                (lambda op: (op, self._call_operation(op, solution))), upnext
            )

            for op, outputs in done_iterator:
                solution.operation_executed(op, outputs)

    def _execute_sequential_method(self, solution: Solution):
        """
        This method runs the graph one operation at a time in a single thread

        :param solution:
            must contain the input values only, gets modified
        """
        for step in self.steps:
            self._check_if_aborted(solution.executed)

            if isinstance(step, Operation):
                log.debug("%sexecuting step: %s", "-" * 32, step.name)

                outputs = self._call_operation(step, solution)
                solution.operation_executed(step, outputs)

            elif isinstance(step, _EvictInstruction):
                # Cache value may be missing if it is optional.
                if step in solution:
                    del solution[step]

            else:
                raise AssertionError(f"Unrecognized instruction.{step}")

    def execute(self, named_inputs, outputs=None, *, method=None) -> Solution:
        """
        :param named_inputs:
            A maping of names --> values that must contain at least
            the compulsory inputs that were specified when the plan was built
            (but cannot enforce that!).
            Cloned, not modified.
        :param outputs:
            If not None, they are just checked if possible, based on :attr:`provides`,
            and scream if not.

        :return:
            The :term:`solution` which contains the results of each operation executed
            +1 for inputs in separate dictionaries.

        :raises ValueError:
            - If plan does not contain any operations, with msg:

                *Unsolvable graph: ...*

            - If given `inputs` mismatched plan's :attr:`needs`, with msg:

                *Plan needs more inputs...*

            - If `outputs` asked cannot be produced by the :attr:`dag`, with msg:

                *Impossible outputs...*
        """
        try:
            self.validate(named_inputs, outputs)

            # choose a method of execution
            executor = (
                self._execute_thread_pool_barrier_method
                if method == "parallel"
                else self._execute_sequential_method
            )

            # If certain outputs asked, put relevant-only inputs in solution,
            # otherwise, keep'em all.
            #
            # Note: clone and keep original `inputs` in the 1st chained-map.
            solution = Solution(
                self,
                {k: v for k, v in named_inputs.items() if k in self.dag.nodes}
                if self.evict
                else named_inputs,
            )
            try:
                executor(solution)
            finally:
                solution.finish()

            # Validate eviction was perfect
            #
            assert (
                not self.evict
                or is_skip_evictions()
                # It is a proper subset when not all outputs calculated.
                or set(solution).issubset(self.provides)
            ), f"Evictions left more data{list(iset(solution) - set(self.provides))} than {self}!"

            return solution
        except Exception as ex:
            jetsam(ex, locals(), "solution")


class Network(Plotter):
    """
    A graph of operations that can :term:`compile` an execution plan.

    :ivar needs:
        the "base", all data-nodes that are not produced by some operation
    :ivar provides:
        the "base", all data-nodes produced by some operation
    """

    def __init__(self, *operations, graph=None):
        """

        :param operations:
            to be added in the graph
        :param graph:
            if None, create a new.
        """
        ## Check for duplicate, operations can only append  once.
        #
        uniques = set(operations)
        if len(operations) != len(uniques):
            dupes = list(operations)
            for i in uniques:
                dupes.remove(i)
            raise ValueError(f"Operations may only be added once, dupes: {list(dupes)}")

        if graph is None:
            # directed graph of operation and data nodes defining the net.
            graph = nx.DiGraph()
        else:
            if not isinstance(graph, nx.Graph):
                raise ValueError(f"Must be a NetworkX graph, was: {graph}")
        self.graph = graph

        for op in operations:
            self._append_operation(graph, op)
        self.needs, self.provides = collect_requirements(self.graph)

        #: Speed up :meth:`compile()` call and avoid a multithreading issue(?)
        #: that is occuring when accessing the dag in networkx.
        self._cached_plans = {}

    def __repr__(self):
        steps = [f"\n  +--{s}" for s in self.graph.nodes]
        return f"Network({''.join(steps)})"

    def _build_pydot(self, **kws):
        from .plot import build_pydot

        kws.setdefault("graph", self.graph)

        return build_pydot(**kws)

    def _append_operation(self, graph, operation: Operation):
        """
        Adds the given operation and its data requirements to the network graph.

        - Invoked during constructor only (immutability).
        - Identities are based on the name of the operation, the names of the operation's needs,
          and the names of the data it provides.

        :param graph:
            the `networkx` graph to append to
        :param operation:
            operation instance to append
        """
        # add nodes and edges to graph describing the data needs for this layer
        for n in operation.needs:
            kw = {}
            if isinstance(n, optional):
                kw["optional"] = True
            if isinstance(n, sideffect):
                kw["sideffect"] = True
                graph.add_node(_DataNode(n), sideffect=True)
            graph.add_edge(_DataNode(n), operation, **kw)

        graph.add_node(operation, **operation.node_props)

        # add nodes and edges to graph describing what this layer provides
        for n in operation.provides:
            kw = {}
            if isinstance(n, sideffect):
                kw["sideffect"] = True
                graph.add_node(_DataNode(n), sideffect=True)
            graph.add_edge(operation, _DataNode(n), **kw)

    def _topo_sort_nodes(self, dag) -> List:
        """Topo-sort dag respecting operation-insertion order to break ties."""
        node_keys = dict(zip(dag.nodes, count()))
        return nx.lexicographical_topological_sort(dag, key=node_keys.get)

    def _unsatisfied_operations(self, dag, inputs: Collection):
        """
        Traverse topologically sorted dag to collect un-satisfied operations.

        Unsatisfied operations are those suffering from ANY of the following:

        - They are missing at least one compulsory need-input.
          Since the dag is ordered, as soon as we're on an operation,
          all its needs have been accounted, so we can get its satisfaction.

        - Their provided outputs are not linked to any data in the dag.
          An operation might not have any output link when :meth:`_prune_graph()`
          has broken them, due to given intermediate inputs.

        :param dag:
            a graph with broken edges those arriving to existing inputs
        :param inputs:
            an iterable of the names of the input values
        :return:
            a list of unsatisfied operations to prune

        """
        # To collect data that will be produced.
        ok_data = set(inputs)
        # To colect the map of operations --> satisfied-needs.
        op_satisfaction = defaultdict(set)
        # To collect the operations to drop.
        unsatisfied = []
        # Topo-sort dag respecting operation-insertion order to break ties.
        sorted_dag = nx.topological_sort(dag)
        for node in sorted_dag:
            if isinstance(node, Operation):
                if not dag.adj[node]:
                    # Prune operations that ended up providing no output.
                    unsatisfied.append(node)
                else:
                    # It's ok not to dig into edge-data("optional") here,
                    # we care about all needs, including broken ones.
                    real_needs = set(
                        n for n in node.needs if not isinstance(n, optional)
                    )
                    if real_needs.issubset(op_satisfaction[node]):
                        # We have a satisfied operation; mark its output-data
                        # as ok.
                        ok_data.update(dag.adj[node])
                    else:
                        # Prune operations with partial inputs.
                        unsatisfied.append(node)
            elif isinstance(node, (_DataNode, str)):  # `str` are givens
                if node in ok_data:
                    # mark satisfied-needs on all future operations
                    for future_op in dag.adj[node]:
                        op_satisfaction[future_op].add(node)
            else:
                raise AssertionError(f"Unrecognized network graph node {node}")

        return unsatisfied

    def _apply_graph_predicate(self, graph, predicate):
        to_del = []
        for node, data in graph.nodes.items():
            try:
                if isinstance(node, Operation) and not predicate(node, data):
                    to_del.append(node)
            except Exception as ex:
                raise ValueError(
                    f"Node-predicate({predicate}) failed due to: {ex}\n  node: {node}, {self}"
                ) from ex
        graph.remove_nodes_from(to_del)

    def _prune_graph(
        self,
        inputs: Items,
        outputs: Items,
        predicate: Callable[[Any, Mapping], bool] = None,
    ) -> Tuple[nx.DiGraph, Collection, Collection, Collection]:
        """
        Determines what graph steps need to run to get to the requested
        outputs from the provided inputs:
        - Eliminate steps that are not on a path arriving to requested outputs;
        - Eliminate unsatisfied operations: partial inputs or no outputs needed;
        - consolidate the list of needs & provides.

        :param inputs:
            The names of all given inputs.
        :param outputs:
            The desired output names.  This can also be ``None``, in which
            case the necessary steps are all graph nodes that are reachable
            from the provided inputs.
        :param predicate:
            the :term:`node predicate` is a 2-argument callable(op, node-data)
            that should return true for nodes to include; if None, all nodes included.

        :return:
            a 3-tuple with the *pruned_dag* & the needs/provides resolved based
            on the given inputs/outputs
            (which might be a subset of all needs/outputs of the returned graph).

            Use the returned `needs/provides` to build a new plan.

        :raises ValueError:
            - if `outputs` asked do not exist in network, with msg:

                *Unknown output nodes: ...*
        """
        # TODO: break cycles here.
        dag = self.graph

        ##  When `inputs` is None, we have to keep all possible input nodes
        #   and this is achieved with 2 tricky locals:
        #
        #   inputs
        #       it is kept falsy, to disable the edge-breaking, so that
        #       the asceding_from_outputs that follows can reach all input nodes;
        #       including intermediate ones;
        #   satisfied_inputs
        #       it is filled with all possible input nodes, to trick `_unsatisfied_operations()`
        #       to assume their operations are satisfied, and keep them.
        #
        if inputs is None and outputs is None:
            satisfied_inputs, outputs = self.needs, self.provides
        else:
            if inputs is None:  # outputs: NOT None
                satisfied_inputs = self.needs - outputs
            else:  # inputs: NOT None, outputs: None
                # Just ignore `inputs` not in the graph.
                satisfied_inputs = inputs = iset(inputs) & dag.nodes

            ## Scream on unknown `outputs`.
            #
            if outputs:
                unknown_outputs = iset(outputs) - dag.nodes
                if unknown_outputs:
                    raise ValueError(
                        f"Unknown output nodes: {list(unknown_outputs)}\n  {self}"
                    )

        assert isinstance(satisfied_inputs, abc.Collection)
        assert inputs is None or isinstance(inputs, abc.Collection)
        assert outputs is None or isinstance(outputs, abc.Collection)

        broken_dag = dag.copy()  # preserve net's graph

        if predicate:
            self._apply_graph_predicate(broken_dag, predicate)

        # Break the incoming edges to all given inputs.
        #
        # Nodes producing any given intermediate inputs are unecessary
        # (unless they are also used elsewhere).
        # To discover which ones to prune, we break their incoming edges
        # and they will drop out while collecting ancestors from the outputs.
        #
        if inputs:
            _break_incoming_edges(broken_dag, inputs)

        # Drop stray input values and operations (if any).
        if outputs is not None:
            # If caller requested specific outputs, we can prune any
            # unrelated nodes further up the dag.
            ending_in_outputs = set()
            for output_name in outputs:
                ending_in_outputs.add(_DataNode(output_name))
                ending_in_outputs.update(nx.ancestors(dag, output_name))
            broken_dag = broken_dag.subgraph(ending_in_outputs)

        # Prune unsatisfied operations (those with partial inputs or no outputs).
        unsatisfied = self._unsatisfied_operations(broken_dag, satisfied_inputs)
        # Clone it, to modify it.
        pruned_dag = dag.subgraph(broken_dag.nodes - unsatisfied).copy()
        # Clean unlinked data-nodes.
        pruned_dag.remove_nodes_from(list(nx.isolates(pruned_dag)))

        inputs = iset(
            _optionalized(pruned_dag, n) for n in satisfied_inputs if n in pruned_dag
        )
        if outputs is None:
            outputs = iset(
                n for n in self.provides if n not in inputs and n in pruned_dag
            )

        assert inputs is not None or isinstance(inputs, abc.Collection)
        assert outputs is not None or isinstance(outputs, abc.Collection)

        return pruned_dag, tuple(inputs), tuple(outputs)

    def narrowed(
        self,
        inputs: Items = None,
        outputs: Items = None,
        predicate: Callable[[Any, Mapping], bool] = None,
    ) -> "Network":
        """
        Return a pruned network supporting just the given `inputs` & `outputs`.

        :param inputs:
            all possible inputs names
        :param outputs:
            all possible output names
        :param predicate:
            the :term:`node predicate` is a 2-argument callable(op, node-data)
            that should return true for nodes to include; if None, all nodes included.

        :return:
            the pruned clone, or this, if both `inputs` & `outputs` were `None`
        """
        if (inputs, outputs, predicate) == (None, None, None):
            return self

        if inputs is not None:
            inputs = astuple(inputs, "outputs", allowed_types=(list, tuple))
        if outputs is not None:
            outputs = astuple(outputs, "outputs", allowed_types=(list, tuple))

        pruned_dag, _needs, _provides = self._prune_graph(inputs, outputs, predicate)
        return Network(graph=pruned_dag)

    def _build_execution_steps(
        self, pruned_dag, inputs: Collection, outputs: Optional[Collection]
    ) -> List:
        """
        Create the list of operation-nodes & *instructions* evaluating all

        operations & instructions needed a) to free memory and b) avoid
        overwritting given intermediate inputs.

        :param pruned_dag:
            The original dag, pruned; not broken.
        :param outputs:
            outp-names to decide whether to add (and which) evict-instructions

        Instances of :class:`_EvictInstructions` are inserted in `steps` between
        operation nodes to reduce the memory footprint of solutions while
        the computation is running.
        An evict-instruction is inserted whenever a *need* is not used
        by any other *operation* further down the DAG.
        """

        steps = []

        def add_step_once(step):
            # For functions with repeated needs, like ['a', 'a'].
            if steps and step == steps[-1] and type(step) == type(steps[-1]):
                log.warning("Skipped dupe step %s in position %i.", step, len(steps))
            else:
                steps.append(step)

        ## Create an execution order such that each layer's needs are provided,
        #  respecting operation-insertion order to break ties;  which means that
        #  the first inserted operations win the `needs`, but
        #  the last ones win the `provides` (and the final solution).
        ordered_nodes = iset(self._topo_sort_nodes(pruned_dag))

        # Add Operations evaluation steps, and instructions to evict data.
        for i, node in enumerate(ordered_nodes):

            if isinstance(node, Operation):
                steps.append(node)

                # NO EVICTIONS when no specific outputs asked.
                if not outputs or is_skip_evictions():
                    continue

                # Add EVICT (1) for operation's needs.
                #
                # Broken links are irrelevant because they are preds of data (provides),
                # but here we scan for preds of the operation (needs).
                #
                for need in pruned_dag.pred[node]:
                    # Do not evict asked outputs or sideffects.
                    if need in outputs:
                        continue

                    # A needed-data of this operation may be evicted if
                    # no future Operations needs it.
                    #
                    for future_node in ordered_nodes[i + 1 :]:
                        if (
                            isinstance(future_node, Operation)
                            and need in pruned_dag.pred[future_node]
                        ):
                            break
                    else:
                        add_step_once(_EvictInstruction(need))

                # Add EVICT (2) for unused operation's provides.
                #
                # A provided-data is evicted if no future operation needs it
                # (and is not an asked output).
                # It MUST use the broken dag, not to evict data
                # that will be pinned(?), but to populate overwrites with them.
                #
                # .. image:: doc/source/images/unpruned_useless_provides.svg
                #
                for provide in node.provides:
                    # Do not evict asked outputs or sideffects.
                    if provide not in outputs and provide not in pruned_dag.nodes:
                        add_step_once(_EvictInstruction(provide))

            else:
                assert isinstance(
                    node, _DataNode
                ), f"Unrecognized network graph node {node, type(node)}"

        return steps

    def compile(
        self, inputs: Items = None, outputs: Items = None, predicate=None
    ) -> ExecutionPlan:
        """
        Create or get from cache an execution-plan for the given inputs/outputs.

        See :meth:`_prune_graph()` and :meth:`_build_execution_steps()`
        for detailed description.

        :param inputs:
            A collection with the names of all the given inputs.
            If `None``, all inputs that lead to given `outputs` are assumed.
            If string, it is converted to a single-element collection.
        :param outputs:
            A collection or the name of the output name(s).
            If `None``, all reachable nodes from the given `inputs` are assumed.
            If string, it is converted to a single-element collection.
        :param predicate:
            the :term:`node predicate` is a 2-argument callable(op, node-data)
            that should return true for nodes to include; if None, all nodes included.

        :return:
            the cached or fresh new execution-plan

        :raises ValueError:
            - If `outputs` asked do not exist in network, with msg:

                *Unknown output nodes: ...*

            - If solution does not contain any operations, with msg:

                *Unsolvable graph: ...*

            - If given `inputs` mismatched plan's :attr:`needs`, with msg:

                *Plan needs more inputs...*

            - If `outputs` asked cannot be produced by the :attr:`dag`, with msg:

                *Impossible outputs...*
        """
        ## Make a stable cache-key.
        #
        if inputs is not None:
            inputs = tuple(
                sorted(astuple(inputs, "inputs", allowed_types=abc.Collection))
            )
        if outputs is not None:
            outputs = tuple(
                sorted(astuple(outputs, "outputs", allowed_types=abc.Collection))
            )
        if not predicate:
            predicate = None

        cache_key = (inputs, outputs, predicate)

        ## Build (or retrieve from cache) execution plan
        #  for the given inputs & outputs.
        #
        if cache_key in self._cached_plans:
            plan = self._cached_plans[cache_key]
        else:
            pruned_dag, needs, provides = self._prune_graph(inputs, outputs, predicate)
            steps = self._build_execution_steps(pruned_dag, needs, outputs or ())
            plan = ExecutionPlan(
                self,
                needs,
                provides,
                pruned_dag,
                tuple(steps),
                evict=outputs is not None,
            )

            self._cached_plans[cache_key] = plan

        return plan
