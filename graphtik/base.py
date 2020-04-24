# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
Generic utilities, or :mod:`.plot` API but without polluting imports.

.. doctest::
    :hide:

    .. Workaround sphinx-doc/sphinx#6590

    >>> from graphtik.base import *
    >>> __name__ = "graphtik.base"
"""

import abc
import inspect
import logging
from collections import defaultdict, namedtuple
from functools import partial, partialmethod, wraps
from typing import Any, Collection, List, Mapping, NamedTuple, Optional, Tuple, Union

Items = Union[Collection, str, None]

log = logging.getLogger(__name__)


class MultiValueError(ValueError):
    def __str__(self):
        """Assuming it has been called with ``MultiValueError(msg, ex1, ...) #"""
        return str(self.args[0])  # pylint: disable=unsubscriptable-object


class Token(str):
    """Guarantee equality, not(!) identity, across processes."""

    __slots__ = ("hashid",)

    def __new__(cls, s):
        return super().__new__(cls, f"<{s}>")

    def __init__(self, *args):
        import random

        self.hashid = random.randint(-(2 ** 32), 2 ** 32 - 1)

    def __eq__(self, other):
        return self.hashid == getattr(other, "hashid", None)

    def __hash__(self):
        return self.hashid

    def __getstate__(self):
        return self.hashid

    def __setstate__(self, state):
        self.hashid = state

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __bool__(self):
        """Always `True`, even if empty string."""
        return True

    def __repr__(self):
        """Avoid 'ticks' around repr."""
        return self.__str__()


#: When an operation function returns this special value,
#: it implies operation has no result at all,
#: (otherwise, it would have been a single result, ``None``).`
NO_RESULT = Token("NO_RESULT")
UNSET = Token("UNSET")


def aslist(i, argname, allowed_types=list):
    """Utility to accept singular strings as lists, and None --> []."""
    if not i:
        return i if isinstance(i, allowed_types) else []

    if isinstance(i, str):
        i = [i]
    elif not isinstance(i, allowed_types):
        try:
            i = list(i)
        except Exception as ex:
            raise ValueError(f"Cannot list-ize {argname}({i!r}) due to: {ex}") from None

    return i


def astuple(i, argname, allowed_types=tuple):
    if not i:
        return i if isinstance(i, allowed_types) else ()

    if isinstance(i, str):
        i = (i,)
    elif not isinstance(i, allowed_types):
        try:
            i = tuple(i)
        except Exception as ex:
            raise ValueError(
                f"Cannot tuple-ize {argname}({i!r}) due to: {ex}"
            ) from None

    return i


def func_name(fn, default=..., mod=None, fqdn=None, human=None) -> Optional[str]:
    """
    FQDN of `fn`, descending into partials to print their args.

    :param default:
        What to return if it fails; by default it raises.
    :param mod:
        when true, prepend module like ``module.name.fn_name``
    :param fqdn:
        when true, use ``__qualname__`` (instead of ``__name__``)
        which differs mostly on methods, where it contains class(es),
        and locals, respectively (:pep:`3155`).
        *Sphinx* uses `fqdn=True` for generating IDs.
    :param human:
        when true, partials denote their args like ``fn({"a": 1}, ...)`` in the returned text,
        otherwise, just the (fqd-)name, appropriate for IDs.

    :return:
        a (possibly dot-separated) string, or `default` (unless this is ``...```).
    :raises:
        Only if default is ``...``, otherwise, errors debug-logged.


    **Examples**

        >>> func_name(func_name)
        'func_name'
        >>> func_name(func_name, mod=1)
        'graphtik.base.func_name'
        >>> func_name(MultiValueError.mro, fqdn=0)
        'mro'
        >>> func_name(MultiValueError.mro, fqdn=1)
        'MultiValueError.mro'

    Even functions defined in docstrings are reported:

        >>> def f():
        ...     def inner():
        ...         pass
        ...     return inner

        >>> func_name(f, mod=1, fqdn=1)
        'graphtik.base.f'
        >>> func_name(f(), fqdn=1)
        'f.<locals>.inner'

    On failures, arg `default` controls the outcomes:

    TBD
    """

    if isinstance(fn, (partial, partialmethod)):
        # Always bubble-up errors.
        fn_name = func_name(fn.func, default, mod, fqdn, human)
        if human:
            args = ", ".join(str(i) for i in fn.args)
            kw = ", ".join(f"{k}={v}" for k, v in fn.keywords.items())
            args_str = ", ".join(i for i in (args, kw) + ("...",) if i)
            fn_name = f"{fn_name}({args_str})"

        return fn_name

    try:
        if human and inspect.isbuiltin(fn):
            return str(fn)
        fn_name = fn.__qualname__ if fqdn else fn.__name__
        assert fn_name

        mod_name = getattr(fn, "__module__", None)
        if mod and mod_name:
            fn_name = ".".join((mod_name, fn_name))
        return fn_name
    except Exception as ex:
        if default is ...:
            raise
        log.debug(
            "Ignored error while inspecting %r name: %s", fn, ex,
        )
        return default


def _un_partial_ize(func):
    """
    Alter functions working on 1st arg being a callable, to descend it if it's a partial.
    """

    @wraps(func)
    def wrapper(fn, *args, **kw):
        if isinstance(fn, (partial, partialmethod)):
            fn = fn.func
        return func(fn, *args, **kw)

    return wrapper


@_un_partial_ize
def func_source(fn, default=..., human=None) -> Optional[Tuple[str, int]]:
    """
    Like :func:`inspect.getsource` supporting partials.

    :param default:
        If given, better be a 2-tuple respecting types,
        or ``...``, to raise.
    :param human:
        when true, denote builtins like python does
    """
    try:
        if inspect.isbuiltin(fn):
            if human:
                return inspect.getdoc(fn).splitlines()[0]
            else:
                return str(fn)
        return inspect.getsource(fn)
    except Exception as ex:
        if default is ...:
            raise
        log.debug(
            "Ignored error while inspecting %r sources: %s", fn, ex,
        )
        return default


@_un_partial_ize
def func_sourcelines(fn, default=..., human=None) -> Optional[Tuple[str, int]]:
    """
    Like :func:`inspect.getsourcelines` supporting partials.

    :param default:
        If given, better be a 2-tuple respecting types,
        or ``...``, to raise.
    """
    try:
        if human and inspect.isbuiltin(fn):
            return [str(fn)], -1
        return inspect.getsourcelines(fn)
    except Exception as ex:
        if default is ...:
            raise
        log.debug(
            "Ignored error while inspecting %r sourcelines: %s", fn, ex,
        )
        return default


def jetsam(ex, locs, *salvage_vars: str, annotation="jetsam", **salvage_mappings):
    """
    Annotate exception with salvaged values from locals() and raise!

    :param ex:
        the exception to annotate
    :param locs:
        ``locals()`` from the context-manager's block containing vars
        to be salvaged in case of exception

        ATTENTION: wrapped function must finally call ``locals()``, because
        *locals* dictionary only reflects local-var changes after call.
    :param annotation:
        the name of the attribute to attach on the exception
    :param salvage_vars:
        local variable names to save as is in the salvaged annotations dictionary.
    :param salvage_mappings:
        a mapping of destination-annotation-keys --> source-locals-keys;
        if a `source` is callable, the value to salvage is retrieved
        by calling ``value(locs)``.
        They take precedence over`salvage_vars`.

    :raises:
        any exception raised by the wrapped function, annotated with values
        assigned as attributes on this context-manager

    - Any attributes attached on this manager are attached as a new dict on
      the raised exception as new  ``jetsam`` attribute with a dict as value.
    - If the exception is already annotated, any new items are inserted,
      but existing ones are preserved.

    **Example:**

    Call it with managed-block's ``locals()`` and tell which of them to salvage
    in case of errors::


        try:
            a = 1
            b = 2
            raise Exception()
        exception Exception as ex:
            jetsam(ex, locals(), "a", b="salvaged_b", c_var="c")
            raise

    And then from a REPL::

        import sys
        sys.last_value.jetsam
        {'a': 1, 'salvaged_b': 2, "c_var": None}

    ** Reason:**

    Graphs may become arbitrary deep.  Debugging such graphs is notoriously hard.

    The purpose is not to require a debugger-session to inspect the root-causes
    (without precluding one).

    Naively salvaging values with a simple try/except block around each function,
    blocks the debugger from landing on the real cause of the error - it would
    land on that block;  and that could be many nested levels above it.
    """
    ## Fail EARLY before yielding on bad use.
    #
    assert isinstance(ex, Exception), ("Bad `ex`, not an exception dict:", ex)
    assert isinstance(locs, dict), ("Bad `locs`, not a dict:", locs)
    assert all(isinstance(i, str) for i in salvage_vars), (
        "Bad `salvage_vars`!",
        salvage_vars,
    )
    assert salvage_vars or salvage_mappings, "No `salvage_mappings` given!"
    assert all(isinstance(v, str) or callable(v) for v in salvage_mappings.values()), (
        "Bad `salvage_mappings`:",
        salvage_mappings,
    )

    ## Merge vars-mapping to save.
    for var in salvage_vars:
        if var not in salvage_mappings:
            salvage_mappings[var] = var

    try:
        annotations = getattr(ex, annotation, None)
        if not isinstance(annotations, dict):
            annotations = {}
            setattr(ex, annotation, annotations)

        ## Salvage those asked
        for dst_key, src in salvage_mappings.items():
            try:
                salvaged_value = src(locs) if callable(src) else locs.get(src)
                annotations.setdefault(dst_key, salvaged_value)
            except Exception as ex:
                log.warning(
                    "Suppressed error while salvaging jetsam item (%r, %r): %r"
                    % (dst_key, src, ex)
                )
    except Exception as ex2:
        log.warning("Suppressed error while annotating exception: %r", ex2, exc_info=1)
        raise ex2


class PlotArgs(NamedTuple):
    """
    All the args of a :meth:`.Plottable.plot()` call,

    check this method for a more detailed  explanation of its attributes.
    """

    #: who is the caller
    plottable: "Plottable" = None
    #: what to plot (or the "overlay" when calling :meth:`Plottable.plot()`)
    graph: "nx.Graph" = None
    #: The name of the graph in the dot-file (important for cmaps).
    name: str = None
    #: the list of execution plan steps.
    steps: Collection = None
    #: the list of input names .
    inputs: Collection = None
    #: the list of output names .
    outputs: Collection = None
    #: Contains the computed results, which might be different from :attr:`plottable`.
    solution: "graphtik.network.Solution" = None
    #: a mapping of nodes to cluster names
    clusters: Mapping = None
    #: If given, overrides :active plotter`.
    plotter: "graphtik.plot.Plotter" = None
    #: If given, overrides :term:`plot theme` plotter will use.
    theme: "graphtik.plot.Theme" = None

    #######
    # Internal item-args for :meth:`.Plotter._make_node()` etall.
    #
    #: Where to add graphviz nodes & stuff.
    dot: "pydot.Dot" = None
    #: The node (data(str) or :class:`Operation`) or edge as gotten from nx-graph.
    nx_item: Any = None
    #: Attributes gotten from nx-graph for the given graph/node/edge.
    #: They are NOT a clone, so any modifications affect the nx `graph`.
    nx_attrs: dict = None
    #: The pydot-node/edge created
    dot_item: Any = None
    #: Collect the actual clustered `dot_nodes` among the given nodes.
    clustered: dict = None

    #######
    # Render args
    #
    #: jupyter configuration overrides
    jupyter_render: Mapping = None
    #: where to write image or show in a matplotlib window
    filename: Union[str, bool, int] = None

    def clone_or_merge_graph(self, base_graph) -> "PlotArgs":
        """
        Overlay :attr:`graph` over `base_graph`, or clone `base_graph`, if no attribute.

        :return:
            the updated plot_args
        """

        if self.graph:
            import networkx as nx

            graph = nx.compose(base_graph, self.graph)
        else:
            graph = base_graph.copy()  # cloned, to freely annotate downstream

        return self._replace(graph=graph)

    def with_defaults(self, *args, **kw) -> "PlotArgs":
        """Replace only fields with `None` values."""
        return self._replace(
            **{k: v for k, v in dict(*args, **kw).items() if getattr(self, k) is None}
        )

    @property
    def kw_render_pydot(self) -> dict:
        return {k: getattr(self, k) for k in self._fields[-2:]}


## Defined here, to avoid subclasses importing `plot` module.
class Plottable(abc.ABC):
    """
    Classes wishing to plot their graphs should inherit this and ...

    implement property ``plot`` to return a "partial" callable that somehow
    ends up calling  :func:`.plot.render_pydot()` with the `graph` or any other
    args bound appropriately.
    The purpose is to avoid copying this function & documentation here around.
    """

    def plot(
        self,
        filename: Union[str, bool, int] = None,
        show=None,
        *,
        plotter: "graphtik.plot.Plotter" = None,
        theme: "graphtik.plot.Theme" = None,
        graph: "networkx.Graph" = None,
        name=None,
        steps=None,
        inputs=None,
        outputs=None,
        solution: "graphtik.network.Solution" = None,
        clusters: Mapping = None,
        jupyter_render: Union[None, Mapping, str] = None,
    ) -> "pydot.Dot":
        """
        Entry-point for plotting ready made operation graphs.

        :param str filename:
            Write a file or open a `matplotlib` window.

            - If it is a string or file, the diagram is written into the file-path

              Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
              call :func:`.plot.supported_plot_formats()` for more.

            - If it IS `True`, opens the  diagram in a  matplotlib window
              (requires `matplotlib` package to be installed).

            - If it equals `-1`, it mat-plots but does not open the window.

            - Otherwise, just return the ``pydot.Dot`` instance.

            :seealso: :attr:`.PlotArgs.filename`, :meth:`.Plotter.render_pydot()`
        :param plottable:
            the :term:`plottable` that ordered the plotting.
            Automatically set downstreams to one of::

                op | netop | net | plan | solution | <missing>

            :seealso: :attr:`.PlotArgs.plottable`
        :param plotter:
            the :term:`plotter` to handle plotting; if none, the :term:`active plotter`
            is used by default.

            :seealso: :attr:`.PlotArgs.plotter`
        :param theme:
            Any :term:`plot theme` overrides; if none, the :attr:`.Plotter.default_theme`
            of the :term:`active plotter` is used.

            :seealso: :attr:`.PlotArgs.theme`
        :param name:
            if not given, dot-lang graph would is named "G"; necessary to be unique
            when referring to generated CMAPs.
            No need to quote it, handled by the plotter, downstream.

            :seealso: :attr:`.PlotArgs.name`
        :param str graph:
            (optional) A :class:`nx.Digraph` with overrides to merge with the graph provided
            by underlying plottables (translated by the :term:`active plotter`).

            It may contain "public" or "private *graph*, *node* & *edge* attributes:

            - "private" attributes: those starting with underscore(``_``),
              handled by :term:`plotter`:

              - ``_fn_link_target`` & ``_fn_link_target`` *(node)*: if truthy,
                override result 2-tuple of :meth:`_get_fn_link()`.
              - ``_op_tooltip`` & ``_fn_tooltip`` *(node)*: if truthy,
                override those derrived from :meth:`_make_op_tooltip()` &
                :meth:`_make_op_tooltip()`.
              - ``_no_plot``: nodes and/or edges skipped from plotting
                (see *"Examples:"* section, below)

            - "public" attributes: reaching `Graphviz`_ as-is, e.g.
              to set ``spline -> ortho`` on the graph attributes
              (this can also be achieved by modified :term:`plot theme`).

              .. Note::

                Remember to escape those values as `Graphviz`_ HTML-Like strings
                (use :func:`.plot.graphviz_html_string()`).

            :seealso: :attr:`.PlotArgs.graph`
        :param inputs:
            an optional name list, any nodes in there are plotted
            as a "house"

            :seealso: :attr:`.PlotArgs.inputs`
        :param outputs:
            an optional name list, any nodes in there are plotted
            as an "inverted-house"

            :seealso: :attr:`.PlotArgs.outputs`
        :param solution:
            an optional dict with values to annotate nodes, drawn "filled"
            (currently content not shown, but node drawn as "filled").
            It extracts more infos from a :class:`.Solution` instance, such as,
            if `solution` has an ``executed`` attribute, operations contained in it
            are  drawn as "filled".

            :seealso: :attr:`.PlotArgs.solution`
        :param clusters:
            an optional mapping of nodes --> cluster-names, to group them
        :param jupyter_render:
            a nested dictionary controlling the rendering of graph-plots in Jupyter cells,
            if `None`, defaults to :data:`jupyter_render`; you may modify it in place
            and apply for all future calls (see :ref:`jupyter_rendering`).

            :seealso: :attr:`.PlotArgs.jupyter_render`
        :param show:
            .. deprecated:: v6.1.1
                Merged with `filename` param (filename takes precedence).

        :return:
            a |pydot.Dot|_ instance
            (for reference to as similar API to |pydot.Dot|_ instance, visit:
            https://pydotplus.readthedocs.io/reference.html#pydotplus.graphviz.Dot)

            The |pydot.Dot|_ instance returned is rendered directly in *Jupyter/IPython*
            notebooks as SVG images (see :ref:`jupyter_rendering`).

        Note that the `graph` argument is absent - Each Plottable provides
        its own graph internally;  use directly :func:`.render_pydot()` to provide
        a different graph.

        .. image:: images/GraphtikLegend.svg
            :alt: Graphtik Legend

        *NODES:*

        oval
            function
        egg
            subgraph operation
        house
            given input
        inversed-house
            asked output
        polygon
            given both as input & asked as output (what?)
        square
            intermediate data, neither given nor asked.
        red frame
            evict-instruction, to free up memory.
        filled
            data node has a value in `solution` OR function has been executed.
        thick frame
            function/data node in execution `steps`.

        *ARROWS*

        solid black arrows
            dependencies (source-data *need*-ed by target-operations,
            sources-operations *provides* target-data)
        dashed black arrows
            optional needs
        blue arrows
            sideffect needs/provides
        wheat arrows
            broken dependency (``provide``) during pruning
        green-dotted arrows
            execution steps labeled in succession

        To generate the **legend**, see :func:`.legend()`.


        **Examples:**

        >>> from graphtik import compose, operation
        >>> from graphtik.modifiers import optional
        >>> from operator import add

        >>> netop = compose("netop",
        ...     operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
        ...     operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(lambda a, b=1: a-b),
        ...     operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
        ... )

        >>> netop.plot(True);                       # plot just the graph in a matplotlib window # doctest: +SKIP
        >>> inputs = {'a': 1, 'b1': 2}
        >>> solution = netop(**inputs)              # now plots will include the execution-plan

        >>> netop.plot('plot1.svg', inputs=inputs, outputs=['asked', 'b1'], solution=solution);           # doctest: +SKIP
        >>> dot = netop.plot(solution=solution);    # just get the `pydot.Dot` object, renderable in Jupyter
        >>> print(dot)
        digraph netop {
        fontname=italic;
        label=<netop>;
        <a> [fillcolor=wheat, margin="0.02,0.02", shape=invhouse, style=filled, tooltip="(int) 1"];
        ...

        .. graphtik::

        .. TODO: move advanced plot examples from API --> tutorial page

        You may use the :attr:`PlotArgs.graph` overlay to skip certain nodes (or edges)
        from the plots:

        >>> import networkx as nx

        >>> g = nx.DiGraph()  # the overlay
        >>> to_hide = netop.net.find_op_by_name("sub")
        >>> g.add_node(to_hide, _no_plot=True)
        >>> dot = netop.plot(graph=g)
        >>> assert "<sub>" not in str(dot), str(dot)

        .. graphtik::

        """
        kw = locals().copy()

        del kw["self"]
        show = kw.pop("show", None)
        if show:
            import warnings

            warnings.warn(
                "Argument `plot` has merged with `filename` and will be deleted soon.",
                DeprecationWarning,
            )
            if not filename:
                kw["filename"] = show

        plot_args = PlotArgs(**kw)

        from .plot import Plotter, get_active_plotter

        if plotter and not isinstance(plotter, Plotter):
            raise ValueError(f"Invalid `plotter` argument given: {plotter}")
        plot_args = plot_args._replace(plotter=plotter or get_active_plotter())

        plot_args = self.prepare_plot_args(plot_args)
        assert plot_args.graph, plot_args

        return plot_args.plotter.plot(plot_args)

    @abc.abstractmethod
    def prepare_plot_args(self, plot_args: PlotArgs) -> PlotArgs:
        """
        Called by :meth:`plot()` to create the nx-graph and other plot-args, e.g. solution.

        - Clone the graph or merge it with the one in the `plot_args`
          (see :meth:`PlotArgs.clone_or_merge_graph()`.

        - For the rest args, prefer :meth:`PlotArgs.with_defaults()` over :meth:`._replace()`,
          not to override user args.
        """
        pass
