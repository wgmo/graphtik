# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import abc
import logging
from collections import namedtuple

log = logging.getLogger(__name__)

from .modifiers import optional


def aslist(i):
    """utility used through out"""
    if i and not isinstance(i, str):
        return list(i)
    return i


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
        They take precendance over`salvae_vars`.

    :raise:
        any exception raised by the wrapped function, annotated with values
        assigned as atrributes on this context-manager

    - Any attrributes attached on this manager are attached as a new dict on
      the raised exception as new  ``jetsam`` attrribute with a dict as value.
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
                    "Supressed error while salvaging jetsam item (%r, %r): %r"
                    % (dst_key, src, ex)
                )
    except Exception as ex2:
        log.warning("Supressed error while annotating exception: %r", ex2, exc_info=1)
        raise ex2

    raise  # noqa #re-raise without ex-arg, not to insert my frame


class Operation(abc.ABC):
    """An abstract class representing a data transformation by :meth:`.compute()`."""

    def __init__(self, name=None, needs=None, provides=None, **kwargs):
        """
        Create a new layer instance.
        Names may be given to this layer and its inputs and outputs. This is
        important when connecting layers and data in a Network object, as the
        names are used to construct the graph.

        :param str name:
            The name the operation (e.g. conv1, conv2, etc..)

        :param list needs:
            Names of input data objects this layer requires.

        :param list provides:
            Names of output data objects this provides.

        """

        # (Optional) names for this layer, and the data it needs and provides
        self.name = name
        self.needs = needs
        self.provides = provides

        # call _after_init as final step of initialization
        self._after_init()

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and self.name == getattr(other, "name", None))

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    @abc.abstractmethod
    def compute(self, named_inputs, outputs=None):
        """
        Compute from a given set of inputs an optional set of outputs.

        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list values representing
            the results of running the feed-forward computation on
            ``inputs``.
        """
        pass

    def _validate(self):
        if not self.name:
            raise ValueError(f"Operation needs a name, got: {self.name}")

        needs = self.needs
        # Allow single value for needs parameter
        if isinstance(needs, str) and not isinstance(needs, optional):
            needs = [needs]
        if not needs:
            raise ValueError(f"Empty `needs` given: {needs!r}")
        if not all(n for n in needs):
            raise ValueError(f"One item in `needs` is null: {needs!r}")
        if not isinstance(needs, (list, tuple)):
            raise ValueError(f"Bad `needs`, not (list, tuple): {needs!r}")
        self.needs = needs

        # Allow single value for provides parameter
        provides = self.provides
        if isinstance(provides, str):
            provides = [provides]
        if provides and not all(n for n in provides):
            raise ValueError(f"One item in `provides` is null: {provides!r}")
        provides = provides or ()
        if not isinstance(provides, (list, tuple)):
            raise ValueError(f"Bad `provides`, not (list, tuple): {provides!r}")
        self.provides = provides or ()

    def _after_init(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``name``, ``needs`` and ``provides``
        attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        clsname = type(self).__name__
        needs = aslist(self.needs)
        provides = aslist(self.provides)
        return f"{clsname}(name={self.name!r}, needs={needs!r}, provides={provides!r})"


class Plotter(abc.ABC):
    """
    Classes wishing to plot their graphs should inherit this and ...

    implement property ``plot`` to return a "partial" callable that somehow
    ends up calling  :func:`plot.render_pydot()` with the `graph` or any other
    args binded appropriately.
    The purpose is to avoid copying this function & documentation here around.
    """

    def plot(self, filename=None, show=False, **kws):
        """
        :param str filename:
            Write diagram into a file.
            Common extensions are ``.png .dot .jpg .jpeg .pdf .svg``
            call :func:`plot.supported_plot_formats()` for more.
        :param show:
            If it evaluates to true, opens the  diagram in a  matplotlib window.
            If it equals `-1`, it plots but does not open the Window.
        :param inputs:
            an optional name list, any nodes in there are plotted
            as a "house"
        :param outputs:
            an optional name list, any nodes in there are plotted
            as an "inverted-house"
        :param solution:
            an optional dict with values to annotate nodes, drawn "filled"
            (currently content not shown, but node drawn as "filled")
        :param executed:
            an optional container with operations executed, drawn "filled"
        :param title:
            an optional string to display at the bottom of the graph
        :param node_props:
            an optional nested dict of Grapvhiz attributes for certain nodes
        :param edge_props:
            an optional nested dict of Grapvhiz attributes for certain edges
        :param clusters:
            an optional mapping of nodes --> cluster-names, to group them

        :return:
            A ``pydot.Dot`` instance.
            NOTE that the returned instance is monkeypatched to support
            direct rendering in *jupyter cells* as SVG.


        Note that the `graph` argument is absent - Each Plotter provides
        its own graph internally;  use directly :func:`render_pydot()` to provide
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
        blue frame
            pinned-instruction, not to overwrite intermediate inputs.
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


        To generate the **legend**, see :func:`legend()`.

        **Sample code:**

        >>> from graphtik import compose, operation
        >>> from graphtik.modifiers import optional
        >>> from operator import add

        >>> graphop = compose(name="graphop")(
        ...     operation(name="add", needs=["a", "b1"], provides=["ab1"])(add),
        ...     operation(name="sub", needs=["a", optional("b2")], provides=["ab2"])(lambda a, b=1: a-b),
        ...     operation(name="abb", needs=["ab1", "ab2"], provides=["asked"])(add),
        ... )

        >>> graphop.plot(show=True);           # plot just the graph in a matplotlib window # doctest: +SKIP
        >>> inputs = {'a': 1, 'b1': 2}
        >>> solution = graphop(inputs)           # now plots will include the execution-plan

        >>> graphop.plot('plot1.svg', inputs=inputs, outputs=['asked', 'b1'], solution=solution);           # doctest: +SKIP
        >>> dot = graphop.plot(solution=solution);   # just get the `pydoit.Dot` object, renderable in Jupyter
        >>> print(dot)
        digraph G {
        fontname=italic;
        label=graphop;
        a [fillcolor=wheat, shape=invhouse, style=filled];
        ...
        ...
        """
        from .plot import render_pydot

        dot = self._build_pydot(**kws)
        return render_pydot(dot, filename=filename, show=show)

    @abc.abstractmethod
    def _build_pydot(self, **kws):
        pass
