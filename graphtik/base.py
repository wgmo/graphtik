# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import logging
from collections import namedtuple

try:
    from collections import abc
except ImportError:
    import collections as abc

from . import plot


log = logging.getLogger(__name__)


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


class Operation(object):
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
        raise NotImplementedError("Abstract %r! cannot compute()!" % self)

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

        def aslist(i):
            if i and not isinstance(i, str):
                return list(i)
            return i

        return u"%s(name='%s', needs=%s, provides=%s)" % (
            self.__class__.__name__,
            getattr(self, "name", None),
            aslist(getattr(self, "needs", None)),
            aslist(getattr(self, "provides", None)),
        )


class NetworkOperation(Operation, plot.Plotter):
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
