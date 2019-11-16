# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""About operation nodes (but not net-ops to break cycle)."""

import abc
from collections.abc import Hashable

from .base import Plotter, aslist, jetsam
from .modifiers import optional, sideffect


def reparse_operation_data(name, needs, provides):
    """
    Validate & reparse operation data as lists.

    As a separate function to be reused by client code
    when building operations and detect errors aearly.
    """

    if not isinstance(name, Hashable):
        raise ValueError(f"Operation needs a hashable object as `name`, got: {name}")

    # Allow single value for needs parameter
    if isinstance(needs, str) and not isinstance(needs, optional):
        needs = [needs]
    if not needs:
        raise ValueError(f"Empty `needs` given: {needs!r}")
    if not all(n for n in needs):
        raise ValueError(f"One item in `needs` is null: {needs!r}")
    if not isinstance(needs, (list, tuple)):
        raise ValueError(f"Bad `needs`, not (list, tuple): {needs!r}")

    # Allow single value for provides parameter
    if isinstance(provides, str):
        provides = [provides]
    if provides and not all(n for n in provides):
        raise ValueError(f"One item in `provides` is null: {provides!r}")
    provides = provides or ()
    if not isinstance(provides, (list, tuple)):
        raise ValueError(f"Bad `provides`, not (list, tuple): {provides!r}")

    return name, needs, provides


class Operation(abc.ABC):
    """An abstract class representing a data transformation by :meth:`.compute()`."""

    def __init__(self, name=None, needs=None, provides=None):
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

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        clsname = type(self).__name__
        needs = aslist(self.needs, "needs")
        provides = aslist(self.provides, "provides")
        return f"{clsname}(name={self.name!r}, needs={needs!r}, provides={provides!r})"


class FunctionalOperation(Operation):
    """Use operation() to build instances of this class instead"""

    def __init__(self, fn=None, name=None, needs=None, provides=None):
        self.fn = fn
        ## Set op-data early, for repr() to work on errors.
        Operation.__init__(self, name=name, needs=needs, provides=provides)
        if not fn or not callable(fn):
            raise ValueError(f"Operation was not provided with a callable: {self.fn}")
        ## Overwrite reparsed op-data.
        self.name, self.needs, self.provides = reparse_operation_data(
            name, needs, provides
        )

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        needs = aslist(self.needs, "needs")
        provides = aslist(self.provides, "provides")
        fn_name = self.fn and getattr(self.fn, "__name__", str(self.fn))
        return f"FunctionalOperation(name={self.name!r}, needs={needs!r}, provides={provides!r}, fn={fn_name!r})"

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
                results = {key: val for key, val in results if key in outputs}
            else:
                results = dict(results)

            return results
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

    This is an example of its use, based on the "builder pattern"::

        >>> from graphtik import operation

        >>> opb = operation(name='add_op')
        >>> opb.withset(needs=['a', 'b'])
        operation(name='add_op', needs=['a', 'b'], provides=[], fn=None)
        >>> opb.withset(provides='SUM', fn=sum)
        operation(name='add_op', needs=['a', 'b'], provides=['SUM'], fn='sum')

    You may keep calling ``withset()`` till you invoke a final ``__call__()``
    on the builder;  then you get the actual :class:`.FunctionalOperation` instance::

        >>> # Create `Operation` and overwrite function at the last moment.
        >>> opb(sum)
        FunctionalOperation(name='add_op', needs=['a', 'b'], provides=['SUM'], fn='sum')
    """

    fn = name = needs = provides = None

    def __init__(self, fn=None, *, name=None, needs=None, provides=None):
        self.withset(fn=fn, name=name, needs=needs, provides=provides)

    def withset(self, fn=None, *, name=None, needs=None, provides=None):
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
        needs = aslist(self.needs, "needs")
        provides = aslist(self.provides, "provides")
        fn_name = self.fn and getattr(self.fn, "__name__", str(self.fn))
        return f"operation(name={self.name!r}, needs={needs!r}, provides={provides!r}, fn={fn_name!r})"
