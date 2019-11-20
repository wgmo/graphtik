# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""About operation nodes (but not net-ops to break cycle)."""

import abc
import logging
from collections.abc import Hashable, Iterable, Mapping

from boltons.setutils import IndexedSet as iset

from .base import Plotter, aslist, jetsam
from .modifiers import optional, sideffect

log = logging.getLogger(__name__)


def reparse_operation_data(name, needs, provides):
    """
    Validate & reparse operation data as lists.

    As a separate function to be reused by client code
    when building operations and detect errors aearly.
    """

    if not isinstance(name, Hashable):
        raise ValueError(f"Operation `name` must be hashable, got: {name}")

    # Allow single string-value for needs parameter
    needs = aslist(needs, "needs", allowed_types=(list, tuple))
    if not all(isinstance(i, str) for i in needs):
        raise ValueError(f"All `needs` must be str, got: {needs!r}")

    # Allow single value for provides parameter
    provides = aslist(provides, "provides", allowed_types=(list, tuple))
    if not all(isinstance(i, str) for i in provides):
        raise ValueError(f"All `provides` must be str, got: {provides!r}")

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

    def __init__(
        self, fn=None, name=None, needs=None, provides=None, *, returns_dict=None
    ):
        self.fn = fn
        self.returns_dict = returns_dict
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
        returns_dict_marker = self.returns_dict and "{}" or ""
        return (
            f"FunctionalOperation(name={self.name!r}, needs={needs!r}, "
            f"provides={provides!r}, fn{returns_dict_marker}={fn_name!r})"
        )

    def _zip_results_with_provides(self, results, real_provides: iset) -> dict:
        """Zip results with expected "real" (without sideffects) `provides`."""
        if not real_provides:  # All outputs were sideffects?
            if results:
                ## Do not scream,
                #  it is common to call a function for its sideffects,
                # which happens to return an irrelevant value.
                log.warning(
                    "Ignoring result(%s) because no `provides` given!\n  %s",
                    results,
                    self,
                )
            results = {}
        elif not self.returns_dict:
            nexpected = len(real_provides)

            if nexpected > 1 and (
                not isinstance(results, Iterable) or len(results) != nexpected
            ):
                raise ValueError(
                    f"Expected x{nexpected} ITERABLE results, got: {results}"
                )

            if nexpected == 1:
                results = [results]

            results = dict(zip(real_provides, results))

        if self.returns_dict:
            if not isinstance(results, Mapping):
                raise ValueError(f"Expected dict-results, got: {results}\n  {self}")
        if set(results) != real_provides:
            raise ValueError(
                f"Results({results}) mismatched provides({real_provides})!\n  {self}"
            )

        return results

    def compute(self, named_inputs, outputs=None) -> dict:
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

            results_fn = self.fn(*args, **optionals)

            provides = iset(n for n in self.provides if not isinstance(n, sideffect))
            results_op = self._zip_results_with_provides(results_fn, provides)

            if outputs:
                outputs = set(n for n in outputs if not isinstance(n, sideffect))
                # Ignore sideffect outputs.
                results_op = {
                    key: val for key, val in results_op.items() if key in outputs
                }

            return results_op
        except Exception as ex:
            jetsam(
                ex,
                locals(),
                "outputs",
                "provides",
                "results_fn",
                "results_op",
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
        If more than one given, those must be returned in an iterable,
        unless `returns_dict` is true, in which cae a dictionary with as many
        elements must be returned
    :param bool returns_dict:
        if true, it means the `fn` returns a dictionary with all `provides`,
        and no further processing is done on them.

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

    def __init__(
        self, fn=None, *, name=None, needs=None, provides=None, returns_dict=None
    ):
        self.withset(
            fn=fn, name=name, needs=needs, provides=provides, returns_dict=returns_dict
        )

    def withset(
        self, *, fn=None, name=None, needs=None, provides=None, returns_dict=None
    ) -> "operation":
        if fn is not None:
            self.fn = fn
        if name is not None:
            self.name = name
        if needs is not None:
            self.needs = needs
        if provides is not None:
            self.provides = provides
        if returns_dict is not None:
            self.returns_dict = returns_dict

        return self

    def __call__(
        self, fn=None, *, name=None, needs=None, provides=None, returns_dict=None
    ) -> FunctionalOperation:
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

        self.withset(
            fn=fn, name=name, needs=needs, provides=provides, returns_dict=returns_dict
        )

        return FunctionalOperation(**vars(self))

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        needs = aslist(self.needs, "needs")
        provides = aslist(self.provides, "provides")
        fn_name = self.fn and getattr(self.fn, "__name__", str(self.fn))
        return f"operation(name={self.name!r}, needs={needs!r}, provides={provides!r}, fn={fn_name!r})"
