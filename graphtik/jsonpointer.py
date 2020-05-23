# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
Utility for :term:`json pointer path` modifier

Copied from pypi/pandalone.


.. doctest::
    :hide:

    .. Workaround sphinx-doc/sphinx#6590

    >>> from graphtik.jsonpointer import *
    >>> __name__ = "graphtik.jsonpointer"
"""
import logging
import operator
from collections import abc as cabc
from functools import partial
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def escape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return part.replace("~", "~0").replace("/", "~1")


def unescape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return part.replace("~1", "/").replace("~0", "~")


def iter_path(jsonpointer: str) -> Iterable[str]:
    """
    Generates the `path` parts according to jsonpointer spec.

    :param path:
        a path to resolve within document
    :return:
        The parts of the path as generator), without
        converting any step to int, and None if None.
        (the 1st step of absolute-paths is always ``''``)

    :author: Julian Berman, ankostis

    **Examples:**

        >>> list(iter_path('a'))
        ['a']
        >>> list(iter_path('a/'))
        ['a', '']
        >>> list(iter_path('a/b'))
        ['a', 'b']

        >>> list(iter_path('/a'))
        ['', 'a']
        >>> list(iter_path('/a/'))
        ['', 'a', '']

        >>> list(iter_path('/'))
        ['', '']

        >>> list(iter_path(''))
        ['']

    """
    return (unescape_jsonpointer_part(part) for part in jsonpointer.split("/"))


class ResolveError(KeyError):
    """
    A :class:`KeyError` raised when a json-pointer path does not :func:`resolve <.resolve_path>`.
    """

    # __init__(path, step, i)

    def __str__(self):
        return f"{self.path!r} @ (#{self.index}) {self.step}"

    def __repr__(self):
        return f"{type(self).__name__}({self})"

    @property
    def path(self):
        """the json-pointer path that failed to resolve"""
        return self.args[0]

    @property
    def step(self):
        """the step where the resolution broke"""
        return self.args[1]

    @property
    def index(self) -> int:
        """the step position where the resolution broke"""
        return self.args[2]


def resolve_path(
    doc: Union[Sequence, Mapping],
    path: Union[str, Iterable[str]],
    default=...,
    root=...,
    descend_objects=None,
):
    """
    Resolve *roughly like* a json-pointer `path` within the referenced `doc`.

    :param doc:
        the current document to start searching `path`
        (which may be different than `root`)
    :param path:
        An absolute or relative json-pointer expression to resolve within `doc` document
        (or just the unescaped steps).

        .. Attention::
            Relative paths DO NOT support the json-pointer extension
            https://tools.ietf.org/id/draft-handrews-relative-json-pointer-00.html

    :param default:
        the value to return if `path` does not resolve; by default, it raises.
    :param root:
        From where to start resolving absolute paths or double-slashes(``//``), or
        final slashes.
        If ``None``, only relative paths allowed; by default,
        the given `doc` is assumed as root (so absolute paths are also accepted).
    :param descend_objects:
        If true, a last ditch effort is made for each part, whether it matches
        the name of an attribute of the parent item.

    :return:
        the resolved doc-item
    :raises ResolveError:
        if `path` cannot resolve and no `default` given
    :raises ValueError:
        if `path` was an absolute path a  ``None`` `root` had been given.

    In order to couple it with a sensible :func:`.set_path_value()`, it departs
    from the standard in these aspects:

    - Supports also relative paths (but not the official extension).
    - For arrays, it tries 1st as an integer, and then falls back to normal indexing
      (usefull when accessing pandas).
    - A ``/`` path does not bring the value of empty``''`` key but the whole document
      (aka the "root").
    - A double slash or a slash at the end of the path restarts from the root.


    **Examples:**

        >>> dt = {
        ...     'pi':3.14,
        ...     'foo':'bar',
        ...     'df': pd.DataFrame(np.ones((3,2)), columns=list('VN')),
        ...     'sub': {
        ...         'sr': pd.Series({'abc':'def'}),
        ...     }
        ... }
        >>> resolve_path(dt, '/pi', default=...)
        3.14

        >>> resolve_path(dt, 'df/V')
        0    1.0
        1    1.0
        2    1.0
        Name: V, dtype: float64

        >>> resolve_path(dt, '/pi/BAD', 'Hi!')
        'Hi!'

    :author: Julian Berman, ankostis
    """
    part_indexers = [
        lambda doc, part: doc[int(part)],
        operator.getitem,
        *((getattr,) if descend_objects else ()),
    ]

    if root is ...:
        root = doc

    parts = iter_path(path) if isinstance(path, str) else path
    for i, part in enumerate(parts):
        if part == "":
            if root is None:
                raise ValueError(
                    f"Absolute step #{i} of json-pointer path {path!r} without `root`!"
                )
            doc = root
            continue

        for indexer in part_indexers:
            try:
                doc = indexer(doc, part)
                break
            except Exception as ex:
                log.debug(
                    "indexer %s failed on step (#%i)%s of json-pointer path %r, due to: %s ",
                    indexer,
                    i,
                    part,
                    path,
                    ex,
                    exc_info=ex,
                )
                pass
        else:
            if default is ...:
                raise ResolveError(path, part, i)

            return default

    return doc


def is_collection(item):
    return not isinstance(item, str) and isinstance(item, cabc.Collection)


def list_scouter(doc, part, container_factory, overwrite):
    """
    Get `doc `list item  by (int) `part`, or create a new one from `container_factory`.

    NOTE: must kick before collection-scouter to handle special non-integer ``.`` index.
    """
    if part == "-":  # It means the position after the last item.
        item = container_factory()
        doc.append(item)
        return item

    part = int(part)  # Let it bubble, to try next (as key-item).
    if not overwrite:
        try:
            child = doc[part]
            if is_collection(child):
                return child
        except LookupError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    item = container_factory()
    doc[part] = item
    return item


def collection_scouter(doc, part, container_factory, overwrite):
    """Get item `part` from `doc` collection, or create a new ome from `container_factory`."""
    if not overwrite:
        try:
            child = doc[part]
            if is_collection(child):
                return child
        except LookupError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    item = container_factory()
    doc[part] = item
    return item


def object_scouter(doc, part, value, container_factory, overwrite):
    """Get attribute `part` from `doc` object, or create a new one from `container_factory`."""
    if not overwrite:
        try:
            child = getattr(doc, part)
            if is_collection(child):
                return child
        except AttributeError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    item = container_factory()
    setattr(doc, part, item)
    return item


def set_path_value(
    doc: Union[Sequence, Mapping],
    path: Union[str, Iterable[str]],
    value,
    container_factory=dict,
    root=...,
    descend_objects=None,
):
    """
    Resolve a ``jsonpointer`` within the referenced ``doc``.

    :param doc:
        the document to extend & insert `value`
    :param path:
        An absolute or relative json-pointer expression to resolve within `doc` document
        (or just the unescaped steps).

        For sequences (arrays), it supports the special index dahs(``-``) char,
        to refer to the position beyond the last item, as by the spec, BUT
        it does not raise - it always add a new item.

        .. Attention::
            Relative paths DO NOT support the json-pointer extension
            https://tools.ietf.org/id/draft-handrews-relative-json-pointer-00.html

    :param container_factory:
        a factory producing the collection instances to extend missing steps,
        (usually a mapping or a sequence)
    :param root:
        From where to start resolving absolute paths, double-slashes(``//``) or
        final slashes.
        If ``None``, only relative paths allowed; by default,
        the given `doc` is assumed as root (so absolute paths are also accepted).
    :param descend_objects:
        If true, a last ditch effort is made for each part, whether it matches
        the name of an attribute of the parent item.

    :raises: ValueError (if jsonpointer empty, missing, invalid-contet)
    """

    part_scouters = [
        list_scouter,
        collection_scouter,
        *((object_scouter,) if descend_objects else ()),
    ]

    if root is ...:
        root = doc

    parts = list(iter_path(path) if isinstance(path, str) else path)
    last_part = len(parts) - 1
    for i, part in enumerate(parts):
        if part == "":
            if root is None:
                raise ValueError(
                    f"Absolute step #{i} of json-pointer path {path!r} without `root`!"
                )
            doc = root
            continue

        if i == last_part:
            fact = lambda: value
            overwrite = True
        else:
            fact = container_factory
            overwrite = False

        for scouter in part_scouters:
            try:
                doc = scouter(doc, part, fact, overwrite)
                break
            # except TypeError:  # Maybe indexing a string... Replace it!
            #     steps[i - 1][parts[i - 1]] = container_factory()
            #     doc = scouter(doc, part, fact, overwrite)
            except Exception as ex:
                log.debug(
                    "scouter %s failed on step (#%i)%s of json-pointer path %r, due to: %s",
                    scouter,
                    i,
                    part,
                    path,
                    ex,
                    exc_info=ex,
                )
                pass
        else:
            raise ValueError(
                f"Failed setting step (#{i}){part} of json pointer path {path!r}"
                "\n  Check debug logs."
            )
