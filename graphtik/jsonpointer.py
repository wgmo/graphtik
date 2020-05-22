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
import operator
from collections.abc import Sequence
from functools import partial
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd


def escape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return part.replace("~", "~0").replace("/", "~1")


def unescape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return part.replace("~1", "/").replace("~0", "~")


# TODO: DROP iter_jsonpointer_parts
def iter_jsonpointer_parts(path: str) -> Iterable[str]:
    """
    Generates the `path` parts according to jsonpointer spec.

    :param path:
        a path to resolve within document
    :return:
        The parts of the path as generator), without
        converting any step to int, and None if None.

    :author: Julian Berman, ankostis

    **Examples:**

        >>> list(iter_jsonpointer_parts('/a/b'))
        ['a', 'b']

        >>> list(iter_jsonpointer_parts('/a//b'))
        ['a', '', 'b']

        >>> list(iter_jsonpointer_parts('/'))
        ['']

        >>> list(iter_jsonpointer_parts(''))
        []


    But paths are strings begining (NOT_MPL: but not ending) with slash('/')::

        >>> list(iter_jsonpointer_parts(None))
        Traceback (most recent call last):
        AttributeError: 'NoneType' object has no attribute 'split'

        >>> list(iter_jsonpointer_parts('a'))
        Traceback (most recent call last):
        ValueError: Jsonpointer-path(a) must start with '/'!

        #>>> list(iter_jsonpointer_parts('/a/'))
        #Traceback (most recent call last):
        #ValueError: Jsonpointer-path(a) must NOT ends with '/'!

    """
    parts = path.split("/")
    if parts.pop(0) != "":
        raise ValueError(f"Jsonpointer-path({path}) must start with '/'!")

    return (unescape_jsonpointer_part(part) for part in parts)


# TODO: rename iter_jsonpointer_parts_relaxed() --> iter_path()
def iter_jsonpointer_parts_relaxed(jsonpointer: str) -> Iterable[str]:
    """
    Like :func:`iter_jsonpointer_parts()` but accepting also non-absolute paths.

    :return:
        The 1st step of absolute-paths is always ''.

    **Examples:**

        >>> list(iter_jsonpointer_parts_relaxed('a'))
        ['a']
        >>> list(iter_jsonpointer_parts_relaxed('a/'))
        ['a', '']
        >>> list(iter_jsonpointer_parts_relaxed('a/b'))
        ['a', 'b']

        >>> list(iter_jsonpointer_parts_relaxed('/a'))
        ['', 'a']
        >>> list(iter_jsonpointer_parts_relaxed('/a/'))
        ['', 'a', '']

        >>> list(iter_jsonpointer_parts_relaxed('/'))
        ['', '']

        >>> list(iter_jsonpointer_parts_relaxed(''))
        ['']

    """
    return (unescape_jsonpointer_part(part) for part in jsonpointer.split("/"))


# TODO: DROP resolve_jsonpointer(), keep only `resolve_path()`.
def resolve_jsonpointer(doc, path: Union[str, Sequence], default=...):
    """
    Resolve a json-pointer `path` within the referenced `doc`.

    :param doc:
        the referrant document
    :param path:
        a jsonpointer to resolve within `doc` document
    :param default:
        the value to return if `path` does not resolve; by default, it raises.

    :return:
        the resolved doc-item
    :raises ResolveError:
        if `path` cannot resolve and no `default` given
    :raises ValueError:
        if path not an absolute path (does not start with a slash(`/``)).

    **Examples:**

        >>> dt = {
        ...     'pi':3.14,
        ...     'foo':'bar',
        ...     'df': pd.DataFrame(np.ones((3,2)), columns=list('VN')),
        ...     'sub': {
        ...         'sr': pd.Series({'abc':'def'}),
        ...     }
        ... }
        >>> resolve_jsonpointer(dt, '/pi', default=...)
        3.14

        >>> resolve_jsonpointer(dt, '/pi/BAD')
        Traceback (most recent call last):
        graphtik.jsonpointer.ResolveError: '/pi/BAD' @ BAD

        >>> resolve_jsonpointer(dt, '/pi/BAD', 'Hi!')
        'Hi!'

    :author: Julian Berman, ankostis
    """
    parts = iter_jsonpointer_parts(path)  # if isinstance(path, str) else path

    for part in parts:
        if isinstance(doc, Sequence) and not isinstance(doc, str):
            # Array indexes should be turned into integers
            try:
                part = int(part)
            except ValueError:
                pass
        try:
            doc = doc[part]
        except (TypeError, LookupError):
            if default is ...:
                raise ResolveError(path, part)
            else:
                return default

    return doc


class ResolveError(KeyError):
    """
    A :class:`KeyError` raised when a json-pointer path does not :func:`resolve <.resolve_path>`.
    """

    # __init__(path, step)

    def __str__(self):
        return f"{self.key!r} @ {self.step}"

    def __repr__(self):
        return f"{type(self).__name__}({self.key!r} @ {self})"

    @property
    def key(self):
        """the json-pointer path that failed to resolve"""
        return self.args[0]

    @property
    def step(self):
        """the step where the resolution stopped"""
        return self.args[1]


class _AbsoluteError(Exception):
    pass


def resolve_path(
    doc: Union[Sequence, Mapping],
    path: Union[str, Iterable[str]],
    default=...,
    root=...,
    index_attributes=None,
):
    """
    Resolve a json-pointer `path` within the referenced `doc`.

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
        From where to start resolving absolute paths or double-slashes(``//``).
        If ``None``, only relative paths allowed; by default,
        the given `doc` is assumed as root (so absolute paths are also accepted).
    :param index_attributes:
        If true, a last ditch effort is made for each part, whether it matches
        the name of an attribute of the parent item.

    :return:
        the resolved doc-item
    :raises ResolveError:
        if `path` cannot resolve and no `default` given
    :raises ValueError:
        if `path` was an absolute path a  ``None`` `root` had been given.

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

    def resolve_root_or_fail(d, p):
        """the last resolver"""
        if p == "":
            if root is None:
                raise _AbsoluteError(
                    f"Absolute json-pointer `path` is not allowed, got: {path!r}"
                )
            return root
        raise ValueError()

    part_resolvers = [
        lambda d, p: d[int(p)],
        operator.getitem,
    ]
    if index_attributes:
        part_resolvers.append(getattr)

    if root is ...:
        root = doc

    for part in iter_jsonpointer_parts_relaxed(path):
        if part == "":
            if root is None:
                raise _AbsoluteError(
                    f"Absolute json-pointer `path` is not allowed, got: {path!r}"
                )
            doc = root
            continue

        # For sequences, try first by index.
        start_i = 0 if isinstance(doc, Sequence) and not isinstance(doc, str) else 1
        for resolver in part_resolvers[start_i:]:
            try:
                doc = resolver(doc, part)
                break
            except _AbsoluteError as ex:
                raise ValueError(str(ex)) from None
            except (ValueError, TypeError, LookupError, AttributeError):
                pass
        else:
            if default is ...:
                raise ResolveError(path, part)

            return default

    return doc


def set_jsonpointer(doc, jsonpointer, value, object_factory=dict, relaxed=False):
    """
    Resolve a ``jsonpointer`` within the referenced ``doc``.

    # FIXME: jsonp_set must also support attributes

    :param doc: the referrant document
    :param str jsonpointer: a jsonpointer to the node to modify
    :param relaxed: when true, json-paths may not start with slash(/)
    :raises: ValueError (if jsonpointer empty, missing, invalid-contet)
    """
    splitter = iter_jsonpointer_parts_relaxed if relaxed else iter_jsonpointer_parts
    parts = list(splitter(jsonpointer))

    # Will scream if used on 1st iteration.
    #
    pdoc = None
    ppart = None
    for i, part in enumerate(parts):
        if isinstance(doc, Sequence) and not isinstance(doc, str):
            # Array indexes should be turned into integers
            #
            doclen = len(doc)
            if part == "-":
                part = doclen
            else:
                try:
                    part = int(part)
                except ValueError:
                    raise TypeError(
                        "Expected numeric index(%s) for sequence at (%r)[%i]"
                        % (part, jsonpointer, i)
                    )
                else:
                    if part > doclen:
                        raise IndexError(
                            "Index(%s) out of bounds(%i) of (%r)[%i]"
                            % (part, doclen, jsonpointer, i)
                        )
        try:
            ndoc = doc[part]
        except (LookupError):
            break  # Branch-extension needed.
        except (TypeError):  # Maybe indexing a string...
            ndoc = object_factory()
            pdoc[ppart] = ndoc
            doc = ndoc
            break  # Branch-extension needed.

        doc, pdoc, ppart = ndoc, doc, part
    else:
        doc = pdoc  # If loop exhausted, cancel last assignment.

    # Build branch with value-leaf.
    #
    nbranch = value
    for part2 in reversed(parts[i + 1 :]):
        ndoc = object_factory()
        ndoc[part2] = nbranch
        nbranch = ndoc

    # Attach new-branch.
    try:
        doc[part] = nbranch
    # Inserting last sequence-element raises IndexError("list assignment index
    # out of range")
    except IndexError:
        doc.append(nbranch)
