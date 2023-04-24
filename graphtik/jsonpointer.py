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
import re
from collections import abc as cabc
from functools import partial
from typing import (
    Any,
    Collection,
    Iterable,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

log = logging.getLogger(__name__)
UNSET = "%%UNSET%%"  # Change this in case of troubles...


def escape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return str(part).replace("~", "~0").replace("/", "~1")


def json_pointer(parts: Sequence[str]) -> str:
    """
    Escape & join `parts` into a jsonpointer path (inverse of :func:`.jsonp_path()`).

    **Examples:**

        >>> json_pointer(["a", "b"])
        'a/b'
        >>> json_pointer(['', "a", "b"])
        '/a/b'

        >>> json_pointer([1, "a", 2])
        '1/a/2'

        >>> json_pointer([""])
        ''
        >>> json_pointer(["a", ""])
        ''
        >>> json_pointer(["", "a", "", "b"])
        '/b'

        >>> json_pointer([])
        ''

        >>> json_pointer(["/", "~"])
        '~1/~0'
    """
    try:
        last_idx = len(parts) - 1 - parts[::-1].index("")
        parts = parts[last_idx:]
    except ValueError:
        pass  # no root after 1st step.
    return "/".join(escape_jsonpointer_part(i) for i in parts)


def prepend_parts(prefix_parts: Sequence[str], parts: Sequence[str]) -> Sequence[str]:
    """
    Prepend `prefix_parts` before given `parts` (unless they are rooted).

    Both `parts` & `prefix_parts` must have been produced by :meth:`.json_path()`
    so that any root(``""``) must come first, and must not be empty
    (except `prefix-parts`).

    **Examples:**

        >>> prepend_parts(["prefix"], ["b"])
        ['prefix', 'b']

        >>> prepend_parts(("", "prefix"), ["b"])
        ['', 'prefix', 'b']
        >>> prepend_parts(["prefix ignored due to rooted"], ("", "b"))
        ('', 'b')

        >>> prepend_parts([], ["b"])
        ['b']
        >>> prepend_parts(["prefix irrelevant"], [])
        Traceback (most recent call last):
        IndexError: list index out of range
    """
    if "" != parts[0]:
        parts = [*prefix_parts, *parts]
    return parts


def unescape_jsonpointer_part(part: str) -> str:
    """convert path-part according to the json-pointer standard"""
    return part.replace("~1", "/").replace("~0", "~")


def jsonp_path(jsonpointer: str) -> List[str]:
    """
    Generates the `path` parts according to jsonpointer spec.

    :param path:
        a path to resolve within document
    :return:
        The parts of the path as generator), without
        converting any step to int, and None if None.
        (the 1st step of absolute-paths is always ``''``)

    In order to support relative & absolute paths along with a sensible
    :func:`.set_path_value()`, it departs from the standard in these aspects:

    - A double slash or a slash at the end of the path restarts from the root.


    :author: Julian Berman, ankostis

    **Examples:**

        >>> jsonp_path('a')
        ['a']
        >>> jsonp_path('a/')
        ['']
        >>> jsonp_path('a/b')
        ['a', 'b']

        >>> jsonp_path('/a')
        ['', 'a']

        >>> jsonp_path('/')
        ['']
        >>> jsonp_path('')
        []

        >>> jsonp_path('a/b//c')
        ['', 'c']
        >>> jsonp_path('a//b////c')
        ['', 'c']
    """
    # Optimization: modifier caches splitted parts as a "_jsonp" attribute.
    parts = getattr(jsonpointer, "_jsonp", None)
    if parts is False:
        parts = [jsonpointer]
    elif parts is None:
        ## For empty-paths, the jsonpointer standard specifies
        #  they must return the whole document (satisfied by `resolve_path()`)
        #  but not what their *parts* should be.
        if jsonpointer == "":
            return []

        parts = [
            unescape_jsonpointer_part(part)
            for part in re.sub(".+(?:/$|/{2})", "/", jsonpointer).split("/")
        ]
        try:
            last_idx = -parts[::-1].index("") - 1 + len(parts)
            parts = parts[last_idx:]
        except ValueError:
            pass  # no root after 1st step.
        if "" in parts[1:]:
            last_idx = len(parts) - 1 - parts[::-1].index("")
            parts = parts[last_idx:]
    return parts


class ResolveError(KeyError):
    """
    A :class:`KeyError` raised when a json-pointer does not :func:`resolve <.resolve_path>`.
    """

    # __init__(path, step, i)

    def __str__(self):
        return (
            f'Failed resolving step (#{self.index}) "{self.part}" of path {self.path!r}.'
            "\n  Check debug logs."
        )

    @property
    def path(self):
        """the json-pointer that failed to resolve"""
        return self.args[0]

    @property
    def part(self):
        """the part where the resolution broke"""
        return self.args[1]

    @property
    def index(self) -> int:
        """the part's position where the resolution broke"""
        return self.args[2]


Doc = TypeVar("Doc", MutableMapping, MutableSequence)


def _log_overwrite(part, doc: Doc, child):
    log.warning(
        "Overwritting json-pointer part %r in a %i-len subdoc over scalar %r.",
        part,
        len(doc),
        child,
    )


def resolve_path(
    doc: Doc,
    path: Union[str, Iterable[str]],
    default=UNSET,
    root: Doc = UNSET,
    descend_objects=True,
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

    In order to support relative & absolute paths along with a sensible
    :func:`.set_path_value()`, it departs from the standard in these aspects:

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
        >>> resolve_path(dt, '/pi')
        3.14

        >>> resolve_path(dt, 'df/V')
        0    1.0
        1    1.0
        2    1.0
        Name: V, dtype: float64

        >>> resolve_path(dt, '/pi/BAD')
        Traceback (most recent call last):
        graphtik.jsonpointer.ResolveError: Failed resolving step (#2) "BAD" of path '/pi/BAD'.
          Check debug logs.


        >>> resolve_path(dt, '/pi/BAD', 'Hi!')
        'Hi!'

    :author: Julian Berman, ankostis
    """
    part_indexers = [
        operator.getitem,
        lambda doc, part: doc[int(part)],
        *((getattr,) if descend_objects else ()),
    ]

    if root is UNSET:
        root = doc

    parts = jsonp_path(path) if isinstance(path, str) else path
    for i, part in enumerate(parts):
        if part == "":
            if root is None:
                raise ValueError(
                    f"Absolute step #{i} of json-pointer {path!r} without `root`!"
                )
            doc = root
            continue

        for ii, indexer in enumerate(part_indexers):
            try:
                doc = indexer(doc, part)
                break
            except Exception as ex:
                if ii > 0:  # ignore int-indexing
                    log.debug(
                        "indexer %s failed on step (#%i)%s of json-pointer(%r) with doc(%s), due to: %s ",
                        indexer,
                        i,
                        part,
                        path,
                        doc,
                        ex,
                        # Don't log int(part) stack-traces.
                        exc_info=type(ex) is not ValueError
                        or not str(ex).startswith("invalid literal for int()"),
                    )
        else:
            if default is UNSET:
                raise ResolveError(path, part, i)

            return default

    return doc


def contains_path(
    doc: Doc,
    path: Union[str, Iterable[str]],
    root: Doc = UNSET,
    descend_objects=True,
) -> bool:
    """Test if `doc` has a value for json-pointer path by calling :func:`.resolve_path()`."""
    try:
        resolve_path(doc, path, root=root, descend_objects=descend_objects)
        return True
    except ResolveError:
        return False


def is_collection(item):
    return isinstance(item, cabc.Collection) and not isinstance(
        item, (str, cabc.ByteString)
    )


def list_scouter(doc: Doc, idx, mother, overwrite) -> Tuple[Any, Optional[Doc]]:
    """
    Get `doc `list item  by (int) `idx`, or create a new one from `mother`.

    :param mother:
        factory producing the child containers to extend missing steps,
        or the "child" value (when `overwrite` is true).

    :return:
        a 2-tuple (child, ``None``)

    NOTE: must come after collection-scouter due to special ``-`` index collision.
    """
    if idx == "-":  # It means the position after the last item.
        item = mother()
        doc.append(item)
        return item, None

    idx = int(idx)  # Let it bubble, to try next (as key-item).
    if overwrite and log.isEnabledFor(logging.WARNING):
        try:
            child = doc[idx]
        except LookupError:
            # Out of bounds, but ok for now, will break also on assignment, below.
            pass
        else:
            if not is_collection(child):
                _log_overwrite(idx, doc, child)
    elif not overwrite:
        try:
            child = doc[idx]
            if is_collection(child):
                return child, None
        except LookupError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    item = mother()
    doc[idx] = item
    return item, None


def collection_scouter(
    doc: Doc, key, mother, overwrite, concat_axis
) -> Tuple[Any, Optional[Doc]]:
    """
    Get item `key` from `doc` collection, or create a new ome from `mother`.

    :param mother:
        factory producing the child containers to extend missing steps,
        or the "child" value (when `overwrite` is true).

    :return:
        a 2-tuple (child, doc) where `doc` is not None if it needs to be replaced
        in its parent container (e.g. due to df-concat with value).
    """
    if (
        overwrite
        and log.isEnabledFor(logging.WARNING)
        and key in doc
        and not is_collection(doc[key])
    ):
        _log_overwrite(key, doc, doc[key])
    elif not overwrite:
        try:
            child = doc[key]
            if is_collection(child):
                return child, None
        except LookupError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    new_doc = None
    value = mother()

    if (
        concat_axis is not None
        and isinstance(doc, NDFrame)
        and isinstance(value, NDFrame)
    ):
        new_doc = pd.concat((doc, value), axis=concat_axis)
    else:
        doc[key] = value

    return value, new_doc


def object_scouter(doc: Doc, attr, mother, overwrite) -> Tuple[Any, Optional[Doc]]:
    """
    Get attribute `attr` in `doc` object, or create a new one from `mother`.

    :param mother:
        factory producing the child containers to extend missing steps,
        or the "child" value (when `overwrite` is true).

    :return:
        a 2-tuple (child, ``None``)
    """
    if overwrite and log.isEnabledFor(logging.WARNING) and hasattr(doc, attr):
        _log_overwrite(attr, doc, getattr(doc, attr))
    elif not overwrite:
        try:
            child = getattr(doc, attr)
            if is_collection(child):
                return child, None
        except AttributeError:
            # Ignore resolve errors, assignment errors below are more important.
            pass

    item = mother()
    setattr(doc, attr, item)
    return item, None


def set_path_value(
    doc: Doc,
    path: Union[str, Iterable[str]],
    value,
    container_factory=dict,
    root: Doc = UNSET,
    descend_objects=True,
    concat_axis: int = None,
):
    """
    Set `value` into a :term:`jsonp` `path` within the referenced `doc`.

    Special treatment (i.e. concat) if must insert a DataFrame into a DataFrame
    with steps ``.`` (vertical) and ``-`` (horizontal) denoting concatenation axis.

    :param doc:
        the document to extend & insert `value`
    :param path:
        An absolute or relative json-pointer expression to resolve within `doc` document
        (or just the unescaped steps).

        For sequences (arrays), it supports the special index dash(``-``) char,
        to refer to the position beyond the last item, as by the spec, BUT
        it does not raise - it always add a new item.

        .. Attention::
            Relative paths DO NOT support the json-pointer extension
            https://tools.ietf.org/id/draft-handrews-relative-json-pointer-00.html

    :param container_factory:
        a factory producing the container to extend missing steps
        (usually a mapping or a sequence).
    :param root:
        From where to start resolving absolute paths, double-slashes(``//``) or
        final slashes.
        If ``None``, only relative paths allowed; by default,
        the given `doc` is assumed as root (so absolute paths are also accepted).
    :param descend_objects:
        If true, a last ditch effort is made for each part, whether it matches
        the name of an attribute of the parent item.
    :param concat_axis:
        if 0 or 1, applies :term:'pandas concatenation` vertically or horizontally,
        by clipping last step when traversing it and doc & value are both Pandas objects.

    :raises ValueError:
        - if jsonpointer empty, missing, invalid-content
        - changed given `doc`/`root` (e.g due to concat-ed with value)

    See :func:`resolve_path()` for departures from the json-pointer standard
    """

    part_scouters = [
        partial(collection_scouter, concat_axis=concat_axis),
        list_scouter,
        *((object_scouter,) if descend_objects else ()),
    ]

    if root is UNSET:
        root = doc

    given_doc: Doc = doc  # Keep user input, to scream if must change (e.g. concat-ed).
    parent: Doc = None
    parent_part = None
    parts = jsonp_path(path) if isinstance(path, str) else list(path)
    last_part = len(parts) - 1
    for i, part in enumerate(parts):
        if part == "":
            if root is None:
                raise ValueError(
                    f"Absolute step #{i} of json-pointer {path!r} without `root`!"
                )
            doc = root
            parent = parent_part = None
            continue

        if i == last_part:
            fact = lambda: value
            overwrite = True
            if concat_axis is not None:
                ## Only the last part supports concatenation.
                part_scouters = part_scouters[:1]
        else:
            fact = container_factory
            overwrite = False

        for ii, scouter in enumerate(part_scouters):
            try:
                child, changed_doc = scouter(doc, part, fact, overwrite)

                ## Replace doc/root upstream if changed
                #  (e.g. df concat-ed with value).
                #
                if changed_doc is not None:
                    if doc is root or doc is given_doc:
                        # Changed-doc would be lost in vain...
                        raise ValueError(
                            f'Cannot modify given doc/root @ step (#{i}) "{part}" of path {path!r}:'
                            f"\n  +--doc: {doc}"
                            f"\n  +--root: {root}"
                        )
                    assert parent is not None and parent_part is not None, locals()

                    parent[  # pylint: disable=unsupported-assignment-operation
                        parent_part
                    ] = changed_doc
                    parent = changed_doc
                else:
                    parent = doc
                parent_part = part
                doc = child
                break
            except Exception as ex:
                if isinstance(ex, ValueError) and str(ex).startswith("Cannot modify"):
                    raise
                log.debug(
                    "scouter %s failed on step (#%i)%s of json-pointer(%r) with doc(%s), due to: %s",
                    scouter,
                    i,
                    part,
                    path,
                    doc,
                    ex,
                    # Don't log int(part) stack-traces.
                    exc_info=type(ex) is not ValueError
                    or not str(ex).startswith("invalid literal for int()"),
                )
        else:
            raise ValueError(
                f'Failed setting step (#{i}) "{part}" of path {path!r}!'
                "\n  Check debug logs."
            )


def _index_or_delay_concat(
    doc: Doc, key: str, value, delayed_concats: Optional[list]
) -> None:
    """
    Set Indexed value, or delay :term:`pandas concatenation`, for the recurse parent to do it.

    :param delayed_concats:
        if given (not ``None``), pandas-concats are enabled and
        should add further values into it
        (may contain past values to be mass-concatenated by the caller)
    """
    if delayed_concats is None or not isinstance(doc, NDFrame):
        doc[key] = value
    else:
        ## Delay further concats or Index non-pandas value.
        #
        if isinstance(value, NDFrame):
            delayed_concats.append(value)
        else:
            assert not delayed_concats, f"Parent left delayed_concats? {locals()}"
            doc[key] = value


def _convey_axes_names(doc, dfs_to_convey):
    """
    Preserve index/column level names by conveying LAST named lavels from `dfs_to_convey`

    TODO: WORKAROUND pandas drop other index-names when concat with unequal axes!
    e.g. pandas#13475, 21629, 27053, 27230
    """
    for axis in (0, 1):
        doc_axis = doc.axes[axis]
        if not any(doc_axis.names):
            for df in reversed(dfs_to_convey):
                ax = df.axes[0 if isinstance(df, pd.Series) else axis]
                if any(ax.names) and len(ax.names) == len(doc_axis.names):
                    doc_axis.names = ax.names
                    return


def _update_paths(
    doc: Doc,
    paths_vals: Collection[Tuple[List[str], Any]],
    container_factory,
    root: Doc,
    descend_objects,
    concat_axis,
) -> Optional[Doc]:
    """
    (recursive) mass-update `path_vals` (jsonp, value) pairs into doc, with ..

    special treatment for :term:`pandas concatenation`.

    :param concat_axis:
        either 0, 1 or None, in which case, it concatenates only when
        both doc & value are DataFrames
    :return:
        `doc` which might have changed, if it as a pandas concatenated.

    FIXME: ROOT in mass-update_paths NOT IMPLEMENTED
    FIXME: SET_OBJECT_ATTR in mass-update_paths NOT IMPLEMENTED
    """
    #: A `group` is a subset of the paths iterated below
    #: that have a common "prefix", i.e. their 1st step,
    #: seen as "root" for each recursed call.
    group: List[Tuple[str, Any]] = ()  # Begin with a blocker value.
    #: The `last_prefix` & `next_prefix` detect when group's 1st step
    #: has changed while iterating (path, value) pairs
    #: (meaning we have proceeded to the next `group)`.
    last_prefix = None
    #: Consecutive Pandas values to mass-concat.
    delayed_concats: list = None if concat_axis is None else []
    for i, (path, value) in enumerate((*paths_vals, ((UNSET,), UNSET))):
        assert len(path) >= 1 or value is UNSET, locals()

        ## Concate any delayed values.
        #
        if delayed_concats and (len(path) > 1 or not isinstance(value, NDFrame)):
            assert concat_axis is not None and isinstance(
                doc, NDFrame
            ), f"Delayed without permission? {locals}"
            delayed_concats = (doc, *delayed_concats)
            doc = pd.concat(delayed_concats, axis=concat_axis)
            _convey_axes_names(doc, delayed_concats)

            delayed_concats = None

        next_prefix = path[0]
        if next_prefix != last_prefix:
            if len(path) == 1 and value is not UNSET:
                # Assign "tip" value of the before proceeding to the next group,
                # THOUGH if a deeper path with this same prefix follows,
                # it will overwrite the value just written.
                _index_or_delay_concat(doc, next_prefix, value, delayed_concats)
            else:
                if last_prefix:  # Is it past the 1st loop?
                    child = None
                    if last_prefix in doc:
                        child = doc[last_prefix]
                        if not is_collection(child):
                            _log_overwrite(last_prefix, doc, child)
                            child = None

                    if child is None:
                        child = doc[last_prefix] = container_factory()

                    ## Recurse into collected sub-group.
                    #
                    sub_group = [(path[1:], value) for path, value in group]
                    new_child = _update_paths(
                        child,
                        sub_group,
                        container_factory,
                        root,
                        descend_objects,
                        concat_axis,
                    )
                    if new_child is not None:
                        doc[last_prefix] = new_child

                # prepare the next group
                last_prefix, group = (next_prefix, [(path, value)])
        else:
            assert len(path) > 1, locals()  # shortest path switches group.
            group.append((path, value))  # pylint: disable=no-member

    return doc


def update_paths(
    doc: Doc,
    paths_vals: Collection[Tuple[str, Any]],
    container_factory=dict,
    root: Doc = UNSET,
    descend_objects=True,
    concat_axis: int = None,
) -> None:
    """
    Mass-update `path_vals` (jsonp, value) pairs into doc.

    Group jsonp-keys by nesting level,to optimize.

    :param concat_axis:
        None, 0 or 1, see :func:`.set_path_value()`.

    :return:
        the updated doc (if it was a dataframe and ``pd.concact`` needed)
    """
    if root is UNSET:
        root = doc
    pvs = [(jsonp_path(p), v) for p, v in sorted(paths_vals, key=lambda pv: pv[0])]
    new_doc = _update_paths(
        doc, pvs, container_factory, root, descend_objects, concat_axis
    )
    if new_doc is not doc:
        # Changed-doc would be lost in vain...
        raise ValueError(
            f"Cannot mass-update Pandas @ ROOT:"
            f"\n  +--(path, values): {pvs}"
            f"\n  +--doc: {doc}"
            f"\n  +--new_doc: {new_doc}"
            + ("" if root is doc else f"\n  +--root: {root}")
        )


def list_popper(doc: MutableSequence, part, do_pop):
    """Call :func:`collection_popper()` with integer `part`."""
    return collection_popper(doc, int(part), do_pop)


def collection_popper(doc: Collection, part, do_pop):
    """Resolve `part` in `doc`, or pop it with `default`if `do_pop`."""
    return doc.pop(part) if do_pop else doc[part]


def object_popper(doc: Collection, part, do_pop):
    """Resolve `part` in `doc` attributes, or ``delattr`` it, returning its value or `default`."""
    item = getattr(doc, part)
    if do_pop:
        delattr(doc, part)
    return item


def pop_path(
    doc: Doc,
    path: Union[str, Iterable[str]],
    default=UNSET,
    root: Doc = UNSET,
    descend_objects=True,
):
    """
    Delete and return the item referenced by json-pointer `path` from the nested `doc` .

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
        the deleted item in `doc`, or `default` if given and `path`  didn't exist
    :raises ResolveError:
        if `path` cannot resolve and no `default` given
    :raises ValueError:
        if `path` was an absolute path a  ``None`` `root` had been given.

    See :func:`resolve_path()` for departures from the json-pointer standard

    **Examples:**

        >>> dt = {
        ...     'pi':3.14,
        ...     'foo':'bar',
        ...     'df': pd.DataFrame(np.ones((3,2)), columns=list('VN')),
        ...     'sub': {
        ...         'sr': pd.Series({'abc':'def'}),
        ...     }
        ... }
        >>> resolve_path(dt, '/pi', default=UNSET)
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
    part_poppers = [
        list_popper,
        collection_popper,
        *((object_popper,) if descend_objects else ()),
    ]

    if root is UNSET:
        root = doc

    parts = jsonp_path(path) if isinstance(path, str) else list(path)
    last_part = len(parts) - 1
    for i, part in enumerate(parts):
        if part == "":
            if root is None:
                raise ValueError(
                    f"Absolute step #{i} of json-pointer {path!r} without `root`!"
                )
            doc = root
            continue

        for popper in part_poppers:
            try:
                doc = popper(doc, part, i == last_part)
                break
            except Exception as ex:
                log.debug(
                    "popper %s failed on step (#%i)%s of json-pointer(%r) with doc(%s), due to: %s ",
                    popper,
                    i,
                    part,
                    path,
                    doc,
                    ex,
                    # Don't log int(part) stack-traces.
                    exc_info=type(ex) is not ValueError
                    or not str(ex).startswith("invalid literal for int()"),
                )
        else:
            if default is UNSET:
                raise ResolveError(path, part, i)

            return default

    return doc
