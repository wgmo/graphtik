import pickle
import re
from collections import namedtuple
from itertools import chain, cycle
from pathlib import Path
from typing import Union

import dill
import pytest

###################################################
# Copied from <sphinx.git>/tests/test_build_html.py

_slow = pytest.mark.slow
_proc = pytest.mark.proc
_thread = pytest.mark.thread
_parallel = pytest.mark.parallel
_marshal = pytest.mark.marshal


_ExeParams = namedtuple("_ExeParams", "parallel, proc, marshal")
_exe_params = _ExeParams(None, None, None)


def flat_dict(d):
    return chain.from_iterable(
        [zip(cycle([fname]), values) for fname, values in d.items()]
    )


def attr_check(attr, *regex, count: Union[int, None, bool] = None):
    """
    Asserts captured nodes have `attr` satisfying `regex` one by one (in a cycle).

    :param count:
        The number-of-nodes expected.
        If ``True``, this number becomes the number-of-`regex`;
        if none, no count check happens.
    """

    rexes = [re.compile(r) for r in regex]

    def checker(nodes):
        nonlocal count

        if count is not None:
            if count is True:
                count = len(rexes)
            n = len(nodes)
            assert len(nodes) == count, f"expected {count} but found {n} nodes: {nodes}"

        for i, (node, rex, rex_pat) in enumerate(
            zip(nodes, cycle(rexes), cycle(regex))
        ):
            txt = node.get(attr)
            assert rex.search(
                txt
            ), f"no0({i}) regex({rex_pat}) missmatched {node.tag}@%{attr}: {txt}"

    return checker


def check_xpath(etree, fname, path, check, be_found=True):
    nodes = list(etree.findall(path))
    if check is None:
        assert nodes == [], "found any nodes matching xpath " "%r in file %s" % (
            path,
            fname,
        )
        return
    else:
        assert nodes != [], "did not find any node matching xpath " "%r in file %s" % (
            path,
            fname,
        )
    if callable(check):
        check(nodes)
    elif not check:
        # only check for node presence
        pass
    else:

        def get_text(node):
            if node.text is not None:
                # the node has only one text
                return node.text
            else:
                # the node has tags and text; gather texts just under the node
                return "".join(n.tail or "" for n in node)

        rex = re.compile(check)
        if be_found:
            if any(rex.search(get_text(node)) for node in nodes):
                return
        else:
            if all(not rex.search(get_text(node)) for node in nodes):
                return

        msg = "didn't match any" if be_found else "matched at least one"
        assert False, (
            f"{Path(fname).absolute()}:\n  {check!r} {msg} node at path {path!r}: "
            f"{[node.text for node in nodes]}"
        )


def dilled(i):
    return dill.loads(dill.dumps(i))


def pickled(i):
    return pickle.loads(pickle.dumps(i))


def addall(*a, **kw):
    "Same as a + b + ...."
    return sum(a) + sum(kw.values())


def abspow(a, p):
    c = abs(a) ** p
    return c
