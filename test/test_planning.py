# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""mostly :mod:`networkx` routines tests"""

import networkx as nx
import pytest
from networkx.readwrite.edgelist import parse_edgelist

from graphtik.planning import (
    yield_also_chaindocs,
    yield_also_subdocs,
    yield_also_superdocs,
    yield_chaindocs,
    yield_subdocs,
    yield_superdocs,
)


@pytest.fixture
def g():
    g = parse_edgelist(
        """
        root    d1                      1
                d1      d11             1
                d1      d12             1
        root    d2                      1
                d2      d21             1
                        d21     d211    1

        # Irrelevant nodes
        root    foo
                d1      bar
                        d11     baz
    """.splitlines(),
        create_using=nx.DiGraph,
        data=[("subdoc", bool)],
    )
    return g


@pytest.fixture(
    params=[
        yield_also_subdocs,
        yield_also_superdocs,
        yield_also_chaindocs,
        lambda g, n, *args: yield_subdocs(g, [n], *args),
        lambda g, n, *args: yield_superdocs(g, [n], *args),
        lambda g, n, *args: yield_chaindocs(g, [n], *args),
    ]
)
def chain_fn(request):
    return request.param


@pytest.mark.parametrize("bad", ["BAD", "/root/BAD", "root/d1/BAD" "root/d1/d11/BAD"])
def test_yield_chained_docs_unknown(g, chain_fn, bad):
    assert list(chain_fn(g, bad)) == []
    assert list(chain_fn(g, bad, ())) == []


@pytest.mark.parametrize("node", "foo bar baz".split())
def test_yield_chained_docs_regular(g, chain_fn, node):
    assert list(chain_fn(g, node)) == [node]
    assert list(chain_fn(g, node, ())) == [node]
    assert list(chain_fn(g, node, (node,))) == []


def test_yield_chained_docs_leaf(g):
    ## leaf-doc
    #
    assert list(yield_also_subdocs(g, "d11")) == ["d11"]
    assert list(yield_also_subdocs(g, "d11", ())) == ["d11"]
    assert list(yield_also_superdocs(g, "d11")) == ["d11", "d1", "root"]
    assert list(yield_also_superdocs(g, "d11", ())) == ["d11", "d1", "root"]
    assert list(yield_also_chaindocs(g, "d11")) == ["d11", "d1", "root"]
    assert list(yield_also_chaindocs(g, "d11", ())) == ["d11", "d1", "root"]

    assert list(yield_subdocs(g, ["d11"])) == ["d11"]
    assert list(yield_subdocs(g, ["d11"], ())) == ["d11"]
    assert list(yield_superdocs(g, ["d11"])) == ["d11", "d1", "root"]
    assert list(yield_superdocs(g, ["d11"], ())) == ["d11", "d1", "root"]
    assert list(yield_chaindocs(g, ["d11"])) == ["d11", "d1", "root"]
    assert list(yield_chaindocs(g, ["d11"], ())) == ["d11", "d1", "root"]


def test_yield_chained_docs_inner(g):
    ## inner-node
    #
    assert list(yield_also_subdocs(g, "d1")) == ["d1", "d11", "d12"]
    assert list(yield_also_subdocs(g, "d1", ())) == ["d1", "d11", "d12"]
    assert list(yield_also_superdocs(g, "d1")) == ["d1", "root"]
    assert list(yield_also_superdocs(g, "d1", ())) == ["d1", "root"]
    assert list(yield_also_chaindocs(g, "d1")) == ["d1", "d11", "d12", "root"]
    assert list(yield_also_chaindocs(g, "d1", ())) == ["d1", "d11", "d12", "root"]

    assert list(yield_subdocs(g, ["d1"])) == ["d1", "d11", "d12"]
    assert list(yield_subdocs(g, ["d1"], ())) == ["d1", "d11", "d12"]
    assert list(yield_superdocs(g, ["d1"])) == ["d1", "root"]
    assert list(yield_superdocs(g, ["d1"], ())) == ["d1", "root"]
    assert list(yield_chaindocs(g, ["d1"])) == ["d1", "d11", "d12", "root"]
    assert list(yield_chaindocs(g, ["d1"], ())) == ["d1", "d11", "d12", "root"]


def test_yield_chained_docs_root(g):
    ## root-doc
    #
    assert list(yield_also_subdocs(g, "root")) == [
        "root",
        "d1",
        "d11",
        "d12",
        "d2",
        "d21",
        "d211",
    ]
    assert list(yield_also_subdocs(g, "root", ())) == [
        "root",
        "d1",
        "d11",
        "d12",
        "d2",
        "d21",
        "d211",
    ]
    assert list(yield_also_superdocs(g, "root")) == ["root"]
    assert list(yield_also_superdocs(g, "root", ())) == ["root"]
    assert list(yield_also_chaindocs(g, "root")) == [
        "root",
        "d1",
        "d11",
        "d12",
        "d2",
        "d21",
        "d211",
    ]
    assert list(yield_also_chaindocs(g, "root", ())) == [
        "root",
        "d1",
        "d11",
        "d12",
        "d2",
        "d21",
        "d211",
    ]
