import re
from itertools import chain, cycle

import pytest

###################################################
# Copied from <sphinx.git>/tests/test_build_html.py


def flat_dict(d):
    return chain.from_iterable(
        [zip(cycle([fname]), values) for fname, values in d.items()]
    )


def attr_check(attr, *regex, count=None):

    rexes = [re.compile(r) for r in regex]

    def checker(nodes):
        if count is not None:
            n = len(nodes)
            assert len(nodes) == count, f"expected {count} but found {n} nodes: {nodes}"

        for i, (node, rex) in enumerate(zip(nodes, cycle(rexes))):
            txt = node.get(attr)
            assert rex.search(
                txt
            ), f"num0-{i} regex({regex}) missmatched {node.tag}@%{attr}: {txt}"

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

        assert False, "%r not found in any node matching " "path %s in %s: %r" % (
            check,
            path,
            fname,
            [node.text for node in nodes],
        )
