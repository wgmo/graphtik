# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Test renames, :term:`operation nesting` & :term:`operation merging`. """
import re
from functools import partial
from operator import add, mul, sub
from textwrap import dedent

import pytest

from graphtik import compose, operation, sfx, sfxed, vararg
from graphtik.fnop import Operation
from graphtik.modifier import dep_renamed

from .helpers import abspow, addall


def test_compose_rename_dict(caplog):
    pip = compose(
        "t",
        operation(str, "op1", provides=["a", "aa"]),
        operation(
            str,
            "op2",
            needs="a",
            provides=["b", sfx("c")],
            aliases=[("b", "B"), ("b", "p")],
        ),
        nest={"op1": "OP1", "op2": lambda n: "OP2", "a": "A", "b": "bb"},
    )
    print(str(pip))
    assert str(pip) == (
        "Pipeline('t', needs=['A'], "
        "provides=['A', 'aa', 'bb', sfx('c'), 'B', 'p'], x2 ops: OP1, OP2)"
    )
    print(str(pip.ops))
    assert (
        str(pip.ops)
        == dedent(
            """
        [FnOp(name='OP1', provides=['A', 'aa'], fn='str'),
         FnOp(name='OP2', needs=['A'], provides=['bb', sfx('c'), 'B', 'p'],
         aliases=[('bb', 'B'), ('bb', 'p')], fn='str')]
    """
        ).replace("\n", "")
    )


def test_compose_rename_dict_non_str(caplog):
    pip = compose(
        "t",
        operation(str, "op1"),
        operation(str, "op2"),
        nest={"op1": 1},
    )
    exp = "Pipeline('t', x2 ops: op1, op2)"
    print(pip)
    assert str(pip) == exp
    exp = "Pipeline('t', x2 ops: t.op1, op2)"
    pip = compose("t", pip, nest={"op1": 1, "op2": 0})
    assert str(pip) == exp
    pip = compose("t", pip, nest={"op1": 1, "op2": ""})
    assert str(pip) == exp
    for record in caplog.records:
        assert "Failed to nest-rename" not in record.message


def test_compose_rename_bad_screamy(caplog):
    def screamy_nester(ren_args):
        raise RuntimeError("Bluff")

    with pytest.raises(RuntimeError, match="Bluff"):
        compose(
            "test_nest_err",
            operation(str, "op1"),
            operation(str, "op2"),
            nest=screamy_nester,
        )
    for record in caplog.records:
        if record.levelname == "WARNING":
            assert "name='op1', parent=None)" in record.message


def test_compose_rename_preserve_ops(caplog):
    pip = compose(
        "t",
        operation(str, "op1"),
        operation(str, "op2"),
        nest=lambda na: f"aa.{na.name}",
    )
    assert str(pip) == "Pipeline('t', x2 ops: aa.op1, aa.op2)"


def test_compose_merge_ops():
    def ops_only(ren_args):
        return ren_args.typ == "op"

    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
    net1 = compose("my network 1", sum_op1, sum_op2, sum_op3)

    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    sol = net1(a=1, b=2, c=4)
    assert sol == exp

    sum_op4 = operation(name="sum_op1", needs=["d", "e"], provides="a")(add)
    sum_op5 = operation(name="sum_op2", needs=["a", "f"], provides="b")(add)

    net2 = compose("my network 2", sum_op4, sum_op5)
    exp = {"a": 3, "b": 7, "d": 1, "e": 2, "f": 4}
    sol = net2(**{"d": 1, "e": 2, "f": 4})
    assert sol == exp

    net3 = compose("merged", net1, net2, nest=ops_only)
    exp = {
        "a": 3,
        "b": 7,
        "c": 5,
        "d": 1,
        "e": 2,
        "f": 4,
        "sum1": 10,
        "sum2": 10,
        "sum3": 15,
    }
    sol = net3(c=5, d=1, e=2, f=4)
    assert sol == exp

    assert repr(net3).startswith(
        "Pipeline('merged', needs=['a', 'b', 'sum1', 'c', 'd', 'e', 'f'], "
        "provides=['sum1', 'sum2', 'sum3', 'a', 'b'], x5 ops"
    )


def test_network_combine():
    sum_op1 = operation(
        name="sum_op1", needs=[vararg("a"), vararg("b")], provides="sum1"
    )(addall)
    sum_op2 = operation(name="sum_op2", needs=[vararg("a"), "b"], provides="sum2")(
        addall
    )
    sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
    net1 = compose("my network 1", sum_op1, sum_op2, sum_op3)
    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    assert net1(a=1, b=2, c=4) == exp
    assert repr(net1).startswith(
        "Pipeline('my network 1', needs=['a'(?), 'b', 'sum1', 'c'], "
        "provides=['sum1', 'sum2', 'sum3'], x3 ops"
    )

    sum_op4 = operation(name="sum_op1", needs=[vararg("a"), "b"], provides="sum1")(
        addall
    )
    sum_op5 = operation(name="sum_op4", needs=["sum1", "b"], provides="sum2")(add)
    net2 = compose("my network 2", sum_op4, sum_op5)
    exp = {"a": 1, "b": 2, "sum1": 3, "sum2": 5}
    assert net2(**{"a": 1, "b": 2}) == exp
    assert repr(net2).startswith(
        "Pipeline('my network 2', needs=['a'(?), 'b', 'sum1'], provides=['sum1', 'sum2'], x2 ops"
    )

    net3 = compose("merged", net1, net2)
    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 5, "sum3": 7}
    assert net3(a=1, b=2, c=4) == exp

    assert repr(net3).startswith(
        "Pipeline('merged', needs=['a'(?), 'b', 'sum1', 'c'], provides=['sum1', 'sum2', 'sum3'], x4 ops"
    )

    ## Reverse ops, change results and `needs` optionality.
    #
    net3 = compose("merged", net2, net1)
    exp = {"a": 1, "b": 2, "c": 4, "sum1": 3, "sum2": 3, "sum3": 7}
    assert net3(**{"a": 1, "b": 2, "c": 4}) == exp

    assert repr(net3).startswith(
        "Pipeline('merged', needs=['a'(?), 'b', 'sum1', 'c'], provides=['sum1', 'sum2', 'sum3'], x4 ops"
    )


def test_network_nest_in_doctests():
    days_count = 3

    weekday = compose(
        "weekday",
        operation(str, name="wake up", needs="backlog", provides="tasks"),
        operation(str, name="sleep", needs="tasks", provides="todos"),
    )

    weekday = compose(
        "weekday",
        operation(
            lambda t: (t[:-1], t[-1:]),
            name="work!",
            needs="tasks",
            provides=["tasks_done", "todos"],
        ),
        operation(str, name="sleep"),
        weekday,
    )
    assert len(weekday.ops) == 3

    weekday = compose("weekday", weekday, excludes="sleep")
    assert len(weekday.ops) == 2

    weekdays = [weekday.withset(name=f"day {i}") for i in range(days_count)]
    week = compose("week", *weekdays, nest=True)
    assert len(week.ops) == 6

    def nester(ren_args):
        if ren_args.name not in ("backlog", "tasks_done", "todos"):
            return True

    week = compose("week", *weekdays, nest=nester)
    assert len(week.ops) == 6
    sol = week.compute({"backlog": "a lot!"})
    assert sol == {
        "backlog": "a lot!",
        "day 0.tasks": "a lot!",
        "tasks_done": "a lot",
        "todos": "!",
        "day 1.tasks": "a lot!",
        "day 2.tasks": "a lot!",
    }


def test_compose_nest_dict(caplog):
    pipe = compose(
        "t",
        compose(
            "p1",
            operation(
                str,
                name="op1",
                needs=[sfx("a"), "aa"],
                provides=[sfxed("S1", "g"), sfxed("S2", "h")],
            ),
        ),
        compose(
            "p2",
            operation(
                str,
                name="op2",
                needs=sfx("a"),
                provides=["a", sfx("b")],
                aliases=[("a", "b")],
            ),
        ),
        nest={
            "op1": True,
            "op2": lambda n: "p2.op2",
            "aa": False,
            sfx("a"): True,
            "b": lambda n: f"PP.{n}",
            sfxed("S1", "g"): True,
            sfxed("S2", "h"): lambda n: dep_renamed(n, "ss2"),
            sfx("b"): True,
        },
    )
    got = str(pipe.ops)
    print(got)
    assert got == re.sub(
        r"[\n ]{2,}",  # collapse all space-chars into a single space
        " ",
        """
        [FnOp(name='p1.op1', needs=[sfx('p1.a'), 'aa'],
         provides=[sfxed('p1.S1', 'g'), sfxed('ss2', 'h')], fn='str'),
        FnOp(name='p2.op2', needs=[sfx('p2.a')],
         provides=['a', sfx('p2.b'), 'PP.b'], aliases=[('a', 'PP.b')], fn='str')]

        """.strip(),
    )
    for record in caplog.records:
        assert record.levelname != "WARNING"


@pytest.mark.parametrize("bools", range(4))
def test_combine_networks(exemethod, bools):
    # Code from `compose.rst` examples
    if not exemethod:
        return

    parallel1 = bools >> 0 & 1
    parallel2 = bools >> 1 & 1

    graphop = compose(
        "graphop",
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="sub1", needs=["a", "ab"], provides=["a-ab"])(sub),
        operation(name="abspow1", needs=["a-ab"], provides=["|a-ab|³"])(
            partial(abspow, p=3)
        ),
        parallel=parallel1,
    )

    assert graphop.compute({"a-ab": -8}) == {"a-ab": -8, "|a-ab|³": 512}

    bigger_graph = compose(
        "bigger_graph",
        graphop,
        operation(name="sub2", needs=["a-ab", "c"], provides="a-ab_minus_c")(sub),
        parallel=parallel2,
        nest=lambda ren_args: ren_args.typ == "op",
    )
    ## Ensure all old-nodes were prefixed.
    #
    old_nodes = graphop.net.graph.nodes
    new_nodes = bigger_graph.net.graph.nodes
    for old_node in old_nodes:
        if isinstance(old_node, Operation):
            assert old_node not in new_nodes
        else:
            assert old_node in new_nodes

    sol = bigger_graph.compute({"a": 2, "b": 5, "c": 5}, ["a-ab_minus_c"])
    assert sol == {"a-ab_minus_c": -13}

    ## Test Plots

    ## Ensure all old-nodes were prefixed.
    #
    # Access all nodes from Network, where no "after pruning" cluster exists.
    old_nodes = [n for n in graphop.net.plot().get_nodes()]
    new_node_names = [n.get_name() for n in bigger_graph.net.plot().get_nodes()]

    for old_node in old_nodes:
        if old_node.get_shape() == "plain":  # Operation
            assert old_node.get_name() not in new_node_names
        else:
            # legend-node included here.`
            assert old_node.get_name() in new_node_names
