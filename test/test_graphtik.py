# Copyright 2016-2020, Yahoo Inc, Kostis Anagnostopoulos.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""General :term:`network` & :term:`execution` tests. """
import math
import re
import sys
from operator import add, floordiv, mul, sub

import networkx as nx
import pytest

from graphtik import (
    NO_RESULT,
    compose,
    modify,
    planning,
    operation,
    optional,
    vararg,
    varargs,
    sfxed,
    sfx,
)
from graphtik.base import IncompleteExecutionError
from graphtik.config import debug_enabled, evictions_skipped, operations_endured

from .helpers import addall


pytestmark = pytest.mark.usefixtures("log_levels")


def scream(*args, **kwargs):
    raise AssertionError(f"Must not have run!\n    args: {args}\n  kwargs: {kwargs}")


def identity(*x):
    return x[0] if len(x) == 1 else x


def filtdict(d, *keys):
    """
    Keep dict items with the given keys

    filtdict({"a": 1, "b": 2}, "b")
    {'b': 2}
    """
    return type(d)(i for i in d.items() if i[0] in keys)


def test_smoke_test():

    # Sum operation, late-bind compute function
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum_ab")(add)

    assert sum_op1.fn(1, 2) == 3

    # Multiply operation, decorate in-place
    @operation(name="mul_op1", needs=["sum_ab", "b"], provides="sum_ab_times_b")
    def mul_op1(a, b):
        return a * b

    # mul_op1 is callable
    assert mul_op1.fn(1, 2) == 2

    # Pow operation
    @operation(
        name="pow_op1", needs="sum_ab", provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"]
    )
    def pow_op1(a, exponent=3):
        return [math.pow(a, y) for y in range(1, exponent + 1)]

    assert pow_op1.compute({"sum_ab": 2}, ["sum_ab_p2"]) == {"sum_ab_p2": 4.0}

    # Partial operation that is bound at a later time
    partial_op = operation(
        name="sum_op2", needs=["sum_ab_p1", "sum_ab_p2"], provides="p1_plus_p2"
    )

    # Bind the partial operation
    sum_op2 = partial_op(add)

    # Sum operation, early-bind compute function
    sum_op_factory = operation(add)

    sum_op3 = sum_op_factory.withset(
        name="sum_op3", needs=["a", "b"], provides="sum_ab2"
    )

    # compose network
    pipeline = compose("my network", sum_op1, mul_op1, pow_op1, sum_op2, sum_op3)

    #
    # Running the network
    #

    # get all outputs
    exp = {
        "a": 1,
        "b": 2,
        "p1_plus_p2": 12.0,
        "sum_ab": 3,
        "sum_ab2": 3,
        "sum_ab_p1": 3.0,
        "sum_ab_p2": 9.0,
        "sum_ab_p3": 27.0,
        "sum_ab_times_b": 6,
    }
    assert pipeline(a=1, b=2) == exp

    # get specific outputs
    exp = {"sum_ab_times_b": 6}
    assert pipeline.compute({"a": 1, "b": 2}, ["sum_ab_times_b"]) == exp

    # start with inputs already computed
    exp = {"sum_ab_times_b": 2}
    assert pipeline.compute({"sum_ab": 1, "b": 2}, ["sum_ab_times_b"]) == exp

    with pytest.raises(ValueError, match="Unknown output node"):
        pipeline.compute({"sum_ab": 1, "b": 2}, "bad_node")
    with pytest.raises(ValueError, match="Unknown output node"):
        pipeline.compute({"sum_ab": 1, "b": 2}, ["b", "bad_node"])


def test_cycle_tip():
    pipe = compose(..., operation(str, "cyclic1", "a", "a"))
    with pytest.raises(nx.NetworkXUnfeasible, match="TIP:"):
        pipe.compute()
    pipe = compose(
        ..., operation(str, "cyclic1", "a", "b"), operation(str, "cyclic2", "b", "a")
    )
    with evictions_skipped(True), pytest.raises(nx.NetworkXUnfeasible, match="TIP:"):
        pipe.compute()


def test_aliases(exemethod):
    aliased = operation(
        lambda: ("A", "B"),
        name="op1",
        provides=["a", sfxed("s1", "foo"), sfxed("s2", "foo", implicit=1)],
        aliases={"a": "b", "s1": "S1", "s2": "S2"},
    )
    assert aliased.provides == ("a", sfxed("s1", "foo"), sfxed("s2", "foo"))
    assert set(aliased.op_provides) == set(
        ("a", "b", "S1", "S2", sfxed("s1", "foo"), sfxed("s2", "foo"))
    )

    op = compose(
        "test_net",
        aliased,
        operation(lambda x: x * 2, name="op2", needs="b", provides="c"),
        parallel=exemethod,
    )
    assert op() == {"a": "A", "s1": "B", "b": "A", "S1": "B", "c": "AA"}


def test_node_predicate_based_prune():
    pipeline = compose(
        "N",
        operation(name="A", needs=["a"], provides=["aa"], node_props={"color": "red"})(
            identity
        ),
        operation(
            name="B", needs=["b"], provides=["bb"], node_props={"color": "green"}
        )(identity),
        operation(name="C", needs=["c"], provides=["cc"])(identity),
        operation(
            name="SUM",
            needs=[optional(i) for i in ("aa", "bb", "cc")],
            provides=["sum"],
        )(addall),
    )
    inp = {"a": 1, "b": 2, "c": 3}
    assert pipeline(**inp)["sum"] == 6
    assert len(pipeline.net.graph.nodes) == 11

    pred = lambda n, d: d.get("color", None) != "red"
    assert pipeline.withset(predicate=pred)(**inp)["sum"] == 5
    assert len(pipeline.withset(predicate=pred).compile().dag.nodes) == 9

    pred = lambda n, d: "color" not in d
    assert pipeline.withset(predicate=pred)(**inp)["sum"] == 3
    assert len(pipeline.withset(predicate=pred).compile().dag.nodes) == 7


def test_input_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if we're provided
    # with data further downstream in the graph as an input.

    sum1 = 2
    sum2 = 5

    # Set up a net such that if sum1 and sum2 are provided directly, we don't
    # need to provide a and b.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["sum1", "sum2"], provides="sum3")(add)
    net = compose("test_net", sum_op1, sum_op2, sum_op3)

    results = net(**{"sum1": sum1, "sum2": sum2})

    # Make sure we got expected result without having to pass a or b.
    assert "sum3" in results
    assert results["sum3"] == add(sum1, sum2)


def test_output_based_pruning(samplenet):
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs.

    c = 2
    d = 3

    results = samplenet.compute({"a": 0, "b": 0, "c": c, "d": d}, ["sum3"])

    # Make sure we got expected result without having to pass a or b.
    assert "sum3" in results
    assert results["sum3"] == add(c, add(c, d))


def test_pruning_avoid_cycles():
    @operation(needs="a", provides="b")
    def f1(x):
        return x

    @operation(needs="b", provides="a")
    def f2(x):
        return 2 * x

    @operation(needs="b", provides="c")
    def f3(x):
        return 3 * x

    @operation(needs="c", provides="d")
    def f4(x):
        return 4 * x

    pipe = compose("", f1, f2, f3, f4)

    assert pipe(a=1) == {"a": 1, "b": 1, "c": 3, "d": 12}
    assert pipe(b=1) == {"b": 1, "a": 2, "c": 3, "d": 12}
    assert pipe.compute({"c": 1}, "d") == {"d": 4}
    assert pipe.compute({"c": 1}, "c") == {"c": 1}


def test_deps_pruning_vs_narrowing(samplenet):
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs or of we're provided with
    # inputs that are further downstream in the graph.

    c = 2
    sum2 = 5

    results = samplenet.compute({"c": c, "sum2": sum2}, ["sum3"])

    # Make sure we got expected result without having to pass a, b, or d.
    assert "sum3" in results
    assert results["sum3"] == add(c, sum2)

    # Compare with both `withset()`.
    net = samplenet.withset(outputs=["sum3"])
    assert net(c=c, sum2=sum2) == results

    # Make sure we got expected result without having to pass a, b, or d.
    assert "sum3" in results
    assert results["sum3"] == add(c, sum2)


def test_pruning_raises_for_bad_output(samplenet):
    # Make sure we get a ValueError during the pruning step if we request an
    # output that doesn't exist.

    # Request two outputs we can compute and one we can't compute.  Assert
    # that this raises a ValueError.
    with pytest.raises(ValueError, match="sum4"):
        samplenet.compute({"a": 1, "b": 2, "c": 3, "d": 4}, ["sum1", "sum3", "sum4"])


def test_impossible_outputs():
    pipeline = compose(
        "test_net",
        operation(name="op1", needs=["a"], provides="aa")(identity),
        operation(name="op2", needs=["aa", "bb"], provides="aabb")(identity),
    )
    with pytest.raises(ValueError, match="Unreachable outputs"):
        pipeline.compute({"a": 1,}, ["aabb"])

    with pytest.raises(ValueError, match="Unreachable outputs"):
        pipeline.compute({"a": 1,}, ["aa", "aabb"])


def test_pruning_not_overrides_given_intermediate(exemethod):
    # Test #25: v1.2.4 overwrites intermediate data when no output asked
    pipeline = compose(
        "pipeline",
        operation(name="not run", needs=["a"], provides=["overridden"])(scream),
        operation(name="op", needs=["overridden", "c"], provides=["asked"])(add),
        parallel=exemethod,
    )

    inputs = {"a": 5, "overridden": 1, "c": 2}
    exp = {"a": 5, "overridden": 1, "c": 2, "asked": 3}
    # v1.2.4.ok
    assert pipeline.compute(inputs, "asked") == filtdict(exp, "asked")
    # FAILs
    # - on v1.2.4 with (overridden, asked): = (5, 7) instead of (1, 3)
    # - on #18(unsatisfied) + #23(ordered-sets) with (overridden, asked) = (5, 7) instead of (1, 3)
    # FIXED on #26
    assert pipeline(**inputs) == exp

    ## Test OVERWRITES
    #
    solution = pipeline.compute(inputs, ["asked"])
    assert solution == filtdict(exp, "asked")
    assert solution.overwrites == {}  # unjust must have been pruned

    solution = pipeline(**inputs)
    assert solution == exp
    assert solution.overwrites == {}  # unjust must have been pruned


def test_pruning_multiouts_not_override_intermediates1(exemethod):
    pipeline = compose(
        "graph",
        operation(name="must run", needs=["a"], provides=["overridden", "calced"])(
            lambda x: (x, 2 * x)
        ),
        operation(name="add", needs=["overridden", "calced"], provides=["asked"])(add),
        parallel=exemethod,
    )

    ## Overwritten values used as downstream inputs.
    #
    inp1 = {"a": 5, "overridden": 1}
    inp2 = {"a": 5, "overridden": 1, "c": 2}
    exp = {"a": 5, "overridden": 5, "calced": 10, "asked": 15}
    exp2 = filtdict(exp, "asked")

    solution = pipeline.compute(inp1)
    assert solution == exp
    assert solution.overwrites == {"overridden": [5, 1]}
    # Check plotting Overwrites.
    assert "SkyBlue" in str(solution.plot())

    solution = pipeline.compute(inp2, "asked")
    assert solution == exp2
    assert solution.overwrites == {}
    # Check not plotting Overwrites.
    assert "SkyBlue" not in str(solution.plot())

    solution = pipeline.compute(inp1, "asked")
    assert solution == exp2
    assert solution.overwrites == {}
    # Check not plotting Overwrites.
    assert "SkyBlue" not in str(solution.plot())


def test_pruning_multiouts_not_override_intermediates2(exemethod):
    pipeline = compose(
        "pipeline",
        operation(name="must run", needs=["a"], provides=["overridden", "e"])(
            lambda x: (x, 2 * x)
        ),
        operation(name="op1", needs=["overridden", "c"], provides=["d"])(add),
        operation(name="op2", needs=["d", "e"], provides=["asked"])(mul),
        parallel=exemethod,
    )

    inputs = {"a": 5, "overridden": 1, "c": 2}
    exp = {"a": 5, "overridden": 5, "c": 2, "e": 10, "d": 7, "asked": 70}

    assert pipeline(**inputs) == exp
    assert pipeline.compute(inputs, "asked") == filtdict(exp, "asked")

    ## Test OVERWRITES
    #
    solution = pipeline.compute(inputs)
    assert solution == exp
    assert solution.overwrites == {"overridden": [5, 1]}
    # No overwrites when evicted.
    #
    solution = pipeline.compute(inputs, "asked")
    assert solution == filtdict(exp, "asked")
    assert solution.overwrites == {}
    # ... but overwrites collected if asked.
    #
    solution = pipeline.compute(inputs, ["asked", "overridden"])
    assert solution == filtdict(exp, "asked", "overridden")
    assert solution.overwrites == {"overridden": [5, 1]}


def test_pruning_with_given_intermediate_and_asked_out(exemethod):
    # Test #24: v1.2.4 does not prune before given intermediate data when
    # outputs not asked, but does so when output asked.
    pipeline = compose(
        "pipeline",
        operation(name="unjustly pruned", needs=["given-1"], provides=["a"])(identity),
        operation(name="shortcut-ed", needs=["a", "b"], provides=["given-2"])(add),
        operation(name="good_op", needs=["a", "given-2"], provides=["asked"])(add),
        parallel=exemethod,
    )

    inps = {"given-1": 5, "b": 2, "given-2": 2}
    exp = {"given-1": 5, "given-2": 2, "a": 5, "b": 2, "asked": 7}

    # v1.2.4 is ok
    assert pipeline(**inps) == exp
    # FAILS
    # - on v1.2.4 with KeyError: 'a',
    # - on #18 (unsatisfied) with no result.
    # FIXED on #18+#26 (new dag solver).
    assert pipeline.compute(inps, "asked") == filtdict(exp, "asked")

    ## Test OVERWRITES
    #
    solution = pipeline.compute(inps)
    assert solution == exp
    assert solution.overwrites == {}

    solution = pipeline.compute(inps, "asked")
    assert solution == filtdict(exp, "asked")
    assert solution.overwrites == {}


def test_same_outputs_operations_order():
    # Test operations providing the same output ordered as given.
    op1 = operation(name="add", needs=["a", "b"], provides=["ab"])(add)
    op2 = operation(name="sub", needs=["a", "b"], provides=["ab"])(sub)
    addsub = compose("add_sub", op1, op2)
    subadd = compose("sub_add", op2, op1)

    inp = {"a": 3, "b": 1}
    assert addsub(**inp) == {"a": 3, "b": 1, "ab": 2}
    assert addsub.compute(inp, "ab") == {"ab": 2}
    assert subadd(**inp) == {"a": 3, "b": 1, "ab": 4}
    sol = subadd.compute(inp, "ab")
    assert sol == {"ab": 4}

    # ## Check it does not duplicate evictions
    assert len(sol.plan.steps) == 4

    ## Add another step to test evictions
    #
    op3 = operation(name="pipe", needs=["ab"], provides=["AB"])(identity)
    addsub = compose("add_sub", op1, op2, op3)
    subadd = compose("sub_add", op2, op1, op3)

    # Notice that `ab` assumed as 2 for `AB` but results in `2`
    solution = addsub.compute(inp)
    assert solution == {"a": 3, "b": 1, "ab": 2, "AB": 2}
    assert solution.overwrites == {"ab": [2, 4]}
    solution = addsub.compute(inp, "AB")
    assert solution == {"AB": 2}
    assert solution.overwrites == {}

    solution = subadd.compute(inp)
    assert solution == {"a": 3, "b": 1, "ab": 4, "AB": 4}
    assert solution.overwrites == {"ab": [4, 2]}
    solution = subadd.compute(inp, "AB")
    assert solution == {"AB": 4}
    assert solution.overwrites == {}

    sol = subadd.compute(inp, "AB")
    assert sol == {"AB": 4}
    assert len(sol.plan.steps) == 6


def test_unsatisfied_operations(exemethod):
    # Test that operations with partial inputs are culled and not failing.
    pipeline = compose(
        "pipeline",
        operation(name="add", needs=["a", "b1"], provides=["a+b1"])(add),
        operation(name="sub", needs=["a", "b2"], provides=["a-b2"])(sub),
        parallel=exemethod,
    )

    exp = {"a": 10, "b1": 2, "a+b1": 12}
    assert pipeline(**{"a": 10, "b1": 2}) == exp
    assert pipeline.compute({"a": 10, "b1": 2}, ["a+b1"]) == filtdict(exp, "a+b1")
    assert pipeline.withset(outputs=["a+b1"])(**{"a": 10, "b1": 2}) == filtdict(
        exp, "a+b1"
    )

    exp = {"a": 10, "b2": 2, "a-b2": 8}
    assert pipeline(**{"a": 10, "b2": 2}) == exp
    assert pipeline.compute({"a": 10, "b2": 2}, ["a-b2"]) == filtdict(exp, "a-b2")


def test_unsatisfied_operations_same_out(exemethod):
    # Test unsatisfied pairs of operations providing the same output.
    pipeline = compose(
        "pipeline",
        operation(name="mul", needs=["a", "b1"], provides=["ab"])(mul),
        operation(name="div", needs=["a", "b2"], provides=["ab"])(floordiv),
        operation(name="add", needs=["ab", "c"], provides=["ab_plus_c"])(add),
        parallel=exemethod,
    )

    #  Parallel FAIL! in #26
    exp = {"a": 10, "b1": 2, "c": 1, "ab": 20, "ab_plus_c": 21}
    assert pipeline(**{"a": 10, "b1": 2, "c": 1}) == exp
    assert pipeline.compute({"a": 10, "b1": 2, "c": 1}, ["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )

    #  Parallel FAIL! in #26
    exp = {"a": 10, "b2": 2, "c": 1, "ab": 5, "ab_plus_c": 6}
    assert pipeline(**{"a": 10, "b2": 2, "c": 1}) == exp
    assert pipeline.compute({"a": 10, "b2": 2, "c": 1}, ["ab_plus_c"]) == filtdict(
        exp, "ab_plus_c"
    )


def test_optional():
    # Test that optional() needs work as expected.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    sum_op = operation(name="sum_op1", needs=["a", "b", optional("c")], provides="sum")(
        addplusplus
    )

    net = compose("test_net", sum_op)

    # Make sure output with optional arg is as expected.
    named_inputs = {"a": 4, "b": 3, "c": 2}
    results = net(**named_inputs)
    assert "sum" in results
    assert results["sum"] == sum(named_inputs.values())

    # Make sure output without optional arg is as expected.
    named_inputs = {"a": 4, "b": 3}
    results = net(**named_inputs)
    assert "sum" in results
    assert results["sum"] == sum(named_inputs.values())


@pytest.mark.parametrize("reverse", [0, 1])
def test_narrow_and_optionality(reverse):
    op1 = operation(name="op1", needs=[optional("a"), optional("bb")], provides="sum1")(
        addall
    )
    op2 = operation(name="op2", needs=["a", optional("bb")], provides="sum2")(addall)
    ops = [op1, op2]
    provides = "'sum1', 'sum2'"
    if reverse:
        ops = list(reversed(ops))
        provides = "'sum2', 'sum1'"
    pipeline_str = f"Pipeline('t', needs=['a', 'bb'(?)], provides=[{provides}], x2 ops"

    pipeline = compose("t", *ops)
    assert repr(pipeline).startswith(pipeline_str)

    ## IO & predicate do not affect network, but solution.

    ## Compose with `inputs`
    #
    pipeline = compose("t", *ops)
    assert repr(pipeline).startswith(pipeline_str)
    assert repr(pipeline.compile("a")).startswith(
        f"ExecutionPlan(needs=['a'], provides=[{provides}], x2 steps:"
    )
    #
    pipeline = compose("t", *ops)
    assert repr(pipeline).startswith(pipeline_str)
    assert repr(pipeline.compile(["bb"])).startswith(
        "ExecutionPlan(needs=['bb'(?)], provides=['sum1'], x1 steps:"
    )

    ## Narrow by `provides`
    #
    pipeline = compose("t", *ops, outputs="sum1")
    assert repr(pipeline).startswith(pipeline_str)
    assert repr(pipeline.compile("bb")).startswith(
        "ExecutionPlan(needs=['bb'(?)], provides=['sum1'], x3 steps:"
    )
    assert repr(pipeline.compile("bb")) == repr(pipeline.compute({"bb": 1}).plan)

    pipeline = compose("t", *ops, outputs=["sum2"])
    assert repr(pipeline).startswith(pipeline_str)
    assert not pipeline.compile("bb").steps
    assert len(pipeline.compile("a").steps) == 3
    assert repr(pipeline.compile("a")).startswith(
        "ExecutionPlan(needs=['a'], provides=['sum2'], x3 steps:"
    )

    ## Narrow by BOTH
    #
    pipeline = compose("t", *ops, outputs=["sum1"])
    assert repr(pipeline.compile(inputs="a")).startswith(
        "ExecutionPlan(needs=['a'(?)], provides=['sum1'], x3 steps:"
    )

    pipeline = compose("t", *ops, outputs=["sum2"])
    with pytest.raises(ValueError, match="Unsolvable graph:"):
        pipeline.compute({"bb": 11})


@pytest.fixture
def quarantine_pipeline(exemethod):
    @operation(
        rescheduled=1, needs="quarantine", provides=["space", "time"], returns_dict=True
    )
    def get_out_or_stay_home(quarantine):
        if quarantine:
            return {"time": "1h"}
        else:
            return {"space": "around the block"}

    @operation(needs="space", provides=["fun", "body"])
    def exercise(where):
        return "refreshed", "strong feet"

    @operation(needs="time", provides=["fun", "brain"])
    def read_book(for_how_long):
        return "relaxed", "popular physics"

    pipeline = compose(
        "covid19", get_out_or_stay_home, exercise, read_book, parallel=exemethod
    )
    return pipeline


def test_rescheduled_quarantine_doctest(quarantine_pipeline):
    pipeline = quarantine_pipeline
    sol = pipeline(quarantine=True)
    assert sol == {
        "quarantine": True,
        "time": "1h",
        "fun": "relaxed",
        "brain": "popular physics",
    }
    sol = pipeline(quarantine=False)
    assert sol == {
        "quarantine": False,
        "space": "around the block",
        "fun": "refreshed",
        "body": "strong feet",
    }


def test_rescheduled_quarantine_with_overwrites(quarantine_pipeline):
    friendly_garden = operation(lambda: "garden", name="friends", provides="space")

    pipeline = compose("quarantine with friends", friendly_garden, quarantine_pipeline)
    fun_overwrites = ["relaxed", "refreshed"]
    space_overwrites = ["around the block", "garden"]

    ## 1st Friendly garden
    #
    sol = pipeline(quarantine=True)
    assert sol == {
        "quarantine": True,
        "space": "garden",
        "time": "1h",
        "fun": "relaxed",
        "body": "strong feet",
        "brain": "popular physics",
    }
    assert sol.overwrites == {"fun": fun_overwrites}
    sol = pipeline(quarantine=False)
    # same as original pipe.
    assert sol == {
        "quarantine": False,
        "space": "around the block",
        "fun": "refreshed",
        "body": "strong feet",
    }
    assert sol.overwrites == {"space": space_overwrites}

    ## Friends 2nd, as a fallback
    #
    pipeline = compose("quarantine with friends", quarantine_pipeline, friendly_garden)

    sol = pipeline(quarantine=True)
    assert sol == {
        "quarantine": True,
        "time": "1h",
        "fun": "refreshed",
        "brain": "popular physics",
        "space": "garden",
        "body": "strong feet",
    }
    assert sol.overwrites == {"fun": fun_overwrites[::-1]}
    sol = pipeline(quarantine=False)
    # garden prevails.
    assert sol == {
        "quarantine": False,
        "space": "garden",
        "fun": "refreshed",
        "body": "strong feet",
    }
    assert sol.overwrites == {"space": space_overwrites[::-1]}


def test_optional_per_function_with_same_output(exemethod):
    # Test that the same need can be both optional and not on different operations.
    #
    ## ATTENTION, the selected function is NOT the one with more inputs
    # but the 1st satisfiable function added in the network.

    add_op = operation(name="add", needs=["a", "b"], provides="a+-b")(add)
    sub_op_optional = operation(
        name="sub_opt", needs=["a", optional("b")], provides="a+-b"
    )(lambda a, b=10: a - b)

    # Normal order
    #
    pipeline = compose("partial_optionals", add_op, sub_op_optional, parallel=exemethod)
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(**named_inputs) == {"a": 1, "a+-b": -1, "b": 2}
    assert pipeline.compute(named_inputs, ["a+-b"]) == {"a+-b": -1}
    #
    named_inputs = {"a": 1}
    assert pipeline.compute(named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline.compute(named_inputs, ["a+-b"]) == {"a+-b": -9}

    # Inverse op order
    #
    pipeline = compose("partial_optionals", sub_op_optional, add_op, parallel=exemethod)
    #
    named_inputs = {"a": 1, "b": 2}
    assert pipeline(**named_inputs) == {"a": 1, "a+-b": 3, "b": 2}
    assert pipeline.compute(named_inputs, ["a+-b"]) == {"a+-b": 3}
    #
    named_inputs = {"a": 1}
    assert pipeline(**named_inputs) == {"a": 1, "a+-b": -9}
    assert pipeline.compute(named_inputs, ["a+-b"]) == {"a+-b": -9}


def test_same_inputs_evictions():
    # Test operations providing the same output ordered as given.
    pipeline = compose(
        "add_sub",
        operation(name="x2", needs=["a", "a"], provides=["2a"])(add),
        operation(name="pipe", needs=["2a"], provides=["@S"])(identity),
    )

    inp = {"a": 3}
    assert pipeline(**inp) == {"a": 3, "2a": 6, "@S": 6}
    sol = pipeline.compute(inp, "@S")
    assert sol == {"@S": 6}
    ## Check it does not duplicate evictions
    assert len(sol.plan.steps) == 4


def test_evict_optional():
    # Test that evictions included for optionals do not raise
    # exceptions when the corresponding input is not provided.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    # Here, an eviction-dependency will be inserted for the optional need 'c'.
    sum_op1 = operation(
        name="sum_op1", needs=["a", "b", optional("c")], provides="sum1"
    )(addplusplus)
    sum_op2 = operation(name="sum_op2", needs=["sum1", "sum1"], provides="sum2")(add)
    net = compose("test_net", sum_op1, sum_op2)

    # Evictions happen only when a subset of outputs are requested.
    results = net.compute({"a": 4, "b": 3}, ["sum2"])
    assert "sum2" in results


def test_evict_instructions_vary_with_inputs():
    # Check #21: eviction-steps positions vary when inputs change.
    def count_evictions(steps):
        return sum(isinstance(n, str) for n in steps)

    pipeline = compose(
        "pipeline",
        operation(name="a free without b", needs=["a"], provides=["aa"])(identity),
        operation(name="satisfiable", needs=["a", "b"], provides=["ab"])(add),
        operation(name="optional ab", needs=["aa", optional("ab")], provides=["asked"])(
            lambda a, ab=10: a + ab
        ),
    )

    inp = {"a": 2, "b": 3}
    exp = inp.copy()
    exp.update({"aa": 2, "ab": 5, "asked": 7})
    res = pipeline(**inp)
    assert res == exp  # ok
    steps11 = pipeline.net.compile(inp).steps
    res = pipeline.compute(inp, ["asked"])
    assert res == filtdict(exp, "asked")  # ok
    steps12 = pipeline.net.compile(inp, ["asked"]).steps

    inp = {"a": 2}
    exp = inp.copy()
    exp.update({"aa": 2, "asked": 12})
    res = pipeline(**inp)
    assert res == exp  # ok
    steps21 = pipeline.net.compile(inp).steps
    res = pipeline.compute(inp, ["asked"])
    assert res == filtdict(exp, "asked")  # ok
    steps22 = pipeline.net.compile(inp, ["asked"]).steps

    # When no outs, no evict-instructions.
    assert steps11 != steps12
    assert count_evictions(steps11) == 0
    assert steps21 != steps22
    assert count_evictions(steps21) == 0

    # Check steps vary with inputs
    #
    # FAILs in v1.2.4 + #18, PASS in #26
    assert steps11 != steps21

    # Check evicts vary with inputs
    #
    # FAILs in v1.2.4 + #18, PASS in #26
    assert count_evictions(steps12) != count_evictions(steps22)


def test_skip_eviction_flag():
    graph = compose(
        "graph",
        operation(name="add1", needs=["a", "b"], provides=["ab"])(add),
        operation(name="add2", needs=["a", "ab"], provides=["aab"])(add),
    )
    with evictions_skipped(True):
        exp = {"a": 1, "b": 3, "ab": 4, "aab": 5}
        assert graph.compute({"a": 1, "b": 3}, "aab") == exp


@pytest.mark.parametrize(
    "endurance, endured", [(None, True), (True, None), (1, 0), (1, 1)],
)
def test_execution_endurance(exemethod, endurance, endured):
    with operations_endured(endurance):
        opb = operation(
            scream, needs=["a", "b"], provides=["a+b", "c"], endured=endured
        )
        scream1 = opb.withset(name="scream1")
        scream2 = opb.withset(name="scream2")
        add1 = operation(name="add1", needs=["a", "b"], provides=["a+b"])(add)
        add2 = operation(name="add2", needs=["a+b", "b"], provides=["a+2b"])(add)
        canceled = operation(name="canceled", needs=["c"], provides="cc")(identity)
        graph = compose(
            "graph", scream1, add1, scream2, add2, canceled, parallel=exemethod
        )

        inp = {"a": 1, "b": 2}
        sol = graph(**inp)
        assert sol.is_failed(scream1) and sol.is_failed(scream2)
        assert "Must not have run!" in str(sol.executed[scream1])
        assert sol == {"a+b": 3, "a+2b": 5, **inp}
        assert "x2 failures" in str(sol.check_if_incomplete())
        with pytest.raises(IncompleteExecutionError, match="x2 failures"):
            assert sol.scream_if_incomplete()

        sol = graph.withset(outputs="a+2b")(**inp)
        assert sol.is_failed(scream1) and sol.is_failed(scream2)
        assert "Must not have run!" in str(sol.executed[scream1])
        assert sol == {"a+2b": 5}

        # SILENTLY failing asked outputs
        sol = graph.compute(inp, outputs=["a+2b", "cc"])
        assert sol == {"a+2b": 5}

        # Check plotting Fail & Cancel.
        #
        dot = str(sol.plot())
        assert "LightCoral" in dot  # failed
        assert "#a9a9a9" in dot  # Canceled


@pytest.mark.parametrize(
    "resched, rescheduled", [(None, True), (True, None), (1, 0), (1, 1)],
)
def test_rescheduling(exemethod, resched, rescheduled):
    canc = operation(lambda: None, name="canc", needs=["b"], provides="cc")
    op = compose(
        "pipeline",
        operation(lambda: [1], name="op1", provides=["a", "b"], rescheduled=1),
        canc,
        operation(
            lambda C=1: C and NO_RESULT,
            name="op2",
            needs=optional("C"),
            provides=["c"],
            rescheduled=1,
        ),
        operation(
            lambda *args: sum(args),
            name="op3",
            needs=["a", vararg("b"), vararg("c")],
            provides=["d"],
        ),
        parallel=exemethod,
    )
    sol = op.compute({})
    assert sol == {"a": 1, "d": 1}
    assert list(sol.canceled) == [canc]
    dot = str(sol.plot())
    assert "#a9a9a9" in dot  # Canceled
    assert 'BORDER="4"' in dot  # Rescheduled
    assert "x2 partial-ops" in str(sol.check_if_incomplete())
    with pytest.raises(IncompleteExecutionError, match="x2 partial-ops"):
        assert sol.scream_if_incomplete()

    ## Check if modified state fails the 2nd time.
    assert op.compute({}) == {"a": 1, "d": 1}

    ## Tell op to cancel just 1 of 2 provides
    #  (the 2n one, 'b').
    #
    sol = op.compute({"C": False})
    assert sol == {"C": False, "a": 1, "c": False, "d": 1}


def test_rescheduling_NO_RESULT(exemethod):
    partial = operation(lambda: NO_RESULT, name="op1", provides=["a"], rescheduled=1)
    canc = operation(lambda: None, name="canc", needs="a", provides="b")
    op = compose("pipeline", partial, canc, parallel=exemethod)
    sol = op()
    assert canc in sol.canceled
    assert partial in sol.executed
    assert "x1 partial-ops" in str(sol.check_if_incomplete())
    with pytest.raises(IncompleteExecutionError, match="x1 partial-ops"):
        assert sol.scream_if_incomplete()


def test_jetsam_n_plot_with_DEBUG():
    pipe = compose(
        "mix",
        operation(
            str,
            "FUNC",
            needs=[
                "a",
                sfxed("b", "foo", keyword="bb"),
                modify("c", implicit=1),
                sfxed("d", "bar", implicit=1),
                vararg("e"),
                varargs("f"),
            ],
            provides=[
                "A",
                sfxed("b", "FOO", keyword="bb"),
                modify("C", implicit=1),
                sfxed("d", "BAR", implicit=1),
                sfx("FOOBAR"),
            ],
            aliases={"A": "aaa", "b": "bbb", "C": "ccc", "d": "ddd"},
        ),
    )

    with debug_enabled(True), pytest.raises(ValueError, match="^Unsolvable"):
        pipe.compute()
    with debug_enabled(True), pytest.raises(
        ValueError, match="^Failed matching inputs <=> needs"
    ) as exc:
        pipe.compute(
            {
                "a": 1,
                sfxed("b", "foo"): 2,
                "c": 3,
                sfxed("d", "bar"): 4,
                "e": 5,
                "f": [6, 7],
            }
        )

    exc.value.jetsam.plot_fpath.unlink()
