# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Test :term:`implicit` & :term:`sideffects`."""
import re
import sys
import types
from itertools import cycle
from operator import add, mul, sub
from textwrap import dedent
from typing import Any, Tuple

import pytest

from graphtik import (
    NO_RESULT,
    NO_RESULT_BUT_SFX,
    compose,
    implicit,
    modify,
    operation,
    sfx,
    sfxed,
)
from graphtik.config import get_execution_pool, is_marshal_tasks
from graphtik.pipeline import Pipeline

pytestmark = pytest.mark.usefixtures("log_levels")


# Function without return value.
def _box_extend(box, *args):
    box.extend([1, 2])


def _box_increment(box):
    for i in range(len(box)):
        box[i] += 1


@pytest.fixture(params=[0, 1])
def pipeline_sideffect1(request, exemethod) -> Pipeline:
    ops = [
        operation(name="extend", needs=["box", sfx("a")], provides=[sfx("b")])(
            _box_extend
        ),
        operation(name="increment", needs=["box", sfx("b")], provides=sfx("c"))(
            _box_increment
        ),
    ]
    if request.param:
        ops = reversed(ops)
    # Designate `a`, `b` as sideffect inp/out arguments.
    graph = compose("sideffect1", *ops, parallel=exemethod)

    return graph


def test_sideffect_no_real_data(pipeline_sideffect1: Pipeline):
    sidefx_fail = is_marshal_tasks() and not isinstance(
        get_execution_pool(), types.FunctionType  # mp_dummy.Pool
    )

    graph = pipeline_sideffect1
    inp = {"box": [0], "a": True}

    ## Normal data must not match sideffects.
    #
    # with pytest.raises(ValueError, match="Unknown output node"):
    #     graph.compute(inp, ["a"])
    # with pytest.raises(ValueError, match="Unknown output node"):
    #     graph.compute(inp, ["b"])

    ## Cannot compile due to missing inputs/outputs
    #
    with pytest.raises(ValueError, match="Unsolvable graph"):
        graph(**inp)
    with pytest.raises(ValueError, match="Unsolvable graph"):
        graph.compute(inp)

    with pytest.raises(ValueError, match="Unreachable outputs"):
        graph.compute(inp, ["box", sfx("b")])

    with pytest.raises(ValueError, match="Unsolvable graph"):
        # Cannot run, since no sideffect inputs given.
        graph.compute(inp)

    box_orig = [0]

    ## OK INPUT SIDEFFECTS
    #
    # ok, no asked out
    sol = graph.compute({"box": [0], sfx("a"): True})
    assert sol == {"box": box_orig if sidefx_fail else [1, 2, 3], sfx("a"): True}
    #
    # Although no out-sideffect asked (like regular data).
    assert graph.compute({"box": [0], sfx("a"): True}, "box") == {"box": [0]}
    #
    # ok, asked the 1st out-sideffect
    sol = graph.compute({"box": [0], sfx("a"): True}, ["box", sfx("b")])
    assert sol == {"box": box_orig if sidefx_fail else [0, 1, 2]}
    #
    # ok, asked the 2nd out-sideffect
    sol = graph.compute({"box": [0], sfx("a"): True}, ["box", sfx("c")])
    assert sol == {"box": box_orig if sidefx_fail else [1, 2, 3]}


@pytest.mark.parametrize("reverse", [0, 1])
def test_sideffect_real_input(reverse, exemethod):
    sidefx_fail = is_marshal_tasks() and not isinstance(
        get_execution_pool(), types.FunctionType  # mp_dummy.Pool
    )

    ops = [
        operation(name="extend", needs=["box", "a"], provides=[sfx("b")])(_box_extend),
        operation(name="increment", needs=["box", sfx("b")], provides="c")(
            _box_increment
        ),
    ]
    if reverse:
        ops = reversed(ops)
    # Designate `a`, `b` as sideffect inp/out arguments.
    graph = compose("mygraph", *ops, parallel=exemethod)

    box_orig = [0]
    assert graph(**{"box": [0], "a": True}) == {
        "a": True,
        "box": box_orig if sidefx_fail else [1, 2, 3],
        "c": None,
    }
    assert graph.compute({"box": [0], "a": True}, ["box", "c"]) == {
        "box": box_orig if sidefx_fail else [1, 2, 3],
        "c": None,
    }


def test_sideffect_steps(exemethod, pipeline_sideffect1: Pipeline):
    sidefx_fail = is_marshal_tasks() and not isinstance(
        get_execution_pool(), types.FunctionType  # mp_dummy.Pool
    )

    pipeline = pipeline_sideffect1.withset(parallel=exemethod)
    box_orig = [0]
    sol = pipeline.compute({"box": [0], sfx("a"): True}, ["box", sfx("c")])
    assert sol == {"box": box_orig if sidefx_fail else [1, 2, 3]}
    assert len(sol.plan.steps) == 4

    ## Check sideffect links plotted as blue
    #  (assumes color used only for this!).
    dot = pipeline.net.plot()
    assert "blue" in str(dot)


def test_sideffect_NO_RESULT(caplog, exemethod):
    # NO_RESULT does not cancel sideffects unless op-rescheduled
    #
    an_sfx = sfx("b")
    op1 = operation(lambda: NO_RESULT, name="do-SFX", provides=an_sfx)
    op2 = operation(lambda: 1, name="ask-SFX", needs=an_sfx, provides="a")
    pipeline = compose("t", op1, op2, parallel=exemethod)
    sol = pipeline.compute({}, outputs=an_sfx)
    assert op1 in sol.executed
    assert op2 not in sol.executed
    assert sol == {}
    sol = pipeline.compute({})
    assert op1 in sol.executed
    assert op2 in sol.executed
    assert sol == {"a": 1}
    sol = pipeline.compute({}, outputs="a")
    assert op1 in sol.executed
    assert op2 in sol.executed
    assert sol == {"a": 1}

    # NO_RESULT cancels sideffects of rescheduled ops.
    #
    pipeline = compose("t", op1, op2, rescheduled=True, parallel=exemethod)
    sol = pipeline.compute({})
    assert op1 in sol.executed
    assert op2 not in sol.executed
    assert sol == {an_sfx: False}
    sol = pipeline.compute({}, outputs="a")
    assert op2 not in sol.executed
    assert op1 in sol.executed
    assert sol == {}  # an_sfx evicted

    # NO_RESULT_BUT_SFX cancels sideffects of rescheduled ops.
    #
    op11 = operation(lambda: NO_RESULT_BUT_SFX, name="do-SFX", provides=an_sfx)
    pipeline = compose("t", op11, op2, rescheduled=True, parallel=exemethod)
    sol = pipeline.compute({}, outputs=an_sfx)
    assert op11 in sol.executed
    assert op2 not in sol.executed
    assert sol == {}
    sol = pipeline.compute({})
    assert op11 in sol.executed
    assert op2 in sol.executed
    assert sol == {"a": 1}
    sol = pipeline.compute({}, outputs="a")
    assert op11 in sol.executed
    assert op2 in sol.executed
    assert sol == {"a": 1}

    ## If NO_RESULT were not translated,
    #  a warning of unknown out might have emerged.
    caplog.clear()
    pipeline = compose("t", operation(lambda: 1, provides=an_sfx), parallel=exemethod)
    pipeline.compute({}, outputs=an_sfx)
    for record in caplog.records:
        if record.levelname == "WARNING":
            assert "Ignoring result(1) because no `provides`" in record.message

    caplog.clear()
    pipeline = compose(
        "t", operation(lambda: NO_RESULT, provides=an_sfx), parallel=exemethod
    )
    pipeline.compute({}, outputs=an_sfx)
    for record in caplog.records:
        assert record.levelname != "WARNING"


def test_sideffect_cancel_sfx_only_operation(exemethod):
    an_sfx = sfx("b")
    op1 = operation(
        lambda: {an_sfx: False},
        name="op1",
        provides=an_sfx,
        returns_dict=True,
        rescheduled=True,
    )
    op2 = operation(lambda: 1, name="op2", needs=an_sfx, provides="a")
    pipeline = compose("t", op1, op2, parallel=exemethod)
    sol = pipeline.compute({})
    assert sol == {an_sfx: False}
    sol = pipeline.compute(outputs=an_sfx)
    assert sol == {an_sfx: False}


def test_sideffect_cancel(exemethod):
    an_sfx = sfx("b")
    op1 = operation(
        lambda: {"a": 1, an_sfx: False},
        name="op1",
        provides=["a", an_sfx],
        returns_dict=True,
        rescheduled=True,
    )
    op2 = operation(lambda: 1, name="op2", needs=an_sfx, provides="b")
    pipeline = compose("t", op1, op2, parallel=exemethod)
    sol = pipeline.compute()
    assert sol == {"a": 1, an_sfx: False}
    sol = pipeline.compute(outputs="a")
    assert sol == {"a": 1}  # an_sfx evicted
    ## SFX both pruned & evicted
    #
    assert an_sfx not in sol.dag.nodes
    assert an_sfx in sol.plan.steps


def test_sideffect_not_canceled_if_not_resched(exemethod):
    # Check op without any provides
    #
    an_sfx = sfx("b")
    op1 = operation(
        lambda: {an_sfx: False}, name="op1", provides=an_sfx, returns_dict=True
    )
    op2 = operation(lambda: 1, name="op2", needs=an_sfx, provides="b")
    pipeline = compose("t", op1, op2, parallel=exemethod)
    # sol = pipeline.compute()
    # assert sol == {an_sfx: False, "b": 1}
    sol = pipeline.compute(outputs="b")
    assert sol == {"b": 1}

    # Check also op with some provides
    #
    an_sfx = sfx("b")
    op1 = operation(
        lambda: {"a": 1, an_sfx: False},
        name="op1",
        provides=["a", an_sfx],
        returns_dict=True,
    )
    op2 = operation(lambda: 1, name="op2", needs=an_sfx, provides="b")
    pipeline = compose("t", op1, op2, parallel=exemethod)
    sol = pipeline.compute()
    assert sol == {"a": 1, an_sfx: False, "b": 1}
    sol = pipeline.compute(outputs="b")
    assert sol == {"b": 1}


DataFrame = Any


@pytest.fixture(params=[0, 1])
def calc_prices_pipeline(request, exemethod):
    """A pipeline that may work even without VAT-rates."""

    @operation(needs="order_items", provides=sfxed("ORDER", "Items", "Prices"))
    def new_order(items: list) -> DataFrame:
        order = {"items": items}
        # Pretend we get the prices from sales.
        order["prices"] = list(range(1, len(order["items"]) + 1))
        return order

    @operation(
        needs=[sfxed("ORDER", "Items"), "vat rate"],
        provides=sfxed("ORDER", "VAT rates"),
    )
    def fill_in_vat_ratios(order: DataFrame, base_vat: float) -> DataFrame:
        order["VAT_rates"] = [
            v for _, v in zip(order["prices"], cycle((base_vat, 2 * base_vat)))
        ]
        return order

    @operation(
        needs=[
            sfxed("ORDER", "Prices"),
            sfxed("ORDER", "VAT rates", optional=True),
        ],
        provides=[sfxed("ORDER", "VAT", "Totals"), "vat owed"],
    )
    def finalize_prices(order: DataFrame) -> Tuple[DataFrame, float]:
        if "VAT_rates" in order:
            order["VAT"] = [p * v for p, v in zip(order["prices"], order["VAT_rates"])]
            order["totals"] = [p + v for p, v in zip(order["prices"], order["VAT"])]
            vat_to_pay = sum(order["VAT"])
        else:
            order["totals"] = order["prices"][::]
            vat_to_pay = None
        return order, vat_to_pay

    ops = [new_order, fill_in_vat_ratios, finalize_prices]
    if request.param:
        ops = reversed(ops)
    return compose("process order", *ops, parallel=exemethod)


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason="Pickling function typing of return annotations fails.",
)
def test_sideffecteds_ok(calc_prices_pipeline):
    inp = {"order_items": "milk babylino toilet-paper".split(), "vat rate": 0.18}
    sol = calc_prices_pipeline.compute(inp)
    print(sol)
    assert sol == {
        "order_items": ["milk", "babylino", "toilet-paper"],
        "vat rate": 0.18,
        "ORDER": {
            "items": ["milk", "babylino", "toilet-paper"],
            "prices": [1, 2, 3],
            "VAT_rates": [0.18, 0.36, 0.18],
            "VAT": [0.18, 0.72, 0.54],
            "totals": [1.18, 2.7199999999999998, 3.54],
        },
        "vat owed": 1.44,
    }
    sol = calc_prices_pipeline.compute(
        inp, [sfxed("ORDER", "VAT"), sfxed("ORDER", "Totals")]
    )
    print(sol)
    assert sol == {
        "ORDER": {
            "items": ["milk", "babylino", "toilet-paper"],
            "prices": [1, 2, 3],
            "VAT_rates": [0.18, 0.36, 0.18],
            "VAT": [0.18, 0.72, 0.54],
            "totals": [1.18, 2.7199999999999998, 3.54],
        },
        # "vat owed": 1.44,
    }

    ## `vat owed` both pruned & evicted
    #
    assert "vat owed" not in sol.dag.nodes
    assert "vat owed" in sol.plan.steps
    # Check Pruned+Evicted data plot as expected.
    #
    dot = str(sol.plot())
    print(dot)
    assert re.search(r"(?s)>vat owed<.+style=dashed", dot)
    assert re.search(r'(?s)>vat owed<.+tooltip=".*\(evicted\)"', dot)


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason="Pickling function typing of return annotations fails.",
)
def test_sideffecteds_endured(calc_prices_pipeline):
    ## Break `fill_in_vat_ratios()`.
    #
    @operation(
        needs=[sfxed("ORDER", "Items"), "vat rate"],
        provides=sfxed("ORDER", "VAT rates"),
        endured=True,
    )
    def fill_in_vat_ratios(order: DataFrame, base_vat: float) -> DataFrame:
        raise ValueError("EC transactions have no VAT!")

    calc_prices_pipeline = compose(
        calc_prices_pipeline.name, fill_in_vat_ratios, calc_prices_pipeline, nest=False
    )

    sol = calc_prices_pipeline.compute(
        {"order_items": "milk babylino toilet-paper".split(), "vat rate": 0.18}
    )

    print(sol)
    assert sol == {
        "order_items": ["milk", "babylino", "toilet-paper"],
        "vat rate": 0.18,
        "ORDER": {
            "items": ["milk", "babylino", "toilet-paper"],
            "prices": [1, 2, 3],
            "totals": [1, 2, 3],
        },
        "vat owed": None,
    }


@pytest.fixture(params=[0, 1])
def sideffected_resched(request, exemethod):
    @operation(provides=sfxed("DEP", "yes", "no"), rescheduled=1, returns_dict=1)
    def half_sfx():
        return {"DEP": 1, sfxed("DEP", "no"): False}

    yes = operation(
        lambda dep: "yes!", name="YES", needs=sfxed("DEP", "yes"), provides="yes"
    )
    no = operation(
        lambda dep: "no!", name="NO", needs=sfxed("DEP", "no"), provides="no"
    )
    ops = [half_sfx, yes, no]
    if request.param:
        ops = reversed(ops)
    return compose("sfxed_resched", *ops, parallel=exemethod)


def test_sideffected_canceled(sideffected_resched):
    """Check if a `returns-dict` op can cancel sideffecteds."""
    sol = sideffected_resched.compute({})
    print(sol)
    assert sol == {"DEP": 1, sfxed("DEP", "no"): False, "yes": "yes!"}


def test_implicit_inp():
    op = operation(str, needs=["A", implicit("a")], provides="b")

    pipe = compose(..., op)
    got = pipe.compute({"A": "val", "a": 1})
    assert got == {"A": "val", "a": 1, "b": "val"}

    with pytest.raises(ValueError, match="Unsolvable graph"):
        pipe.compute({"A": "val"})

    assert "(implicit)" in str(op.plot())


def test_implicit_out():
    op = operation(str, "hh", provides=["A", implicit("a")])

    pipe = compose(..., op)
    got = pipe.compute()
    assert got == {"A": ""}

    assert "(implicit)" in str(op.plot())
