# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Test :term:`parallel`, :term:`marshalling` and other :term:`execution` related stuff. """
import os
from functools import partial
from operator import mul, sub
from multiprocessing import cpu_count
from multiprocessing import dummy as mp_dummy
from time import sleep, time

import pytest

from graphtik import AbortedException, compose, operation, optional
from graphtik.config import abort_run, execution_pool_plugged
from graphtik.execution import _OpTask, task_context

from .helpers import _exe_params, abspow


@pytest.mark.xfail(reason="Spurious passes when threading with on low-cores?")
def test_task_context(exemethod, request):
    def check_task_context():
        sleep(0.15)
        assert task_context.get().op == next(iop), "Corrupted task-context"

    n_ops = 10
    pipe = compose(
        "t",
        *(
            operation(check_task_context, f"op{i}", provides=f"{i}")
            for i in range(n_ops)
        ),
        parallel=exemethod,
    )
    iop = iter(pipe.ops)

    print(_exe_params, cpu_count())
    err = None
    if _exe_params.proc and _exe_params.marshal:
        err = Exception("^Error sending result")
    elif _exe_params.parallel and _exe_params.marshal:
        err = AssertionError("^Corrupted task-context")
    elif _exe_params.parallel and not os.environ.get("TRAVIS"):
        # Travis has low parallelism and error does not surface
        err = AssertionError("^Corrupted task-context")

    if err:
        with pytest.raises(type(err), match=str(err)):
            pipe.compute()
        raise pytest.xfail("Cannot marshal parallel processes with `task_context` :-(.")
    else:
        pipe.compute()
        with pytest.raises(StopIteration):
            next(iop)


@pytest.mark.xfail(
    reason="Spurious copied-reversed graphs in Travis, with dubious cause...."
)
def test_multithreading_plan_execution():
    # Compose the mul, sub, and abspow operations into a computation graph.
    # From Huygn's test-code given in yahoo/graphkit#31
    graph = compose(
        "graph",
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
        operation(
            name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"]
        )(partial(abspow, p=3)),
    )

    with mp_dummy.Pool(int(2 * cpu_count())) as pool, execution_pool_plugged(pool):
        pool.map(
            # lambda i: graph.withset(name='graph').compute(
            lambda i: graph.compute(
                {"a": 2, "b": 5}, ["a_minus_ab", "abs_a_minus_ab_cubed"]
            ),
            range(300),
        )


@pytest.mark.slow
def test_parallel_execution(exemethod):
    if not exemethod:
        return

    delay = 0.5

    def fn(x):
        sleep(delay)
        print("fn %s" % (time() - t0))
        return 1 + x

    def fn2(a, b):
        sleep(delay)
        print("fn2 %s" % (time() - t0))
        return a + b

    def fn3(z, k=1):
        sleep(delay)
        print("fn3 %s" % (time() - t0))
        return z + k

    pipeline = compose(
        "l",
        # the following should execute in parallel under threaded execution mode
        operation(name="a", needs="x", provides="ao")(fn),
        operation(name="b", needs="x", provides="bo")(fn),
        # this should execute after a and b have finished
        operation(name="c", needs=["ao", "bo"], provides="co")(fn2),
        operation(name="d", needs=["ao", optional("k")], provides="do")(fn3),
        operation(name="e", needs=["ao", "bo"], provides="eo")(fn2),
        operation(name="f", needs="eo", provides="fo")(fn),
        operation(name="g", needs="fo", provides="go")(fn),
        nest=False,
    )

    t0 = time()
    result_threaded = pipeline.withset(parallel=True).compute(
        {"x": 10}, ["co", "go", "do"]
    )
    # print("threaded result")
    # print(result_threaded)

    t0 = time()
    pipeline = pipeline.withset(parallel=False)
    result_sequential = pipeline.compute({"x": 10}, ["co", "go", "do"])
    # print("sequential result")
    # print(result_sequential)

    # make sure results are the same using either method
    assert result_sequential == result_threaded


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Spurious copied-reversed graphs in Travis, with dubious cause...."
)
def test_multi_threading_computes():
    import random

    def op_a(a, b):
        sleep(random.random() * 0.02)
        return a + b

    def op_b(c, b):
        sleep(random.random() * 0.02)
        return c + b

    def op_c(a, b):
        sleep(random.random() * 0.02)
        return a * b

    pipeline = compose(
        "pipeline",
        operation(name="op_a", needs=["a", "b"], provides="c")(op_a),
        operation(name="op_b", needs=["c", "b"], provides="d")(op_b),
        operation(name="op_c", needs=["a", "b"], provides="e")(op_c),
        nest=False,
    )

    def infer(i):
        # data = open("616039-bradpitt.jpg").read()
        outputs = ["c", "d", "e"]
        results = pipeline.compute({"a": 1, "b": 2}, outputs)
        assert tuple(sorted(results.keys())) == tuple(sorted(outputs)), (
            outputs,
            results,
        )
        return results

    N = 33
    for i in range(13, 61):
        with mp_dummy.Pool(i) as pool:
            pool.map(infer, range(N))


def test_abort(exemethod):
    pipeline = compose(
        "pipeline",
        operation(fn=None, name="A", needs=["a"], provides=["b"]),
        operation(name="B", needs=["b"], provides=["c"])(lambda x: abort_run()),
        operation(fn=None, name="C", needs=["c"], provides=["d"]),
        parallel=exemethod,
    )
    with pytest.raises(AbortedException) as exinfo:
        pipeline(a=1)

    exp = {"a": 1, "b": 1, "c": None}
    solution = exinfo.value.args[0]
    assert solution == exp
    assert exinfo.value.jetsam["solution"] == exp
    executed = {op.name: val for op, val in solution.executed.items()}
    assert executed == {"A": None, "B": None}

    pipeline = compose(
        "pipeline",
        operation(fn=None, name="A", needs=["a"], provides=["b"]),
        parallel=exemethod,
    )
    assert pipeline.compute({"a": 1}) == {"a": 1, "b": 1}
