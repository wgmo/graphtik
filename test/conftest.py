import os
from collections import namedtuple
from multiprocessing import Pool
from multiprocessing import dummy as mp_dummy
from multiprocessing import get_context
from operator import add

import pytest

from graphtik import compose, operation
from graphtik.config import debug_enabled, execution_pool_plugged, tasks_marshalled

from .helpers import (
    _exe_params,
    _ExeParams,
    _marshal,
    _parallel,
    _proc,
    _slow,
    _thread,
    dilled,
    pickled,
)

# Enable pytest-sphinx fixtures
# See https://www.sphinx-doc.org/en/master/devguide.html#unit-testing
pytest_plugins = "sphinx.testing.fixtures"

# TODO: is this needed along with norecursedirs?
# See https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
collect_ignore = ["helpers.py"]


@pytest.fixture
def debug_mode():
    from graphtik.config import debug_enabled

    with debug_enabled(True):
        yield


@pytest.fixture(params=[dilled, pickled])
def ser_method(request):
    return request.param


@pytest.fixture(
    params=[
        # PARALLEL?, Thread/Proc?, Marshalled?
        (None, None, None),
        pytest.param((1, 0, 0), marks=(_parallel, _thread)),
        pytest.param((1, 0, 1), marks=(_parallel, _thread, _marshal)),
        pytest.param((1, 1, 1), marks=(_parallel, _proc, _marshal, _slow)),
        pytest.param(
            (1, 1, 0),
            marks=(
                _parallel,
                _proc,
                _slow,
                pytest.mark.xfail(reason="ProcessPool non-marshaled may fail."),
            ),
        ),
    ]
)
def exemethod(request):
    """Returns (exemethod, marshal) combinations"""
    global _exe_params
    parallel, proc_pool, marshal = request.param
    _exe_params = _ExeParams(*request.param)

    nsharks = None  # number of pool swimmers....

    with tasks_marshalled(marshal):
        if parallel:
            if proc_pool:
                if os.name == "posix":  # Allthough it is the default ...
                    # NOTE: "spawn" DEADLOCKS!.
                    pool = get_context("fork").Pool(nsharks)
                else:
                    pool = Pool(nsharks)
            else:
                pool = mp_dummy.Pool(nsharks)

            with execution_pool_plugged(pool), pool:
                yield parallel
        else:
            yield parallel


@pytest.fixture
def samplenet():
    # Set up a network such that we don't need to provide a or b d if we only
    # request sum3 as output and if we provide sum2.
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
    return compose("test_net", sum_op1, sum_op2, sum_op3)
