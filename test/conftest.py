import logging
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
    _marshal,
    _parallel,
    _proc,
    _slow,
    _thread,
    dilled,
    exe_params,
    pickled,
)

# Enable pytest-sphinx fixtures
# See https://www.sphinx-doc.org/en/master/devguide.html#unit-testing
pytest_plugins = "sphinx.testing.fixtures"

# TODO: is this needed along with norecursedirs?
# See https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
collect_ignore = ["helpers.py"]

########
## From https://stackoverflow.com/a/57002853/548792
##
def pytest_addoption(parser):
    """Add a command line option to disable logger."""
    parser.addoption(
        "--logger-disabled",
        action="append",
        default=[],
        help="disable specific loggers",
    )


def pytest_configure(config):
    """Disable the loggers from CLI and silence sphinx markers warns."""
    for name in config.getoption("--logger-disabled", default=[]):
        logger = logging.getLogger(name)
        logger.propagate = False

    config.addinivalue_line("markers", "sphinx: parametrized sphinx test-launches")
    config.addinivalue_line(
        "markers", "test_params: for parametrized sphinx test-launches"
    )


##
########


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
    """Returns exe-method combinations, and store them globally, for xfail checks."""
    parallel, proc_pool, marshal = request.param
    exe_params.parallel, exe_params.proc, exe_params.marshal = request.param

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
    """sum1 = (a + b), sum2 = (c + d), sum3 = c + (c + d)"""
    sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
    sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
    sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
    return compose("test_net", sum_op1, sum_op2, sum_op3)


@pytest.fixture(params=[10, 20])
def log_levels(request, caplog):
    caplog.set_level(request.param)
    return
