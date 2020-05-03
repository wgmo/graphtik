import sys
from functools import partial
from pathlib import Path
from operator import mul, sub
import pytest

from graphtik import compose, operation

# Enable pytest-sphinx fixtures
# See https://www.sphinx-doc.org/en/master/devguide.html#unit-testing
pytest_plugins = "sphinx.testing.fixtures"

# TODO: is this needed along with norecursedirs?
# See https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
collect_ignore = ["helpers.py"]


@pytest.fixture
def debug_mode():
    from graphtik import debug_enabled

    with debug_enabled(True):
        yield


def abspow(a, p):
    c = abs(a) ** p
    return c


@pytest.fixture
def merge_pipes():
    graphop = compose(
        "graphop",
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
        operation(
            name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"]
        )(partial(abspow, p=3)),
    )

    another_graph = compose(
        "another_graph",
        operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
        operation(name="mul2", needs=["c", "ab"], provides=["cab"])(mul),
    )

    return graphop, another_graph
