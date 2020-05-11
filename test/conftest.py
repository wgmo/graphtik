import pytest
from .helpers import dilled, pickled


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
