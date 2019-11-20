# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import pytest

from graphtik import operation, optional
from graphtik.op import Operation, reparse_operation_data


@pytest.fixture(params=[None, ["some"]])
def opname(request):
    return request.param


@pytest.fixture(params=[None, ["some"]])
def opneeds(request):
    return request.param


@pytest.fixture(params=[None, ["some"]])
def opprovides(request):
    return request.param


class MyOp(Operation):
    def compute(self):
        pass


def test_operation_repr_smoke(opname, opneeds, opprovides):
    # Simply check __repr__() does not crash on partial attributes.
    kw = locals().copy()
    kw = {name[2:]: arg for name, arg in kw.items()}

    op = operation(**kw)
    str(op)

    op = MyOp(**kw)
    str(op)


def test_operation_repr_returns_dict():
    assert (
        str(operation(lambda: None, returns_dict=True)())
        == "FunctionalOperation(name=None, needs=[], provides=[], fn{}='<lambda>')"
    )


@pytest.mark.parametrize(
    "opargs, exp",
    [
        ((None, None, None), (None, [], [])),
        ## Check name
        (("_", "a", ("A",)), ("_", ["a"], ("A",))),
        (((), ("a",), None), ((), ("a",), [])),
        ((("a",), "a", "b"), (("a",), ["a"], ["b"])),
        ((("a",), None, None), (("a",), [], [])),
        ## Check needs
        (((), (), None), ((), (), [])),
        (((), [], None), ((), [], [])),
        (("", object(), None), ValueError("Argument 'needs' not an iterable")),
        (("", [None], None), ValueError("All `needs` must be str")),
        (("", [()], None), ValueError("All `needs` must be str")),
        ## Check provides
        (((), "a", ()), ((), ["a"], ())),
        (((), "a", []), ((), ["a"], [])),
        (("", "a", object()), ValueError("Argument 'provides' not an iterable")),
        (("", "a", (None,)), ValueError("All `provides` must be str")),
        (("", "a", [()]), ValueError("All `provides` must be str")),
    ],
)
def test_operation_validation(opargs, exp):
    if isinstance(exp, Exception):
        with pytest.raises(type(exp), match=str(exp)):
            reparse_operation_data(*opargs)
    else:
        assert reparse_operation_data(*opargs) == exp


def test_operation_returns_dict():
    result = {"a": 1}

    op = operation(lambda: result, provides="a", returns_dict=True)()
    assert op.compute({}) == result

    op = operation(lambda: 1, provides="a", returns_dict=False)()
    assert op.compute({}) == result
