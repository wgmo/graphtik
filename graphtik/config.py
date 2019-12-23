# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
""":term:`Configurations` for network execution, and utilities on them."""
import ctypes
from contextvars import ContextVar
from multiprocessing import Value
from typing import Optional

from boltons.iterutils import first

_abort: ContextVar[Optional[bool]] = ContextVar(
    "abort", default=Value(ctypes.c_bool, lock=False)
)
_skip_evictions: ContextVar[Optional[bool]] = ContextVar("skip_evictions", default=None)
_execution_pool: ContextVar[Optional["Pool"]] = ContextVar(
    "execution_pool", default=None
)
_parallel_tasks: ContextVar[Optional[bool]] = ContextVar("parallel_tasks", default=None)
_marshal_tasks: ContextVar[Optional[bool]] = ContextVar("marshal_tasks", default=None)
_endure_operations: ContextVar[Optional[bool]] = ContextVar(
    "endure_operations", default=None
)
_reschedule_operations: ContextVar[Optional[bool]] = ContextVar(
    "reschedule_operations", default=None
)


def abort_run():
    """Signal to the 1st running network to stop :term:`execution`."""
    _abort.get().value = True


def _reset_abort():
    _abort.get().value = False


def is_abort():
    """Return `True` if networks have been signaled to stop :term:`execution`."""
    return _abort.get().value


def set_skip_evictions(skipped):
    """If :term:`eviction` is true, keep all intermediate solution values, regardless of asked outputs."""
    return _skip_evictions.set(bool(skipped))


def is_skip_evictions():
    """Return `True` if keeping all intermediate solution values, regardless of asked outputs."""
    return _skip_evictions.get()


def set_execution_pool(pool: "Optional[Pool]"):
    """
    Set the process-pool for :term:`parallel` plan executions.

    You may have to :also func:`set_marshal_tasks()` to resolve
    pickling issues.
    """
    return _execution_pool.set(pool)


def get_execution_pool() -> "Optional[Pool]":
    """Get the process-pool for :term:`parallel` plan executions."""
    return _execution_pool.get()


def set_parallel_tasks(enable: Optional[bool]):
    """
    Enable/disable globally :term:`parallel` execution of operations.

    :param enable:
        - If ``None`` (default), respect the flag on each operation.
        - If true/false, force it for operations.

    :return:
        a "reset" token (see :meth:`.ContextVar.set`)
    """
    return _parallel_tasks.set(bool(enable))


def is_parallel_tasks() -> Optional[bool]:
    """see :meth:`set_parallel_tasks()`"""
    return _parallel_tasks.get()


def set_marshal_tasks(enable: Optional[bool]):
    """
    Enable/disable globally :term:`marshalling` of :term:`parallel` operations, ...

    inputs & outputs with :mod:`dill`,  which might help for pickling problems.

    :param enable:
        - If ``None`` (default), respect the flag on each operation.
        - If true/false, force it for operations.

    :return:
        a "reset" token (see :meth:`.ContextVar.set`)
    """
    return _marshal_tasks.set(bool(enable))


def is_marshal_tasks() -> Optional[bool]:
    """see :meth:`set_marshal_tasks()`"""
    return _marshal_tasks.get()


def set_endure_operations(endure):
    """
    If :term:`endurance` set to true, keep executing even if some operations fail

    :param enable:
        - If ``None`` (default), respect the flag on each operation.
        - If true/false, force it for operations.

    :return:
        a "reset" token (see :meth:`.ContextVar.set`)

    ."""
    return _endure_operations.set(bool(endure))


def is_endure_operations():
    """see :meth:`set_endure_operations`"""
    return _endure_operations.get()


def set_reschedule_operations(reschedule):
    """
    If :term:`reschedule` set to true, operations may provide *partial outputs*.

    :param enable:
        - If ``None`` (default), respect the flag on each operation.
        - If true/false, force it for operations.

    :return:
        a "reset" token (see :meth:`.ContextVar.set`)

   ."""
    return _reschedule_operations.set(bool(reschedule))


def is_reschedule_operations():
    """see :meth:`set_reschedule_operations`"""
    return _reschedule_operations.get()


def is_solid_true(*tristates, default=False):
    return first(tristates, default=default, key=lambda i: i is not None)


def is_op_rescheduled(op) -> bool:
    return is_solid_true(is_reschedule_operations(), op.rescheduled)


def is_op_endured(op) -> bool:
    return is_solid_true(is_endure_operations(), op.endured)


def is_op_parallel(op) -> bool:
    return is_solid_true(is_parallel_tasks(), op.parallel)


def is_op_marshalled(op) -> bool:
    return is_solid_true(is_marshal_tasks(), op.marshalled)
