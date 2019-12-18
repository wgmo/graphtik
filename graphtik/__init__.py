# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Lightweight :term:`computation` graphs for Python."""

__author__ = "hnguyen, ankostis"
__version__ = "4.3.0"
__release_date__ = "16 Dec 2019, 16:16"
__license__ = "Apache-2.0"
__title__ = "graphtik"
__summary__ = __doc__.splitlines()[0]
__uri__ = "https://github.com/pygraphkit/graphtik"

NO_RESULT = object()

from .modifiers import *  # noqa, on purpose to include any new modifiers
from .network import (
    AbortedException,
    abort_run,
    is_abort,
    is_endure_execution,
    is_skip_evictions,
    set_endure_execution,
    set_skip_evictions,
)
from .netop import compose
from .op import operation
