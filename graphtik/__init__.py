# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Lightweight computation graphs for Python."""

__author__ = "hnguyen"
__version__ = "3.0.0"
__release_date__ = "2 Dec 2019, 17:25"
__license__ = "Apache-2.0"
__title__ = "graphtik"
__summary__ = __doc__.splitlines()[0]
__uri__ = "https://github.com/pygraphkit/graphtik"

from .modifiers import *  # noqa, on purpose to include any new modifiers
from .network import (
    AbortedException,
    abort_run,
    is_abort,
    is_skip_evictions,
    set_evictions_skipped,
)
from .netop import compose
from .op import operation
