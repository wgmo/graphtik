# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Lightweight :term:`computation` graphs for Python."""

__author__ = "hnguyen, ankostis"
__version__ = "4.0.1"
__release_date__ = "12 Dec 2019, 00:51"
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
