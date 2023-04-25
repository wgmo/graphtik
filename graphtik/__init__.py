# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
:term:`computation` graphs for Python & Pandas

.. import-speeds-start
.. tip::
     The module *import-time* dependencies have been carefully optimized so that
     importing all from package takes the minimum time (e.g. *<10ms* in a 2019 laptop)::

        >>> %time from graphtik import *     # doctest: +SKIP
        CPU times: user 8.32 ms, sys: 34 µs, total: 8.35 ms
        Wall time: 7.53 ms

     Still, constructing your :term:`pipeline`\\s on import time would take
     considerable more time (e.g. *~300ms* for the 1st pipeline).
     So prefer to construct them in "factory" module functions
     (remember to annotate them with :pep:`typing hints <484>` to denote their retun type).

.. import-speeds-stop

.. seealso::
    :func:`.plot.active_plotter_plugged()`, :func:`.plot.set_active_plotter()` &
    :func:`.plot.get_active_plotter()` configs, not imported, unless plot is needed.
"""

__version__ = "10.5.0"
__release_date__ = "24 Apr 2023, 18:01"
__title__ = "graphtik"
__summary__ = __doc__.splitlines()[0]
__license__ = "Apache-2.0"
__uri__ = "https://github.com/pygraphkit/graphtik"
__author__ = "hnguyen, ankostis"  # chronologically ordered


from .base import AbortedException, IncompleteExecutionError
from .fnop import NO_RESULT, NO_RESULT_BUT_SFX, operation
from .modifier import (
    hcat,
    implicit,
    keyword,
    modify,
    optional,
    sfx,
    sfxed,
    sfxed_vararg,
    sfxed_varargs,
    vararg,
    varargs,
    vcat,
)
from .pipeline import compose
