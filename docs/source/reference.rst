=============
API Reference
=============

.. autosummary::

     graphtik
     graphtik.op
     graphtik.netop
     graphtik.network
     graphtik.plot
     graphtik.base

Module: `op`
====================

.. automodule:: graphtik.op
     :members: Operation, reparse_operation_data
     :undoc-members:

Module: `netop`
====================

.. automodule:: graphtik.netop
     :members:
     :undoc-members:

Module: `network`
=================

.. automodule:: graphtik.network
     :members: AbortedException, _unsatisfied_operations, _execution_configs,
               abort_run, is_abort, is_endure_execution, is_skip_evictions,
               set_endure_execution, set_execution_pool, set_skip_evictions,
               _do_task
     :private-members:
.. autoclass:: graphtik.network.Network
     :members:
     :private-members:
     :special-members:
     :undoc-members:
.. autoclass:: graphtik.network.ExecutionPlan
     :members:
     :private-members:
     :special-members:
     :undoc-members:
.. autoclass:: graphtik.network._OpTask
     :members:
     :private-members:
     :special-members:
     :undoc-members:
.. autoclass:: graphtik.network.Solution
     :members:
     :special-members:

Module: `plot`
===============

.. automodule:: graphtik.plot
     :members:
     :undoc-members:

Module: `base`
==============

.. automodule:: graphtik.base
     :members:
     :undoc-members:
