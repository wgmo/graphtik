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
     :members: AbortedException, abort_run, is_abort, is_skip_evictions, set_skip_evictions,
               set_endure_execution, is_endure_execution, _execution_configs, _unsatisfied_operations
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
.. autoclass:: graphtik.network.Solution
     :members:
     :special-members:
.. autoclass:: graphtik.network._Rescheduler
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
