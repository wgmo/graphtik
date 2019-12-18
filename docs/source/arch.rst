.. _arch:

============
Architecture
============


.. default-role:: term
.. glossary::

    COMPUTE
    computation
        |v410-flowchart|
        The definition & execution of networked operation is splitted in 1+2 phases:

        - `COMPOSITION`
        - `COMPILATION`
        - `EXECUTION`

        ... it is constrained by these IO data-structures:

        - `operation`\(s) (with `needs` & `provides` for each one)

        - given `inputs`
        - asked `outputs`

        ... populates these low-level data-structures:

        - `network graph` (COMPOSE time)
        - `execution dag` (COMPILE time)
        - `execution steps` (COMPILE time)
        - `solution` (EXECUTE time)

        ... and utilizes these main classes:

        .. autosummary::

            graphtik.op.FunctionalOperation
            graphtik.netop.NetworkOperation
            graphtik.network.Network
            graphtik.network.ExecutionPlan
            graphtik.network.Solution


    compose
    COMPOSITION
        The *phase* where `operation`\s are constructed and grouped into `netop`\s and
        corresponding `network`\s.

        .. Tip::
            - Use :class:`~.graphtik.operation()` builder class to construct
              :class:`.FunctionalOperation` instances.
            - Use :func:`~.graphtik.compose()` factory to prepare the `net` internally,
              and build :class:`.NetworkOperation` instances.

    compile
    COMPILATION
        The *phase* where the :class:`.Network` creates a new `execution plan`
        by `pruning` all `graph` nodes into a subgraph `dag`, and  derriving
        the `execution steps`.

    execute
    EXECUTION
        The *phase* where the :class:`.ExecutionPlan` calls sequentially or parallel
        the underlying functions of all `operation`\s contained in `execution steps`,
        with `inputs`/`outputs` taken from the `solution`.

    configurations
        A global :data:`._execution_configs` affecting `execution`
        stored in a :class:`contextvars.ContextVar`.

        .. Tip::
            Instead of directly modifying ``_execution_configs``, prefer
            the special ``set_...()`` & ``is_...()`` methods exposed from ``graptik`` package.

    graph
    network graph
        The :attr:`.Network.graph` (currently a DAG) contains all :class:`FunctionalOperation`
        and :class:`_DataNode` nodes of some `netop`.

        They are layed out and connected by repeated calls of
        :meth:`.Network._append_operation()` by Network constructor.

        This graph is then `prune`\d to extract the `dag`, and the `execution steps`
        are calculated, all ingridents for a new :class:`ExecutionPlan`.

    dag
    execution dag
        There are 2 *directed-acyclic-graphs* instances used:

        - the :attr:`.ExecutionPlan.dag`,  in the `execution plan`, which contains
          the `prune`\d  nodes, used to decide the `execution steps`;
        - the :attr:`.Solution.dag` in the `solution`, which contains
          the `reschedule`\d nodes.

    steps
    execution steps
        The :attr:`ExecutionPlan.steps` contains a list of the operation-nodes only
        from the `dag`, topologically sorted, and interspersed with
        *instruction steps* needed to `compute` the asked `outputs` from the given `inputs`.

        It is built by :meth:`.Network._build_execution_steps()` based on
        the subgraph `dag`.

        The only *instruction* step is for performing `eviction`.

    evict
    eviction
        The :class:`_EvictInstruction` `steps` erase items from
        `solution` as soon as they are not needed further down the dag,
        to reduce memory footprint while computing.

    solution
        A :class:`.Solution` instance created internally by :meth:`.NetworkOperation.compute()`
        to hold the values both `inputs` & `outputs`, and the status of *executed* operations.
        It is based on a :class:`collections.ChainMap`, to keep one dictionary
        for each `operation` executed +1 for inputs.

        The results of the last operation executed "wins" in the final *outputs* produced,
        BUT while executing, the `needs` of each operation receive the *solution* values
        in **reversed order**, that is, the 1st operation result (or given input) wins
        for some *needs* name.

        Rational:

            During execution we want stability (the same input value used by all operations),
            and that is most important when consuming input values - otherwise,
            we would use (possibly *overwritten* and thus changing)) intermediate ones.

            But at the end we want to affect the calculation results by adding
            operations into some *netop* - furthermore, it wouldn't be very usefull
            to get back the given inputs in case of `overwrites`.

    overwrites
        Values in the `solution` that have been written by more than one `operation`\s,
        accessed by :attr:`Solution.overwrites`:

    net
    network
        the :class:`.Network` contains a `graph` of `operation`\s and can
        `compile` an `execution plan` or `prune` a cloned *network* for
        given `inputs`/`outputs`/`node predicate`.

    plan
    execution plan
        Class :class:`.ExecutionPlan` perform the `execution` phase which contains
        the `dag` and the `steps`.

        `Compile`\ed *execution plans* are cached in :attr:`.Network._cached_plans`
        across runs with (`inputs`, `outputs`, `predicate`) as key.

    inputs
        a dictionary of named input values given to :meth:`.NetworkOperation.compute()`

    outputs
        A dictionary of computed values returned by :meth:`.NetworkOperation.compute()`,
        or the actual (*partial* or complete) `provides` returned by
        some :class:`FunctionalOperation`.

        All computed values are retained in it when no specific outputs requested,
        to :meth:`.NetworkOperation.compute()`, that is, no data-eviction happens.

    operation
        Either the abstract notion of an action with specified `needs` and `provides`,
        or the concrete wraper :class:`.FunctionalOperation` for arbitrary functions
        (any :class:`callable`).

    netop
    network operation
        The :class:`.NetworkOperation` class holding a `network` of `operation`\s.

    needs
        A list of names of the compulsory/optional values or `sideffects` an operation's
        underlying callable requires to execute.

    provides
        A list of names of the values produced when the `operation`'s
        underlying callable executes.

    sideffects
        Fictive `needs` or `provides` not consumed/produced by the underlying function
        of an `operation`, annotated with :class:`.sideffect`.
        A *sideffect* participates in the solution of the graph but is never
        given/asked to/from functions.

    prune
    pruning
        A subphase of `compilation` performed by method :meth:`.Network._prune_graph()`,
        which extracts a subgraph `dag` that does not contain any `unsatisfied operation`\s.

        It topologically sorts the `graph`, and *prunes* based on given `inputs`,
        asked `outputs`, `node predicate` and `operation` `needs` & `provides`.

    unsatisfied operation
        The core of `pruning` & `rescheduling`, performed by method
        :func:`.network._unsatisfied_operations()`, which collects all `operation`\s
        that fall into any of these 2 cases:

        - they have `needs` that do not correspond to any of the given `inputs` or
          the intermediately `compute`\d `outputs` of the `solution`;
        - all threir `provides` are NOT needed by any other operation, nor are asked
          as *outputs*.

    reschedule
    rescheduling
    partial outputs
        The partial `pruning` of the `solution`'s dag during `execution`.
        It happens when any of these 2 conditions apply:

        - an `operation` is marked with the :attr:`FunctionalOperation.reschedule`
          attribute, which means that its underlying *callable* may produce
          only a subset of its `provides` (*partial outputs*);
        - `endurance` is enabled, either globally (in the `configurations`), or
          for a specific *operation*.

        the *solution* must then *reschedule* the remaining operations downstreams,
        and *cancel* some of those.

    endurance
        Keep executing as many `operation`\s as possible, even if some of them fail.
        If :func:`.set_endure_execution()` in the `configurations` is set to true,
        you may interogate :class:`.Solution` properties to discover whether
        an operation may be:

        - executed successfully,
        - *failed*, or
        - *canceled*, if another operation, the sole provider of a compulsory `needs`,
          has failed upstreams, and this operation was `reschedule`\d.

    predicate
    node predicate
        A callable(op, node-data) that should return true for nodes not to be
        :meth:`~.NetworkOperation.narrowed`.

.. default-role:: obj
.. |v410-flowchart| raw:: html
    :file: images/GraphtikFlowchart-v4.1.0.svg
