.. _arch:

============
Architecture
============


.. default-role:: term
.. glossary::

    compute
    computation
    phase
        |v410-flowchart|
        The definition & execution of networked operation is split in 1+2 phases:

        - `composition`
        - `compilation`
        - `execution`

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
    composition
        The :term:`phase` where `operation`\s are constructed and grouped into `netop`\s and
        corresponding `network`\s.

        .. Tip::
            - Use :class:`.operation` builder class to construct
              :class:`.FunctionalOperation` instances.
            - Use :func:`~.graphtik.compose()` factory to prepare the `net` internally,
              and build :class:`.NetworkOperation` instances.

    compile
    compilation
        The :term:`phase` where the :class:`.Network` creates a new `execution plan`
        by `pruning` all `graph` nodes into a subgraph `dag`, and  deriving
        the `execution steps`.

    execute
    execution
    sequential
        The :term:`phase` where the :class:`.ExecutionPlan` calls the underlying functions
        of all `operation`\s contained in `execution steps`, with `inputs`/`outputs`
        taken from the `solution`.

        Currently there are 2 ways to execute:

        - *sequential*
        - *parallel*, with a :class:`multiprocessing.pool.ProcessPool`

        Plans may abort their execution by setting the `abort run` global flag.

    parallel
    parallel execution
    execution pool
    task
        `execute` `operation`\s *in parallel*, with a `thread pool` or `process pool`
        (instead of `sequential`).
        Operations and `netop` are marked as such on construction, or enabled globally
        from `configurations`.

        Note a `sideffects` are not expected to function with *process pools*,
        certainly not when `marshalling` is enabled.

    process pool
        When the :class:`multiprocessing.pool.Pool` class is used for `parallel` execution,
        the `task`\s  must be communicated to/from the worker process, which requires
        `pickling <https://docs.python.org/library/pickle.html>`_, and that may fail.
        With pickling failures you may try `marshalling` with *dill* library,
        and see if that helps.

        Note that `sideffects` are not expected to function at all.
        certainly not when `marshalling` is enabled.

    thread pool
        When the :func:`multiprocessing.dummy.Pool` class is used for `parallel` execution,
        the `task`\s are run *in process*, so no `marshalling` is needed.

    marshalling
        Pickling `parallel` `operation`\s and their `inputs`/`outputs` using
        the :mod:`dill` module. It is `configured <configurations>` either globally
        with :func:`.set_marshal_tasks()` or set with a flag on each
        operation / `netop`.

        Note that `sideffects` do not work when this is enabled.

    configurations
    graphtik configuration
        The functions controlling `compile` & `execution` globally  are defined
        in :mod:`.config` module and +1 in :mod:`graphtik.plot` module;
        the underlying global data are stored in :class:`contextvars.ContextVar` instances,
        to allow for nested control.

        All *boolean* configuration flags are **tri-state** (``None, False, True``),
        allowing to "force" all operations, when they are not set to the ``None``
        value.  All of them default to ``None`` (false).

    graph
    network graph
        The :attr:`.Network.graph` (currently a DAG) contains all :class:`.FunctionalOperation`
        and :class:`._DataNode` nodes of some `netop`.

        They are layed out and connected by repeated calls of
        :meth:`.Network._append_operation()` by Network constructor.

        This graph is then `prune`\d to extract the `dag`, and the `execution steps`
        are calculated, all ingredients for a new :class:`.ExecutionPlan`.

    dag
    execution dag
    solution dag
        There are 2 *directed-acyclic-graphs* instances used:

        - the :attr:`.ExecutionPlan.dag`,  in the `execution plan`, which contains
          the `prune`\d  nodes, used to decide the `execution steps`;
        - the :attr:`.Solution.dag` in the `solution`, which derives the
          `canceled operation`\s due to `reschedule`\d/failed operations upstream.

    steps
    execution steps
        The :attr:`.ExecutionPlan.steps` contains a list of the operation-nodes only
        from the `dag`, topologically sorted, and interspersed with
        *instruction steps* needed to `compute` the asked `outputs` from the given `inputs`.

        It is built by :meth:`.Network._build_execution_steps()` based on
        the subgraph `dag`.

        The only *instruction* step is for performing `evictions`.

    evictions
        A memory footprint optimization where intermediate `inputs` & `outputs`
        are erased from `solution` as soon as they are not needed further down the `dag`.

        *Evictions* are pre-calculated during `compilation`, where :class:`._EvictInstruction`
        `steps` are inserted in the `execution plan`.

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
            operations into some *netop* - furthermore, it wouldn't be very useful
            to get back the given inputs in case of `overwrites`.

    overwrites
        Values in the `solution` that have been written by more than one `operation`\s,
        accessed by :attr:`.Solution.overwrites`.
        Note that `sideffected` dependencies are, by definition, *overwrites*.

    net
    network
        the :class:`.Network` contains a `graph` of `operation`\s and can
        `compile` an `execution plan` or `prune` a cloned *network* for
        given `inputs`/`outputs`/`node predicate`.

    plan
    execution plan
        Class :class:`.ExecutionPlan` perform the `execution` phase which contains
        the `dag` and the `steps`.

        `compile`\ed *execution plans* are cached in :attr:`.Network._cached_plans`
        across runs with (`inputs`, `outputs`, `predicate`) as key.

    inputs
        The named input values that are fed into an `operation` (or `netop`)
        through :meth:`.Operation.compute()` method according to its `needs`.

        These values are either:

        - given by the user to the outer `netop`, at the start of a `computation`, or
        - derived from `solution` using *needs* as keys, during intermediate `execution`.

    outputs
        The dictionary of computed values returned by an `operation` (or a `netop`)
        matching its `provides`, when method :meth:`.Operation.compute()` is called.

        Those values are either:

        - retained in the `solution`, internally during `execution`, keyed by
          the respective *provide*, or
        - returned to user after the outer *netop* has finished `computation`.

        When no specific outputs requested from a *netop*, :meth:`.NetworkOperation.compute()`
        returns all intermediate `inputs` along with the *outputs*, that is,
        no `evictions` happens.

        An *operation* may return `partial outputs`.

    returns dictionary
        When an operation is marked with this flag, the underlying function is not
        expected to return a sequence but a dictionary; hence, no "zipping"
        of outputs/provides takes place.

    operation
    dependency
        Either the abstract notion of an action with specified `needs` and `provides`,
        *dependencies*, or the concrete wrapper :class:`.FunctionalOperation` for
        (any :func:`callable`), that feeds on `inputs` and update `outputs`,
        from/to `solution`, or given-by/returned-to the user by a `netop`.

        The distinction between *needs*/*provides* and *inputs*/*outputs* is akin to
        function *parameters* and *arguments* during define-time and run-time,
        respectively.

    netop
    network operation
        The :class:`.NetworkOperation` class holding a `network` of `operation`\s.

    needs
        A list of (positionally ordered) `dependency` names an `operation`'s
        underlying callable requires as input arguments.  The respective `input <inputs>`
        values will be extract from `solution` (or directly given by the user) when
        :meth:`.Operation.compute()` is called during `execution`.

        `modifiers` may annotate certain names as `optionals`, `sideffects`,
        or map them to differently named function arguments.

        The `graph` is laid out by matching the *needs* & `provides` of all *operations*.

    provides
        A list of `dependency` names to be zipped with the `output <outputs>` values
        produced when the `operation`'s underlying callable executes.
        The resulting dictionary will be stored into the `solution` or returned
        to the user after :meth:`.Operation.compute()` is called during `execution`.

        `modifiers` may annotate certain names as `sideffects`.

        The `graph` is laid out by matching the `needs` & *provides* of all *operations*.

    modifiers
        Annotations on specific arguments of `needs` and/or `provides` such as
        `optionals` & `sideffects` (see :mod:`graphtik.modifiers` module).

    optionals
        `needs` corresponding either:

        - to function arguments-with-defaults (annotated with :class:`.optional`), or
        - to ``*args`` (annotated with :class:`.vararg` & :class:`.varargs`),

        that do not hinder execution of the `operation` if absent from `inputs`.

    sideffects
        Fictive `needs` or `provides` not consumed/produced by the underlying function
        of an `operation`.
        A *sideffect* participates in the `compilation` of the graph, and is updated
        into the `solution`, but is never given/asked to/from functions.

        - An *abstract* *sideffect*, unrelated to any other `dependency` is annotated
          with :class:`.sideffect` modifier. `sideffected` dependency.
        - An *sideffect* applied on an existing `dependency` is annotated on that
          with :class:`.sideffected` modifier.

    sideffected
        An existing `dependency` that is linked to some `sideffects`.

        In all cases, the *dependency* must be declared in the `needs`.
        If it's the operation that applies the side-effect modification, then
        it must be declared also in its `provides`.

        All *sideffected* `outputs` are, by definition, `overwrites`.

    prune
    pruning
        A subphase of `compilation` performed by method :meth:`.Network._prune_graph()`,
        which extracts a subgraph `dag` that does not contain any `unsatisfied operation`\s.

        It topologically sorts the `graph`, and *prunes* based on given `inputs`,
        asked `outputs`, `node predicate` and `operation` `needs` & `provides`.

    unsatisfied operation
        The core of `pruning` & `rescheduling`, performed by
        :func:`.network._unsatisfied_operations()` function, which collects
        all `operation`\s that fall into any of these 2 cases:

        - they have `needs` that do not correspond to any of the given `inputs` or
          the intermediately `compute`\d `outputs` of the `solution`;
        - all their `provides` are NOT needed by any other operation, nor are asked
          as *outputs*.

    reschedule
    rescheduling
    partial outputs
    partial operation
    canceled operation
        The partial `pruning` of the `solution`'s dag during `execution`.
        It happens when any of these 2 conditions apply:

        - an `operation` is marked with the :attr:`.FunctionalOperation.rescheduled`
          attribute, which means that its underlying *callable* may produce
          only a subset of its `provides` (*partial outputs*);
        - `endurance` is enabled, either globally (in the `configurations`), or
          for a specific *operation*.

        the *solution* must then *reschedule* the remaining operations downstream,
        and possibly *cancel* some of those ( assigned in :attr:`.Solution.canceled`).

    endurance
    endured
        Keep executing as many `operation`\s as possible, even if some of them fail.
        Endurance for an operation  is enabled if :func:`.set_endure_operations()`
        is true globally in the `configurations` or if :attr:`.FunctionalOperation.endured`
        is true.

        You may interrogate :attr:`.Solution.executed` to discover the status
        of each executed operations or call one of :meth:`.check_if_incomplete()`
        or :meth:`.scream_if_incomplete()`.

    predicate
    node predicate
        A callable(op, node-data) that should return true for nodes to be
        included in `graph` during `compilation`.

    abort run
        A global `configurations` flag that when set with :func:`.abort_run()` function,
        it halts the execution of all currently or future `plan`\s.

        It is reset automatically on every call of :meth:`.NetworkOperation.compute()`
        (after a successful intermediate :term:`compilation`), or manually,
        by calling :func:`.reset_abort()`.

    plottable
        Objects that can plot their graph network, such as those inheriting :class:`.Plottable`,
        (:class:`.NetworkOperation`, :class:`.Network`, :class:`.ExecutionPlan`, :class:`.Solution`)
        or a |pydot.Dot|_ instance (the result of the :meth:`.Plottable.plot()` method).

        Such objects may render as SVG in *Jupter notebooks* (through their ``plot()`` method)
        and can render in a Sphinx site with with the :rst:dir:`graphtik` *RsT directive*.
        You may control the rendered image as explained in the *tip*  of
        the :ref:`plotting` section.

        SVGs are in rendered with the `zoom-and-pan javascript library
        <https://github.com/ariutta/svg-pan-zoom>`_

        .. Attention::
            Zoom-and-pan does not work in Sphinx sites for Chrome locally - serve
            the HTML files through some HTTP server, e.g. launch this command
            to view the site of this project::

                python -m http.server 8080 --directory build/sphinx/html/

    plotter
        A :class:`.Plotter` is responsible for rendering `plottable`\s as images.
        It is the `active plotter` that does that, unless overridden in a
        :meth:`.Plottable.plot()` call.
        Plotters can be customized by :ref:`various means <plot-customizations>`,
        such `plot theme`.

    active plotter
    default active plotter
        The `plotter` currently installed "in-context" of the respective `graphtik
        configuration` - this term implies also any :ref:`plot-customizations`
        done on the active plotter (such as `plot theme`).

        Installation happens by calling one of :func:`.active_plotter_plugged()` or
        :func:`.set_active_plotter` functions.

        The **default** *active plotter* is the plotter instance that this project
        comes pre-configured with, ie, when no *plot-customizations* have yet happened.

    plot theme
        The attributes of :class:`.plot.Theme` class.
        The actual theme in-use is the :attr:`.Plotter.theme` attribute of
        the `active plotter`.


.. default-role:: obj
.. |v410-flowchart| raw:: html
    :file: images/GraphtikFlowchart-v4.1.0.svg
