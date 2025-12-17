==============================
Implementing Internal Builders
==============================

Per-step work inside BATTER (``prepare_equil``, ``prepare_fe``, etc.) is handled by
classes under ``batter/_internal``. This section outlines the directory conventions,
registries, and shared ops so you can extend builders without reverse-engineering the
codebase.

Directory Conventions
---------------------

Every builder receives a :class:`~batter._internal.builders.interfaces.BuildContext`
that points at a per-ligand working directory::

   simulations/<LIGAND>/
   ├── q_build_files/         # Shared build artifacts (build.pdb, anchors, dum.*)
   ├── q_amber_files/         # Amber templates for equilibration
   ├── q_run_files/           # Job scripts, logs
   ├── e_build_files/         # Per-component equivalents (e, v, o, z, y, ...)
   └── ...

The ``ctx.build_dir`` / ``ctx.amber_dir`` helpers abstract these paths so ops like
``create_box`` can stage files without duplicating directory logic. Always write
intermediate artifacts into the component’s build directory and keep final AMBER
inputs under ``ctx.window_dir``.

Registry Hooks
--------------

Registries live in :mod:`batter._internal.builders.fe_registry` and map component codes
to functions:

* ``@register_build_complex('e')`` – adds a factory for the component’s
  ``_build_complex`` hook.
* ``@register_create_box('z')`` – selects the correct ``create_box`` helper.
* ``@register_restraints(...)``, ``@register_sim_files(...)``, etc. – route the remaining
  hooks.

When introducing a new component or overriding an existing hook, register it here so
the orchestrator picks it up automatically.

Reusable Ops
------------

Common tasks are centralised under ``batter/_internal/ops``:

* ``box.py`` – ``create_box`` helpers (AMBER tleap scripts, solvation, ion placement).
* ``restraints.py`` – Writers for disang/cv inputs.
* ``sim_files.py`` – Template renderers for MD input decks (equilibration emits the
  standard ``eqnpt.in``/``eqnpt0.in`` plus the appear/disappear variants used by the
  run scripts).
* ``simprep.py`` – Build directory initialisation and window copying.

Prefer importing and extending these modules instead of duplicating tleap/parmed code.
Many helpers already expect ``BuildContext`` objects and honour the directory layout
described above.

Sim file templates and ``total_steps``
--------------------------------------

Equilibration and production templates written by ``sim_files.py`` begin with a
``! total_steps=<int>`` comment (``#`` also works). ``write_sim_files`` uses
``eq_steps`` for equilibration; per-component ``n_steps`` are used for production
mdin templates. The runtime scripts (``run-local.bash``, ``run-local-vacuum.bash``,
``run-equil.bash``) call ``parse_total_steps`` in ``check_run.bash`` to read that
marker and ``parse_nstlim`` to pick the first ``nstlim`` as the chunk length. They
roll ``md-current.rst7``/``md-previous.rst7`` between segments until
``total_steps`` is reached, so avoid deleting or renaming the comment when hand
editing templates.

Config field consumers
----------------------

Key :class:`~batter.config.simulation.SimulationConfig` fields and where they land:

* ``eq_steps`` → ``write_sim_files`` (equil mdin-template ``total_steps`` marker).
* ``<comp>_n_steps`` → component writers in ``sim_files.py`` (per-window mdin-template markers).
* ``analysis_start_step`` → stored in FE records and passed to analysis writers to skip early frames.
* ``ntpr``/``ntwr``/``ntwx`` → substituted into mdin templates in ``sim_files.py`` (restart/trajectory cadence).
* ``hmr``/``enable_mcwat`` → toggles template selection and template flags in ``write_amber_templates``.

Builder Lifecycle
-----------------

Each :class:`batter._internal.builders.base.BaseBuilder` subclass implements the hook
methods executed by :meth:`BaseBuilder.build`:

1. ``_build_complex`` – Align systems, detect anchors, and populate ``build_dir``.
2. ``_create_box`` – Write solvated/vacuum topologies using the registered ``create_box`` op.
3. ``_restraints`` – Generate component-specific restraints.
4. ``_pre_sim_files`` (optional) – Any additional preprocessing before rendering inputs.
5. ``_sim_files`` – Produce AMBER mdin/mini/eq decks.
6. ``_run_files`` – Emit SLURM/local job scripts.

Builder Testing
---------------

Builders operate strictly inside the per-ligand working directory, which keeps them
easy to reason about. To test a new builder in isolation:

1. Construct a :class:`BuildContext` with temporary directories for ``working_dir``,
   ``system_root``, and ``param_dir_dict``.
2. Call the individual hook (or ``build()``) and assert that the expected files appear
   under ``ctx.working_dir``.
3. Re-run ``create_box`` and ``sim_files`` to ensure idempotency—many ops reuse cached
   artifacts when files already exist.
