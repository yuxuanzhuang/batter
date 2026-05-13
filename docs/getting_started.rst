================
Getting Started
================

Welcome to **BATTER**. This short guide provides the key resources new users and
developers should consult in order to become productive quickly.

.. important::

   Before submitting jobs through Slurm, make sure the seeded files under
   ``~/.batter/`` are edited for your cluster so Amber/AmberTools loads
   successfully. In practice this usually means updating
   ``job_manager.header`` and ``SLURMM-Am.header`` (and
   ``SLURMM-BATCH-remd.header`` if you plan to use REMD) so the right modules,
   environment activation, and executable paths are set for your site.

   If the folder has not been created yet, run ``batter seed-headers`` first.
   If you use a custom header directory, the same requirement applies to the
   location pointed to by ``run.slurm_header_dir``. See
   :doc:`cookbook/slurm_headers` for the full layout and examples.

   If you need to compile or patch AMBER GPU builds for BATTER runs, review
   :doc:`cookbook/amber_compilation` before launching production windows.

Follow the Tutorial
===================

The best way to see BATTER in action is to follow :doc:`tutorial/index`. It walks through setting up
an environment, preparing input files, and running full ABFE and RBFE examples. Completing
the tutorial ensures all required dependencies are installed and introduces the core
workflow.

Configuration Reference
=======================

For a complete description of the YAML schemas, default values, and validation rules,
refer to :doc:`cookbook/configuration`. It details every field in
:class:`~batter.config.run.RunConfig` and :class:`~batter.config.simulation.SimulationConfig`,
including advanced options such as restraint overrides, REMD setup, or SLURM-specific
controls.

Understanding the outputs
=========================

``batter`` generates several output files and directories during execution.
Use :doc:`cookbook/results_folder` to inspect the output layout, and consult
:doc:`developer_guide` when debugging orchestration or analysis internals.


Developing Pipelines
====================

If you plan to modify or extend BATTER's internals—for example, adding new pipeline
steps or integrating a custom backend—read the :doc:`developer_guide`. These
documents explain the architecture, typed payloads, metadata model, and orchestration
flow used by the codebase.

Additional Resources
====================

* ``examples/`` – Ready-to-run YAML configurations demonstrating ABFE (membrane
  and soluble), RBFE, ASFE, and plain MD scenarios.
* ``tests/`` – A suite of unit tests that exercise pipeline handlers, state registry,
  and metadata helpers; useful for understanding expected behaviour.
* ``README.rst`` – Project overview, installation, and quickstart commands.
* ``TODO`` – Current developer to-do list and open enhancements.
* GitHub Issues / Discussions – Report bugs or request features; link available from
  the repository homepage.
