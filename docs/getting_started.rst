================
Getting Started
================

Welcome to **BATTER**. This short guide provides the key resources new users and
developers should consult in order to become productive quickly.

Follow the Tutorial
===================

The best way to see BATTER in action is to follow :doc:`tutorial`. It walks through setting up
an environment, preparing input files, and running a full ABFE example. Completing
the tutorial ensures all required dependencies are installed and introduces the core
workflow.

Configuration Reference
=======================

For a complete description of the YAML schemas, default values, and validation rules,
refer to :doc:`configuration`. It details every field in
:class:`~batter.config.run.RunConfig` and :class:`~batter.config.simulation.SimulationConfig`,
including advanced options such as restraint overrides, REMD setup, or SLURM-specific
controls.

Developing Pipelines
====================

If you plan to modify or extend BATTER's internals—for example, adding new pipeline
steps or integrating a custom backend—read the :doc:`developer_guide` (and the focused
:doc:`developer_guide/pipeline_payloads_and_metadata` chapter). These documents explain
the architecture, typed payloads, metadata model, and orchestration flow used by the
codebase.

Public API
==========

The main user-facing entry points live in :mod:`batter.api`. Highlights include:

* :func:`batter.api.run_from_yaml` – Execute a full workflow from a top-level YAML.
* :func:`batter.api.load_sim_config` / :func:`batter.api.save_sim_config` – Read and
  write simulation configurations.
* :class:`batter.api.ArtifactStore`, :class:`batter.api.FEResultsRepository` – Inspect
  or reuse portable results.

Each function includes docstrings and type hints; consult the module for up-to-date
signatures.

Additional Resources
====================

* ``examples/`` – Ready-to-run YAML configurations demonstrating ABFE (membrane
  and soluble) and ASFE scenarios.
* ``tests/`` – A suite of unit tests that exercise pipeline handlers, state registry,
  and metadata helpers; useful for understanding expected behaviour.
* ``README.rst`` – Project overview, installation, and quickstart commands.
* ``TODO`` – Current developer to-do list and open enhancements.
* GitHub Issues / Discussions – Report bugs or request features; link available from
  the repository homepage.
