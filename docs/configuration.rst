Configuration Overview
======================

BATTER's configuration layer is split into two complementary models:

* :class:`batter.config.run.RunConfig` describes a *run* – the system to build,
  the FE protocol to execute, runtime options, and any backend preferences.
* :class:`batter.config.simulation.SimulationConfig` represents the fully
  resolved simulation knobs consumed by the orchestrator and execution engines.

Run Configuration Schema
------------------------

The run YAML file is divided into four top-level sections:

``system``
    Paths and metadata that identify where artifacts are written. This section
    is validated by :class:`batter.config.run.SystemSection`.
``create``
    Inputs for system preparation (paths to protein/ligand files, force-field
    selections, optional restraint files). The structure maps directly to
    :class:`batter.config.run.CreateArgs`.
``fe_sim``
    Overrides and controls for free-energy simulation stages. For ABFE/ASFE runs
    these map to :class:`batter.config.run.FESimArgs`. MD-only runs automatically
    coerce this section into :class:`batter.config.run.MDSimArgs`, so fields like
    ``lambdas`` or SDR restraints are no longer required.
``run``
    Execution behaviour such as SLURM options, dry-run toggles, and failure
    policies. These fields populate :class:`batter.config.run.RunSection`.

The helper :func:`batter.config.load_run_config` loads a YAML file into a
validated :class:`~batter.config.run.RunConfig`, expanding environment variables
and ``~`` home shortcuts along the way.

Simulation Configuration Derivation
-----------------------------------

``RunConfig.resolved_sim_config()`` produces the
:class:`batter.config.simulation.SimulationConfig` that downstream components
expect. Internally this delegates to
:meth:`batter.config.simulation.SimulationConfig.from_sections`, merging the
``create`` and ``fe_sim`` sections while performing additional coercions such as
normalising yes/no flags and expanding lambda schedules.

When you need to persist or reload a resolved simulation configuration, use the
helper functions:

* :func:`batter.config.load_simulation_config`
* :func:`batter.config.dump_simulation_config`

They mirror the behaviour of :func:`load_run_config`, ensuring environment
variables and user-relative paths are expanded consistently.

Quick Reference
---------------

.. autosummary::
   :toctree: autosummary/config
   :nosignatures:

   batter.config.run.RunConfig
   batter.config.run.CreateArgs
   batter.config.run.FESimArgs
   batter.config.run.MDSimArgs
   batter.config.simulation.SimulationConfig
   batter.config.load_run_config
   batter.config.dump_run_config
   batter.config.load_simulation_config
   batter.config.dump_simulation_config
