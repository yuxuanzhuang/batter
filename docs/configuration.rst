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
    ``lambdas`` or SDR restraints are no longer required.  Equilibration controls
    are expressed via ``eq_steps`` (steps per segment) and ``num_equil_extends``
    (how many additional segments to run). The total number of equilibration MD steps
    therefore scales as ``num_equil_extends * eq_steps``. For FE production the
    ``num_fe_extends`` field multiplies the stage-2 component steps (e.g.,
    ``z_steps2``) so each window ultimately samples
    ``num_fe_extends * <component>_steps2`` steps before moving on.
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

Component-Specific Inputs
-------------------------

Although the ``create`` block is shared by ABFE, MASFE, and MD pipelines, some fields
are consumed only by particular builders. The table below highlights the ones that
feed into the low-level ops documented in :doc:`developer_guide/internal_builders`:

.. list-table::
   :header-rows: 1

   * - Field
     - Used by
     - Purpose
   * - ``buffer_x/y/z``
     - ``create_box`` (protein systems)
     - Controls rectangular solvent box sizing.
   * - ``solv_shell``
     - ``create_box`` (ligand-only runs)
     - Sets cubic padding for standalone ligands.
   * - ``water_model``
     - ``create_box`` helpers
     - Selects the ``leaprc.water.*`` template.
   * - ``cation`` / ``anion``
     - ``create_box`` helpers
     - Define ion names that ``addionsrand`` inserts.
   * - ``ion_conc``
     - :attr:`SimulationConfig.ion_def` → ``create_box``
     - Drives salt concentration when ``neutralize_only = "no"``.
   * - ``neutralize_only``
     - :attr:`SimulationConfig.neut` → ``create_box``
     - Toggles between neutralisation-only or salt+neutralisation workflows.
   * - ``extra_restraints``
     - Restraint ops
     - Adds positional restraints for ABFE builders.
   * - ``extra_conformation_restraints``
     - Restraint ops
     - JSON specification for conformational restraints.
   * - ``lipid_mol``
     - Build/ops helpers
     - Identifies membrane residues when trimming waters.

Linking configuration fields to their downstream consumers makes it easier to reason
about which parts of the file structure (build directories, solvation scripts,
restraint writers) are affected when you toggle individual knobs.

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
  The ``buffer_z`` value also determines the SDR translation distance: ligands are
  shifted so they sit near the midpoint of the solvent slab, with an extra 5 Å of
  clearance (see :func:`batter.systemprep.helpers.get_sdr_dist`).  For membrane systems
  the builder enforces a minimum effective ``buffer_z`` of ~25 Å to keep the ligand in
  bulk solvent above the membrane even if the YAML specifies a smaller buffer.
