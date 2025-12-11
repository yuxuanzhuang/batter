Configuration Overview
======================

BATTER's configuration layer is driven by :class:`batter.config.run.RunConfig`,
the user-facing schema that describes the system to build, the FE protocol to
execute, runtime options, and backend preferences. Derived simulation knobs are
produced by :meth:`RunConfig.resolved_sim_config`; the resulting
:class:`~batter.config.simulation.SimulationConfig` is documented in the
developer guide (:doc:`../developer_guide`).

Run Configuration Schema
------------------------

The run YAML file is divided into three sections grouped inside
``RunConfig``:

``run``
    Execution controls that include runtime behaviour, SLURM settings,
    notification preferences, and artifact destination. ``run.output_folder`` is
    required and becomes the base path for ``<run.output_folder>/executions/<run_id>/``.
    ``run.system_type`` optionally overrides the builder selection inferred from the
    protocol (``MABFE`` for ABFE/MD, ``MASFE`` for ASFE). This section is validated
    by :class:`batter.config.run.RunSection`.
``create``
    Inputs required for system staging (protein/topology paths, ligands, force fields,
    optional restraints). The structure maps directly to
    :class:`batter.config.run.CreateArgs`.
``fe_sim``
    Overrides and controls for free-energy simulation stages. For ABFE/ASFE runs
    these map to :class:`batter.config.run.FESimArgs`. MD-only runs automatically
    coerce this section into :class:`batter.config.run.MDSimArgs`, so fields like
    ``lambdas`` or SDR restraints are no longer required. Equilibration controls
    are expressed via ``eq_steps`` which now represents the **total** equilibration
    steps. The legacy ``num_equil_extends`` knob is ignored (retained only for
    backward compatibility). FE production no longer chunks into extends; set
    ``steps2`` to the total per-window production steps. The legacy
    ``num_fe_extends`` knob is ignored.

See Quick Reference below for links to individual config classes.

Per-component steps and lambdas
-------------------------------

Stage-1/Stage-2 component steps are supplied via ``fe_sim.steps1`` and
``fe_sim.steps2`` as dicts keyed by the single-letter component (e.g. ``z: 50000``).
Keys like ``z_steps1``/``y_steps2`` are also accepted and folded into these
maps automatically. Each protocol enforces the required components: ABFE fills
``z`` defaults if omitted, and ASFE fills ``y``/``m`` defaults.

Lambda schedules can be customized per component using ``fe_sim.component_lambdas``
(or ``<comp>_lambdas`` keys). When a component is missing from that map, it
inherits the top-level ``fe_sim.lambdas`` list. Values can be written as YAML lists
or comma/space separated strings; validation ensures ascending order.

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

The ``buffer_z`` value also determines the SDR translation distance: ligands are
shifted so they sit near the midpoint of the solvent slab, with an extra 5 Å of
clearance (see :func:`batter.systemprep.helpers.get_sdr_dist`).  For membrane systems
the builder enforces a minimum effective ``buffer_z`` of ~25 Å to keep the ligand in
bulk solvent above the membrane even if the YAML specifies a smaller buffer.

Equilibration options
---------------------

Two frequently toggled equilibration knobs live under ``fe_sim`` and flow into the
resolved :class:`~batter.config.simulation.SimulationConfig`:

* ``hmr`` – ``"yes"`` enables hydrogen mass repartitioning. The builder swaps in HMR
  parameter files and switches equilibration/production mdins to the HMR topology
  (``full.hmr.prmtop``).
* ``enable_mcwat`` – ``"yes"`` (default) enables Monte Carlo water moves during
  equilibration. The flag populates the ``mcwat`` setting in AMBER input decks via
  :func:`batter._internal.ops.amber.write_amber_templates`.

REMD runs
---------

REMD inputs (mdins/groupfiles) are always written during preparation so you can decide at
submit time whether to run them. Use ``fe_sim.remd`` to tune segment length and exchange
frequency (``nstlim`` / ``numexchg``). Control execution with ``run.remd`` (``yes`` or
``no``); when ``run.remd: no`` the files are still generated but no REMD jobs are
scheduled. REMD jobs submit one Slurm job per component via ``SLURMM-BATCH-remd`` and
monitor ``FINISHED``/``FAILED`` sentinels in the component folder. See
:doc:`remd_submission` for operational details.

SLURM header templates
----------------------

BATTER renders SLURM scripts by combining a user-editable header with a packaged body.
Headers are copied into ``~/.batter`` on first use.
You can also seed them explicitly:

.. code-block:: bash

   batter seed-headers           # seeds into ~/.batter
   batter seed-headers --dest /path/to/dir
   batter seed-headers --force   # overwrite existing headers

To check how your headers differ from the packaged defaults:

.. code-block:: bash

   batter diff-headers           # compares ~/.batter headers to defaults
   batter diff-headers --dest /path/to/dir

Edit the headers to match your cluster defaults (queue/partition, env exports,
executable paths). Bodies remain managed by the package. Header files:

* ``SLURMM-Am.header`` (equil/FE runs)
* ``SLURMM-BATCH-remd.header`` (REMD runs)
* ``job_manager.header`` (manager script for ``batter --slurm-submit``)

The header lookup/seed location is controlled by ``run.slurm_header_dir``; when omitted it
defaults to ``~/.batter``.

Per-run SLURM overrides
-----------------------

Simulation submit scripts inherit the header settings above, but you can also control SLURM
resources per run via the ``run.slurm`` block (partition, time, nodes, ntasks_per_node, mem, etc.).
Those values are substituted into SLURM scripts when rendered. Combine the two mechanisms by
setting cluster defaults in the headers and per-run overrides in the YAML when needed.

Batch mode (single allocation)
------------------------------

If you prefer to request a multi-GPU allocation once and submit per-window jobs from a
manager process, set ``run.batch_mode: true``. The manager will render ``SLURMM-BATCH``
scripts into ``executions/<run_id>/batch_run`` and submit them with ``sbatch``; each script
``cd``s into the component/window folder and runs ``run-local.bash`` (or ``run-local-remd.bash``).
Equilibration and FE-equil run as normal per-ligand submits; FE production is bundled into a
single batch submission per ligand when REMD is disabled.
Set ``run.batch_gpus`` to request GPUs on the sbatch line (via ``--gres gpu:<batch_gpus>``)
for the per-ligand FE batch submission; ``run.batch_gpus_per_task`` controls the per-task
allocation used inside the batch helper.

The batch wrapper header is seeded to ``~/.batter/SLURMM-BATCH.header`` (similar to other
headers); edit it to match your cluster defaults (GPUs, partition, modules).

Remember to request GPUs in your job manager header (``job_manager.header``) so the manager
allocation has the resources it needs.

Executable resolution
---------------------

BATTER launches external tools by name (e.g., ``pmemd.cuda``, ``pmemd.cuda.MPI``,
``pmemd``, ``sander``, ``tleap``, ``antechamber``, ``cpptraj``, ``parmchk2``,
``obabel``, ``vmd``). Ensure they are on ``PATH`` or exported in your SLURM headers
if cluster modules are required. The package ships ``USalign`` internally and calls
it via the baked-in path. For the Python-side tooling you can override executables
via environment variables so overrides propagate into subprocesses:

* ``BATTER_ANTECHAMBER`` (default: ``antechamber``)
* ``BATTER_TLEAP`` (default: ``tleap``)
* ``BATTER_CPPTRAJ`` (default: ``cpptraj``)
* ``BATTER_PARMCHK2`` (default: ``parmchk2``)
* ``BATTER_CHARMM_LIPID2AMBER`` (default: ``charmmlipid2amber.py``)
* ``BATTER_USALIGN`` (default: packaged ``USalign``)
* ``BATTER_OBABEL`` (default: ``obabel``)
* ``BATTER_VMD`` (default: ``vmd``)


Quick Reference
---------------

.. autosummary::
   :toctree: autosummary/config
   :nosignatures:

   batter.config.run.RunConfig
   batter.config.run.CreateArgs
   batter.config.run.FESimArgs
   batter.config.run.MDSimArgs
   batter.config.run.RunSection
   batter.config.load_run_config
   batter.config.dump_run_config
