Configuration Overview
======================

BATTER's configuration layer is driven by :class:`batter.config.run.RunConfig`,
the user-facing schema that describes the system to build, the FE protocol to
execute, runtime options, and execution settings. Derived simulation knobs are
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
    protocol (``MABFE`` for ABFE/RBFE/MD-family runs, ``MASFE`` for ASFE). This
    section is validated by :class:`batter.config.run.RunSection`. Set
    ``run.clean_failures: true`` to remove ``FAILED`` sentinels,
    ``job_attempt.txt`` retry counters, and progress caches before rerunning an
    existing execution. Set ``run.store_debug_files: true`` to preserve
    intermediate scratch files that are otherwise pruned after successful stages.
``create``
    Inputs required for system staging (protein/topology paths, ligands, force fields,
    optional anchors/restraints). The structure maps directly to
    :class:`batter.config.run.CreateArgs`.
``fe_sim``
    Overrides and controls for simulation stages. FE protocols map this section to
    :class:`batter.config.run.FESimArgs`. MD-only runs automatically coerce it into
    :class:`batter.config.run.MDSimArgs`, so fields like ``lambdas`` or SDR
    restraints are no longer required. Equilibration controls are expressed via
    ``eq_steps`` which now represents the **total** equilibration steps. The value
    is written into ``mdin-template`` as ``! total_steps=<total>``,
    letting runtime scripts determine the target length without regenerating inputs.
    Legacy production extend knobs (``num_fe_extends``) are rejected; set
    ``n_steps`` to total steps instead. ``analysis_range`` is likewise
    disallowed—use ``analysis_start_step`` to skip early production frames. FE
    production no longer chunks into extends; set ``n_steps`` to the total per-window
    production steps. Those mdin templates also include ``! total_steps=<total>``;
    ``run-local*.bash`` reads that marker plus the first ``nstlim`` it finds to choose
    the segment length. Each invocation runs one segment, writes an explicit
    numbered restart such as ``md-01.rst7`` or ``md-02.rst7`` plus the matching
    ``md-*.out``, and returns; rerun the script to continue until
    ``total_steps`` is reached. Numbered ``md-*.rst7`` restart files are removed
    after the production window is successfully marked ``FINISHED``.

See Quick Reference below for links to individual config classes.

Per-component steps and lambdas
-------------------------------

Component steps are supplied via ``fe_sim.n_steps`` as dicts keyed by the
single-letter component (e.g. ``z: 100000``). Keys like ``y_n_steps`` are also
accepted and folded into this map automatically. Each FE protocol enforces the
components it needs: ABFE requires ``z``, standard and SEPTOP RBFE require ``x``,
and ASFE requires ``y``/``m``. Set the corresponding ``<comp>_n_steps`` or
``fe_sim.n_steps`` entry explicitly in production YAMLs.

Lambda schedules can be customized per component using ``fe_sim.component_lambdas``
(or ``<comp>_lambdas`` keys). When a component is missing from that map, it
inherits the top-level ``fe_sim.lambdas`` list. Values can be written as YAML lists
or comma/space separated strings; validation ensures ascending order.

RBFE mapping options
--------------------

For ``protocol: rbfe`` or ``protocol: rbfe_septop``, the ``rbfe`` block controls
network planning and atom mapping/scoring.

* ``rbfe.mapping`` – mapping strategy (for example ``default`` or ``konnektor``).
* ``rbfe.mapping_file`` – explicit pair list file; takes precedence over ``mapping``.
* ``rbfe.atom_mapping_file`` – optional JSON/YAML atom mapping overrides for
  selected pairs; uncovered pairs use ``rbfe.atom_mapper``.
* ``rbfe.network_scorer`` – edge scorer for Konnektor planning. ``auto`` uses the
  LoMap scorer for standard RBFE and pocket-shape scoring for ``rbfe_septop``.
  Other accepted values include ``lomap``, ``shape_difference`` and
  ``pocket_shape``.
* ``rbfe.direction_policy`` – ``larger_volume`` (default) or ``preserve``. Generated
  networks are oriented with the larger grid-volume ligand as reference unless a
  mapping file explicitly fixes the direction.
* ``rbfe.minimal_mapping_atom`` – minimum mapped-atom count required for standard
  RBFE edges. This check is not used to reject ``rbfe_septop`` edges because SEPTOP
  uses full-ligand softcore setup.
* ``rbfe.add_atom_mapping_edges`` – default ``false``; append valid
  atom-mapping override pairs when neither direction was selected by network
  planning.
* ``rbfe.konnektor_layout`` – optional Konnektor layout when ``mapping: konnektor``.
* ``rbfe.both_directions`` – when true, run both directions for each mapped edge.
* ``rbfe.atom_mapper`` – atom mapper backend used for RBFE mapping:

  - ``kartograf`` (default), configured as ``KartografAtomMapper(atom_max_distance=0.95, map_hydrogens_on_hydrogens_only=True, atom_map_hydrogens=False, map_exact_ring_matches_only=True, allow_partial_fused_rings=True, allow_bond_breaks=False, additional_mapping_filter_functions=[filter_element_changes])`` during network planning.
  - ``lomap``, using ``LomapAtomMapper(time=20, threed=True, max3d=1.5, element_change=False, shift=True)``.

Mapper constructor options can be overridden in nested blocks. Omitted values keep
BATTER's previous Kartograf/LoMap defaults documented in :doc:`rbfe`.

.. code-block:: yaml

   rbfe:
     atom_mapper: lomap
     lomap:
       time: 7
       max3d: 2.0
       shift: false
     kartograf:
       atom_max_distance: 1.1
       allow_bond_breaks: true
       filter_element_changes: false

See :doc:`rbfe` for RBFE-specific examples.

Anchor selection
----------------

``create.anchor_atoms`` is optional. If it is omitted, BATTER resolves the anchor
triplet during ``system_prep`` and stores the resolved global selections in
``executions/<run_id>/all-ligands/manifest.json``. Prepared-system anchor masks
used by later equilibration/FE setup are also written per ligand to
``equil/anchors.json``.

Both an omitted field and ``anchor_atoms: null`` request full automatic
selection. A scalar selection string is normalized to a one-entry P1 hint, but
the list form is preferred for clarity.

* For runs with real ligands, the first available real ligand pose drives a
  ligand-guided receptor-anchor heuristic.
* For apo-only MD, BATTER switches to a protein-only heuristic so dummy ligand
  coordinates do not determine the anchor geometry.

Provide three distinct, unambiguous selections only when you need fully manual
P1/P2/P3 geometry. One selection is a P1 hint and BATTER chooses P2/P3. Two
entries are treated as incomplete: BATTER keeps the first as the P1 hint and
chooses P2/P3 together. Invalid or ambiguous selections produce a warning and
fall back to automatic selection; if more than three are supplied, only the
first three are considered.

When automatic selection is active and a charged protein-ligand contact is
detected, BATTER prefers that salt bridge for P1/L1. For L2/L3 it first prefers
valid ring atoms connected to at least two heavy atoms, then atoms connected to
more than two heavy atoms, and then other nonterminal heavy atoms. Receptor
P1/P2/P3 candidates come from stable non-loop Cα atoms and are screened to avoid
near-linear frames. If the preferred receptor spacing or ligand-anchor distance
window has no solution, BATTER warns and uses the best compact, non-collinear
frame instead.

Component-Specific Inputs
-------------------------

Although the ``create`` block is shared by ABFE, MASFE, and MD pipelines, some fields
are consumed only by particular builders. The table below highlights the ones that
feed into the low-level ops documented in :doc:`../developer_guide/internal_builders`:

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
     - Drives salt concentration when ``neutralize_only = "no"``; defaults to
       ``0.05`` M (50 mM).
   * - ``neutralize_only``
     - :attr:`SimulationConfig.neut` → ``create_box``
     - Toggles between neutralisation-only or salt+neutralisation workflows.
   * - ``extra_restraints``
     - Restraint ops
     - Adds positional restraints for ABFE builders.
   * - ``extra_conformation_restraints``
     - Restraint ops
     - JSON specification for conformational restraints.
   * - ``anchor_atoms``
     - ``system_prep`` / restraint ops
     - Optional receptor-anchor override. Empty means BATTER selects anchors
       heuristically; one selection pins P1 and auto-selects P2/P3; three valid,
       distinct selections provide explicit P1/P2/P3 geometry. Incomplete or
       invalid input warns and falls back to automatic selection.
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

FE ion guard
------------

``fe_sim.ion_guard`` defaults to ``"yes"`` for ABFE, RBFE, and RBFE-SEPTOP FE
windows. When enabled, BATTER appends ``#Ion_Guard`` lower-wall distance
restraints to ``disang.rest`` for ``z`` and ``x`` components. Each configured
bulk ion (from ``create.cation`` / ``create.anion``) is restrained from coming
within 10 Å of the ligand reference atom in the binding site. Set
``fe_sim.ion_guard: no`` to disable this
FE-stage guard; equilibration restraints are unchanged.

Equilibration options
---------------------

Two frequently toggled equilibration knobs live under ``fe_sim`` and flow into the
resolved :class:`~batter.config.simulation.SimulationConfig`:

* ``hmr`` – ``"yes"`` enables hydrogen mass repartitioning during preparation.
  BATTER may still generate HMR helper files, but generated local, batch, and
  REMD launch scripts use ``full_merged.prmtop`` as the runtime topology.
* ``enable_mcwat`` – ``"yes"`` (default) enables Monte Carlo water moves during
  equilibration. The flag populates the ``mcwat`` setting in AMBER input decks via
  :func:`batter._internal.ops.amber.write_amber_templates`; MC-water
  equilibration templates place the shifted ligand dummy 15 Å from the site.
* ``mcwat_fe`` – ``"no"`` (default) enables the same MC-water move block in FE
  production input templates when set to ``"yes"``. Runtime ``mdin-current``,
  REMD, and batch inputs inherit it from the generated production template.

REMD runs
---------

REMD inputs (mdins/groupfiles) are always written during preparation so you can decide at
submit time whether to run them. Use ``fe_sim.remd.nstlim`` to set the exchange interval
and segment length; the default is ``1000`` MD steps. The exchange count is derived from
the remaining steps so total runtime is controlled by ``n_steps``. Runtime REMD launchers
preserve the ``nstlim`` already present in ``mdin-remd-template`` and update
``numexchg`` for each segment. The MBAR ``bar_intervall`` and center-of-mass
``DUMPFREQ`` intervals are both ``1000`` steps by default and are capped at
``nstlim`` for shorter exchange blocks. Control execution with ``run.remd``
(``yes`` or ``no``); when
``run.remd: no`` the files are still generated but no REMD jobs are scheduled. REMD jobs
submit one Slurm job per component via ``SLURMM-BATCH-remd`` and monitor
``FINISHED``/``FAILED`` sentinels in the component folder. See
:doc:`remote_bundled_jobs` for operational details.

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
* ``job_manager.header`` (manager script for ``batter run --slurm-submit``)

The header lookup/seed location is controlled by ``run.slurm_header_dir``; when omitted it
defaults to ``~/.batter``.

Per-run SLURM overrides
-----------------------

Simulation submit scripts inherit the header settings above, but you can also control SLURM
resources per run via the ``run.slurm`` block (partition, time, nodes, ntasks_per_node, mem, etc.).
Those values are substituted into SLURM scripts when rendered. Combine the two mechanisms by
setting cluster defaults in the headers and per-run overrides in the YAML when needed.

SLURM configuration block
-------------------------

The ``run.slurm`` block maps directly onto ``sbatch`` flags. All fields are optional; if a
value is omitted it will not be added to the submission command. You can also use
``run.slurm_header_dir`` to point at a custom header directory seeded with
``batter seed-headers``. Keep ``backend: local`` in the YAML; submit the manager
through SLURM with ``batter run ... --slurm-submit`` when you want cluster execution.

Example:

.. code-block:: yaml

   backend: local
   run:
     slurm_header_dir: /path/to/slurm_headers
     slurm:
       partition: gpu
       time: "08:00:00"
       nodes: 2
       ntasks_per_node: 8
       mem_per_cpu: "8G"
       gres: "gpu:8"
       account: my-account
       qos: normal
       constraint: "a100"
       extra_sbatch:
         - "--exclusive"
         - "--mail-type=FAIL"

Supported keys in ``run.slurm``:

* ``partition`` – SLURM partition/queue name (``-p``).
* ``time`` – Walltime in ``HH:MM:SS`` (``-t``).
* ``nodes`` – Node count (``-N``).
* ``ntasks_per_node`` – Tasks per node (``--ntasks-per-node``).
* ``mem_per_cpu`` – Memory per CPU (``--mem-per-cpu``).
* ``gres`` – Generic resources, e.g. GPUs (``--gres``).
* ``account`` – Account/project string (``--account``).
* ``qos`` – QoS string (``--qos``).
* ``constraint`` – Constraint string (``--constraint``).
* ``extra_sbatch`` – Additional ``sbatch`` flags appended verbatim.

Run notifications
-----------------

Set ``run.email_on_completion`` to receive a best-effort email when the BATTER
manager finishes normally or exits with an uncaught failure. BATTER sends that
message through ``localhost`` SMTP and uses ``run.email_sender`` as the sender
address (default: ``nobody@stanford.edu``).

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
   batter.config.simulation.SimulationConfig
   batter.config.load_run_config
   batter.config.dump_run_config
   batter.config.load_simulation_config
   batter.config.dump_simulation_config
