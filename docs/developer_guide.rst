======================
BATTER Developer Guide
======================

This guide provides an up-to-date overview of the internal architecture of **BATTER**.
It targets contributors who need to understand or extend the codebase.

The project powers absolute binding (ABFE) and solvation (ASFE) free-energy workflows,
supports both local execution and SLURM clusters, and packages results in a portable
artifact store.

.. toctree::
   :maxdepth: 1
   :caption: Focus Topics

   developer_guide/pipeline_payloads_and_metadata
   developer_guide/authoring_pipelines
   developer_guide/internal_builders

.. contents::
   :depth: 2
   :local:

Code Layout
===========

.. code-block:: text

   batter/                     # Modern package (public API, pipelines, builders)
   batter_v1/                  # Legacy BAT.py compatibility layer (frozen)
   docs/                       # Sphinx sources (user + developer guides)
   examples/                   # Reference YAML workflows and restraint templates
   tests/                      # Pytest suite covering configs, pipelines, exec, etc.
   extern/                     # Vendored dependencies (editable installs)
   devtools/                   # Helper scripts + conda envs for development
   scripts/                    # Misc automation helpers
   README.rst                  # Project overview
   TODO                        # Open engineering tasks / ideas
   pyproject.toml / setup.cfg  # Build and packaging metadata
   environment*.yml            # Conda environments for the main stack

The ``batter/`` package itself is organised as:

.. code-block:: text

   batter/
   ├── api.py                 # Public entry points (run_from_yaml, FE repos, etc.)
   ├── cli/                   # click-based CLI commands (run, fe, fek-schedule, ...)
   ├── config/                # Pydantic models for run/simulation YAML + helpers
   ├── systems/               # System descriptors and builders (MABFE / MASFE)
   ├── _internal/             # Low-level build ops (create_box, restraints, sim files)
   ├── param/                 # Ligand parameterisation helpers
   ├── pipeline/              # Steps, payloads, pipeline factories
   ├── exec/                  # Local/SLURM backends and step handlers
   ├── orchestrate/           # High-level orchestration + pipeline wiring
   ├── runtime/               # Portable artifact store and FE repository
   ├── analysis/              # Post-processing & convergence utilities
   └── utils/                 # Shared helpers (Amber wrappers, file ops, etc.)

Further Reading
---------------

The following reference chapters live elsewhere in the docs but are useful when
working on internal builders or pipelines:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   developer_guide/pipeline_payloads_and_metadata
   developer_guide/execution_model
   developer_guide/analysis
   developer_guide/parameterisation

High-Level Execution Flow
=========================

A run triggered via :func:`batter.orchestrate.run.run_from_yaml` progresses through
the stages below:

1. **Configuration** – Parse the top-level YAML into a
   :class:`~batter.config.run.RunConfig` and resolve the composed
   :class:`~batter.config.simulation.SimulationConfig`.
2. **System build** – Use a :class:`~batter.systems.core.SystemBuilder`
   (``MABFEBuilder`` or ``MASFEBuilder``) to stage shared inputs under
   ``<output_folder>/executions/<run_id>/``.
3. **Ligand staging** – Copy ligand files under ``executions/<run_id>/simulations/<LIG>/inputs``.
4. **Parameterisation** – Run the ``param_ligands`` step once to populate
   ``executions/<run_id>/artifacts/ligand_params``.
5. **Pipeline construction** – Select an ABFE/ASFE pipeline using
   :func:`~batter.orchestrate.pipeline_utils.select_pipeline`.
6. **Execution** – Drive each pipeline phase on the chosen backend
   (:class:`~batter.exec.local.LocalBackend` or :class:`~batter.exec.slurm.SlurmBackend`).
   Step handlers consume typed :class:`~batter.pipeline.payloads.StepPayload` objects.
7. **Result packaging** – Persist window outputs and summary statistics using
   :class:`~batter.runtime.fe_repo.FEResultsRepository`, enabling portable analysis.

Configuration Layer
===================

*Module:* ``batter.config``

- :class:`~batter.config.run.RunSection` – Execution controls that include the
  artifact destination (``run.output_folder``) and optional builder override
  (``run.system_type``), along with backend/dry-run/failure policy knobs.
- :class:`~batter.config.run.CreateArgs` – Inputs required to stage the system
  (protein, topology, ligands, restraints).
- :class:`~batter.config.run.RunConfig` – Aggregates the sections, exposes helpers such as
  :meth:`load`, :meth:`model_validate_yaml`, and :meth:`resolved_sim_config`, and resolves
  relative paths when a YAML is loaded.
- :class:`~batter.config.simulation.SimulationConfig` – Fully merged simulation
  specification produced by :meth:`RunConfig.resolved_sim_config`. The
  developer-facing configuration never includes this model directly, but the
  developer guide documents available fields and the protocol-specific
  validations (e.g., ABFE requires ``z_steps*``; ASFE requires ``y_steps*``).

Systems and Builders
====================

*Modules:* ``batter.systems.core``, ``batter.systems.mabfe``, ``batter.systems.masfe``

- :class:`~batter.systems.core.SimSystem` – Immutable descriptor of a system on disk.
  Metadata is stored in :class:`~batter.systems.core.SystemMeta` which offers
  structured accessors and ``merge`` semantics for propagating ligand-specific data.
- :class:`~batter.systems.mabfe.MABFEBuilder` – Prepares shared ABFE systems and creates
  per-ligand children under ``simulations/<LIG>/``.
- :class:`~batter.systems.masfe.MASFEBuilder` – MASFE counterpart that stages ligands
  without a protein topology.

Binding (ABFE) components
-------------------------

ABFE simulations decouple the ligand from the bound complex using the ``z`` component
(restraints + decoupling in complex). Both ``z_steps1``/``z_steps2`` are required
via ``fe_sim.steps1/steps2``. Ensure the
restraints and lambdas in the run YAML align with your chosen decoupling scheme.

Solvation (ASFE) components
---------------------------

ASFE simulations run two FE components:

- ``y`` – ligand-in-solvent decoupling.
- ``m`` – ligand-in-vacuum decoupling.

Both components require step counts in ``fe_sim.steps1``/``steps2``. The orchestrator
enforces that both are positive before pipeline execution.

Practical constraints
---------------------

- Water boxes require ``buffer_x/y/z >= 15 Å``; the validator will reject smaller
  padding to avoid vacuum artifacts. For membranes, automatic Z padding is applied if
  needed.
- Resume semantics rely on ``run_id`` plus the stored configuration signature (only
  ``create`` and ``fe_sim`` fields). Changing execution knobs under ``run`` will not
  trigger a new run_id, so bump the run_id yourself when you want a clean workspace.
- Lambda overrides: provide a default ``lambdas`` list and override per component via
  ``component_lambdas`` when needed. Missing components inherit the default schedule.

Parameterisation
================

*Module:* ``batter.param.ligand``

- :func:`~batter.param.ligand.batch_ligand_process` – Performs ligand force-field
  assignment, producing a content-addressed store under ``artifacts/ligand_params``.
  Used by the ``param_ligands`` handler to distribute parameter files.

Pipelines and Payloads
======================

*Modules:* ``batter.pipeline.step``, ``batter.pipeline.pipeline``,
``batter.pipeline.factory``, ``batter.pipeline.payloads``

- :class:`~batter.pipeline.step.Step` – Encapsulates a DAG node with ``name``,
  ``requires`` and a :class:`~batter.pipeline.payloads.StepPayload`.
  ``step.params`` remains as a compatibility alias.
- :class:`~batter.pipeline.pipeline.Pipeline` – Topologically orders steps and invokes
  the backend through :meth:`run`.
- :mod:`batter.pipeline.factory` – Builds canonical ABFE/ASFE pipelines. Pipelines are
  expressed in terms of :class:`~batter.pipeline.payloads.StepPayload` and
  :class:`~batter.pipeline.payloads.SystemParams`.
- :mod:`batter.pipeline.payloads` – Defines the typed payload and system-parameter models.
  See :doc:`developer_guide/pipeline_payloads_and_metadata` for details.

Execution Backends
==================

*Modules:* ``batter.exec.base``, ``batter.exec.local``, ``batter.exec.slurm``,
``batter.exec.handlers``.

- :class:`~batter.exec.base.ExecBackend` – Shared protocol implemented by backends.
- :class:`~batter.exec.local.LocalBackend` – Runs Python handlers directly (serial or joblib).
- :class:`~batter.exec.slurm.SlurmBackend` – Submits SLURM jobs via
  :class:`~batter.exec.slurm_mgr.SlurmJobManager`.
- Handler modules under ``batter/exec/handlers`` implement step-specific logic
  (system prep, parameterisation, equilibration, FE production, analysis). Each handler
  receives a :class:`~batter.pipeline.payloads.StepPayload` and a
  :class:`~batter.systems.core.SimSystem`.

Orchestration
=============

*Module:* ``batter.orchestrate.run``

:func:`~batter.orchestrate.run.run_from_yaml` wires every layer together:

1. Load the run YAML and apply optional overrides.
2. Instantiate a system builder inferred from the selected protocol (abfe/md → MABFE, asfe → MASFE; overrides via ``run.system_type`` remain for backward compatibility).
3. Resolve staged ligands (supporting resume) and regenerate the system if required.
4. Construct the ABFE/ASFE pipeline using :func:`select_pipeline
   <batter.orchestrate.pipeline_utils.select_pipeline>`.
5. Execute parent-only steps (``system_prep``, ``param_ligands``).
6. Clone the pipeline for per-ligand execution, injecting the SLURM job manager when needed.
7. Run phases sequentially, enforcing skip/resume semantics via
   :mod:`batter.orchestrate.markers`.
8. Persist FE results using :class:`~batter.runtime.fe_repo.FEResultsRepository`.

Run identifiers and config signatures
-------------------------------------

Each execution lives under ``<output_folder>/executions/<run_id>/``. When a run_id
already exists, :func:`batter.orchestrate.run._compute_run_signature` compares the
current YAML against the stored signature under ``artifacts/config/run_config.hash``.
Only the simulation inputs are hashed (``create`` and ``fe_sim``/``fe``); ``run`` and
override flags do not affect the signature. A normalized JSON snapshot of the hashed
payload is also written to ``artifacts/config/run_config.normalized.json`` to aid
debugging. If the signatures differ and ``run_id`` was requested explicitly, the
orchestrator raises unless ``--allow-run-id-mismatch`` is set; in ``auto`` mode it
will automatically pick a fresh run_id and log a brief diff of the mismatched fields.

Runtime & Portability
=====================

*Modules:* ``batter.runtime.portable``, ``batter.runtime.fe_repo``.

- :class:`~batter.runtime.portable.ArtifactStore` – Manages a relocatable manifest of
  files/directories produced during the run.
- :class:`~batter.runtime.fe_repo.FEResultsRepository` – Indexes and stores
  :class:`~batter.runtime.fe_repo.FERecord` objects, capturing total ΔG, per-window data,
  and copies of analysis outputs.

Directory Layout Example
========================

The structure below illustrates an ABFE execution root (``<output_folder>/executions/<run_id>/``)::

   executions/<run_id>/
   ├── artifacts/
   │   ├── config/
   │   │   ├── sim_overrides.json
   │   │   └── sim.resolved.yaml
   │   └── ligand_params/
   │       ├── index.json
   │       └── LIG1/
   │           ├── lig.mol2
   │           └── metadata.json
   ├── simulations/
   │   ├── LIG1/
   │   │   ├── inputs/ligand.sdf
   │   │   └── fe/...
   │   └── LIG2/
   │       └── ...
   ├── batter.run.log
   └── fe/Results/Results.dat     (per-ligand directories once analysis finishes)

Refer to :mod:`batter.orchestrate.markers` for the sentinel files used to detect
completion or failure of each phase.
