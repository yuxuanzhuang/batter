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

- :class:`~batter.config.run.SystemSection` – Static system metadata (type, output folder).
- :class:`~batter.config.run.CreateArgs` – Inputs required to stage the system
  (protein, topology, ligands, restraints).
- :class:`~batter.config.run.RunSection` – Execution controls
  (backend, dry-run, failure policy).
- :class:`~batter.config.run.RunConfig` – Aggregates the sections and exposes helpers
  such as :meth:`load` and :meth:`resolved_sim_config`.
- :class:`~batter.config.simulation.SimulationConfig` – Fully merged simulation
  specification used by handlers (temperature, λ-schedule, REMD, restraints).

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
  ``requires`` and a :class:`StepPayload <batter.pipeline.payloads.StepPayload>`.
  ``step.params`` remains as a compatibility alias.
- :class:`~batter.pipeline.pipeline.Pipeline`` – Topologically orders steps and invokes
  the backend through :meth:`run`.
- :mod:`batter.pipeline.factory` – Builds canonical ABFE/ASFE pipelines. Pipelines are
  expressed in terms of :class:`StepPayload` and :class:`SystemParams
  <batter.pipeline.payloads.SystemParams>`.
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
  receives a :class:`StepPayload` and a :class:`SimSystem`.

Orchestration
=============

*Module:* ``batter.orchestrate.run``

:func:`~batter.orchestrate.run.run_from_yaml` wires every layer together:

1. Load the run YAML and apply optional overrides.
2. Instantiate a system builder inferred from the selected protocol (abfe/md → MABFE, asfe → MASFE; overrides via ``system.type`` remain for backward compatibility).
3. Resolve staged ligands (supporting resume) and regenerate the system if required.
4. Construct the ABFE/ASFE pipeline using :func:`select_pipeline
   <batter.orchestrate.pipeline_utils.select_pipeline>`.
5. Execute parent-only steps (``system_prep``, ``param_ligands``).
6. Clone the pipeline for per-ligand execution, injecting the SLURM job manager when needed.
7. Run phases sequentially, enforcing skip/resume semantics via
   :mod:`batter.orchestrate.markers`.
8. Persist FE results using :class:`~batter.runtime.fe_repo.FEResultsRepository`.

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
