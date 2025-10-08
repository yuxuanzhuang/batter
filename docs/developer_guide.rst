======================
BATTER Developer Guide
======================

Overview
========

This document describes the internal architecture and developer workflow for **BATTER**, 
a modular framework for running and managing free-energy calculations. It supports
**absolute binding free energy (ABFE) for both membrane and soluble protein** and
**absolute solvation free energy (ASFE) for smalle molecules**.

It is intended for contributors who wish to understand or extend
the orchestration, parameterization, and execution logic.

.. contents::
   :depth: 2
   :local:

High-Level Flow
===============

A BATTER run proceeds in seven conceptual stages:

1. **Parse** a user YAML (:class:`~batter.config.run.RunConfig`) into validated objects.
2. **Build** a shared simulation system under ``work/<system_name>/``.
3. **Stage all ligands** under ``work/<system_name>/ligands/<LIG>/inputs/ligand.sdf``.
4. **Parameterize** all ligands with given forcefield (``param_ligands`` step).
5. **Assemble** the appropriate pipeline (:func:`~batter.pipeline.factory.make_abfe_pipeline`, etc.).
6. **Run** the pipeline on a backend (:class:`~batter.exec.local.LocalBackend` or :class:`~batter.exec.slurm.SlurmBackend`).
7. **Collect** and store results into the portable FE repository.

Each step is represented by a :class:`~batter.pipeline.step.Step` and executed in dependency order.

Code Architecture
=================

The core packages are organized as follows::

   batter/
   ├── config/          # Pydantic models for user YAMLs
   ├── systems/         # System objects and builders
   ├── param/           # Ligand parameterization utilities
   ├── pipeline/        # Step, Pipeline, and factories
   ├── exec/            # Execution backends (local / slurm)
   ├── orchestrate/     # High-level run orchestration
   └── runtime/         # Portable artifact and results store

Configuration Layer
-------------------

*Module:* ``batter.config.run``

- :class:`~batter.config.run.SystemSection`
  Holds static system info (type, output folder).

- :class:`~batter.config.run.CreateArgs`
  Defines inputs used to *create* or stage the system, such as ``protein_input``,
  ``system_input``, and ``ligand_paths`` or ``ligand_input`` (JSON).

- :class:`~batter.config.run.RunSection`
  Contains runtime switches: ``dry_run``, ``only_fe_preparation``, ``run_id``.

- :class:`~batter.config.run.RunConfig`
  Top-level object that aggregates all sections and exposes :meth:`load`
  and :meth:`model_validate_yaml` helpers.

Simulation Layer
----------------

*Module:* ``batter.config.simulation``

- :class:`~batter.config.simulation.SimulationConfig`  
  A detailed specification of a simulation (temperature, λ-schedule, restraint parameters, etc.).
  Referenced by ``sim_config_path`` in the top-level YAML.

System Layer
------------

*Modules:* ``batter.systems.core``, ``batter.systems.mabfe``

- :class:`~batter.systems.core.SimSystem`
  Immutable record of a system on disk (name, root path, protein, ligands, anchors, metadata).

- :class:`~batter.systems.mabfe.MABFEBuilder`
  Builds shared systems and stages per-ligand subsystems under ``work/<sys>/ligands/``.
  Provides helpers such as :meth:`build` and :meth:`make_child_for_ligand`.

Parameterization
----------------

*Module:* ``batter.param.ligand``

- :func:`~batter.param.ligand.batch_ligand_process`  
  Performs bulk ligand parameterization for all staged ligands.
  Generates ``.mol2``, ``.frcmod``, ``.lib``, etc. into ``<work>/ligand_params`` and
  links them into each ligand child under ``ligands/<LIG>/params/``.

Pipeline Layer
--------------

*Modules:* ``batter.pipeline.step``, ``batter.pipeline.pipeline``, ``batter.pipeline.factory``

- :class:`~batter.pipeline.step.Step`  
  Describes one computational step: ``name``, ``requires``, ``params``.

- :class:`~batter.pipeline.pipeline.Pipeline``  
  An ordered collection of :class:`~batter.pipeline.step.Step` objects.
  The :meth:`~batter.pipeline.pipeline.Pipeline.run` method drives execution:
  
  1. Call the backend handler for each step.
  2. If it returns job IDs, invoke :meth:`backend.wait` to block until completion.
  3. Collect results in an :class:`~batter.pipeline.step.ExecResult` mapping.

- :mod:`batter.pipeline.factory`  
  Builds canonical pipelines:
  
  * ABFE → ``param_ligands → prepare_equil → equil → prepare_fe → prepare_fe_windows → fe_equil → fe → analyze``
  * ASFE → ``param_ligands → prepare_fe → solvation → analyze``

Execution Backends
------------------

*Modules:* ``batter.exec.local``, ``batter.exec.slurm``

Backends determine *how* each step is executed.

- :class:`~batter.exec.local.LocalBackend`  
  Directly executes registered Python callables.  
  Used for quick tests and debugging.

- :class:`~batter.exec.slurm.SlurmBackend`  
  Generates and submits SLURM job scripts, waits until completion
  by polling ``squeue`` or ``sacct``.  
  Provides:
  
  * :meth:`submit(payload, resources, workdir, job_name)`
  * :meth:`wait(job_ids, step, system)`

Each backend maintains a registry of step handlers:

.. code-block:: python

   backend.register("equil", equil_handler)
   backend.register("windows", windows_handler)

Orchestration
-------------

*Module:* ``batter.orchestrate.run``

The **conductor** that wires all layers together.

Steps performed by :func:`~batter.orchestrate.run.run_from_yaml`:

1. **Load configuration** via :class:`RunConfig`.
2. **Select backend** (local or slurm).
3. **Build** shared system using :class:`MABFEBuilder`.
4. **Resolve ligands** from YAML or JSON, stage them under ``ligands/``.
5. **Run** the single batch ``param_ligands`` step.
6. **Construct** per-ligand pipeline (dropping the parameterization step).
7. **Execute** all remaining steps per ligand.
8. **Save** :class:`~batter.runtime.fe_repo.FERecord` per ligand.

Runtime & Portability
---------------------

*Modules:* ``batter.runtime.portable``, ``batter.runtime.fe_repo``

- :class:`~batter.runtime.portable.ArtifactStore`  
  Provides a stable, portable layout for storing artifacts and manifests.
  Enables runs to be transferred between clusters without path breakage.

- :class:`~batter.runtime.fe_repo.FEResultsRepository`  
  Interface for saving/loading :class:`~batter.runtime.fe_repo.FERecord` objects
  that capture per-ligand free-energy results.

On-Disk Layout
--------------

A typical ABFE run produces::

   work/
   └── adrb2_A/
       ├── artifacts/
       │   ├── equil.rst7
       │   ├── windows.json
       │   └── analyze.ok
       ├── ligand_params/
       │   ├── LIG1.mol2
       │   ├── LIG1.lib
       │   └── ...
       ├── ligands/
       │   ├── LIG1/
       │   │   ├── inputs/ligand.sdf
       │   │   └── params/...
       │   └── LIG2/
       │       ├── inputs/ligand.sdf
       │       └── params/...
       └── record.json

Backend Behavior
================

Local Backend
-------------

Executes handlers synchronously inside the same process.  
Each handler returns an :class:`ExecResult` with optional artifacts.

SLURM Backend
-------------

1. Writes a temporary job script in the step's working directory.
2. Submits via ``sbatch`` with desired resources.
3. Returns the job ID(s) in :class:`ExecResult`.
4. The pipeline then calls :meth:`wait` to poll job status until all jobs finish.

Example handler for SLURM:

.. code-block:: python

   def equil_handler(step, system, params):
       payload = f"""
       pmemd.MPI -O -i equil.in -o equil.out -p top.prmtop -c start.rst7 -r end.rst7
       """
       job_ids = backend.submit(payload, SlurmResources(time="4:00:00", cpus=8, mem="32G"),
                                workdir=system.root/"equil", job_name=f"eq-{system.name}")
       return ExecResult(job_ids=job_ids, artifacts={"rst7": system.root/"artifacts/equil.rst7"})

Extending BATTER
================

Adding a New Step
-----------------

1. Define a new :class:`~batter.pipeline.step.Step` in your pipeline factory.
2. Implement a backend handler (local, slurm, or both).
3. Register it:

   .. code-block:: python

      backend.register("my_new_step", my_handler)

4. Add logic in :mod:`batter/orchestrate/run.py` if it affects orchestration.

Adding a New Protocol
---------------------

Create a new factory function (e.g. ``make_rbfe_pipeline``)
and extend ``_select_pipeline`` in :mod:`batter/orchestrate/run.py`:

.. code-block:: python

   if p == "rbfe":
       return make_rbfe_pipeline(sim_cfg, only_fe_prep)

Testing
=======

Use the local backend for rapid iteration:

.. code-block:: bash

   new_batter run examples/run_local.yaml

To test job submission logic without an actual cluster,
use ``backend: slurm`` and mock ``sbatch``/``squeue`` commands.

Portability
===========

All artifacts are written relative to the system root.  
The :class:`ArtifactStore` ensures that FE results and manifests
can be copied between clusters (e.g., Frontier → Sherlock)
without path rewriting.

In summary
==========

- **Orchestration** (`run.py`) drives the workflow.
- **Pipeline** defines the logical steps.
- **Backend** defines *how* those steps run.
- **System** organizes inputs and outputs.
- **Runtime store** makes everything portable.

Together, these layers let BATTER execute free-energy pipelines reproducibly
on both local machines and large HPC clusters.