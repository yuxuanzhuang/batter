.. _tutorial:

Tutorial
========

Absolute Binding Free Energy (ABFE) Workflow with ``batter``
------------------------------------------------------------

This tutorial walks through a typical membrane ABFE run powered by ``batter``.
We will reference the maintained example configuration (``examples/mabfe.yaml``)
so you can reproduce the steps locally and later adapt them to your own system.

Installation
------------

#. *(Optional)* Set a persistent pip cache (helps on shared clusters)::

       export PIP_CACHE_DIR=$SCRATCH/.cache

#. Clone the repository and initialize submodules::

       git clone https://github.com/yuxuanzhuang/batter.git
       cd batter
       git submodule update --init --recursive

#. Create and activate a Conda environment (matches ``environment.yml``)::

       conda env create -n batter_env python=3.12 -y
       conda env update -n batter_env -f environment.yml
       conda activate batter_env

#. Install editable copies of bundled dependencies and ``batter`` itself::

       pip install -e ./extern/alchemlyb
       pip install -e ./extern/rocklinc
       pip install -e .

Preparing the System
--------------------

We will use ``examples/mabfe.yaml`` as our starting configuration. Each field is
documented in :mod:`batter.config.run`, but you should review the following inputs
before running anything:

Required Files
~~~~~~~~~~~~~~

1. **Protein structure** – ``protein_input.pdb``  
   Protonated protein exported from Maestro (or an equivalent pipeline).  
   Water and ligand coordinates may be included but will be removed during preparation.

2. **Ligand structures** – One or more ligand ``.sdf`` files  
   Ensure that all hydrogens are present and protonation states are correct.  
   Multiple poses of the same ligand are acceptable.  
   Use your preferred cheminformatics toolkit (RDKit, OpenEye, etc.) to verify protonation.

3. **Membrane system (optional)** – Dabble-generated system files  
   ``system_input.pdb`` and ``system_input.inpcrd`` describing the protein–membrane complex.  
   Only the topology is required if ``system_coordinate`` is provided.  
   Refer to the `Dabble repository <https://github.com/Eigenstate/dabble>`_ for membrane generation.

Generating Simulation Inputs
----------------------------

#. Copy ``examples/mabfe.yaml`` to your own path:

   - Set ``create.system_name`` and ``system.output_folder`` so outputs land in a dedicated folder.
   - Update ``create.anchor_atoms`` with receptor-specific selections.
   - Point the ``create.*`` paths at your protein/ligand/system inputs.

#. From the repository root, validate the configuration before launching real work::

       batter run examples/mabfe.yaml --dry-run

#. Once staging completes, inspect ``<system.output_folder>/executions/<run_id>/``:

   - ``executions/<run_id>/simulations/<LIGAND>/inputs`` contains per-ligand copies of your structures.
   - ``executions/<run_id>/artifacts`` holds shared topology/coordinate assets.
   - Review “build.pdb” and intermediate logs before moving on to production sampling.

Running on SLURM
~~~~~~~~~~~~~~~~

To submit via SLURM instead of running locally::

    batter run examples/mabfe.yaml --slurm-submit

Provide ``--slurm-manager-path`` if you keep a custom SLURM header template (accounts,
modules, partitions, etc.). The job manager will stage the system locally, write an
``sbatch`` script derived from the YAML hash, and stream status updates as windows finish.

Handy CLI Flags
---------------

``batter run`` exposes many overrides so you rarely have to edit YAML mid-iteration:

``--on-failure {prune,raise,retry}``
    Decide how to handle per-ligand failures. ``retry`` clears ``FAILED`` sentinels and reruns that phase once.
``--only-equil / --full``
    Stop after shared prep/equilibration—useful for debugging system setup before FE windows.
``--run-id`` and ``--output-folder``
    Override execution paths without touching ``system.*`` fields.
``--slurm-submit`` / ``--slurm-manager-path``
    Switch between local execution and SLURM submission (with an optional custom header).

Run ``batter run --help`` anytime you need the full list of switches and defaults.

You can check the status of running jobs with::

    batter report-jobs

Note if you kill your current batter process,
the SLURM jobs will continue running in the background.
You have to manually cancel them via::

    batter cancel-jobs --contains <system_path_reported_above>


Optional: Additional Conformational Restraints
----------------------------------------------

#. Use the restraint-generation notebook from `bat_mem <https://github.com/yuxuanzhuang/bat_mem/blob/main/tutorial/TEMPLATES/generate_restraints.ipynb>`_
   (or an equivalent script) to author a ``restraints.json`` describing the distance constraints you need.

#. Point ``create.extra_conformation_restraints`` at the resulting JSON file::

       extra_conformation_restraints: path/to/restraints.json

Analysis
--------

Completed runs automatically write MBAR summaries under ``executions/<run_id>/results``.
Use the CLI helpers to inspect them::

    batter fe list <system.output_folder>
    batter fe show <system.output_folder> <run_id>

``fe list`` prints a high-level table (ΔG, SE, components) for every stored run, while
``fe show`` dives into per-window data. CSV/JSON exports live alongside the results on disk.
Also inspect the convergence plots saved under ``results/<run-id>/<ligand-name>/Results``
See :doc:`developer_guide/analysis` for deeper post-processing (plots, REMD diagnostics, etc.).
