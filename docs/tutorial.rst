.. _tutorial:

Tutorial
========

Absolute Binding Free Energy (ABFE) Workflow with ``batter``
------------------------------------------------------------

This tutorial walks through a typical membrane ABFE run powered by ``batter``.
The workflow applies λ-dependent Boresch restraints, runs the simultaneous decoupling/
recoupling (SDR) protocol with both interacting and dummy ligands present, and uses
softcore electrostatics/van der Waals potentials so the entire calculation completes
in a single leg. We will reference the maintained example configuration
(``examples/mabfe.yaml``) so you can reproduce the steps locally and later adapt them
to your own system.

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

We will use ``examples/mabfe.yaml``
as our starting configuration. Each field is documented in :mod:`batter.config.run`,
but you should review the following inputs
before running anything:

Required Files
~~~~~~~~~~~~~~

1. **Protein structure** – ``protein_input.pdb``  
   Protein exported from Maestro (or an equivalent pipeline). the protonation states of 
   the titratable residues will be assigned based on the residue name (AMBER-based), e.g. ASH will be
   protonated ASP.
   Water and ligand coordinates may be included but will be removed during preparation.

2. **Ligand structures** – One or more ligand ``.sdf`` files with its 3D coordinates.
   The ligand can be provided from docked pose(s) to the provided ``protein_input.pdb`` structure.
   It can also be from experimental structures and aligned to the provided ``protein_input.pdb`` structure.
   It can also be generated from co-folding models and then aligned to the provided ``protein_input.pdb`` structure.  
   Each ``.sdf`` file should contain only one ligand molecule.
   Ensure that all hydrogens are present and protonation states are correct.  
   Use your preferred cheminformatics toolkit (e.g. openbabel) to determine protonation
   or use `unipKa` in ``scripts/get_protonation.ipynb``.

3. **Membrane system (optional)** – Dabble-generated system files  
   ``system_input.pdb`` and ``system_input.inpcrd`` describing the protein–membrane complex.
   The box information is expected in the ``system_input.pdb`` file if ``system_input.inpcrd`` is not provided.
   The coordinates from ``system_input.inpcrd`` will be used if it is provided.
   preferably, the system is preferred with the same ``protein_input.pdb`` structure.
   Refer to the `Dabble repository <https://github.com/Eigenstate/dabble>`_ for membrane generation.
   Embedded system prepared from other tools, e.g. CHARMM-GUI and Maestro, may be successfully used
   but have not been extensively tested.


Generating Simulation Inputs
----------------------------

#. Copy the content of ``examples/mabfe.yaml`` (https://github.com/yuxuanzhuang/batter/blob/main/examples/mabfe.yaml)
   to your own path. Note all the paths in the YAML are relative to the YAML file location or absolute paths.

   - Set ``system.output_folder`` so outputs land in a dedicated folder.
   - The ``create.system_name`` is a label for your system (used in reports).
   - Create a ``ligand_dict.json`` mapped unique ligand identifiers to their ``.sdf`` paths.
     See ``examples/ligand_dict.json`` for formatting.
   - Point the ``create.*`` paths at your protein/ligand/system inputs.
   - Update ``create.anchor_atoms`` with receptor-specific selections. Pick the anchor atoms (P1, P2, P3) based on the following criteria:
        - They should be backbone atoms (CA, C or N) and part of stable secondary structure
          such as an alpha-helix or a beta sheet. Avoid loop regions.
        - P1: preferably choose a residue that forms consistent
          electrostatics interactions with the ligand (e.g., a salt bridge).
        - P2, P3: P1-P2 and P2-P3 distances should be at least 8 Å.
        - P1-P2_P3 angle should NOT be close to 0 or 180 degrees. preferably
            around 90 degrees.
        - For GPCR-orthosteric ligand, I often choose
            P1: 3x32
            P2: 2x53
            P3: 7x42

#. From the repository root, validate the configuration before launching real work::

       batter run examples/mabfe.yaml --dry-run
    
    Warning: This will also run heavy-loading steps like ligand parameterization.
    So the job may take several minutes to complete.

    You should preferably run it with a compute node if you are on a shared cluster. 

#. Once staging completes, inspect ``<system.output_folder>/executions/<run_id>/``:

   - ``executions/<run_id>/simulations/<LIGAND>/inputs`` contains per-ligand copies of your structures.
   - ``executions/<run_id>/artifacts`` holds shared topology/coordinate assets.
   - Review “build.pdb” and intermediate logs before moving on to production sampling.

#. If satisfied, launch the full workflow::

       batter run examples/mabfe.yaml

    This will take several hours to days depending on system size, number of ligands,
    and available hardware. It will tracks progress in the terminal and write logs to
    ``<system.output_folder>/executions/<run_id>/logs/batter.log``.

    Alternatively, you can submit this "manager job" to a SLURM scheduler::
    
        batter run examples/mabfe.yaml --slurm-submit

Provide ``--slurm-manager-path`` if you keep a custom SLURM header template (accounts,
modules, partitions, etc.). The job manager will stage the system locally, write an
``sbatch`` script derived from the YAML hash, and submit the manager job to SLURM.

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
