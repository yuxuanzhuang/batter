.. _tutorial:

Tutorial
========

Absolute Binding Free Energy (ABFE) Workflow with ``batter``
------------------------------------------------------------

This tutorial demonstrates how to prepare and run an Absolute Binding Free
Energy Perturbation (ABFEP) workflow using ``batter`` for a membrane protein-
ligand system.

It mirrors the example configuration distributed with the repository:
``examples/mabfe.yaml``.

Installation
------------

0. *(Optional)* Set a persistent PIP cache directory to speed up future installs::

       export PIP_CACHE_DIR=$SCRATCH/.cache

1. Clone the repository and initialize submodules::

       git clone https://github.com/yuxuanzhuang/batter.git
       cd batter
       git submodule update --init --recursive

2. Create and activate a Conda environment::

       conda env create -n batter_env python=3.12 -y
       conda env update -n batter_env -f environment.yml
       conda activate batter_env

3. Install dependencies in editable mode::

       pip install -e ./extern/alchemlyb
       pip install -e ./extern/rocklinc
       pip install -e .

Preparing the System
--------------------

We will use ``mabfe.yaml`` as the base configuration file.  
See :mod:`batter.config.run` for detailed descriptions of each YAML field.

Required Files
~~~~~~~~~~~~~~

1. **Protein structure** – ``protein_input.pdb``  
   Protonated protein exported from Maestro (or an equivalent pipeline).  
   Water and ligand coordinates may be included but will be removed during preparation.

2. **Ligand structures** – One or more ligand ``.sdf`` files  
   Ensure that all hydrogens are present and protonation states are correct.  
   Multiple poses of the same ligand are acceptable.  
   Helper scripts such as ``get_protonation.ipynb`` can assist with protonation.

3. **Membrane system (optional)** – Dabble-generated system files  
   ``system_input.pdb`` and ``system_input.inpcrd`` describing the protein–membrane complex.  
   Only the topology is required if ``system_coordinate`` is provided.  
   Refer to the `Dabble repository <https://github.com/Eigenstate/dabble>`_ for membrane generation.

Generating Simulation Inputs
----------------------------

1. Edit ``mabfe.yaml``:
   - Set ``create.system_name``.
   - Update ``create.anchor_atoms`` with the three anchors appropriate for your receptor.
   - Adjust any additional fields as needed.

2. From the repository root, run::

       batter run mabfe.yaml --dry-run

   Replace ``mabfe.yaml`` with your own configuration path.  
   Remove ``--dry-run`` to start the actual workflow.

3. When no errors are reported, simulation input files will appear under::

       output/mabfe/create/

   Inspect these files to confirm correctness:
   - Verify protein–ligand placement in the binding site.
   - Check box dimensions, ions, and solvent configuration.
   - Check if membrane lipids are correctly placed (if applicable).

Running on SLURM
~~~~~~~~~~~~~~~~

To launch the workflow through a SLURM scheduler::

    batter run mabfe.yaml --slurm-submit

This command starts a SLURM job manager that handles submission and monitoring.  
You may supply a custom SLURM header template (for account, partition, etc.) using::

    --slurm-manager-path <path-to-slurm-header>

Optional: Additional Conformational Restraints
----------------------------------------------

1. Use ``generate_restraints.ipynb`` as a template to generate a
   ``restraints.json`` file.

2. Enable these restraints by adding the following line under the ``create`` section
   of your YAML file::

       extra_conformation_restraints: restraints.json

Analysis
--------

Future tutorials will describe how to run analysis notebooks (e.g.,
``analysis.ipynb``) to visualize and interpret results from completed workflows.