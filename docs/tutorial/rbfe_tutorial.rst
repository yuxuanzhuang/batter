.. _rbfe_tutorial:

RBFE Tutorial
=============

Relative Binding Free Energy (RBFE) Workflow with ``batter``
------------------------------------------------------------

This tutorial walks through a membrane RBFE run powered by ``batter``. The workflow
applies a hybrid topology that behaves like dual-topology with a shared core.
It uses the simultaneous decoupling/recoupling
(SDR) protocol with both ligands present, and relies on softcore
electrostatics/van der Waals potentials so the entire calculation completes in a
single leg. We reference ``examples/rbfe.yaml`` so you can reproduce the run locally
before adapting it to your own system.

Quick walkthrough
-----------------

``batter`` orchestrates an end-to-end AMBER RBFE workflow that starts from protein +
embedded protein-membrane system (if applicable) + ligand(s) (3D coordinates) overlayed to the
protein binding site. The main steps are:

#. **system staging and loading** – An execution folder will be created under ``<run.output_folder>/executions/``
   to hold all intermediate files, logs, and results. If a run ID is not provided, a timestamp-based unique ID is generated. If the same run ID already exists, the execution is
   resumed from the last successful step.
#. **Ligand parameterisation** – supports both GAFF/GAFF2 and OpenFF force fields with
   options to choose charges (AM1-BCC by default)
#. **Equilibration system preparation** – builds solvated/membrane-embedded
   systems with the ligand in the binding site.
#. **Equilibration** – Steps to run before FE production run. During this phase,
   the ligand and protein are not restrained (unless explicitly configured).
   If the ligand unbound from the binding site during equilibration, the run
   is marked as unbound and skipped during FE production.
#. **Equilibrium analysis** - Find a representative frame from the equilibrated trajectory
   to start the FE windows from. RMSD analysis is also performed and saved in the equil folder. Adjust the bound/unbound cutoff via ``fe_sim.unbound_threshold`` if your system requires a different distance threshold.
#. **Network planning** – Build the RBFE transformation map (pair list) based on the selected scheme.
#. **FE window generation and submission** – λ windows are created based on the configuration.
#. **FE equilbration** - very short equilibration runs to allow water relaxation. If flag ``--only-equil`` is provided, the workflow stops after this step.
#. **FE production runs** – Each window is submitted as an independent SLURM job.
   The main process monitors job status and streams updates to the terminal.
   Set ``run.max_active_jobs`` in your YAML (default 1000, ``0`` disables throttling)
   to cap how many SLURM jobs Batter keeps active at once and avoid overloading the scheduler.
#. **Analysis** – Once all windows complete, MBAR analysis is performed and

Network planning schemes
------------------------

RBFE mappings can be created in a few ways:

* **Default** – maps the first ligand to all others (star topology).
* **Konnektor** – uses the ``konnektor`` library to build a network; configure with
  ``rbfe.mapping: konnektor`` and optionally ``rbfe.konnektor_layout``.
  The available layouts are listed in the `Konnektor documentation <https://konnektor.openfree.energy/en/latest/api/konnektor.planners.html>`_.
  provide inputs can be either `MinimalSpanningTreeNetworkGenerator` or `minimalspanningtree`.
  See detailed tutorial in `Konnektor tutorial <https://konnektor.openfree.energy/en/latest/tutorials/basic_network_generation.html>`_.
* **Mapping file** – provide explicit pairs via ``rbfe.mapping_file`` (JSON/YAML list or
  text file with one pair per line).

Set ``rbfe.both_directions: true`` if you want to run both directions for every edge.

Installation
------------

#. *(Optional)* set a persistent pip cache (helpful on shared clusters)::

       export PIP_CACHE_DIR=$SCRATCH/.cache

#. Clone the repository with ssh (or HTTPS if SSH is unavailable) and initialize submodules::

       git clone git@github.com:yuxuanzhuang/batter.git
       # If SSH is unavailable, use HTTPS instead:
       # git clone https://github.com/yuxuanzhuang/batter.git
       # For SSH setup tips:
       # https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

       cd batter
       git submodule update --init --recursive

#. Create and activate a Conda environment (with ``environment.yml``)::

       conda create -n batter python=3.12 -y
       conda env update -n batter -f environment.yml
       conda activate batter

#. Install editable copies of the bundled dependencies plus ``batter`` itself::

       pip install -e ./extern/alchemlyb
       pip install -e ./extern/rocklinc
       pip install -e .

#. Verify the installation::

       batter --help

Preparing the System
--------------------

Use ``examples/rbfe.yaml`` as the starting configuration. Each field is documented in
:doc:`configuration` , but review the inputs below before running anything:

Required Files
~~~~~~~~~~~~~~

1. **Protein structure** – ``protein_input.pdb``
   It can be prepared from e.g. Maestro or an equivalent software. Protonation states are
   inferred from the residue name (AMBER conventions, e.g., ASH denotes protonated ASP).
   Water or ligand coordinates may remain in the file—they are stripped during staging.

2. **Ligand structures** – one ligand per ``.sdf`` file with 3D coordinates.
   Docked poses, aligned experimental structures, or co-folding models all work as
   long as the coordinates align with the provided ``protein_input.pdb``. Ensure hydrogens/protonation states
   are correct (Open Babel, `unipKa <https://github.com/yuxuanzhuang/batter/blob/main/scripts/get_protonation.ipynb>`_, or a similar tool can help).

3. **System topology and coordinates (optional)** – ``system_input.pdb`` / ``system_input.inpcrd``
   Needed for membrane protein system.

   The membrane-embedded system can be generated via `Dabble <https://github.com/Eigenstate/dabble>`_ (preferred with ``protein_input.pdb``).
   ``system_input.pdb`` must encode the correct unit-cell vectors (box information) if ``system_input.inpcrd`` is not provided (Dabble does this by default).
   If ``system_input.inpcrd`` is provided its coordinates and box information take precedence.

   ``protein_input.pdb`` **does not** need to be aligned to ``system_input.pdb``; it can be helpful in cases e.g.,
   the protein structure used for docking (so all the docked poses are superposed to this protein) is oriented differently
   from the membrane system. During system staging, the protein will be aligned to the membrane system, and the alignment
   will be done automatically based on the ``create.protein_align`` config setting.

   Systems from other builders (CHARMM-GUI, Maestro, etc.) may work but are not extensively tested.

   Command to generate POPC-embedded systems with Dabble::

       dabble -i protein_input.mae -o system_input.prmtop --hmr -w 20 -O -ff charmm

   In ``batter`` preparation process, the membrane molecules will be extracted (controlled by `create.lipid_mols`);
   water and ion molecules around ``create.solv_shell`` will also be extracted.

Generating Simulation Inputs
----------------------------

1. **Copy and edit the template.**
   Start from `examples/rbfe.yaml <https://github.com/yuxuanzhuang/batter/blob/main/examples/rbfe.yaml>`_
   and save a copy beside your project data. Update:

   - ``run.output_folder`` – dedicated directory for outputs/logs.
   - ``create.system_name`` – label used in reports.
   - ``create.ligand_input`` – JSON file mapping unique ligand IDs to ``.sdf`` files (see ``examples/reference/ligand_dict.json``).
   - ``create.*`` paths – point at your receptor, system, membrane, and restraint files.
   - ``create.anchor_atoms`` – it is strictly not needed but saved for consistency. Choose stable backbone atoms (CA/C/N) with the guidelines below.

     Anchors (P1, P2, P3) should avoid loop regions, keep P1–P2 and P2–P3 ≥ 8 Å, and target
     ∠(P1–P2–P3) near 90°.

     P1 should preferably form a consistent electrostatics interaction with available
     bound ligands (e.g., a salt bridge).

     For GPCR orthosteric sites, a common choice is P1=3x32,
     P2=2x53, P3=7x42.

   Additional field that may need adjustment based on your cluster environment:

   - ``run.email_on_completion`` – email address to notify when SLURM jobs complete.
   - ``run.email_sender`` – email address to send notifications from. Default to ``nobody@stanford.edu`` if unset.
   - ``run.slurm.partition`` – SLURM partition/queue to submit jobs to.
   - ``run.max_active_jobs`` – cap on how many SLURM jobs to keep active at once (default 1000, ``0`` disables throttling).
   - ``rbfe.mapping`` / ``rbfe.mapping_file`` – choose your network planning scheme.

      The available schemes are described in :ref:`Network planning schemes <rbfe_tutorial>`.

2. **Validate the configuration before heavy computation (Optional)**::

       batter run examples/rbfe.yaml --dry-run

   This command runs ligand parameterisation (WARNING: heavy load), and equilibration system preparation.
   On shared clusters, run the dry-run on a compute node if possible to avoid overloading login nodes.

3. **Inspect the staged system (Optional)**
   Once the dry-run completes, review ``<run.output_folder>/executions/<run_id>/``:

   - ``simulations/<LIGAND>/equil/full.pdb`` – ligand-specific equilibration systems.
     Check if the ligand is correctly placed in the binding site,
     and that membranes/solvent boxes look reasonable.

4. **Launch the full workflow manager (local execution)**::

       batter run examples/rbfe.yaml

   Production runs take hours to days depending on system size, the number of ligands,
   and available hardware. Progress is streamed to the terminal and to
   ``executions/<run_id>/logs/batter.log``.

Submitting the manager job via SLURM (RECOMMENDED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit the same run through SLURM::

    batter run examples/rbfe.yaml --slurm-submit

Provide ``--slurm-manager-path`` if you maintain a custom SLURM header template
(accounts, modules, partitions, etc.). Copy and modify the default template from
``batter/data/job_manager.header`` + ``job_manager.body``.

The job manager stages the system locally,
writes an ``sbatch`` script based on the YAML hash, and streams updates as windows
finish.

Handy CLI Flags
---------------

``batter run`` exposes many overrides so you rarely have to edit YAML mid-iteration:

``--on-failure {prune,raise,retry}``
    Decide how to handle per-ligand failures. ``retry`` clears ``FAILED`` sentinels and reruns that phase once.
``--only-equil / --full``
    Stop after shared prep/equilibration—useful for debugging system setup before FE windows.
``--dry-run``
    Stage the system and prepare equilibration inputs without running any MD.
``--run-id`` and ``--output-folder``
    Override execution paths without touching ``system.*`` fields.
``--slurm-submit`` / ``--slurm-manager-path``
    Switch between local execution and SLURM submission (with an optional custom header).
