.. _tutorial:

Tutorial
========

Currently, only the tutorial of absolute binding free energy (ABFE) calculation
for the membrane protein system is available. More tutorials will be added in the future.


Absolute Binding Free Energy (ABFE) Workflow with ``batter``
------------------------------------------------------------

This tutorial walks through a membrane ABFE run powered by ``batter``. The workflow
applies λ-dependent Boresch restraints, uses the simultaneous decoupling/recoupling
(SDR) protocol with both interacting and dummy ligands present, and relies on softcore
electrostatics/van der Waals potentials so the entire calculation completes in a
single leg. We reference ``examples/mabfe_example.yaml`` so you can reproduce the run locally
before adapting it to your own system.

Quick walkthrough
-----------------

``batter`` orchestrates an end-to-end AMBER ABFE workflow that starts from protein +
embedded protein-membrane system (if applicable) + ligand(s) (3D coordinates) overlayed to the
protein binding site. The main steps are:

#. **system staging and loading** – A executon folder will be created under ``<run.output_folder>/executions/``
   to hold all intermediate files, logs, and results. If a run ID is not provided, a timestamp-based unique ID is generated. If the same run ID already exists, the execution is
   resumed from the last successful step.
#. **Ligand parameterisation** – supports both GAFF/GAFF2 and OpenFF force fields with
   options to choose charges (AM1-BCC by default)
#. **Equilbration system preparation** – builds solvated/membrane-embedded
   systems with the ligand in the binding site.
#. **Equilibration** – Steps to run before FE production run. During this phase,
   the ligand and protein are not restrained (unless explicitly configured).
   If the ligand unbound from the binding site during equilibration, the run
   is marked as unbound and skipped during FE production.
#. **Equilibrium analysis** - Find a representative frame from the equilibrated trajectory
   to start the FE windows from. RMSD analysis is also performed and saved in the equil folder. Adjust the bound/unbound cutoff via ``fe_sim.unbound_threshold`` if your system requires a different distance threshold.
#. **FE window generation and submission** – λ windows are created based on the configuration.
#. **FE equilbration** - very short equilibration runs to allow water relaxation. If flag ``--only-equil`` is provided, the workflow stops after step 6.
#. **FE production runs** – Each window is submitted as an independent SLURM job.
   The main process monitors job status and streams updates to the terminal.
   Set ``run.max_active_jobs`` in your YAML (default 1000, ``0`` disables throttling)
   to cap how many SLURM jobs Batter keeps active at once and avoid overloading the scheduler.
#. **Analysis** – Once all windows complete, MBAR analysis is performed and
   results are summarised in CSV/JSON formats with convergence plots. The worker pool for this stage follows ``run.max_workers``; optionally limit the trajectory range per window via ``fe_sim.analysis_fe_range`` (``[start, end]`` defaults to ``[2, -1]`` or ``[0, -1]`` when ``num_fe_extends < 4``).

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

       conda env create -n batter_env python=3.12 -y
       conda env update -n batter_env -f environment.yml
       conda activate batter_env

#. Install editable copies of the bundled dependencies plus ``batter`` itself::

       pip install -e ./extern/alchemlyb
       pip install -e ./extern/rocklinc
       pip install -e .

#. Verify the installation::

       batter --help

Preparing the System
--------------------

Use ``examples/mabfe_example.yaml`` as the starting configuration. Each field is documented in
``batter.config.run``, but review the inputs below before running anything:

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
   ``system_input.pdb`` must encode the correct unit-cell vectors (box information).
   If ``system_input.inpcrd`` is provided its coordinates take precedence.
   
   The protein **does not** need to be aligned to ``protein_input.pdb`` and the alignment
   will be done automatically based on the ``create.protein_align`` config setting.

   Systems from other builders (CHARMM-GUI, Maestro, etc.) may work but are not extensively tested.

   Command to generate POPC-embedded systems with Dabble::

       dabble -i protein_input.mae -o system_input.prmtop --hmr -w 20 -O -ff charmm

   In ``batter`` preparation process, the membrane molecules will be extracted (controlled by `create.lipid_mols`);
   water and ion molecules around ``create.solv_shell`` will also be extracted.

Generating Simulation Inputs
----------------------------

1. **Copy and edit the template.**  
   Start from `examples/mabfe_example.yaml <https://github.com/yuxuanzhuang/batter/blob/main/examples/mabfe_example.yaml>`_
   and save a copy beside your project data. Update:

   - ``run.output_folder`` – dedicated directory for outputs/logs.
   - ``create.system_name`` – label used in reports.
   - ``create.ligand_input`` – JSON file mapping unique ligand IDs to ``.sdf`` files (see ``examples/ligand_dict.json``).
   - ``create.*`` paths – point at your receptor, system, membrane, and restraint files.
   - ``create.anchor_atoms`` – choose stable backbone atoms (CA/C/N) with the guidelines below.

     Anchors (P1, P2, P3) should avoid loop regions, keep P1–P2 and P2–P3 ≥ 8 Å, and target
     ∠(P1–P2–P3) near 90°.

     P1 should preferably form a consistent electrostatics interaction with available
     bound ligands (e.g., a salt bridge).

     For GPCR orthosteric sites, a common choice is P1=3x32,
     P2=2x53, P3=7x42.
   
   Additional field that may need adjustment based on your system:

   - ``run.amber_setup_command`` – command to load AMBER on your cluster (e.g., source a setup script). Default to ``source $GROUP_HOME/software/amber24/setup_amber.sh``.
   - ``run.email_on_completion`` – email address to notify when SLURM jobs complete.
   - ``run.email_sender`` – email address to send notifications from. Default to "nobody@stanford.edu" if unset.
   - ``run.slurm.partition`` – SLURM partition/queue to submit jobs to.
   - ``run.max_active_jobs`` – cap on how many SLURM jobs to keep active at once (default 1000, ``0`` disables throttling).
   

2. **Validate the configuration before heavy computation (Optional)**::

       batter run examples/mabfe_example.yaml --dry-run

   This command runs ligand parameterisation (WARNING: heavy load), and equilibration system preparation.
   On shared clusters, run the dry-run on a compute node if possible to avoid overloading login nodes.

3. **Inspect the staged system (Optional)**  
   Once the dry-run completes, review ``<run.output_folder>/executions/<run_id>/``:

   - ``simulations/<LIGAND>/equil/full.pdb`` – ligand-specific equilibration systems.
     Check if the ligand is correctly placed in the binding site,
     and that membranes/solvent boxes look reasonable.

4. **Launch the full workflow manager (local execution)**::

       batter run examples/mabfe_example.yaml

   Production runs take hours to days depending on system size, the number of ligands,
   and available hardware. Progress is streamed to the terminal and to
   ``executions/<run_id>/logs/batter.log``.

Submitting the manager job via SLURM (RECOMMENDED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit the same run through SLURM::

    batter run examples/mabfe_example.yaml --slurm-submit

Provide ``--slurm-manager-path`` if you maintain a custom SLURM header template
(accounts, modules, partitions, etc.).

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
``--run-id`` and ``--output-folder``
    Override execution paths without touching ``system.*`` fields.
``--slurm-submit`` / ``--slurm-manager-path``
    Switch between local execution and SLURM submission (with an optional custom header).

Run ``batter run --help`` anytime you need the full list of switches and defaults.

Monitoring Jobs
~~~~~~~~~~~~~~~

Keep an eye on SLURM progress with::

    batter report-jobs

It summarises queued, running, and completed windows per system. If you stop the
manager process but the SLURM jobs keep running, cancel them via::

    batter cancel-jobs --contains <system_path_reported_above>

Optional: Additional Conformational Restraints
----------------------------------------------

#. Use the restraint-generation notebook from
   `bat_mem <https://github.com/yuxuanzhuang/bat_mem/blob/main/tutorial/TEMPLATES/generate_restraints.ipynb>`_
   (or an equivalent script) to build a ``restraints.json`` describing the distance
   constraints you need.

#. Point ``create.extra_conformation_restraints`` at the resulting JSON file::

       extra_conformation_restraints: path/to/restraints.json

See ``examples/conformational_restraints`` for a full example.

Optional: Additional Positioinal Restraints
----------------------------------------------

#. Add selection string for the atoms to be positionally restraint to  ``create.extra_restraints`` at the resulting JSON file::

       extra_restraints: "selection_string"

See ``examples/extra_restraints`` for a full example.

Analysis
--------

Completed runs automatically write MBAR summaries under ``executions/<run_id>/results``.
Use the CLI helpers to inspect them::

    batter fe list <run.output_folder>
    batter fe show <run.output_folder> <run_id> --ligand <ligand>
    batter fe analyze <run.output_folder> <run_id> --ligand <ligand> --workers 4

``fe list`` prints a high-level table (ΔG, SE, protocol, originals, status) for every stored run, while
``fe show`` dives into per-window data. ``fe analyze`` re-runs the post-processing with
optional worker controls; use ``--ligand`` when the run produced multiple ligand
records. CSV/JSON exports live alongside the results on
disk, and convergence plots appear under ``results/<run_id>/<ligand>/Results``. See
:doc:`developer_guide/analysis` for deeper post-processing (MBAR diagnostics and REMD
parsing).
Use ``--no-raise-on-error`` if you want the analysis sweep to continue when individual
ligand artifacts fail to parse.

Additional Resources
--------------------

- Start from SMILES and protein sequence (with or without available structures) to absolute
binding free energy: `bat_mem <https://github.com/yuxuanzhuang/bat_mem/blob/main/tutorial/>`_

- Unsure about the protonation state of the ligand: `unipKa <https://github.com/yuxuanzhuang/batter/blob/main/scripts/get_protonation.ipynb>`_.
