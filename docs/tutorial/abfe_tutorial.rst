.. _abfe_tutorial:

ABFE Tutorial
=============

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
embedded protein-membrane system (if applicable) + ligand(s) (3D coordinates) overlaid to the
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
   If the ligand unbinds from the binding site during equilibration, the run
   is marked as unbound and skipped during FE production.
#. **Equilibrium analysis** - Find a representative frame from the equilibrated trajectory
   to start the FE windows from. RMSD analysis is also performed and saved in the equil folder. Adjust the bound/unbound cutoff via ``fe_sim.unbound_threshold`` if your system requires a different distance threshold.
#. **FE window generation and submission** – λ windows are created based on the configuration.
#. **FE equilibration** - very short equilibration runs to allow water relaxation. If flag ``--only-equil`` is provided, the workflow stops after this step.
#. **FE production runs** – Each window is submitted as an independent SLURM job.
   The main process monitors job status and streams updates to the terminal.
   Set ``run.max_active_jobs`` in your YAML (default 1000, ``0`` disables throttling)
   to cap how many SLURM jobs BATTER keeps active at once and avoid overloading the scheduler.
#. **Analysis** – Once all windows complete, MBAR analysis is performed and the final
   results are written to the portable ``results/`` repository.

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

Use ``examples/mabfe_example.yaml`` as the starting configuration. Each field is documented in
:doc:`../cookbook/configuration`, but review the inputs below before running anything:

Required Files
~~~~~~~~~~~~~~

1. **Protein structure** – ``protein_input.pdb``  
   It can be prepared in Maestro or equivalent software. Protonation states are
   inferred from residue names using AMBER conventions (for example, ASH denotes
   protonated ASP). When explicit hydrogens are present, BATTER also uses them to
   distinguish protonation states.
   Water or non-protein small-molecule coordinates may remain in the file; they are
   stripped during staging. BATTER currently does not support cofactors or other
   non-protein residues in ``protein_input.pdb``.

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

   In ``batter`` preparation process, the membrane molecules will be extracted (controlled by ``create.lipid_mol``);
   water and ion molecules around ``create.solv_shell`` will also be extracted.

Generating Simulation Inputs
----------------------------

1. **Copy and edit the template.**  
   Start from `examples/mabfe_example.yaml <https://github.com/yuxuanzhuang/batter/blob/main/examples/mabfe_example.yaml>`_
   and save a copy beside your project data. Update:

   - ``run.output_folder`` – dedicated directory for outputs/logs.
   - ``create.system_name`` – label used in reports.
   - ``create.ligand_input`` – JSON file mapping unique ligand IDs to ``.sdf`` files (see ``examples/reference/ligand_dict.json``).
   - ``create.*`` paths – point at your receptor, system, membrane, and restraint files.
   - ``create.anchor_atoms`` – choose stable backbone atoms (CA/C/N) with the guidelines below.

     Anchors (P1, P2, P3) should avoid loop regions, keep P1–P2 and P2–P3 ≥ 8 Å, and target
     ∠(P1–P2–P3) near 90°.

     P1 should preferably form a consistent electrostatics interaction with available
     bound ligands (e.g., a salt bridge).

     For GPCR orthosteric sites, a common choice is P1=3x32,
     P2=2x53, P3=7x42.
   
   Additional field that may need adjustment based on your cluster environment:

   - ``run.email_on_completion`` – email address to notify when the BATTER manager finishes or aborts with an uncaught failure.
   - ``run.email_sender`` – sender address for those notifications. Defaults to ``nobody@stanford.edu``.
   - ``run.slurm.partition`` – SLURM partition/queue to submit jobs to.
   - ``run.max_active_jobs`` – cap on how many SLURM jobs to keep active at once (default 1000, ``0`` disables throttling).

   Use :doc:`../cookbook/configuration` for the full YAML field reference. If you plan
   to submit through Slurm, also review :doc:`../cookbook/slurm_headers` before the
   first production run.

2. **Validate the configuration before heavy computation (Optional)**::

       batter run examples/mabfe_example.yaml --dry-run

   This command runs ligand parameterisation (a heavy step) and prepares the
   equilibration systems.
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

Before submitting, make sure the SLURM header files have been seeded and edited for
your cluster. BATTER stores them in ``~/.batter/`` by default (or under
``run.slurm_header_dir`` if you configured a custom location). In particular,
update ``job_manager.header`` and ``SLURMM-Am.header`` so they load Amber/AmberTools
successfully and match your site environment (modules, conda activation, partitions,
MPI launcher, executable paths, account settings, etc.). If you plan to run REMD,
also review ``SLURMM-BATCH-remd.header``. The dedicated
:doc:`../cookbook/slurm_headers` page summarizes what each header controls and how
the seeded files relate to the packaged script bodies.

Seed the default headers if needed::

    batter seed-headers

To submit the same run through SLURM::

    batter run examples/mabfe_example.yaml --slurm-submit

Provide ``--slurm-manager-path`` if you maintain a custom SLURM header template
(accounts, modules, partitions, etc.). Copy and modify the default template from
``batter/data/job_manager.header`` + ``job_manager.body``. See :doc:`../cookbook/slurm_headers`
for the full header layout and override rules.

The job manager stages the system locally,
writes an ``sbatch`` script based on the YAML hash, and streams updates as windows
finish.

Handy CLI Flags
---------------

``batter run`` exposes many overrides so you rarely have to edit YAML mid-iteration:

``--on-failure {prune,raise,retry}``
    Decide how to handle per-ligand failures. ``retry`` clears ``FAILED`` sentinels and reruns that phase once.
``--clean-failures / --no-clean-failures``
    Remove ``FAILED`` sentinels, ``job_attempt.txt`` retry counters, and progress caches before rerunning a previous execution.
``--only-equil / --full``
    Stop after shared prep/equilibration—useful for debugging system setup before FE windows.
``--dry-run``
    Stage the system and prepare equilibration inputs without running any MD.
``--run-id`` and ``--output-folder``
    Override execution paths without touching ``system.*`` fields.
``--slurm-submit`` / ``--slurm-manager-path``
    Switch between local execution and SLURM submission (with an optional custom header).

Some failures are transient cluster issues rather than setup problems, for example a
job landing on a bad node or hitting a temporary GPU/filesystem problem. In that
case, rerun the same command with ``--clean-failures`` to clear stale failure
markers before resuming. If you want BATTER to clear phase sentinels and retry once
within the run manager, use ``--on-failure retry``.

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
   `bat_mem restraint notebook <https://github.com/yuxuanzhuang/bat_mem/blob/main/tutorial/TEMPLATES/generate_restraints.ipynb>`_
   or an equivalent script to build a ``restraints.json`` describing the distance
   constraints you need.

#. Point ``create.extra_conformation_restraints`` at the resulting JSON file::

       extra_conformation_restraints: path/to/restraints.json

See ``examples/conformational_restraints`` for a full example.

Optional: Additional Positional Restraints
----------------------------------------------

#. Add a selection string for the atoms to be positionally restrained in
   ``create.extra_restraints``::

       extra_restraints: "selection_string"

See ``examples/extra_restraints`` for a full example.

Analysis
--------

Completed runs automatically write MBAR summaries under ``results/<run_id>``.
Use the CLI helpers to inspect them::

    batter fe list <run.output_folder>
    batter fe show <run.output_folder> <run_id> --ligand <ligand>

``fe list`` prints a high-level table (ΔG, SE, protocol, originals, status) for every stored run, while
``fe show`` dives into per-window data; use ``--ligand`` when the run produced multiple ligand
records. CSV/JSON exports live alongside the results on disk, and convergence plots
appear under ``results/<run_id>/<ligand>/Results``. See
:doc:`../developer_guide/analysis` for deeper post-processing (MBAR diagnostics and REMD
parsing). For a file-by-file description of the portable repository written under
``<run.output_folder>/results/``, see :doc:`../cookbook/results_folder`.

For final error estimation, it is usually better to run three independent repeats
of the full simulation and estimate the uncertainty across those replicate runs,
rather than relying only on the per-run bootstrap uncertainty from a single run.
The per-run bootstrapping remains useful as a within-run diagnostic, but it should
not be treated as a substitute for repeat-run error estimation.

BATTER does not apply any automatic symmetry correction to the reported free
energies. If your ligand or restraint setup requires a symmetry correction, inspect
the relevant states and add that correction separately when interpreting the final
result.

Additional Resources
--------------------

- Start from SMILES and protein sequence (with or without available structures) to absolute
  binding free energy: `bat_mem <https://github.com/yuxuanzhuang/bat_mem/blob/main/tutorial/>`_

- Unsure about the protonation state of the ligand: `unipKa <https://github.com/yuxuanzhuang/batter/blob/main/scripts/get_protonation.ipynb>`_.

Lambda-Schedule Tuning
----------------------

If you already have a good estimate for how many lambda windows your system needs,
you can keep that window count fixed and use ``batter fek-schedule`` to optimize
the spacing. The current recipe is documented in :doc:`../cookbook/fek_schedule`.

The cookbook example is written for an RBFE transformation path, but the same idea
applies to ABFE components once you point ``batter fek-schedule`` at the relevant
analysis-ready FE directory. In practice, this is most useful after an initial pilot
run has shown that you want to keep the same total number of windows but redistribute
them more efficiently.

48 windows has worked well in testing:

.. code-block:: yaml

   lambdas: [0.00000000, 0.12542000, 0.16637000, 0.19653000, 0.22148000, 0.24326000,
             0.26289000, 0.28094000, 0.29779000, 0.31370000, 0.32884000, 0.34336000,
             0.35737000, 0.37095000, 0.38416000, 0.39707000, 0.40971000, 0.42215000,
             0.43441000, 0.44652000, 0.45852000, 0.47043000, 0.48228000, 0.49410000,
             0.50590000, 0.51772000, 0.52958000, 0.54150000, 0.55351000, 0.56563000,
             0.57790000, 0.59036000, 0.60303000, 0.61596000, 0.62920000, 0.64280000,
             0.65684000, 0.67140000, 0.68659000, 0.70254000, 0.71944000, 0.73754000,
             0.75722000, 0.77906000, 0.80408000, 0.83431000, 0.87533000, 1.00000000]
