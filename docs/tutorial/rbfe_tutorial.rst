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
single leg. We reference ``examples/rbfe_example.yaml`` so you can reproduce the run locally
before adapting it to your own system.

Quick walkthrough
-----------------

``batter`` orchestrates an end-to-end AMBER RBFE workflow that starts from protein +
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
#. **Network planning** – Build the RBFE transformation map (pair list) based on the selected scheme.
#. **FE window generation and submission** – λ windows are created based on the configuration.
#. **FE equilibration** - very short equilibration runs to allow water relaxation. If flag ``--only-equil`` is provided, the workflow stops after this step.
#. **FE production runs** – Each window runs as an independent local task or
   scheduler job, depending on how you launch the workflow. The main process
   monitors job status and streams updates to the terminal.
   Set ``run.max_active_jobs`` in your YAML (default 1000, ``0`` disables throttling)
   to cap how many SLURM jobs BATTER keeps active at once and avoid overloading the scheduler.
#. **Analysis** – Once all windows complete, MBAR analysis is performed and the final
   results are written to the portable ``results/`` repository.

.. _rbfe_network_planning_schemes:

Network planning schemes
------------------------

RBFE mappings can be created in a few ways:

* **Default** – maps the first ligand to all others (star topology).
* **Konnektor** – uses the ``konnektor`` library to build a network; configure with
  ``rbfe.mapping: konnektor`` and optionally ``rbfe.konnektor_layout``.
  Choose atom mapping backend via ``rbfe.atom_mapper`` (``kartograf`` or ``lomap``).
  The exact Kartograf/LoMap mapper parameters and YAML option blocks are documented in
  :ref:`rbfe_atom_mapper_options`.
  The available layouts are listed in the `Konnektor documentation <https://konnektor.openfree.energy/en/latest/api/konnektor.planners.html>`_.
  In BATTER, ``rbfe.konnektor_layout`` can be written either as the full class name
  such as ``MinimalSpanningTreeNetworkGenerator`` or as the lowercase shorthand
  ``minimalspanningtree``.
  See detailed tutorial in `Konnektor tutorial <https://konnektor.openfree.energy/en/latest/tutorials/basic_network_generation.html>`_.
* **Mapping file** – provide explicit pairs via ``rbfe.mapping_file`` (JSON/YAML list or
  text file with one pair per line).

Set ``rbfe.both_directions: true`` if you want to run both directions for every edge.
During ``prepare_rbfe``, BATTER writes
``executions/<run_id>/artifacts/config/rbfe_network.html`` plus
``artifacts/config/rbfe_mappings/<LIG1~LIG2>/mapping.*`` so the planned graph and
atom maps can be inspected before ligand equilibration and later transformation
setup.

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

#. Install ``batter`` itself after the environment update (which already installs the bundled ``extern/*`` dependencies)::

       pip install -e .

#. Verify the installation::

       batter --help

Preparing the System
--------------------

Use ``examples/rbfe_example.yaml`` as the starting configuration. Each field is documented in
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
   are correct (Open Babel, the ``scripts/get_protonation.ipynb`` notebook, or a similar tool can help).
   If you use ``rbfe.atom_mapper: kartograf`` (the BATTER default), the ligands should
   preferably be pre-aligned in a consistent binding pose, since well-aligned
   molecules are one of Kartograf's core assumptions for finding a good mapping. See
   the `Kartograf mapping tutorial <https://kartograf.openfree.energy/en/latest/tutorial/mapping_tutorial.html>`_
   for the upstream guidance.

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
   Start from ``examples/rbfe_example.yaml``
   and save a copy beside your project data. Update:

   - ``run.output_folder`` – dedicated directory for outputs/logs.
   - ``create.system_name`` – label used in reports.
   - ``create.ligand_input`` – JSON file mapping unique ligand IDs to ``.sdf`` files (see ``examples/reference/ligand_dict.json``).
   - ``create.*`` paths – point at your receptor, system, membrane, and restraint files.
   - ``create.anchor_atoms`` – Optional three atoms that define the binding site and
     anchor geometry used during staging and validation. If omitted, BATTER
     auto-selects stable backbone anchors from the first ligand pose with the
     guidelines below.

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
   - ``rbfe.mapping`` / ``rbfe.mapping_file`` – choose your network planning scheme.
   - ``rbfe.atom_mapper`` – choose RBFE atom mapper backend: ``kartograf`` (default) or ``lomap``.

      The available schemes are described in :ref:`rbfe_network_planning_schemes`.
      Mapper options can be overridden under ``rbfe.kartograf`` and ``rbfe.lomap``;
      see :ref:`rbfe_atom_mapper_options` for the accepted keys and defaults.
      For mapper-specific behavior and examples, see the `Kartograf documentation <https://kartograf.openfree.energy/en/latest/>`_
      and the `LoMap documentation <https://lomap.openfree.energy/en/stable/>`_.
      As a practical default, start with ``kartograf`` unless you have a reason to prefer
      ``lomap`` for a particular ligand series. ``lomap`` remains available and can
      still be a better fit for some chemotypes or mapping preferences.

   Use :doc:`../cookbook/configuration` for the full YAML field reference and
   :doc:`../cookbook/rbfe` for the RBFE-specific mapping examples and defaults. If you
   plan to submit through Slurm, also review :doc:`../cookbook/slurm_headers`.

2. **Validate the configuration before heavy computation (Optional)**::

       batter run examples/rbfe_example.yaml --dry-run

   This command runs ligand parameterisation (a heavy step) and prepares the
   equilibration systems.
   On shared clusters, run the dry-run on a compute node if possible to avoid overloading login nodes.

3. **Inspect the staged system (Optional)**
   Once the dry-run completes, review ``<run.output_folder>/executions/<run_id>/``:

   - ``simulations/<LIGAND>/equil/full.pdb`` – ligand-specific equilibration systems.
     Check if the ligand is correctly placed in the binding site,
     and that membranes/solvent boxes look reasonable.

4. **Launch the full workflow manager (local execution)**::

       batter run examples/rbfe_example.yaml

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

    batter run examples/rbfe_example.yaml --slurm-submit

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

Results and Analysis
--------------------

Completed runs automatically write MBAR summaries under ``results/<run_id>``.
For RBFE runs, per-run analysis also writes a Cinnabar bundle under
``results/cinnabar/<run_id>/``. The most direct ways to inspect those outputs are:

* Open ``results/cinnabar/<run_id>/cinnabar_dashboard.html`` in a browser. That
  dashboard includes the network view, the absolute ranking view, and the clickable
  ligand / mapping panels.
* Read ``edge_summary.csv`` when you want the combined edge-level ``ΔΔG`` table.
* Read ``cinnabar_relative.csv`` and ``cinnabar_absolute.csv`` when you want the
  FEMap-exported relative and absolute values.
* Open ``cinnabar_network.png`` and ``cinnabar_absolute_sorted.png`` for static
  figures suitable for slides or quick sharing.
* Use the `RBFE Cinnabar analysis notebook
  <https://github.com/yuxuanzhuang/batter/blob/main/examples/plot_rbfe.ipynb>`_
  when you want notebook-based tables, plots, and optional experimental
  comparisons.

If you later merge multiple RBFE runs with ``batter fe cinnabar``, the combined
bundle is written separately from the per-run subdirectory. Same-work-dir
replicates default to ``results/cinnabar/``; cross-work-dir combinations should
use an explicit ``--out-dir``.

Use the CLI helpers to inspect them::

    batter fe list <run.output_folder>
    batter fe show <run.output_folder> <run_id> --ligand <ligand_pair>

For cross-run RBFE benchmarking or Cinnabar plotting, convert stored BATTER
records into a Cinnabar bundle. The recommended form treats each run as an atomic
``WORK_DIR`` + ``RUN_ID`` input, so runs from different work directories can be
combined:

.. code-block:: console

    batter fe cinnabar \
        --run work/adrb2 rep1 \
        --run work/adrb2_retry rep2 \
        --out-dir combined_cinnabar

Per-run RBFE analysis already writes a default bundle under
``results/cinnabar/<run_id>/``. Use explicit ``--run`` inputs when you want to
merge replicate runs into one Cinnabar view. If all runs are in the same work
directory, this shortcut is equivalent::

    batter fe cinnabar <run.output_folder> --run-id rep1 --run-id rep2

The same workflow is available from Python via
:func:`batter.analysis.cinnabar.build_batter_rbfe_cinnabar_from_runs`. This is the
function to use when you want to combine replicate run ids programmatically or
connect networks from different work directories. BATTER matches ligand endpoints
by ligand name plus canonical SMILES: matching name/SMILES pairs merge into one
node, while same-name but different-SMILES endpoints remain separate suffixed
nodes.

See :doc:`cinnabar` for the dedicated Cinnabar workflow page, including the
default per-run output layout and the Python API for combined replicate bundles.

Those commands read the saved ``results/index.csv`` rows, combine the selected
RBFE edges, and write a derived bundle. Use ``--split-runs`` only with the
same-work-dir shortcut if you want one bundle per run instead of collapsing
repeats.
If you have experimental absolute affinities, pass them with
``--experimental-csv`` so Cinnabar can emit DG/DDG comparison plots. BATTER merges
``A~B`` and ``B~A`` into one canonical edge by default; add ``--split-directions``
if you want to keep the two stored directions separate in the Cinnabar export.
BATTER also writes ``cinnabar_absolute_sorted.png`` from the Cinnabar MLE absolute
values; use ``--absolute-offset`` if you want to shift that ranking plot onto a
chosen absolute reference level.

``fe list`` prints a high-level table for every stored run, while ``fe show`` opens
the saved record for one transformation pair such as ``LIG1~LIG2``. For a file-by-file
description of the portable repository, including the RBFE-only ``mapping.*``,
``rbfe_network.png``, ``rbfe_network.html``, and ``Equil_ref`` / ``Equil_alt``
exports, see
:doc:`../cookbook/results_folder`.

For final error estimation, it is usually better to run three independent repeats
of the full simulation and estimate the uncertainty across those replicate runs,
rather than relying only on the per-run bootstrap uncertainty from a single run.
The per-run bootstrapping remains useful as a within-run diagnostic, but it should
not be treated as a substitute for repeat-run error estimation.

BATTER does not apply any automatic symmetry correction to the reported free
energies. If your transformation needs a symmetry correction, inspect the end
states and add that correction separately when interpreting the final result.

Lambda-Schedule Tuning
----------------------

If you already know the approximate number of windows your ligand series needs, you
can keep that count fixed and use ``batter fek-schedule`` to optimize the spacing.
The current recipe is documented in :doc:`../cookbook/fek_schedule`.

For the small-molecule RBFE cases documented so far, 24 windows often seem to be
enough, using a simple evenly spaced schedule:

.. code-block:: yaml

   lambdas: [0.0, 0.04347826, 0.08695652, 0.13043478, 0.17391304,
             0.2173913, 0.26086957, 0.30434783, 0.34782609, 0.39130435,
             0.43478261, 0.47826087, 0.52173913, 0.56521739, 0.60869565,
             0.65217391, 0.69565217, 0.73913043, 0.7826087, 0.82608696,
             0.86956522, 0.91304348, 0.95652174, 1.0]

For more complex transformations, 48 windows has worked well in testing:

.. code-block:: yaml

   lambdas: [0.00000000, 0.12542000, 0.16637000, 0.19653000, 0.22148000, 0.24326000,
             0.26289000, 0.28094000, 0.29779000, 0.31370000, 0.32884000, 0.34336000,
             0.35737000, 0.37095000, 0.38416000, 0.39707000, 0.40971000, 0.42215000,
             0.43441000, 0.44652000, 0.45852000, 0.47043000, 0.48228000, 0.49410000,
             0.50590000, 0.51772000, 0.52958000, 0.54150000, 0.55351000, 0.56563000,
             0.57790000, 0.59036000, 0.60303000, 0.61596000, 0.62920000, 0.64280000,
             0.65684000, 0.67140000, 0.68659000, 0.70254000, 0.71944000, 0.73754000,
             0.75722000, 0.77906000, 0.80408000, 0.83431000, 0.87533000, 1.00000000]
