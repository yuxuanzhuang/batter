=====================
Command-Line Usage
=====================

BATTER ships with a ``batter`` command-line tool powered by ``click``. This page
summarises the key commands and options. Run ``batter --help`` or append ``--help`` to
any subcommand for the full syntax.

Run Workflows
=============

.. code-block:: console

   batter run examples/mabfe_example.yaml

Options:

``--on-failure {prune,raise,retry}``
   Control how ligand failures are handled (default: ``raise``). ``retry`` clears FAILED sentinels and reruns that phase once.
``--output-folder PATH``
   Override the system output directory.
``--run-id TEXT``
   Override the execution run id (use ``auto`` to reuse the latest).
``--allow-run-id-mismatch / --no-allow-run-id-mismatch``
   Allow reuse of a run id even when the stored configuration hash differs.
``--dry-run / --no-dry-run``
   Toggle ``run.dry_run`` from the YAML.
``--clean-failures / --no-clean-failures``
   Clear ``FAILED`` sentinels, ``job_attempt.txt`` retry counters, and progress caches before rerunning an execution.
``--only-equil / --full``
   Run only equilibration preparation steps. FE preparation is still performed (up to
   ``prepare_fe_windows``), but the FE equilibration/production/analyse phases are skipped.
``--slurm-submit``
   Emit an ``sbatch`` script and submit the job instead of running locally.
``--slurm-manager-path PATH``
   Optional path to a custom SLURM header/body pair for the manager submission.
``--partition / -p``
   Override ``run.slurm.partition`` from the YAML.

Resume an existing execution (no YAML needed once seeded)::

   batter run-exec work/adrb2/executions/rep1

Notes:

* The first ``batter run`` stores a copy of the YAML plus any external restraint files (e.g.,
  ``extra_conformation_restraints``) under ``artifacts/config/``. ``run-exec`` reuses that copy.

Generate Batch Scripts
======================

Use ``batter batch`` to emit an ``sbatch`` script that runs ``run-local-batch.bash`` (non-REMD)
across one or more execution folders::

   batter batch -e work/adrb2/executions/rep1 -e work/adrb2/executions/rep2

Use ``--remd`` to switch to REMD mode (runs ``run-local-remd.bash``)::

   batter batch --remd -e work/adrb2/executions/rep1 -e work/adrb2/executions/rep2

The command writes ``run-local-batch.bash`` into each component folder using the packaged
template and skips components that already contain ``FINISHED`` (or where all windows
are marked ``FINISHED``).
Key options:

``--gpus``
   Total GPUs to request (defaults to the total window count detected).
``--gpus-per-node``
   GPUs available per node (default: 8). Used to size per-task node allocations when
   ``MPI_EXEC`` is ``srun``.
``--nodes``
   Override the total node count in the header.
``--remd``
   Use REMD execution mode (``run-local-remd.bash``) instead of standard batch mode.
``--auto-resubmit`` / ``--no-auto-resubmit``
   When enabled (default), the generated sbatch traps a pre-timeout signal,
   regenerates the batch script, and resubmits it until all components finish
   or the max resubmission count is reached.
``--signal-mins``
   Minutes before the time limit to trigger auto-resubmit (default: 90).
``--max-resubmit-count``
   Maximum total submissions for the script (including the first run; default: 4).
``--current-submission-time``
   Internal counter for auto-resubmit; increments on each resubmission (default: 0).

Inspect Free-Energy Results
===========================

List results stored in a portable work directory::

   batter fe list work/adrb2

Use ``--format`` to output JSON/CSV/TSV instead of a table.

Display a specific FE record::

   batter fe show work/adrb2 run-20240101 --ligand LIG1

If the run contains multiple ligands, pass ``--ligand`` to disambiguate.

Re-run FE analysis for a saved execution::

   batter fe analyze work/adrb2 run-20240101 --ligand LIG1 --workers 4

To analyze every run under ``work/adrb2/executions`` (instead of one run), omit
``run_id``::

   batter fe analyze work/adrb2 --workers 4

Use ``--workers`` to control parallelism and ``--analysis-start-step`` to skip early
production steps in each window. By default existing analysis outputs are preserved;
pass ``--overwrite`` to regenerate them. Pass ``--n-bootstrap`` to request MBAR
bootstrap resamples and ``--no-raise-on-error`` to continue if one ligand fails.

For RBFE runs, ``batter fe analyze`` also writes a per-run Cinnabar bundle under
``work/adrb2/results/cinnabar/<run_id>/`` by default. When the work directory
contains replicate RBFE runs, BATTER prints a follow-up note with the matching
``batter fe cinnabar ... --run-id ...`` command to combine same-work-dir
replicates into one bundle.

Re-run FE analysis for exactly one ligand folder::

   batter fe ligand-analyze work/adrb2/executions/run-20240101/simulations/LIG1 --overwrite

``ligand-analyze`` also accepts directories outside ``executions/<run_id>`` as long
as they contain an ``fe/`` folder.

Interactively enable or disable stored FE rows for aggregate analysis::

   batter fe analysis-inclusion work/adrb2

This command edits the ``include_in_analysis`` flag in ``results/index.csv``.
Rows set to ``False`` are preserved on disk but skipped by Cinnabar export and
other aggregate analyses.

Convert stored RBFE records into `Cinnabar <https://cinnabar.openfree.energy/en/latest/>`_
inputs and plots. Prefer explicit atomic run inputs when collecting production
replicates, especially if the runs may live in different BATTER work directories::

   batter fe cinnabar \
       --run work/adrb2 rep1 \
       --run work/adrb2_retry rep2 \
       --out-dir combined_cinnabar

If all selected runs are in the same work directory, the older shortcut remains
available::

   batter fe cinnabar work/adrb2 --run-id rep1 --run-id rep2

By default this aggregates the selected RBFE records into a single FEMap. For the
same-work-dir shortcut, BATTER writes under ``work/adrb2/results/cinnabar/``. For
explicit ``--run`` inputs from multiple work directories, pass ``--out-dir``; if it
is omitted, BATTER writes to ``./cinnabar``. Common files include:

* ``edge_summary.csv`` – combined edge-level DDG estimates and uncertainties
* ``raw_signed.csv`` – signed per-measurement table after BATTER canonicalizes edge direction
* ``cinnabar_relative.csv`` – relative measurements exported from the FEMap
* ``cinnabar_absolute.csv`` – MLE-derived absolute values when the network is connected
* ``cinnabar_absolute_sorted.png`` – BATTER-rendered absolute ΔG ranking plot, sorted by energy
* ``cinnabar_network.png`` – best-effort network visualisation
* ``cinnabar_dg.png`` / ``cinnabar_ddg.png`` – plots when experimental data is provided

Use ``--split-runs`` with the same-work-dir shortcut to emit one Cinnabar bundle
per run instead of combining them::

   batter fe cinnabar work/adrb2 --split-runs --run-id rep1 --run-id rep2

This writes bundles under ``work/adrb2/results/cinnabar/<run_id>/``. Use that mode
when you want to inspect run-to-run variation directly instead of collapsing repeats.

To compare BATTER RBFE results against experiment, pass a CSV with absolute
affinities::

   batter fe cinnabar work/adrb2 \
       --experimental-csv experimental.csv \
       --exp-ligand-column ligand \
       --exp-abfe-column abfe \
       --exp-error-column se

Use ``--combine-by-run-first`` (default) to collapse repeated measurements within
each run before combining runs. Switch to ``--pool-all-measurements`` if you want to
weight every stored edge measurement directly. ``--uncertainty-mode`` controls the
repeat-combination rule (``ivw``, ``sample``, or ``max``).

By default BATTER also merges opposite-direction rows such as ``LIGA~LIGB`` and
``LIGB~LIGA`` into one canonical edge before constructing the FEMap. Use
``--split-directions`` if you want those two stored transformations to remain
separate directional measurements in the exported Cinnabar bundle.

``--absolute-offset`` adds a constant shift to the computed MLE absolute energies in
``cinnabar_absolute_sorted.png``. This is useful when you want to place the
arbitrary RBFE-derived absolute scale onto a chosen reference level before
comparing or presenting rankings.

Clone Executions
================

Duplicate the directory structure of an execution (useful for what-if experiments)::

   batter clone-exec work/adrb2 run-20240101 run-20240101-test --mode symlink --only-equil

Only metadata, inputs/params, equilibration artifacts, and a minimal ``fe/`` scaffold are copied,
so the clone can start new simulations without recreating large FE dumps.

SLURM Utilities
===============

* ``batter seed-headers`` – Seed packaged SLURM headers into ``~/.batter`` (or
  a custom directory via ``--dest``). Use ``--force`` to overwrite existing headers.
* ``batter diff-headers`` – Show a unified diff between your headers and the packaged
  defaults (use ``--dest`` to point at a custom header directory).
* ``batter report-jobs`` – Summarise SLURM jobs launched by BATTER. Use ``--detailed``
  to show per-job information and ``--partition`` to filter by queue.
* ``batter cancel-jobs --contains TEXT`` – Cancel SLURM jobs whose name contains the
  supplied substring (matches the ``fep_...`` job names produced by BATTER).

FE Toolkit Schedules
====================

BATTER wraps the ``fetkutils-tischedule.py`` script from AMBERTOOLS via ``batter fek-schedule`` so
you can optimise or analyse lambda schedules without leaving the main CLI.

.. code-block:: console

   batter fek-schedule \
       --opt 96 \
       --ar --ssc --sym \
       --start 4 --stop 8 \
       -T 310 \
       --out sched.ar.z.dat \
       --plot sched.ar.z.png \
       ADRB2_I/rep1/fe/pose0/sdr/z

Key options:

``--opt N`` / ``--read FILE``
   Optimise a fresh schedule with ``N`` lambda values or analyse an existing schedule.
``--pso`` | ``--ar`` | ``--kl``
   Choose the optimisation metric (phase-space overlap, replica-exchange ratio, or exp(-KL)).
``--ssc`` / ``--sym`` / ``--alpha0`` / ``--alpha1``
   Restrict optimisation to SSC(alpha) families, optionally symmetric around 0.5.
``--plot PATH`` / ``--out PATH``
   Save the interpolated heatmap and the digitised schedule to disk.

All options mirror the original script, so existing workflows can switch to the new command
without rewriting job scripts. Use ``--verbose`` to surface the diagnostic output emitted by
the underlying ``fetkutils`` routines.

Environment Notes
=================

* The CLI uses :mod:`batter.api` and is safe to call from activated conda/virtualenv
  installations.
* ``batter run`` validates the YAML before invoking the orchestration layer.
* When submitting to SLURM, the generated manager script is rendered from
  ``job_manager.header``/``job_manager.body`` into ``<hash>_job_manager.sbatch`` using a
  hash of the YAML contents and overrides.
