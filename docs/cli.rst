=====================
Command-Line Usage
=====================

BATTER ships with a ``batter`` command-line tool powered by ``click``. This page
summarises the key commands and options. Run ``batter --help`` or append ``--help`` to
any subcommand for the full syntax.

Run Workflows
=============

.. code-block:: console

   batter run examples/mabfe.yaml

Options:

``--on-failure {prune,raise,retry}``
   Control how ligand failures are handled (default: ``raise``). ``retry`` clears FAILED sentinels and reruns that phase once.
``--output-folder PATH``
   Override the system output directory.
``--run-id TEXT``
   Override the execution run id (use ``auto`` to reuse the latest).
``--dry-run / --no-dry-run``
   Toggle ``run.dry_run`` from the YAML.
``--only-equil / --full``
   Run only equilibration preparation steps. FE preparation is still performed (up to
   ``prepare_fe_windows``), but the FE equilibration/production/analyse phases are skipped.
``--slurm-submit``
   Emit an ``sbatch`` script and submit the job instead of running locally.

Resume an existing execution (no YAML needed once seeded)::

   batter run-exec work/adrb2/executions/rep1

Notes:

* The first ``batter run`` stores a copy of the YAML plus any external restraint files (e.g.,
  ``extra_conformation_restraints``) under ``artifacts/config/``. ``run-exec`` reuses that copy.

Generate REMD Batch Scripts
===========================

Use ``batter remd-batch`` to emit an ``sbatch`` script that runs ``run-local-remd.bash``
across one or more execution folders::

   batter remd-batch -e work/adrb2/executions/rep1 -e work/adrb2/executions/rep2

The command skips component folders that already contain ``FINISHED`` and writes the
script to the current directory. Key options:

``--gpus``
   Total GPUs to request (defaults to the total window count detected).
``--gpus-per-node``
   GPUs available per node (default: 8). Used to size per-task node allocations when
   ``MPI_EXEC`` is ``srun``.
``--nodes``
   Override the total node count in the header.
``--auto-resubmit`` / ``--no-auto-resubmit``
   When enabled (default), the generated sbatch traps a pre-timeout signal,
   regenerates the remd-batch script, and resubmits it until all components finish
   or the max resubmission window is reached.
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

Use ``--workers`` to control parallelism and ``--sim-range`` (``start,end``) to restrict
which windows are parsed.
Pass ``--no-raise-on-error`` to continue even if a ligand's analysis fails.

Clone Executions
================

Duplicate the directory structure of an execution (useful for what-if experiments)::

   batter clone-exec work/adrb2 run-20240101 run-20240101-test --mode symlink --only-equil

Only metadata, inputs/params, equilibration artifacts, and a minimal ``fe/`` scaffold are copied,
so the clone can start new simulations without recreating large FE dumps.

SLURM Utilities
===============

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
