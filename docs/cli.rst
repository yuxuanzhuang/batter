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

``--on-failure {prune,raise}``
   Control how ligand failures are handled (default: ``raise``).
``--output-folder PATH``
   Override the system output directory.
``--run-id TEXT``
   Override the execution run id (use ``auto`` to reuse the latest).
``--dry-run / --no-dry-run``
   Toggle ``run.dry_run`` from the YAML.
``--only-equil / --full``
   Run only equilibration preparation steps.
``--slurm-submit``
   Emit an ``sbatch`` script and submit the job instead of running locally.

Inspect Free-Energy Results
===========================

List results stored in a portable work directory::

   batter fe list work/adrb2

Use ``--format`` to output JSON/CSV/TSV instead of a table.

Display a specific FE record::

   batter fe show work/adrb2 run-20240101

Clone Executions
================

Duplicate the directory structure of an execution (useful for what-if experiments)::

   batter clone-exec work/adrb2 run-20240101 run-20240101-test --symlink --only-equil

SLURM Utilities
===============

* ``batter report-jobs`` – Summarise SLURM jobs launched by BATTER. Use ``--detailed``
  to show per-job information and ``--partition`` to filter by queue.
* ``batter cancel-jobs --contains TEXT`` – Cancel SLURM jobs whose name contains the
  supplied substring (matches the ``fep_...`` job names produced by BATTER).

Environment Notes
=================

* The CLI uses :mod:`batter.api` and is safe to call from activated conda/virtualenv
  installations.
* ``batter run`` validates the YAML before invoking the orchestration layer and emits
  friendly :class:`click.ClickException` error messages.
* When submitting to SLURM, the generated script is named
  ``<hash>_job_manager.sbatch`` using a hash of the YAML contents and overrides.
