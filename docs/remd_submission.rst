REMD submission flow
====================

When ``run.remd`` is enabled, production jobs are submitted per-component rather than
per-window. REMD input files are always prepared; the toggle only controls whether they
are submitted.
Key behaviours:

* Under ``fe/<comp>/``, REMD groupfiles live in ``remd/`` with component-agnostic names
  (``mdin.in.remd.groupfile``, ``mdin.in.stageXX.remd.groupfile``).
* The run script ``run-local-remd.bash`` tracks completion with standard
  ``FINISHED``/``FAILED`` sentinels and expects window folders ``<comp>00/`` etc. for restarts.
* The Slurm submit script for REMD is ``SLURMM-BATCH-remd`` sitting in the component folder.
  Submissions use this script directly and rely on the same sentinels above.
* Job names are formatted as ``fep_<system_root>_<comp>_remd`` so CLI reporting
  (:func:`batter.cli.run.report_jobs`) can classify them with stage ``remd``.

Keep these conventions in mind if modifying the REMD pipeline or backends.
