REMD submission flow
====================

When ``sim.remd`` is enabled, production jobs are submitted per-component rather than per-window.
Key behaviours:

* Under ``fe/<comp>/``, REMD groupfiles live in ``remd/`` with component-agnostic names
  (``mdin.in.remd.groupfile``, ``mdin.in.stageXX.remd.groupfile``).
* The run script ``run-local-remd.bash`` tracks completion with ``remd_FINISHED/FAILED``
  and expects window folders ``<comp>00/`` etc. for restarts.
* The Slurm submit script for REMD is ``SLURMM-BATCH-remd`` sitting in the component folder.
  Submissions use this script directly and set the finished/failed sentinels to the REMD files.
* Job names are formatted as ``fep_<system_root>_<comp>_remd`` so CLI reporting
  (:func:`batter.cli.run.report_jobs`) can classify them with stage ``remd``.

Keep these conventions in mind if modifying the REMD pipeline or backends.
