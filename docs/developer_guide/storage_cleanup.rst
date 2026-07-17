==========================
Storage Cleanup Semantics
==========================

BATTER can prune generated scratch files during successful workflow stages to keep
large production runs from exhausting inode and storage quotas. The cleanup code lives
in :mod:`batter._internal.ops.cleanup` and is controlled by the run-level
``store_debug_files`` flag.

Configuration
=============

Cleanup is enabled by default::

   run:
     store_debug_files: false

Set ``store_debug_files: true`` to preserve all intermediate build, LEaP, ParmEd,
runtime, and analysis scratch files. This is useful while developing new builders,
debugging AMBER setup failures, or comparing generated topology/coordinate staging.

Cleanup only runs at stage boundaries where the downstream files are already present.
It should not delete files needed by the current running job. In particular, production
``md-*`` files and REST ``cmass*`` traces are always preserved.

Global keep rules
=================

The cleanup helpers preserve these classes of files across stages:

* status and resume markers such as ``FINISHED``, ``FAILED``, ``UNBOUND``,
  ``JOBID``, ``EQ_FINISHED``, ``job_attempt.txt`` and ``production-start.ps``;
* run scripts and templates such as ``run-local.bash``, ``run-local-batch.bash``,
  ``run-local-remd.bash``, ``check_run.bash``, ``SLURMM-run``,
  ``SLURMM-BATCH-remd``, ``lambda.sch``, ``mdin-template``,
  ``mdin-batch-template`` and ``mdin-remd-template``;
* final topology/coordinate inputs such as ``full.prmtop``, ``full.inpcrd``,
  ``full.hmr.prmtop``, ``full_merged.prmtop`` and ``full.pdb``;
* restraint and component metadata such as ``disang.rest``, ``cv.in``,
  ``restraints.in``, ``sdr_info.txt`` and ``anchors*.txt/json``;
* all files whose name starts with ``md-``;
* all files whose name starts with ``cmass``;
* analysis artifacts such as ``representative.*``, ``equil-reference.pdb``,
  ``equilibration_analysis_results.npz``, ``simulation_analysis.png``, ProLIF
  outputs and stable-pair metadata;
* FE result files, including ``Results.dat``, ``*_results.json``,
  ``*_results.pickle``, ``*_df_list.pickle``, convergence plots, and
  ``fe_timeseries.*``.

Intermediate ``eq*.rst7`` files are treated specially. Cleanup preserves the restart
needed by the next stage and removes older equilibration restarts:

* equilibration analysis keeps the highest-priority final equilibration restart,
  usually ``eqnpt_appear.rst7`` for full equilibration workflows;
* FE and pre-FE ``<comp>-1`` equilibration keeps ``eqnpt04.rst7`` when present,
  because current FE and REMD launchers reference it;
* production windows keep ``eq.rst7``.

Stage behavior
==============

``prepare_equil``
-----------------

Runs after the equilibration system has been prepared.

Removed:

* ``q_amber_files/``, ``q_build_files/`` and ``q_run_files/``;
* debug/cache directories such as ``.restraintmask_cache/``, ``amber_files/``,
  ``ARCHIVED_LOGS/`` and ``WRONG_FAIL/``;
* transient LEaP, ParmEd, build, solvation and repair scratch, including
  ``tleap*.in``, ``tleap*.log``, ``leap.log``, ``parmed-hmr.*``,
  ``assign.*``, ``build*.pdb``, ``rec_amber*.pdb``, ``solvate_*``,
  ``full_pre.pdb`` and repair JSON reports.

Preserved:

* prepared system files under ``equil/`` such as ``full*.prmtop``,
  ``full.inpcrd``, ``full.pdb`` and ``equil-<resname>.pdb``;
* renumbering files, run scripts, mdin files and stage markers.

``equil_analysis``
------------------

Runs after equilibration has finished or a terminal equilibration state is detected.

Removed:

* debug/cache directories such as ``WRONG_FAIL/``, ``ARCHIVED_LOGS/``,
  ``.restraintmask_cache/`` and ``amber_files/``;
* stage scheduler logs such as ``STAGE-POSE-*.out``, ``STAGE-POSE-*.err`` and
  ``mdinfo``;
* non-production equilibration trajectories and outputs such as ``traj*.nc``,
  ``eqnpt*.nc``, ``eqnvt.nc``, ``mini*.nc``, ``eqnpt*.out``, ``eqnvt.out`` and
  ``mini*.out``;
* intermediate ``eq*.rst7`` files, leaving only the selected final restart.

Preserved:

* all ``md-*`` files;
* final equilibration restart selected by priority;
* representative structures, analysis ``npz/png`` files, ProLIF outputs,
  stable Boresch-pair metadata, topology/input/run files and status markers.

``pre_prepare_fe`` and ``prepare_fe``
-------------------------------------

Run after FE component scaffolds are built. ``pre_prepare_fe`` is used by workflows
that need an intermediate ``pre_fe`` system before final FE setup.

Removed:

* component run scratch such as ``<comp>_run_files/``;
* debug/cache directories and transient build files at the component root and
  window roots;
* for ``pre_prepare_fe`` only, ``<comp>_amber_files/`` and
  ``<comp>_build_files/`` are also removed immediately because no later window
  expansion depends on them.

Preserved:

* ``<comp>-1`` scaffold directories;
* final topology/coordinate files, restraint files, mdin templates, run scripts and
  stage markers.

``prepare_fe_windows``
----------------------

Runs after lambda windows are expanded.

Removed:

* ``<comp>_amber_files/``, ``<comp>_build_files/`` and ``<comp>_run_files/``;
* remaining component/window-level prep scratch and cache directories.

Preserved:

* all lambda window directories;
* per-window topology/coordinate files, ``mdin-template``,
  ``mdin-batch-template``, ``mdin-remd-template``, run scripts, REMD groupfiles,
  ``lambda.sch``, ``artifacts/windows.json`` and ``prepare_fe_windows.ok``.

``pre_fe_equil`` and ``fe_equil``
---------------------------------

Run after each ``<comp>-1`` equilibration job has written ``EQ_FINISHED``.

Removed:

* equilibration-only trajectories and outputs such as ``traj*.nc``, ``eq*.nc``,
  ``mini*.nc``, ``eq*.out`` and ``mini*.out``;
* scheduler logs, ``mdinfo``, ``logfile`` and retry archives;
* intermediate ``eq*.rst7`` files, preserving the restart used by downstream FE
  setup and launchers.

Preserved:

* ``EQ_FINISHED`` and other markers;
* ``eqnpt04.rst7`` when present, otherwise the first available fallback from
  ``eq.rst7`` or ``eqnpt_eq.rst7``;
* all ``md-*`` files and all ``cmass*`` files;
* topology/coordinate inputs, restraint files, mdin files and run scripts.

``fe`` production
-----------------

No cleanup runs while FE production is active or resumable. Production continuation
depends on rolling restart files and segment outputs, so these are left intact until
analysis has consumed the data.

Preserved during production:

* ``md-current.rst7`` and ``md-previous.rst7``;
* all ``md-*`` files;
* all ``cmass*`` files;
* ``job_attempt.txt``, ``production-start.ps``, ``run.log`` and status markers.

``analyze``
-----------

Runs after FE analysis has successfully produced ``fe/Results/Results.dat``.

Removed:

* remaining component build/run scratch directories such as ``<comp>_amber_files/``,
  ``<comp>_build_files/`` and ``<comp>_run_files/``;
* debug/cache directories and prep scratch;
* non-``md-*`` runtime outputs such as ``traj*.nc``, ``mdin-*.nc``, ``mini*.nc``,
  ``eq*.nc``, ``mdin-*.out``, ``mini*.out``, ``eq*.out``, ``*.mdinfo``,
  ``*.log``, ``*.mden``, ``mdinfo`` and ``logfile``;
* generated cpptraj REST scratch such as ``restraints_curr.in``,
  ``restraints.dat`` and ``restraints.log``;
* intermediate ``eq*.rst7`` files using the restart rules above.

Preserved:

* all ``md-*`` files and all ``cmass*`` files;
* ``Results.dat``, ``*_results.json``, ``*_results.pickle``,
  ``*_df_list.pickle``, convergence plots, ``fe_timeseries.*`` and
  ``analyze.ok``;
* final topology/restart/input/run files needed for inspection or manual reruns.

Developer notes
===============

When adding new cleanup rules:

* add stage-specific deletion patterns in :mod:`batter._internal.ops.cleanup`;
* keep deletion gated by a successful stage sentinel such as ``EQ_FINISHED`` or
  ``Results.dat``;
* do not delete ``md-*`` or ``cmass*`` files;
* preserve any file referenced by generated run scripts, REMD groupfiles, analysis
  code or phase-state checks;
* validate changes with a synthetic directory test before applying cleanup to live
  runs.
