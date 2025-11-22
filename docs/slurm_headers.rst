SLURM header templates
======================

BATTER renders SLURM scripts by combining a user-editable header with a packaged body.
On first use of the CLI, headers are copied into ``~/.batter`` (or ``run.slurm_header_dir`` if set).
You can also seed them explicitly:

.. code-block:: bash

   batter seed-headers           # seeds into ~/.batter
   batter seed-headers --dest /path/to/dir
   batter seed-headers --force   # overwrite existing headers

Once seeded, edit the headers to match your cluster defaults (queue/partition, env exports,
executable paths). Do not edit the body files; they are managed by the package.

Header files
------------

* ``SLURMM-Am.header``: equilibration/FE runs
* ``SLURMM-BATCH-remd.header``: REMD runs
* ``job_manager.header``: manager script used by ``batter --slurm-submit``

Common environment overrides
----------------------------

Headers include commented examples for overriding executables:

* ``PMEMD_EXEC`` (default: pmemd.cuda)
* ``PMEMD_MPI_EXEC`` (default: pmemd.cuda.MPI)
* ``PMEMD_DPFP_EXEC`` (default: pmemd.cuda_DPFP)
* ``PMEMD_CPU_EXEC`` (default: pmemd)
* ``PMEMD_CPU_MPI_EXEC`` (default: pmemd.MPI)
* ``SANDER_EXEC`` (default: sander)
* ``MPI_EXEC`` (default: mpirun)

Edit exports in the header files to point to site-specific binaries/modules. The packaged
bodies will be appended during rendering; only the headers are meant for customization.
