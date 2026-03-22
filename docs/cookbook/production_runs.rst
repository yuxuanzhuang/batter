Running for Production
======================

For larger cluster campaigns, a common pattern is:

.. code-block:: console

   batter run xxx.yaml --on-failure prune --run-id repxxx --slurm-submit

Replace ``xxx.yaml`` with your workflow YAML file, and replace ``repxxx`` with a
concrete run label such as ``rep1``, ``rep2``, or ``rep3``.

Why This Pattern Helps
----------------------

``--on-failure prune``
   Keep the overall campaign moving if one ligand or transformation fails. BATTER
   prunes failed entries from later phases instead of terminating the whole run,
   which is often the right tradeoff for production sweeps.

``--run-id repxxx``
   Control the execution directory name under
   ``<run.output_folder>/executions/repxxx/``. This is useful for independent
   repeats because each replicate can live in a clearly named directory such as
   ``rep1``, ``rep2``, and ``rep3``.

``--slurm-submit``
   Submit the generated job-manager script with ``sbatch`` instead of running the
   orchestration process locally in your shell. Before using this mode, make sure
   your seeded SLURM headers under ``~/.batter/`` or ``run.slurm_header_dir`` load
   Amber correctly for your cluster; see :doc:`slurm_headers`.

Practical Notes
---------------

* If you are running replicate calculations for uncertainty estimation, give each
  repeat a distinct ``--run-id``.
* ``--on-failure prune`` is useful when you want one bad ligand, mapping, or node
  issue to stop affecting the rest of the campaign.
* If you want BATTER to stop immediately when something fails, use
  ``--on-failure raise`` instead.
