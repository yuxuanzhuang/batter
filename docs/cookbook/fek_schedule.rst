Optimizing FEP Schedules from AR Data
=====================================

This recipe shows how to generate a denser SSC(alpha) lambda schedule from
existing analysis-ready overlap or replica-exchange data with
``batter fek-schedule``.

Example Command
---------------

.. code-block:: console

   batter fek-schedule --opt 48 -o sched.ar.z.dat --plot sched.ar.z.png -T 298.15 --start 0 --stop 8 --ssc --ar /path/to/executions/run-id/simulations/transformations/lig1~lig2/fe/comp/

What This Does
--------------

* ``--opt 48`` optimizes a fresh schedule with 48 lambda values.
* ``--ar`` targets the predicted replica-exchange ratio.
* ``--ssc`` restricts the search to SSC(alpha) schedules.
* ``-o sched.ar.z.dat`` writes the optimized schedule to disk.
* ``--plot sched.ar.z.png`` writes the corresponding heatmap/path projection plot.
* ``-T 298.15`` sets the temperature in Kelvin.
* ``--start 0 --stop 8`` trims the data range passed into the scheduling utility.

Input Directory
---------------

Point the final positional argument at the component folder containing the FE
analysis data. For RBFE transformations this is typically the ``x`` component,
for example:

.. code-block:: text

   /path/to/executions/run-id/simulations/transformations/lig1~lig2/fe/x/

If you use a placeholder such as ``comp`` in scripts or notes, replace it with
the actual component directory before running the command.

Related CLI Options
-------------------

See :doc:`../cli` for other schedule metrics and options such as ``--pso``,
``--kl``, ``--sym``, ``--alpha0``, ``--alpha1``, and ``--read``.
