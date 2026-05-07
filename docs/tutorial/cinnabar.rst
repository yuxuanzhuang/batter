.. _cinnabar_tutorial:

Using Cinnabar with RBFE Results
================================

BATTER can convert stored RBFE records into
`Cinnabar <https://cinnabar.openfree.energy/>`_ ``FEMap`` objects and write
analysis-ready tables, static plots, and an interactive dashboard.

Default per-run output
----------------------

When RBFE analysis finishes for a run, BATTER writes a default Cinnabar bundle for
that run id under::

   <run.output_folder>/results/cinnabar/<run_id>/

The most useful files are:

* ``cinnabar_dashboard.html`` - interactive network and absolute-ranking dashboard.
* ``edge_summary.csv`` - combined edge-level ``DDG`` estimates and uncertainties.
* ``raw_signed.csv`` - signed per-measurement rows after edge-direction handling.
* ``cinnabar_relative.csv`` - relative measurements exported from the Cinnabar ``FEMap``.
* ``cinnabar_absolute.csv`` - MLE-derived absolute values when the network is connected.
* ``cinnabar_network.png`` - static network plot.
* ``cinnabar_absolute_sorted.png`` - sorted absolute free-energy ranking plot.

This per-run folder is the first place to look after a normal RBFE run. For example::

   open <run.output_folder>/results/cinnabar/<run_id>/cinnabar_dashboard.html

Combine replicates and connect networks
---------------------------------------

If you ran replicate RBFE runs, or if different runs contain overlapping ligands
that should connect into one larger network, build a combined bundle with
``build_batter_rbfe_cinnabar_from_runs``. This API treats each run as an atomic
``(work_dir, run_id)`` input, so it works even when the runs are stored in
different BATTER work directories:

.. code-block:: python

   from pathlib import Path
   from batter.analysis.cinnabar import (
       build_batter_rbfe_cinnabar_from_runs,
       write_cinnabar_outputs,
   )

   runs = [
       (Path("work/adrb2"), "rep1"),
       (Path("work/adrb2_retry"), "rep2"),
   ]

   result = build_batter_rbfe_cinnabar_from_runs(runs)
   write_cinnabar_outputs(
       result,
       Path("combined_cinnabar"),
       method_name="BATTER",
       target_name="adrb2 replicates",
   )

The equivalent CLI command is:

.. code-block:: console

   batter fe cinnabar \
       --run work/adrb2 rep1 \
       --run work/adrb2_retry rep2 \
       --out-dir combined_cinnabar

If all runs are stored in one work directory, the same-work-dir shortcut remains
available::

   batter fe cinnabar work/adrb2 --run-id rep1 --run-id rep2 --run-id rep3

For the shortcut, the combined bundle is written under
``<run.output_folder>/results/cinnabar/``. For explicit ``--run`` inputs from
multiple work directories, pass ``--out-dir``; if it is omitted, BATTER writes to
``./cinnabar``.

By default BATTER combines repeated measurements within each run first, then
combines runs into one FEMap. Ligand nodes are matched by ligand name plus
canonical SMILES. If two runs use the same ligand name and the same canonical
SMILES, those endpoints merge into one node, connecting the networks. If the name
matches but the canonical SMILES differs, BATTER keeps separate nodes with
deterministic suffixed labels so different molecules are not silently merged.

Enable or disable rows
----------------------

Each saved FE row in ``results/index.csv`` has an ``include_in_analysis`` flag.
Rows with ``include_in_analysis`` set to ``False`` are skipped when BATTER builds
Cinnabar bundles. This is useful when you want to remove a bad transformation from
aggregate analysis without deleting its raw result artifacts.

Use the interactive CLI to change the flag:

.. code-block:: console

   batter fe analysis-inclusion work/adrb2

The command prints numbered rows and accepts commands such as ``disable 3``,
``disable 1,4``, ``enable 2``, ``disable all``, ``show``, and ``quit``. Use
``--run-id`` one or more times to restrict the interactive table to specific run
ids.

Split or merge directionality
-----------------------------

BATTER merges opposite-direction rows such as ``LIGA~LIGB`` and ``LIGB~LIGA`` into
one canonical edge by default. Use ``merge_bidirectional=False`` in Python, or
``--split-directions`` in the CLI, when you want the two stored transformations to
remain separate directional measurements:

.. code-block:: python

   result = build_batter_rbfe_cinnabar(
       "work/adrb2",
       run_ids=["rep1", "rep2"],
       merge_bidirectional=False,
   )

For one Cinnabar bundle per run instead of one combined bundle, use
``build_batter_rbfe_cinnabar_by_run`` in Python or ``--split-runs`` with the
same-work-dir CLI shortcut.

Experimental comparisons
------------------------

To add experimental absolute affinities, pass an experimental table when building
the result:

.. code-block:: python

   import pandas

   exp = pandas.read_csv("experimental.csv")
   result = build_batter_rbfe_cinnabar(
       "work/adrb2",
       run_ids=["rep1", "rep2"],
       experimental_df=exp,
       exp_ligand_column="ligand",
       exp_abfe_column="abfe",
       exp_error_column="se",
   )

The CLI accepts the same information with ``--experimental-csv``,
``--exp-ligand-column``, ``--exp-abfe-column``, and ``--exp-error-column``.
