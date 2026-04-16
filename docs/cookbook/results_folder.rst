Results Folder Layout
=====================

BATTER writes portable analysis outputs under::

   <run.output_folder>/results/

This tree is separate from ``executions/<run_id>/``. The execution directory holds
the live working files for a run, while ``results/`` stores the analysis products
that are meant to be queried later with ``batter fe`` or copied elsewhere.

Top-Level Layout
----------------

A typical results tree looks like::

   results/
   ├── index.csv
   └── <run_id>/
       ├── <ligand_or_pair>/
       │   ├── record.json
       │   ├── Results/
       │   ├── Equil/
       │   └── ...
       └── <another_ligand_or_pair>/

The exact per-ligand contents depend on whether the record succeeded, failed, or
came from an RBFE transformation.

``index.csv``
-------------

``results/index.csv`` is the repository-wide summary table used by
``batter fe list``. Each row corresponds to one saved FE record and includes fields
such as:

* ``run_id`` and ``ligand``
* ``mol_name`` and ``system_name``
* ``temperature``
* ``total_dG`` and ``total_se``
* ``canonical_smiles``, ``original_name``, and ``original_path`` when available
* ``protocol``
* ``analysis_start_step`` and ``n_bootstraps``
* ``status`` and ``failure_reason``
* ``created_at``

When BATTER saves the same ``(run_id, ligand, analysis_start_step, n_bootstraps)``
combination again, the row is replaced rather than duplicated.

Successful Record Directory
---------------------------

For a successful result, BATTER writes::

   results/<run_id>/<ligand_or_pair>/
   ├── record.json
   ├── Results/
   └── Equil/          # ABFE / ASFE / single-ligand runs

``record.json``
   Portable structured summary of the FE result. This is what
   ``batter fe show`` loads.

``Results/``
   Copy of the raw analysis folder from ``executions/<run_id>/.../fe/Results``.
   Common files include:

   * ``Results.dat`` with total and per-component FE values
   * ``fe_timeseries.json`` and ``fe_timeseries.png``
   * ``<component>_results.json``
   * ``<component>_convergence.png``
   * ``<component>_df_list_attrs.json``
   * ``<component>_df_list.pickle``

``Results.dat``
   Plain-text summary table with one line per analyzed component plus a final
   ``Total`` row. Each line stores ``LABEL FE SE``.

``fe_timeseries.json`` and ``fe_timeseries.png``
   Overall progress diagnostic for the assembled FE estimate. The JSON file stores
   the cumulative FE value and uncertainty arrays across increasing fractions of the
   trajectory. The PNG plots those values against simulation progress, adds error
   bars, and overlays the final FE estimate as a dashed reference line with a
   shaded ``±1 kcal/mol`` band.

``<component>_results.json``
   Per-component scalar summary written by the MBAR analysis stage. It stores the
   final FE value, its uncertainty, and the component-level FE timeseries used to
   build the overall summary.

``<component>_convergence.png``
   Three-panel convergence diagnostic for one FE component. BATTER plots:

   * **Time Convergence**: forward/backward FE estimates as increasing fractions of
     the trajectory are included
   * **Overlap Matrix**: MBAR state overlap between neighboring windows
   * **Block Convergence**: FE estimates from block-averaged chunks of the data

   These plots are intended for checking whether a component is equilibrated,
   whether adjacent windows overlap adequately, and whether the estimate is stable
   to block partitioning.

``<component>_df_list.pickle``
   Python pickle cache containing the list of alchemlyb-style reduced-potential
   DataFrames, one per lambda window, that MBAR consumes. BATTER writes this file so
   repeated analysis or notebook-based debugging can reload the parsed data without
   re-reading all AMBER outputs.

``<component>_df_list_attrs.json``
   Sidecar metadata for the pickle cache. It records the component name,
   temperature, analysis start step, whether automatic equilibration detection was
   enabled, the sampling interval, and the number of frames retained per window.

``Equil/``
   Selected equilibration artifacts copied for reference. Common files include:

   * ``README.txt`` describing the copied artifacts
   * ``equilibration_analysis_results.npz``
   * ``simulation_analysis.png``
   * ``dihed_hist.png``
   * ``representative.pdb``
   * ``representative_complex.pdb``
   * ``representative_pose.pdb``
   * ``initial_pose.pdb``
   * ``initial_complex.pdb``
   * ``<ligand>.sdf``, ``<ligand>.prmtop``, and ``<ligand>.pdb`` when present

Failure or Unbound Record Directory
-----------------------------------

If BATTER cannot extract totals from the analysis outputs, or if a ligand is marked
failed/unbound, it still creates a per-ligand directory::

   results/<run_id>/<ligand_or_pair>/
   ├── failure.json
   ├── Results/        # only when raw analysis outputs exist to copy
   └── Equil/          # or Equil_ref / Equil_alt for RBFE

``failure.json`` stores the status and reason string. The matching ``index.csv`` row
has ``status = failed`` or ``status = unbound`` and leaves FE totals empty.

If a later rerun succeeds, BATTER removes the stale ``failure.json`` before writing
the new ``record.json``.

RBFE-Specific Additions
-----------------------

RBFE records use the same top-level pattern, but the payload is slightly different::

   results/<run_id>/<ligand_ref~ligand_alt>/
   ├── record.json
   ├── Results/
   │   ├── Results.dat
   │   ├── rbfe_network.png
   │   ├── mapping.json
   │   ├── mapping.pkl
   │   └── mapping.png
   ├── Equil_ref/
   └── Equil_alt/

``Results/rbfe_network.png``
   Copy of the resolved RBFE network plot from
   ``executions/<run_id>/artifacts/config/rbfe_network.png``.

``Results/mapping.*``
   Atom-mapping artifacts copied from the transformation setup directory.

``Equil_ref/`` and ``Equil_alt/``
   Equilibration reference material for the two ligands that define the
   transformation pair.

How to Read the Repository
--------------------------

The main CLI entry points for this tree are:

.. code-block:: bash

   batter fe list <run.output_folder>
   batter fe show <run.output_folder> <run_id> --ligand <ligand_or_pair>

``fe list`` reads ``results/index.csv`` and prints one row per stored record.
``fe show`` opens ``record.json`` for a single record.

Derived Cinnabar Exports
------------------------

The portable FE repository above is BATTER's canonical stored-results layer. The
optional command::

   batter fe cinnabar <run.output_folder>

writes an additional derived export tree under::

   <run.output_folder>/results/cinnabar/

This tree is not used by BATTER's core FE record loader. It is a convenience export
layer for downstream Cinnabar notebooks, figures, and benchmarking workflows.

A combined export typically looks like::

   results/
   ├── index.csv
   ├── <run_id>/
   │   └── ...
   └── cinnabar/
       ├── manifest.json
       ├── raw_signed.csv
       ├── edge_summary.csv
       ├── cinnabar_relative.csv
       ├── cinnabar_absolute.csv      # when the network is connected
       ├── cinnabar_network.png
       ├── cinnabar_dg.png            # when experiment is provided
       └── cinnabar_ddg.png           # when experiment is provided

When ``--split-runs`` is used, BATTER writes one subdirectory per run instead::

   results/cinnabar/<run_id>/

``raw_signed.csv``
   Measurement-level table after BATTER filters to RBFE rows, resolves the edge
   label, and assigns a sign convention based on sorted ligand names.

``edge_summary.csv``
   Edge-level DDG estimates after repeat combination. This is the main table to
   inspect when you want one summarized value per perturbation.

``cinnabar_relative.csv``
   Relative measurements exported from the constructed ``FEMap``.

``cinnabar_absolute.csv``
   MLE-derived absolute values from Cinnabar. BATTER writes this only when the RBFE
   network is connected strongly enough for Cinnabar to solve absolute values.

``manifest.json``
   Lightweight summary of what BATTER wrote, including whether experimental data was
   merged and whether absolute values were successfully generated.

Where This Differs From ``executions/``
---------------------------------------

``executions/<run_id>/`` is the working directory for an active or completed run.
It contains staged inputs, Slurm scripts, logs, raw trajectories, and in-progress
markers.

``results/`` is the compact, portable summary layer derived from those executions.
If you only need final FE values plus the most important analysis artifacts, this is
usually the tree to archive or inspect first.

Visualizing Trajectories
------------------------

The portable ``results/`` repository does not duplicate the production trajectories.
For molecular visualization, go back to the matching directory under
``executions/<run_id>/`` and open the staged coordinate and trajectory files there.

For equilibration trajectories, a common workflow is:

.. code-block:: bash

   vmd full.pdb md-*.nc

Then load ``full.prmtop`` inside VMD to recover bonded information.

For FE production trajectories, inspect one window at a time from the corresponding
component directory. BATTER stores these trajectories without water, so a common
workflow is:

.. code-block:: bash

   vmd vac.pdb md-*.nc

Then load ``vac.prmtop`` inside VMD to recover bonded information. ``vac.pdb`` also
uses 1-based residue numbering, which is often convenient when comparing against the
analysis outputs and restraint setup.

It is often useful to visualize the first and last lambda windows explicitly. This
can help identify cases where an additional symmetry correction may be needed for the
transformation or restraint interpretation.
