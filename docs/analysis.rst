=====================
Analysis Toolkit
=====================

BATTER ships with a small but opinionated analysis package that focuses on
post-processing ABFE/ASFE outputs, validating replica-exchange logs, and
parsing legacy BAT.py result files. This page summarises the most common entry
points so you can reuse them in notebooks or bespoke pipelines.

Component Free-Energy Analysis
==============================

``batter.analysis.analysis.MBARAnalysis`` encapsulates the workflow of
collecting AMBER window energies, trimming equilibration, and running an
``alchemlyb`` MBAR estimator.

Key capabilities:

* **Deterministic caching** – reduced potentials are written to
  ``<lig_folder>/Results/<component>_df_list.pickle`` and re-used when
  ``load=True``.
* **Equilibration detection** – each window can be truncated using
  :func:`pymbar.timeseries.detect_equilibration` by setting
  ``detect_equil=True``.
* **Convergence summaries** – MBAR diagnostics (forward/backward convergence,
  block averages, window overlap) are stored in ``analysis.results`` and can be
  plotted via helper methods such as :meth:`MBARAnalysis.plot_convergence`.

Minimal example::

    from batter.analysis.analysis import MBARAnalysis

    analysis = MBARAnalysis(
        lig_folder="work/adrb2/simulations/LIG1",
        component="e",
        windows=list(range(21)),
        temperature=310.0,
        energy_unit="kcal/mol",
        detect_equil=True,
    )
    analysis.run_analysis()
    print(analysis.fe, "+/-", analysis.fe_error)
    analysis.plot_convergence(save_path="lig1_e_convergence.png")

Handling Restrained Components
==============================

``RESTMBARAnalysis`` extends ``MBARAnalysis`` for the restraint components
(``a``, ``l``, ``t``, ``c``, ``r``, ``m``, ``n``). It reads the window-specific
``disang.rest`` files, extracts cpptraj restraint traces, and converts them into
reduced potentials before running MBAR. Use it exactly like ``MBARAnalysis``;
the class automatically detects the additional files it needs.

Replica-Exchange Diagnostics
============================

``batter.analysis.remd.RemdLog`` provides a lightweight parser for AMBER
``remlog`` files. Instantiate it with the logfile path and call
:meth:`RemdLog.analyze` to obtain round-trip statistics and neighbour
acceptance ratios. You can also poke at ``RemdLog.replica_trajectory`` to produce
custom plots.

For quick visual checks, call :func:`batter.analysis.remd.plot_trajectory`, which
renders either a single combined plot or a grid of per-replica subplots.

Legacy Result Files
===================

Many BAT.py workflows still emit human-readable ``results.dat`` summaries.
``batter.analysis.results.FEResult`` parses those files into a structured form:

* Component values are exposed as properties (``fe``, ``attach``, ``elec``, …).
* ``FEResult.is_unbound`` tells you whether the analysis flagged an unbound
  complex.
* ``FEResult.to_dict`` produces a JSON-friendly representation that you can
  embed in reports or store alongside the newer portable artifacts.

Testing the Analysis Layer
==========================

Unit tests covering the analysis helpers live in ``tests/test_analysis_*.py``:

* ``test_analysis_utils.py`` validates ``exclude_outliers`` chunking and NaN
  handling.
* ``test_analysis_remd.py`` checks the round-trip metrics computed by
  ``RemdLog``.
* ``test_analysis_results.py`` exercises the legacy result parser, including the
  unbound sentinel.

Add new regression tests next to these files whenever you extend the analysis
toolkit. The fixtures only depend on ``numpy``/``pandas`` so they run without a
full MD stack.
