=====================
Analysis Toolkit
=====================

BATTER ships with an analysis package that focuses on
post-processing ABFE/ASFE outputs with MBAR and alchemlyb, validating replica-exchange logs.
This page summarises the most common entry points so you can reuse them in notebooks or bespoke pipelines.

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
  block averages, window overlap) are stored alongside the analysis object and can be
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

Config-driven trimming
======================

``analysis_start_step`` in ``SimulationConfig`` (set via ``fe_sim.analysis_start_step``)
controls how much of each FE window is ignored before MBAR runs. The orchestrator
passes that value into record writers so downstream consumers (CLI, notebooks) can
respect the same cutoff. ``analysis_range`` is deprecated and rejected at load time;
use ``analysis_start_step`` instead. To mirror the config in standalone scripts, set
``start_step`` when instantiating ``MBARAnalysis``::

    analysis = MBARAnalysis(..., start_step=sim_cfg.analysis_start_step)
    analysis.run_analysis()

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
