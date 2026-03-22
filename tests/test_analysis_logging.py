from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from batter.analysis import analysis as analysis_mod


def test_mbar_extract_window_does_not_remove_global_logger(
    tmp_path: Path, monkeypatch
) -> None:
    win_dir = tmp_path / "z00"
    win_dir.mkdir()
    (win_dir / "md-00.out").write_text("dummy\n")

    def _fail_remove(*args, **kwargs):
        raise AssertionError("logger.remove should not be called during FE analysis")

    monkeypatch.setattr(analysis_mod.logger, "remove", _fail_remove)
    monkeypatch.setattr(analysis_mod.logger, "debug", lambda *a, **k: None)
    monkeypatch.setattr(analysis_mod.logger, "warning", lambda *a, **k: None)
    monkeypatch.setattr(analysis_mod, "exclude_outliers", lambda df, iclam: df)

    index = pd.MultiIndex.from_arrays(
        [[0.0, 1.0], [0.0, 0.0]],
        names=["time", "lambdas"],
    )
    parsed = pd.DataFrame(
        {0.0: [0.0, 1.0], 1.0: [0.5, 1.5]},
        index=index,
    )
    monkeypatch.setattr(analysis_mod, "extract_u_nk", lambda *a, **k: parsed)

    out = analysis_mod.MBARAnalysis._extract_all_for_window(
        win_i=0,
        comp_folder=str(tmp_path),
        component="z",
        temperature=300.0,
        analysis_start_step=0,
        truncate=False,
    )

    assert not out.empty


def test_rest_mbar_extract_window_does_not_remove_global_logger(
    tmp_path: Path, monkeypatch
) -> None:
    win_dir = tmp_path / "a00"
    win_dir.mkdir()
    (win_dir / "mdin-00.nc").write_text("")

    def _fail_remove(*args, **kwargs):
        raise AssertionError("logger.remove should not be called during FE analysis")

    def _fake_generate_results_rest(nc_list, component, blocks=5, top="full"):
        Path("restraints.dat").write_text("0 1.0\n1 1.5\n")

    monkeypatch.setattr(analysis_mod.logger, "remove", _fail_remove)
    monkeypatch.setattr(analysis_mod.logger, "debug", lambda *a, **k: None)
    monkeypatch.setattr(analysis_mod, "generate_results_rest", _fake_generate_results_rest)

    out = analysis_mod.RESTMBARAnalysis._extract_all_for_window(
        win_i=0,
        comp_folder=str(tmp_path),
        component="a",
        temperature=300.0,
        analysis_start_step=0,
        rfc=np.array([[1.0]]),
        req=np.array([[1.0]]),
        rty=["d"],
        num_rest=1,
        num_win=1,
        truncate=False,
        dt=0.004,
        ntwx=0,
    )

    assert not out.empty
