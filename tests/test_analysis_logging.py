from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

from batter.analysis import analysis as analysis_mod
import batter
import logging


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


def test_allow_loguru_record_suppresses_alchemlyb_info() -> None:
    record = {
        "name": "alchemlyb.parsing.amber",
        "level": type("L", (), {"no": logging.INFO})(),
    }
    assert batter._allow_loguru_record(record) is False


def test_allow_loguru_record_keeps_alchemlyb_warning() -> None:
    record = {
        "name": "alchemlyb.parsing.amber",
        "level": type("L", (), {"no": logging.WARNING})(),
    }
    assert batter._allow_loguru_record(record) is True


def test_silence_alchemlyb_only_sets_python_loggers_to_warning() -> None:
    amber_logger = logging.getLogger("alchemlyb.parsing.amber")
    prev_level = amber_logger.level
    amber_logger.setLevel(logging.INFO)

    try:
        with analysis_mod.SilenceAlchemlybOnly():
            assert logging.getLogger("alchemlyb").level == logging.WARNING
            assert logging.getLogger("alchemlyb.parsing").level == logging.WARNING
            assert amber_logger.level == logging.WARNING
        assert amber_logger.level == logging.INFO
    finally:
        amber_logger.setLevel(prev_level)
