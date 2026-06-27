from __future__ import annotations

import json
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


def test_component_results_json_includes_backward_and_convergence_data(
    tmp_path: Path,
) -> None:
    class DummyAnalysis(analysis_mod.FEAnalysisBase):
        def run_analysis(self):
            pass

        def plot_convergence(self, ax=None, **kwargs):
            pass

    ana = DummyAnalysis()
    ana.results["fe"] = 1.5
    ana.results["fe_error"] = 0.2
    ana.results["fe_timeseries"] = np.array([[1.0, 0.1], [1.5, 0.2]])
    ana.results["fe_timeseries_backward"] = np.array([[2.0, 0.3], [1.5, 0.2]])
    ana.results["convergence"]["time_convergence"] = pd.DataFrame(
        {
            "Forward": [1.0, 1.5],
            "Forward_Error": [0.1, 0.2],
            "Backward": [2.0, 1.5],
            "Backward_Error": [0.3, 0.2],
        },
        index=pd.Index([0.5, 1.0], name="data_fraction"),
    )
    ana.results["convergence"]["block_convergence"] = pd.DataFrame(
        {"FE": [1.4, 1.6], "FE_Error": [0.2, 0.25]},
        index=pd.Index([1, 2], name="block"),
    )
    ana.results["convergence"]["block_timeseries"] = pd.DataFrame(
        {"FE": [1.4, 1.6], "FE_Error": [0.2, 0.25]},
        index=pd.Index([0.5, 1.0], name="fraction"),
    )
    ana.results["convergence"]["overlap_matrix"] = np.array([[1.0, 0.2], [0.2, 1.0]])

    out = tmp_path / "z_results.json"
    ana.dump(out)

    payload = json.loads(out.read_text())
    assert payload["fe_timeseries"] == [[1.0, 0.1], [1.5, 0.2]]
    assert payload["fe_timeseries_backward"] == [[2.0, 0.3], [1.5, 0.2]]
    assert payload["convergence"]["time_convergence"]["columns"] == [
        "data_fraction",
        "Forward",
        "Forward_Error",
        "Backward",
        "Backward_Error",
    ]
    assert payload["convergence"]["time_convergence"]["records"][0]["Backward"] == 2.0
    assert payload["convergence"]["block_convergence"]["records"][1]["FE"] == 1.6
    assert payload["convergence"]["overlap_matrix"] == [[1.0, 0.2], [0.2, 1.0]]


def test_analyze_lig_task_writes_backward_fe_timeseries(
    tmp_path: Path, monkeypatch
) -> None:
    lig_path = tmp_path / "lig"
    (lig_path / "x").mkdir(parents=True)

    class FakeMBARAnalysis:
        def __init__(self, **kwargs):
            self.results = {}

        def run_analysis(self):
            self.results = {
                "fe": 4.0,
                "fe_error": 0.4,
                "fe_timeseries": np.array([[1.0, 0.1], [2.0, 0.2]]),
                "fe_timeseries_backward": np.array([[3.0, 0.3], [4.0, 0.4]]),
            }

        def plot_convergence(self, save_path=None, title=None):
            if save_path:
                Path(save_path).write_bytes(b"png")

    monkeypatch.setattr(analysis_mod, "MBARAnalysis", FakeMBARAnalysis)

    analysis_mod.analyze_lig_task(
        lig_path=str(lig_path / "fe"),
        lig="LIG",
        components=["x"],
        rest=(0.0, 0.0, 0.0, 0.0, 0.0),
        temperature=300.0,
        water_model="TIP3P",
        component_windows_dict={"x": [0, 1]},
        raise_on_error=True,
        dt=0.004,
    )

    payload = json.loads((lig_path / "fe" / "Results" / "fe_timeseries.json").read_text())
    assert payload["fe_value"][:2] == [1.0, 2.0]
    assert payload["fe_std"][:2] == [0.1, 0.2]
    assert payload["backward_fe_value"][:2] == [3.0, 4.0]
    assert payload["backward_fe_std"][:2] == [0.3, 0.4]


def test_ligand_rest_component_direction_registered() -> None:
    assert analysis_mod.COMPONENT_DIRECTION_DICT["l"] == 1


def test_rest_mbar_extracts_keyed_ligand_restraints(tmp_path: Path) -> None:
    comp_dir = tmp_path / "l"
    for win, force in [(0, 0.0), (1, 10.0)]:
        win_dir = comp_dir / f"l{win:02d}"
        win_dir.mkdir(parents=True)
        win_dir.joinpath("disang.rest").write_text(
            "&rst iat=101,202,              "
            "r1=     0.0000, r2=    2.5000, r3=    2.5000, r4=  999.0000, "
            f"rk2= {force:10.7f}, rk3= {force:10.7f}, &end #Lig_TR\n"
            "&rst iat=301,101,202,          "
            "r1=     0.0000, r2=   90.0000, r3=   90.0000, r4=  180.0000, "
            f"rk2= {force:10.7f}, rk3= {force:10.7f}, &end #Lig_TR\n"
            "&rst iat=401,301,101,202,      "
            "r1= -120.0000, r2=   60.0000, r3=   60.0000, r4=  240.0000, "
            f"rk2= {force:10.7f}, rk3= {force:10.7f}, &end #Lig_TR\n"
            "&rst iat=4363,4364,4365,4366,    "
            "r1= -179.5156, r2=    0.4844, r3=    0.4844, r4=  180.4844, "
            f"rk2= {force:10.7f}, rk3= {force:10.7f}, &end #Lig_D\n"
        )

    ana = analysis_mod.RESTMBARAnalysis(
        lig_folder=str(tmp_path),
        component="l",
        windows=[0, 1],
        temperature=300.0,
        detect_equil=False,
        dt=0.004,
    )

    rfc, req, rty, num_rest = ana._extract_restraints_from_windows()

    assert num_rest == 4
    assert rty == ["d", "a", "t", "t"]
    assert np.allclose(req[:, 0], [2.5, 2.5])
    assert np.allclose(req[:, 1], [90.0, 90.0])
    assert np.allclose(req[:, 2], [60.0, 60.0])
    assert np.allclose(req[:, 3], [0.4844, 0.4844])
    assert np.allclose(rfc[0], np.zeros(4))
    assert np.isclose(rfc[1, 0], 10.0)
    assert np.isclose(rfc[1, 1], 10.0 * (np.pi / 180.0) ** 2)
    assert np.isclose(rfc[1, 2], 10.0 * (np.pi / 180.0) ** 2)
    assert np.isclose(rfc[1, 3], 10.0 * (np.pi / 180.0) ** 2)


def test_mbar_extract_window_skips_incomplete_mdout(tmp_path: Path, monkeypatch) -> None:
    win_dir = tmp_path / "z00"
    win_dir.mkdir()
    bad = win_dir / "md-01.out"
    good = win_dir / "md-02.out"
    bad.write_text("job started but amber never wrote headers\n")
    good.write_text("parseable amber output\n")

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

    def _fake_extract(path, *args, **kwargs):
        if Path(path).name == "md-01.out":
            raise ValueError(f'no "CONTROL DATA" section found in file {path}')
        return parsed

    monkeypatch.setattr(analysis_mod, "extract_u_nk", _fake_extract)

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
