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


def test_mbar_convergence_fallback_repeats_final_estimate(tmp_path: Path) -> None:
    (tmp_path / "z").mkdir()
    ana = analysis_mod.MBARAnalysis(
        lig_folder=str(tmp_path),
        component="z",
        windows=[0, 1],
        temperature=300.0,
        detect_equil=False,
        dt=0.004,
    )
    ana.results["fe"] = 1.25
    ana.results["fe_error"] = 0.5

    ana._set_convergence_fallback("too few frames", n_points=3)

    assert ana.results["fe_timeseries"].shape == (3, 2)
    assert ana.results["fe_timeseries_backward"].shape == (3, 2)
    assert np.allclose(ana.results["fe_timeseries"][:, 0], 1.25)
    assert np.allclose(ana.results["fe_timeseries"][:, 1], 0.5)
    assert list(ana.results["convergence"]["time_convergence"].columns) == [
        "Forward",
        "Forward_Error",
        "Backward",
        "Backward_Error",
    ]


def test_boresch_analysis_selects_ligand_specific_tag(
    tmp_path: Path, monkeypatch
) -> None:
    disang = tmp_path / "disang.rest"
    disang.write_text(
        "\n".join(
            [
                f"&rst iat=1,2, r2={value:.1f}, &end #Lig_TR_REF"
                for value in range(1, 7)
            ]
            + [
                f"&rst iat=1,2, r2={value:.1f}, &end #Lig_TR_ALT"
                for value in range(11, 17)
            ]
        )
    )
    seen: list[tuple[float, float, float, float, float, float]] = []

    def _fake_fe_int(r0, a1_0, t1_0, a2_0, t2_0, t3_0, *args):
        seen.append((r0, a1_0, t1_0, a2_0, t2_0, t3_0))
        return r0

    monkeypatch.setattr(
        analysis_mod.BoreschAnalysis, "fe_int", staticmethod(_fake_fe_int)
    )

    ana = analysis_mod.BoreschAnalysis(
        disangfile=disang,
        k_r=1.0,
        k_a=1.0,
        temperature=300.0,
        restraint_tag="Lig_TR_ALT",
    )
    ana.run_analysis()

    assert ana.results["fe"] == 11.0
    assert seen == [(11.0, 12.0, 13.0, 14.0, 15.0, 16.0)]


def test_boresch_fe_int_uses_numpy_trapezoid_without_trapz(monkeypatch) -> None:
    real_trapezoid = analysis_mod.np.trapezoid
    calls = 0

    def _trapezoid(y, x):
        nonlocal calls
        calls += 1
        return real_trapezoid(y, x)

    monkeypatch.setattr(analysis_mod.np, "trapezoid", _trapezoid)
    monkeypatch.delattr(analysis_mod.np, "trapz", raising=False)

    result = analysis_mod.BoreschAnalysis.fe_int(
        5.0,
        90.0,
        180.0,
        90.0,
        180.0,
        180.0,
        10.0,
        10.0,
        298.15,
    )

    assert np.isfinite(result)
    assert calls == 6


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
    assert payload["components"]["x"]["fe_timeseries"] == [[1.0, 0.1], [2.0, 0.2]]
    assert payload["components"]["x"]["fe_timeseries_backward"] == [
        [3.0, 0.3],
        [4.0, 0.4],
    ]


def test_analyze_lig_task_adds_septop_boresch_corrections(
    tmp_path: Path, monkeypatch
) -> None:
    lig_path = tmp_path / "fe"
    x_boresch = lig_path / "x" / "x-1" / "disang.rest"
    x_boresch.parent.mkdir(parents=True)
    x_boresch.write_text(
        "\n".join(
            [
                f"&rst iat=1,2, r2={value:.1f}, &end #Lig_TR_REF"
                for value in [2, 20, 21, 22, 23, 24]
            ]
            + [
                f"&rst iat=1,2, r2={value:.1f}, &end #Lig_TR_ALT"
                for value in [3, 30, 31, 32, 33, 34]
            ]
        )
    )

    class FakeMBARAnalysis:
        def __init__(self, **kwargs):
            self.results = {}

        def run_analysis(self):
            self.results = {
                "fe": 10.0,
                "fe_error": 1.0,
                "fe_timeseries": np.array([[10.0, 1.0], [10.0, 1.0]]),
                "fe_timeseries_backward": np.array([[10.0, 1.0], [10.0, 1.0]]),
            }

        def plot_convergence(self, save_path=None, title=None):
            if save_path:
                Path(save_path).write_bytes(b"png")

    def _fake_fe_int(r0, *args):
        return r0

    monkeypatch.setattr(analysis_mod, "MBARAnalysis", FakeMBARAnalysis)
    monkeypatch.setattr(
        analysis_mod.BoreschAnalysis, "fe_int", staticmethod(_fake_fe_int)
    )

    analysis_mod.analyze_lig_task(
        lig_path=str(lig_path),
        lig="LIG",
        components=["x"],
        rest=(0.0, 0.0, 5.0, 250.0, 0.0),
        temperature=300.0,
        water_model="TIP3P",
        component_windows_dict={"x": [0, 1]},
        raise_on_error=True,
        dt=0.004,
    )

    results = (lig_path / "Results" / "Results.dat").read_text().splitlines()
    assert "Boresch_REF\t-2.00\t0.00" in results
    assert "Boresch_ALT\t3.00\t0.00" in results
    assert "Total\t11.00\t1.00" in results

    payload = json.loads((lig_path / "Results" / "fe_timeseries.json").read_text())
    assert payload["fe_value"][:2] == [11.0, 11.0]


def test_disang_restraint_tag_count_matches_exact_tag(tmp_path: Path) -> None:
    disang = tmp_path / "disang.rest"
    disang.write_text(
        "&rst iat=1,2, r2=1.0, &end #Lig_TR\n"
        "&rst iat=1,2, r2=2.0, &end #Lig_TR_REF\n"
        "&rst iat=1,2, r2=3.0, &end #Lig_TR_ALT\n"
    )

    assert analysis_mod._disang_restraint_tag_count(disang, "Lig_TR") == 1
    assert analysis_mod._disang_restraint_tag_count(disang, "Lig_TR_REF") == 1
    assert analysis_mod._disang_restraint_tag_count(disang, "Lig_TR_ALT") == 1
    assert not analysis_mod._disang_has_complete_boresch_block(disang, "Lig_TR")


def test_analyze_lig_task_adds_reduced_abfe_correction(
    tmp_path: Path, monkeypatch
) -> None:
    lig_path = tmp_path / "fe"
    z_boresch = lig_path / "z" / "z-1" / "disang.rest"
    z_boresch.parent.mkdir(parents=True)
    z_boresch.write_text(
        "\n".join(
            f"&rst iat=1,2, r2={value:.1f}, &end #Lig_TR"
            for value in [2, 20, 21]
        )
    )

    class FakeMBARAnalysis:
        def __init__(self, **kwargs):
            self.results = {}

        def run_analysis(self):
            self.results = {
                "fe": 10.0,
                "fe_error": 1.0,
                "fe_timeseries": np.array([[10.0, 1.0], [10.0, 1.0]]),
                "fe_timeseries_backward": np.array([[10.0, 1.0], [10.0, 1.0]]),
            }

        def plot_convergence(self, save_path=None, title=None):
            if save_path:
                Path(save_path).write_bytes(b"png")

    monkeypatch.setattr(analysis_mod, "MBARAnalysis", FakeMBARAnalysis)
    monkeypatch.setattr(
        analysis_mod.ReducedExternalRestraintAnalysis,
        "fe_int",
        staticmethod(lambda values, *args: float(values[0])),
    )

    analysis_mod.analyze_lig_task(
        lig_path=str(lig_path),
        lig="SOD",
        components=["z"],
        rest=(0.0, 0.0, 5.0, 250.0, 0.0),
        temperature=300.0,
        water_model="TIP3P",
        component_windows_dict={"z": [0, 1]},
        raise_on_error=True,
        dt=0.004,
    )

    results = (lig_path / "Results" / "Results.dat").read_text()
    assert "Boresch" not in results
    assert "Reduced_TR\t-2.00\t0.00" in results
    assert "z\t-10.00\t1.00" in results
    assert "Total\t-12.00\t1.00" in results


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


def test_mbar_extract_window_skips_detect_equilibration_for_single_sample(
    tmp_path: Path, monkeypatch
) -> None:
    win_dir = tmp_path / "z00"
    win_dir.mkdir()
    (win_dir / "md-01.out").write_text("short smoke-test amber output\n")

    index = pd.MultiIndex.from_arrays(
        [[0.0], [0.0]],
        names=["time", "lambdas"],
    )
    parsed = pd.DataFrame({0.0: [0.0], 1.0: [0.5]}, index=index)

    monkeypatch.setattr(analysis_mod.logger, "debug", lambda *a, **k: None)
    monkeypatch.setattr(analysis_mod.logger, "warning", lambda *a, **k: None)
    monkeypatch.setattr(analysis_mod, "extract_u_nk", lambda *a, **k: parsed)
    monkeypatch.setattr(analysis_mod, "exclude_outliers", lambda df, iclam: df.iloc[0:0])
    monkeypatch.setattr(
        analysis_mod,
        "detect_equilibration",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("detect_equilibration should be skipped")
        ),
    )

    out = analysis_mod.MBARAnalysis._extract_all_for_window(
        win_i=0,
        comp_folder=str(tmp_path),
        component="z",
        temperature=300.0,
        analysis_start_step=0,
        truncate=True,
    )

    assert len(out) == 1


def test_rest_mbar_extract_window_does_not_remove_global_logger(
    tmp_path: Path, monkeypatch
) -> None:
    win_dir = tmp_path / "a00"
    win_dir.mkdir()
    (win_dir / "md-01.nc").write_text("")
    seen_nc_lists: list[list[str]] = []

    def _fail_remove(*args, **kwargs):
        raise AssertionError("logger.remove should not be called during FE analysis")

    def _fake_generate_results_rest(
        nc_list, component, blocks=5, top="full", workdir=None
    ):
        seen_nc_lists.append(list(nc_list))
        Path(workdir).joinpath("restraints.dat").write_text("0 1.0\n1 1.5\n")

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
    assert seen_nc_lists == [["md-01.nc"]]


def test_rest_mbar_extract_window_reads_segmented_cmass_without_cpptraj(
    tmp_path: Path, monkeypatch
) -> None:
    win_dir = tmp_path / "l00"
    win_dir.mkdir()
    (win_dir / "cmass.txt").write_text("100 99.0 99.0\n")
    (win_dir / "cmass-02.txt").write_text(
        "# step r1 r2\n"
        "0 19.0 29.0\n"
        "100 2.0 3.0\n"
        "200 2.5 3.5\n"
    )
    (win_dir / "cmass-01.txt").write_text(
        "# step r1 r2\n"
        "0 9.0 9.0\n"
        "0 9.1 9.1\n"
        "100 1.0 1.0\n"
        "200 1.5 1.5\n"
    )

    def _fail_generate_results_rest(*args, **kwargs):
        raise AssertionError("segmented cmass traces should avoid cpptraj extraction")

    monkeypatch.setattr(analysis_mod, "generate_results_rest", _fail_generate_results_rest)

    kT = 0.0019872041 * 300.0
    out = analysis_mod.RESTMBARAnalysis._extract_all_for_window(
        win_i=0,
        comp_folder=str(tmp_path),
        component="l",
        temperature=300.0,
        analysis_start_step=200,
        rfc=np.array([[1.0, 1.0], [1.0, 1.0]]),
        req=np.array([[2.0, 3.0], [2.5, 3.5]]),
        rty=["d", "d"],
        num_rest=2,
        num_win=2,
        truncate=False,
        dt=0.004,
        ntwx=100,
    )

    assert len(out) == 2
    assert list(out.columns) == [0.0, 1.0]
    np.testing.assert_allclose(out[0.0].to_numpy(), [0.0, 0.5 / kT])
    np.testing.assert_allclose(out[1.0].to_numpy(), [0.5 / kT, 0.0])


def test_generate_results_rest_accepts_successful_run_with_log(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "restraints.in").write_text(
        "parm vac.prmtop\n"
        "trajin md02.nc\n"
        "distance d0 :1@C :2@C out restraints.dat\n"
    )
    monkeypatch.chdir(tmp_path)

    calls: list[str] = []

    def _fake_run_with_log(command: str, working_dir=None):
        calls.append(command)
        assert Path(working_dir) == tmp_path
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(analysis_mod, "run_with_log", _fake_run_with_log)

    analysis_mod.generate_results_rest(["md-01.nc"], "l", top="full")

    assert calls == [
        f"{analysis_mod.cpptraj} -i restraints_curr.in > restraints.log 2>&1"
    ]
    assert (tmp_path / "restraints_curr.in").read_text().splitlines() == [
        "parm ../l-1/full.prmtop",
        "trajin md-01.nc",
        "distance d0 :1@C :2@C out restraints.dat",
    ]


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
