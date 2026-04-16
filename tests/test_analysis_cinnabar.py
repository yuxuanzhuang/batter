from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from batter.analysis import cinnabar as cinnabar_mod
from batter.cli.run import cli


class _FakeQuantity:
    def __init__(self, magnitude: float, unit: "_FakeUnit") -> None:
        self.m = float(magnitude)
        self.u = unit
        self.dimensionality = unit.dimensionality

    def to(self, unit: "_FakeUnit") -> "_FakeQuantity":
        return _FakeQuantity(self.m, unit)


class _FakeUnit:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dimensionality = name

    def __rmul__(self, value: float) -> _FakeQuantity:
        return _FakeQuantity(float(value), self)


class _FakeUnitModule:
    kilocalorie_per_mole = _FakeUnit("kcal/mol")
    kilojoule_per_mole = _FakeUnit("kJ/mol")
    kelvin = _FakeUnit("kelvin")


def _quantity_value(value) -> float:
    if hasattr(value, "m"):
        return float(value.m)
    return float(value)


class _FakeFEMap:
    def __init__(self) -> None:
        self.relative_rows: list[dict[str, object]] = []
        self.experimental_rows: list[dict[str, object]] = []
        self.absolute_df = pd.DataFrame()

    def add_relative_calculation(
        self,
        *,
        labelA,
        labelB,
        value,
        uncertainty,
        source="",
        temperature=None,
    ) -> None:
        self.relative_rows.append(
            {
                "labelA": labelA,
                "labelB": labelB,
                "DDG (kcal/mol)": _quantity_value(value),
                "uncertainty (kcal/mol)": _quantity_value(uncertainty),
                "source": source,
                "temperature": _quantity_value(temperature) if temperature is not None else 298.15,
            }
        )

    def add_experimental_measurement(
        self,
        *,
        label,
        value,
        uncertainty,
        source="",
        temperature=None,
    ) -> None:
        self.experimental_rows.append(
            {
                "label": label,
                "DG (kcal/mol)": _quantity_value(value),
                "uncertainty (kcal/mol)": _quantity_value(uncertainty),
                "source": source,
                "temperature": _quantity_value(temperature) if temperature is not None else 298.15,
            }
        )

    def generate_absolute_values(self) -> None:
        labels = sorted(
            {row["labelA"] for row in self.relative_rows}
            | {row["labelB"] for row in self.relative_rows}
        )
        self.absolute_df = pd.DataFrame(
            [
                {
                    "label": label,
                    "DG (kcal/mol)": float(idx),
                    "uncertainty (kcal/mol)": 0.1,
                    "source": "MLE",
                    "computational": True,
                }
                for idx, label in enumerate(labels)
            ]
        )

    def get_relative_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.relative_rows)

    def get_absolute_dataframe(self) -> pd.DataFrame:
        return self.absolute_df.copy()

    def draw_graph(self, *, filename: str | None = None, title: str = "") -> None:
        if filename:
            Path(filename).write_bytes(b"png")

    def to_legacy_graph(self) -> dict[str, int]:
        return {"n_edges": len(self.relative_rows)}


class _FakePlotting:
    @staticmethod
    def plot_DGs(graph, **kwargs) -> None:
        filename = kwargs.get("filename")
        if filename:
            Path(filename).write_bytes(b"png")

    @staticmethod
    def plot_DDGs(graph, **kwargs) -> None:
        filename = kwargs.get("filename")
        if filename:
            Path(filename).write_bytes(b"png")


@pytest.fixture()
def fake_cinnabar_stack(monkeypatch):
    monkeypatch.setattr(
        cinnabar_mod,
        "_import_cinnabar_stack",
        lambda: (_FakeFEMap, _FakePlotting, _FakeUnitModule),
    )


@pytest.fixture()
def rbfe_index_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run1",
                "ligand": "A~B",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": 1.0,
                "total_se": 0.2,
                "temperature": 300.0,
                "status": "success",
            },
            {
                "run_id": "run1",
                "ligand": "A~B",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": 1.4,
                "total_se": 0.2,
                "temperature": 300.0,
                "status": "success",
            },
            {
                "run_id": "run2",
                "ligand": "A~B",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": 0.8,
                "total_se": 0.4,
                "temperature": 300.0,
                "status": "success",
            },
            {
                "run_id": "run2",
                "ligand": "ignored",
                "original_name": "",
                "protocol": "abfe",
                "total_dG": -5.0,
                "total_se": 0.1,
                "temperature": 300.0,
                "status": "success",
            },
        ]
    )


def test_build_batter_rbfe_cinnabar_combines_runs(
    monkeypatch, fake_cinnabar_stack, rbfe_index_df: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: rbfe_index_df.copy())

    result = cinnabar_mod.build_batter_rbfe_cinnabar(tmp_path, run_ids=["run1", "run2"])

    assert len(result.edge_summary) == 1
    edge = result.edge_summary.iloc[0]
    assert edge["labelA"] == "A"
    assert edge["labelB"] == "B"
    assert edge["n_runs"] == 2
    assert edge["n_measurements"] == 3
    assert pytest.approx(edge["calc_DDG"], rel=1e-6) == 1.12
    assert result.absolute_summary is not None
    assert set(result.absolute_summary["label"]) == {"A", "B"}


def test_build_batter_rbfe_cinnabar_by_run_splits_runs(
    monkeypatch, fake_cinnabar_stack, rbfe_index_df: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: rbfe_index_df.copy())

    bundles = cinnabar_mod.build_batter_rbfe_cinnabar_by_run(
        tmp_path,
        run_ids=["run1", "run2"],
    )

    assert set(bundles) == {"run1", "run2"}
    assert bundles["run1"].edge_summary.iloc[0]["n_runs"] == 1
    assert bundles["run2"].edge_summary.iloc[0]["n_runs"] == 1


def test_build_batter_rbfe_cinnabar_can_split_bidirectional_edges(
    monkeypatch, fake_cinnabar_stack, tmp_path: Path
) -> None:
    df = pd.DataFrame(
        [
            {
                "run_id": "run1",
                "ligand": "A~B",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": 1.0,
                "total_se": 0.2,
                "temperature": 300.0,
                "status": "success",
            },
            {
                "run_id": "run1",
                "ligand": "B~A",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": -0.9,
                "total_se": 0.3,
                "temperature": 300.0,
                "status": "success",
            },
        ]
    )
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: df.copy())

    merged = cinnabar_mod.build_batter_rbfe_cinnabar(
        tmp_path,
        run_ids=["run1"],
        merge_bidirectional=True,
    )
    split = cinnabar_mod.build_batter_rbfe_cinnabar(
        tmp_path,
        run_ids=["run1"],
        merge_bidirectional=False,
    )

    assert len(merged.edge_summary) == 1
    assert merged.merge_bidirectional is True
    assert split.merge_bidirectional is False
    assert {(row.labelA, row.labelB) for row in split.edge_summary.itertuples(index=False)} == {
        ("A", "B"),
        ("B", "A"),
    }
    assert len(split.edge_summary) == 2


def test_write_cinnabar_outputs_writes_expected_files(
    monkeypatch, fake_cinnabar_stack, rbfe_index_df: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: rbfe_index_df.copy())

    result = cinnabar_mod.build_batter_rbfe_cinnabar(tmp_path)
    outputs = cinnabar_mod.write_cinnabar_outputs(result, tmp_path / "out")

    expected = {
        "raw_signed_csv",
        "edge_summary_csv",
        "cinnabar_relative_csv",
        "cinnabar_absolute_csv",
        "absolute_sorted_png",
        "network_png",
        "manifest_json",
    }
    assert expected.issubset(outputs)
    for key in expected:
        assert outputs[key].exists()


def test_write_cinnabar_outputs_manifest_records_split_directionality(
    monkeypatch, fake_cinnabar_stack, tmp_path: Path
) -> None:
    df = pd.DataFrame(
        [
            {
                "run_id": "run1",
                "ligand": "A~B",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": 1.0,
                "total_se": 0.2,
                "temperature": 300.0,
                "status": "success",
            },
            {
                "run_id": "run1",
                "ligand": "B~A",
                "original_name": "",
                "protocol": "rbfe",
                "total_dG": -0.9,
                "total_se": 0.3,
                "temperature": 300.0,
                "status": "success",
            },
        ]
    )
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: df.copy())

    result = cinnabar_mod.build_batter_rbfe_cinnabar(
        tmp_path,
        run_ids=["run1"],
        merge_bidirectional=False,
    )
    outputs = cinnabar_mod.write_cinnabar_outputs(result, tmp_path / "out")
    manifest = json.loads(outputs["manifest_json"].read_text())

    assert manifest["direction_mode"] == "split"
    assert manifest["n_directional_edges"] == 2
    assert manifest["n_reciprocal_pairs"] == 1
    assert manifest["reciprocal_pairs"] == ["A~B"]


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_fe_cinnabar_combined(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    called: dict[str, object] = {}
    sentinel = object()

    def _fake_build(**kwargs):
        called["build_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar",
        _fake_build,
    )

    def _fake_write(result, out_dir, **kwargs):
        called["write_result"] = result
        called["write_out_dir"] = out_dir
        called["write_kwargs"] = kwargs
        return {"manifest_json": Path(out_dir) / "manifest.json"}

    monkeypatch.setattr("batter.cli.fe_cmds.write_cinnabar_outputs", _fake_write)

    result = runner.invoke(cli, ["fe", "cinnabar", str(tmp_path)])

    assert result.exit_code == 0
    assert called["build_kwargs"]["work_dir"] == tmp_path
    assert called["build_kwargs"]["merge_bidirectional"] is True
    assert called["write_result"] is sentinel
    assert called["write_out_dir"] == tmp_path / "results" / "cinnabar"
    assert called["write_kwargs"]["absolute_offset"] == 0.0
    assert "combined Cinnabar bundle" in result.output


def test_cli_fe_cinnabar_split_runs(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    calls: list[tuple[object, Path]] = []
    bundles = {"run1": object(), "run2": object()}
    called: dict[str, object] = {}

    def _fake_build(**kwargs):
        called["build_kwargs"] = kwargs
        return bundles

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar_by_run",
        _fake_build,
    )
    monkeypatch.setattr(
        "batter.cli.fe_cmds.write_cinnabar_outputs",
        lambda result, out_dir, **kwargs: calls.append((result, out_dir)) or {},
    )

    result = runner.invoke(
        cli,
        [
            "fe",
            "cinnabar",
            str(tmp_path),
            "--split-runs",
            "--run-id",
            "run1",
            "--run-id",
            "run2",
            "--split-directions",
            "--absolute-offset",
            "1.5",
        ],
    )

    assert result.exit_code == 0
    assert called["build_kwargs"]["merge_bidirectional"] is False
    assert calls == [
        (bundles["run1"], tmp_path / "results" / "cinnabar" / "run1"),
        (bundles["run2"], tmp_path / "results" / "cinnabar" / "run2"),
    ]
    assert "2 per-run Cinnabar bundle" in result.output


def test_cli_fe_cinnabar_split_runs_passes_absolute_offset(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    calls: list[dict[str, object]] = []
    bundles = {"run1": object()}

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar_by_run",
        lambda **kwargs: bundles,
    )

    def _fake_write(result, out_dir, **kwargs):
        calls.append({"result": result, "out_dir": out_dir, "kwargs": kwargs})
        return {}

    monkeypatch.setattr("batter.cli.fe_cmds.write_cinnabar_outputs", _fake_write)

    result = runner.invoke(
        cli,
        [
            "fe",
            "cinnabar",
            str(tmp_path),
            "--split-runs",
            "--absolute-offset",
            "-2.0",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "result": bundles["run1"],
            "out_dir": tmp_path / "results" / "cinnabar" / "run1",
            "kwargs": {
                "method_name": "BATTER",
                "target_name": f"{tmp_path.name}:run1",
                "write_plots": True,
                "absolute_offset": -2.0,
            },
        }
    ]


def test_cli_fe_cinnabar_split_directions_warns_without_reciprocals(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    result_obj = type(
        "Result",
        (),
        {"edge_summary": pd.DataFrame([{"labelA": "A", "labelB": "B"}])},
    )()

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar",
        lambda **kwargs: result_obj,
    )
    monkeypatch.setattr(
        "batter.cli.fe_cmds.write_cinnabar_outputs",
        lambda result, out_dir, **kwargs: {},
    )

    result = runner.invoke(
        cli,
        ["fe", "cinnabar", str(tmp_path), "--split-directions"],
    )

    assert result.exit_code == 0
    assert "contain no reciprocal A~B/B~A transformations" in result.output
