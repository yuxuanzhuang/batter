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


class _FakeFEMapNoAbsolute(_FakeFEMap):
    def generate_absolute_values(self) -> None:
        raise RuntimeError("network is disconnected")


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


def test_build_batter_rbfe_cinnabar_warns_when_absolute_solution_missing(
    monkeypatch, rbfe_index_df: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cinnabar_mod,
        "_import_cinnabar_stack",
        lambda: (_FakeFEMapNoAbsolute, _FakePlotting, _FakeUnitModule),
    )
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: rbfe_index_df.copy())

    result = cinnabar_mod.build_batter_rbfe_cinnabar(tmp_path, run_ids=["run1", "run2"])

    assert result.absolute_summary is None
    assert result.absolute_warning is not None
    assert "Could not build a full absolute ΔG solution" in result.absolute_warning


def test_resolve_label_positions_separates_overlapping_boxes() -> None:
    specs = [
        {
            "base": cinnabar_mod.np.array([0.0, 0.0]),
            "tangent": cinnabar_mod.np.array([1.0, 0.0]),
            "normal": cinnabar_mod.np.array([0.0, 1.0]),
        },
        {
            "base": cinnabar_mod.np.array([0.0, 0.0]),
            "tangent": cinnabar_mod.np.array([0.0, 1.0]),
            "normal": cinnabar_mod.np.array([1.0, 0.0]),
        },
    ]

    resolved = cinnabar_mod._resolve_label_positions(specs, box_size=(60.0, 42.0))

    assert len(resolved) == 2
    assert not cinnabar_mod._label_rects_overlap(
        resolved[0],
        (60.0, 42.0),
        resolved[1],
        (60.0, 42.0),
    )


def test_network_graph_layout_spreads_dense_nodes() -> None:
    rows = []
    labels = [f"L{i}" for i in range(8)]
    for idx, label_a in enumerate(labels):
        for label_b in labels[idx + 1 :]:
            rows.append(
                {
                    "labelA": label_a,
                    "labelB": label_b,
                    "calc_DDG": 1.0,
                    "calc_dDDG": 0.2,
                    "n_runs": 1,
                    "n_measurements": 1,
                }
            )
    edge_summary = pd.DataFrame(rows)

    graph, pos = cinnabar_mod._network_graph_with_layout(edge_summary)
    radii = cinnabar_mod._layout_node_radii(graph)

    assert len(pos) == len(labels)
    nodes = list(pos)
    for idx, node_a in enumerate(nodes):
        for node_b in nodes[idx + 1 :]:
            dist = float(cinnabar_mod.np.linalg.norm(pos[node_b] - pos[node_a]))
            assert dist >= radii[node_a] + radii[node_b] - 1e-6


def test_png_layout_scale_grows_for_dense_networks() -> None:
    sparse_graph = cinnabar_mod._import_networkx().DiGraph()
    sparse_graph.add_edge("A", "B")
    sparse_graph.add_edge("B", "C")

    dense_graph = cinnabar_mod._import_networkx().complete_graph(8, create_using=cinnabar_mod._import_networkx().DiGraph())

    assert cinnabar_mod._png_layout_scale(dense_graph) > cinnabar_mod._png_layout_scale(sparse_graph)


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
        "dg_values_png",
        "network_png",
        "dashboard_html",
        "manifest_json",
    }
    assert expected.issubset(outputs)
    for key in expected:
        assert outputs[key].exists()
    dashboard_html = outputs["dashboard_html"].read_text().lower()
    assert "tabbar" in dashboard_html
    assert "sticky-note" in dashboard_html
    assert "event.stoppropagation()" in dashboard_html
    assert "openedgesticky" in dashboard_html
    assert "edgeassets" in dashboard_html
    assert "<polygon points=" in dashboard_html
    assert "marker-end=" not in dashboard_html
    assert "network-viewport" in dashboard_html
    assert "fitnetworkviewport" in dashboard_html
    assert "network-zoom-in" in dashboard_html


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


def test_write_cinnabar_outputs_also_writes_cycle_closure(
    monkeypatch, fake_cinnabar_stack, tmp_path: Path
) -> None:
    df = pd.DataFrame(
        [
            {
                "ligand": "A~B",
                "total_dG": 1.0,
                "total_se": 0.2,
                "status": "success",
                "temperature": 300.0,
            },
            {
                "ligand": "B~C",
                "total_dG": 1.0,
                "total_se": 0.3,
                "status": "success",
                "temperature": 300.0,
            },
            {
                "ligand": "C~A",
                "total_dG": -3.0,
                "total_se": 0.4,
                "status": "success",
                "temperature": 300.0,
            },
        ]
    )
    result = cinnabar_mod.dataframe_to_cinnabar(df)

    outputs = cinnabar_mod.write_cinnabar_outputs(result, tmp_path / "out")

    assert {
        "cycle_closure_nodes_csv",
        "cycle_closure_edges_csv",
        "cycle_closure_cycles_csv",
        "cycle_closure_network_png",
        "cycle_closure_dg_values_png",
        "cycle_closure_errors_png",
    }.issubset(outputs)
    for key in (
        "cycle_closure_network_png",
        "cycle_closure_dg_values_png",
        "cycle_closure_errors_png",
    ):
        assert outputs[key].exists()
    assert set(pd.read_csv(outputs["cycle_closure_nodes_csv"])["label"]) == {
        "A",
        "B",
        "C",
    }
    manifest = json.loads(outputs["manifest_json"].read_text())
    assert manifest["cycle_closure"]["status"] == "success"
    assert manifest["cycle_closure"]["n_cycles"] >= 1
    assert manifest["cycle_closure"]["edge_value_column"] == "ddG_wcc1"
    assert manifest["cycle_closure"]["node_value_column"] == "dG_wcc1"
    assert "cycle_closure_nodes_csv" in manifest["outputs"]
    assert "cycle_closure_network_png" in manifest["outputs"]

    dashboard_html = outputs["dashboard_html"].read_text().lower()
    assert "cycle-closure-toggle" in dashboard_html
    assert 'body class="show-cycle-closure"' in dashboard_html
    assert 'id="cycle-closure-toggle" type="checkbox" checked' in dashboard_html
    assert "show-cycle-closure" in dashboard_html
    assert "result-view-cycle" in dashboard_html
    assert "cycle-network-svg" in dashboard_html
    assert "cycle-network-viewport" in dashboard_html
    assert "cycle-network-zoom-in" in dashboard_html
    assert "cycle_closure_dg_values.png" in dashboard_html
    assert "cycle_closure_errors.png" in dashboard_html


def test_write_cinnabar_outputs_can_disable_cycle_closure(
    monkeypatch, fake_cinnabar_stack, rbfe_index_df: pd.DataFrame, tmp_path: Path
) -> None:
    monkeypatch.setattr(cinnabar_mod, "list_fe_runs", lambda work_dir: rbfe_index_df.copy())

    result = cinnabar_mod.build_batter_rbfe_cinnabar(tmp_path)
    outputs = cinnabar_mod.write_cinnabar_outputs(
        result,
        tmp_path / "out",
        write_cycle_closure=False,
    )

    assert "cycle_closure_nodes_csv" not in outputs
    manifest = json.loads(outputs["manifest_json"].read_text())
    assert manifest["cycle_closure"]["status"] == "disabled"


def test_auto_write_rbfe_cinnabar_for_run_uses_run_scoped_output_and_replicate_note(
    monkeypatch, tmp_path: Path
) -> None:
    fake_result = cinnabar_mod.CinnabarConversionResult(
        femap=object(),
        edge_summary=pd.DataFrame([{"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.2}]),
        raw_signed=pd.DataFrame(),
        absolute_warning="abs warning",
    )
    called: dict[str, object] = {}

    def _fake_build(work_dir, **kwargs):
        called["build_work_dir"] = work_dir
        called["build_kwargs"] = kwargs
        return fake_result

    def _fake_write(result, out_dir, **kwargs):
        called["write_result"] = result
        called["write_out_dir"] = out_dir
        called["write_kwargs"] = kwargs
        return {"dashboard_html": Path(out_dir) / "cinnabar_dashboard.html"}

    monkeypatch.setattr(cinnabar_mod, "build_batter_rbfe_cinnabar", _fake_build)
    monkeypatch.setattr(cinnabar_mod, "write_cinnabar_outputs", _fake_write)
    monkeypatch.setattr(
        cinnabar_mod,
        "list_fe_runs",
        lambda work_dir: pd.DataFrame(
            [
                {"run_id": "rep1", "protocol": "rbfe", "system_name": "sys"},
                {"run_id": "rep2", "protocol": "rbfe", "system_name": "sys"},
            ]
        ),
    )

    export = cinnabar_mod.auto_write_rbfe_cinnabar_for_run(tmp_path, "rep1")

    assert called["build_work_dir"] == tmp_path
    assert called["build_kwargs"] == {
        "run_ids": ["rep1"],
        "combine_by_run_first": True,
        "merge_bidirectional": True,
    }
    assert called["write_result"] is fake_result
    assert called["write_out_dir"] == tmp_path / "results" / "cinnabar" / "rep1"
    assert export["output_dir"] == tmp_path / "results" / "cinnabar" / "rep1"
    assert export["absolute_warning"] == "abs warning"
    note = str(export["replicate_note"])
    assert "batter fe cinnabar" in note
    assert "--run-id rep1" in note
    assert "--run-id rep2" in note


def test_auto_write_rbfe_cinnabar_for_run_omits_replicate_note_for_single_run(
    monkeypatch, tmp_path: Path
) -> None:
    fake_result = cinnabar_mod.CinnabarConversionResult(
        femap=object(),
        edge_summary=pd.DataFrame([{"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.2}]),
        raw_signed=pd.DataFrame(),
    )

    monkeypatch.setattr(
        cinnabar_mod,
        "build_batter_rbfe_cinnabar",
        lambda work_dir, **kwargs: fake_result,
    )
    monkeypatch.setattr(
        cinnabar_mod,
        "write_cinnabar_outputs",
        lambda result, out_dir, **kwargs: {"dashboard_html": Path(out_dir) / "cinnabar_dashboard.html"},
    )
    monkeypatch.setattr(
        cinnabar_mod,
        "list_fe_runs",
        lambda work_dir: pd.DataFrame(
            [{"run_id": "rep1", "protocol": "rbfe", "system_name": "sys"}]
        ),
    )

    export = cinnabar_mod.auto_write_rbfe_cinnabar_for_run(tmp_path, "rep1")

    assert export["replicate_note"] is None


def test_read_cinnabar_outputs_reads_all_bundle_csv_tables(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "cinnabar"
    bundle_dir.mkdir()
    relative = pd.DataFrame(
        [{"labelA": "A", "labelB": "B", "DDG (kcal/mol)": 1.0}]
    )
    absolute = pd.DataFrame(
        [{"label": "A", "DG (kcal/mol)": -5.0}]
    )
    cycle_nodes = pd.DataFrame(
        [{"label": "A", "dG_cc": -5.1, "path_dependent_error": 0.2}]
    )
    custom = pd.DataFrame([{"source": "plugin", "value": 3}])
    relative.to_csv(bundle_dir / "cinnabar_relative.csv", index=False)
    absolute.to_csv(bundle_dir / "cinnabar_absolute.csv", index=False)
    cycle_nodes.to_csv(bundle_dir / "cycle_closure_nodes.csv", index=False)
    custom.to_csv(bundle_dir / "custom_table.csv", index=False)

    tables = cinnabar_mod.read_cinnabar_outputs(bundle_dir)

    assert tables["relative"] is not None
    pd.testing.assert_frame_equal(tables["relative"], relative)
    assert tables["absolute"] is not None
    pd.testing.assert_frame_equal(tables["absolute"], absolute)
    assert tables["cycle_closure_nodes"] is not None
    pd.testing.assert_frame_equal(tables["cycle_closure_nodes"], cycle_nodes)
    assert tables["cycle_closure_edges"] is None
    assert tables["raw_signed"] is None
    assert tables["custom_table"] is not None
    pd.testing.assert_frame_equal(tables["custom_table"], custom)


def test_convert_cinnabar_outputs_to_csv_writes_all_available_csvs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "cinnabar"
    out_dir = tmp_path / "exported"
    bundle_dir.mkdir()
    relative = pd.DataFrame(
        [{"labelA": "A", "labelB": "B", "DDG (kcal/mol)": 1.0}]
    )
    absolute = pd.DataFrame(
        [{"label": "A", "DG (kcal/mol)": -5.0}]
    )
    cycle_edges = pd.DataFrame(
        [{"labelA": "A", "labelB": "B", "ddG_cc": 1.2, "pair_error": 0.1}]
    )
    relative.to_csv(bundle_dir / "cinnabar_relative.csv", index=False)
    absolute.to_csv(bundle_dir / "cinnabar_absolute.csv", index=False)
    cycle_edges.to_csv(bundle_dir / "cycle_closure_edges.csv", index=False)

    outputs = cinnabar_mod.convert_cinnabar_outputs_to_csv(
        bundle_dir,
        out_dir,
        relative_name="rbfe_relative.csv",
        absolute_name="rbfe_absolute.csv",
    )

    assert outputs["relative_csv"] == out_dir / "rbfe_relative.csv"
    assert outputs["absolute_csv"] == out_dir / "rbfe_absolute.csv"
    assert outputs["cycle_closure_edges_csv"] == out_dir / "cycle_closure_edges.csv"
    pd.testing.assert_frame_equal(pd.read_csv(outputs["relative_csv"]), relative)
    pd.testing.assert_frame_equal(pd.read_csv(outputs["absolute_csv"]), absolute)
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs["cycle_closure_edges_csv"]),
        cycle_edges,
    )


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
    assert called["write_kwargs"]["write_cycle_closure"] is True
    assert "combined Cinnabar bundle" in result.output


def test_cli_fe_cinnabar_can_disable_cycle_closure(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar",
        lambda **kwargs: object(),
    )

    def _fake_write(result, out_dir, **kwargs):
        called["write_kwargs"] = kwargs
        return {}

    monkeypatch.setattr("batter.cli.fe_cmds.write_cinnabar_outputs", _fake_write)

    result = runner.invoke(cli, ["fe", "cinnabar", str(tmp_path), "--no-cycle-closure"])

    assert result.exit_code == 0
    assert called["write_kwargs"]["write_cycle_closure"] is False


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
                "write_cycle_closure": True,
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


def test_cli_fe_cinnabar_warns_when_absolute_solution_missing(
    monkeypatch, tmp_path: Path, runner: CliRunner
) -> None:
    result_obj = type(
        "Result",
        (),
        {
            "edge_summary": pd.DataFrame([{"labelA": "A", "labelB": "B"}]),
            "absolute_warning": "Could not build a full absolute ΔG solution from the RBFE network.",
        },
    )()

    monkeypatch.setattr(
        "batter.cli.fe_cmds.build_batter_rbfe_cinnabar",
        lambda **kwargs: result_obj,
    )
    monkeypatch.setattr(
        "batter.cli.fe_cmds.write_cinnabar_outputs",
        lambda result, out_dir, **kwargs: {},
    )

    result = runner.invoke(cli, ["fe", "cinnabar", str(tmp_path)])

    assert result.exit_code == 0
    assert "Could not build a full absolute ΔG solution" in result.output
