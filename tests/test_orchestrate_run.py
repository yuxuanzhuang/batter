from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from batter.config.simulation import SimulationConfig
from batter.orchestrate.results_io import extract_ligand_metadata
from batter.orchestrate.run import save_fe_records
from batter.runtime.fe_repo import FERecord, FEResultsRepository
from batter.runtime.portable import ArtifactStore
from batter.systems.core import SimSystem, SystemMeta
from batter.orchestrate import run as run_mod
from batter.orchestrate import run_support as rs


def _make_sim_cfg() -> SimulationConfig:
    return SimulationConfig.model_validate(
        {
            "system_name": "sys",
            "fe_type": "rest",
            "dec_int": "mbar",
            "components": ["z"],
            "component_lambdas": {"z": [0.0, 1.0]},
            "lambdas": [0.0, 1.0],
            "temperature": 300.0,
            "analysis_start_step": 0,
            "buffer_x": 15.0,
            "buffer_y": 15.0,
            "buffer_z": 15.0,
        }
    )


def test_rbfe_network_review_note_mentions_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"

    note = run_mod._rbfe_network_review_note(run_dir)

    assert str(run_dir / "artifacts" / "config" / "rbfe_network.html") in note
    assert str(run_dir / "artifacts" / "config" / "rbfe_network.json") in note
    assert "edit rbfe_network.json" in note
    assert "reloads its pairs field" in note
    assert "fall back to the configured atom mapper" in note
    assert "Identical duplicate ligands and full-map edges are omitted" in note
    assert "run.only_rbfe_network" in note
    assert "--full-rbfe" in note


@pytest.mark.parametrize("has_results", [False])
def test_save_fe_records_failure(tmp_path: Path, has_results: bool) -> None:
    run_dir = tmp_path / "run1"
    child_root = run_dir / "simulations" / "lig1"
    (child_root / "fe" / "Results").mkdir(parents=True, exist_ok=True)
    ligand_names_path = run_dir / "artifacts" / "ligand_names.json"
    ligand_names_path.parent.mkdir(parents=True, exist_ok=True)
    ligand_names_path.write_text(json.dumps({"lig1": "Ligand One Original"}))

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:lig1:run1",
        root=child_root,
        meta=SystemMeta(ligand="lig1", residue_name="lig1"),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="abfe",
    )

    assert failures
    df = pd.read_csv(run_dir / "results" / "index.csv")
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "no_totals_found"
    assert row["original_name"] == "Ligand One Original"
    failure_json = run_dir / "results" / "run1" / "lig1" / "failure.json"
    assert failure_json.exists()


def test_save_fe_records_uses_stored_original_name_for_success(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run1"
    child_root = run_dir / "simulations" / "lig1"
    results_dir = child_root / "fe" / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "Results.dat").write_text("Total\t-1.0\t0.1\n")
    ligand_names_path = run_dir / "artifacts" / "ligand_names.json"
    ligand_names_path.parent.mkdir(parents=True, exist_ok=True)
    ligand_names_path.write_text(json.dumps({"lig1": "Ligand One Original"}))

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:lig1:run1",
        root=child_root,
        meta=SystemMeta(ligand="lig1", residue_name="lig1"),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="abfe",
    )

    assert not failures
    record = repo.load("run1", "lig1")
    assert record.original_name == "Ligand One Original"
    df = pd.read_csv(run_dir / "results" / "index.csv")
    row = df[(df["run_id"] == "run1") & (df["ligand"] == "lig1")].iloc[0]
    assert row["original_name"] == "Ligand One Original"


def test_extract_ligand_metadata_records_both_rbfe_endpoints(tmp_path: Path) -> None:
    ref_dir = tmp_path / "params" / "ref"
    alt_dir = tmp_path / "params" / "alt"
    ref_dir.mkdir(parents=True)
    alt_dir.mkdir(parents=True)
    (ref_dir / "metadata.json").write_text(
        json.dumps(
            {
                "canonical_smiles": "CC",
                "input_path": "/inputs/ref.sdf",
                "aliases": ["Ref Ligand"],
            }
        )
    )
    (alt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "canonical_smiles": "CCC",
                "input_path": "/inputs/alt.sdf",
                "aliases": ["Alt Ligand"],
            }
        )
    )
    child = SimSystem(
        name="sys:pair:run1",
        root=tmp_path / "pair",
        meta=SystemMeta(
            ligand="ref~alt",
            residue_name="REF",
            mode="RBFE",
            param_dir_dict={"REF": str(ref_dir), "ALT": str(alt_dir)},
            extras={
                "ligand_ref": "ref",
                "ligand_alt": "alt",
                "residue_ref": "REF",
                "residue_alt": "ALT",
            },
        ),
    )

    canonical_smiles, original_name, original_path = extract_ligand_metadata(child)

    assert canonical_smiles == "CC~CCC"
    assert original_name == "Ref Ligand~Alt Ligand"
    assert original_path == "/inputs/ref.sdf~/inputs/alt.sdf"


def test_save_fe_records_copies_rbfe_network_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    config_dir = run_dir / "artifacts" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "rbfe_network.png").write_text("png")
    (config_dir / "rbfe_network.html").write_text("<html></html>")

    child_root = run_dir / "simulations" / "pair1"
    results_dir = child_root / "fe" / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "Results.dat").write_text("Total\t-1.0\t0.1\n")

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:pair1:run1",
        root=child_root,
        meta=SystemMeta(
            ligand="pair1",
            residue_name="lig1",
            mode="RBFE",
            extras={
                "ligand_ref": "A",
                "ligand_alt": "B",
                "residue_ref": "A",
                "residue_alt": "B",
            },
        ),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="rbfe",
    )

    assert not failures
    out = run_dir / "results" / "run1" / "pair1" / "Results" / "rbfe_network.png"
    assert out.exists()
    html_out = run_dir / "results" / "run1" / "pair1" / "Results" / "rbfe_network.html"
    assert html_out.exists()


def test_build_rbfe_network_plan_writes_planned_html(
    monkeypatch, tmp_path: Path
) -> None:
    pytest.importorskip("networkx")
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"], ["A", "C"]]}))
    atom_mapping_file = tmp_path / "atom_mapping.json"
    atom_mapping_file.write_text(json.dumps({"A~B": {"0": 1, "2": 3}}))
    cfg = RBFENetworkArgs(
        mapping_file=mapping_file,
        atom_mapping_file=atom_mapping_file,
    )
    seen: dict[str, object] = {}

    def _fake_mapping_artifacts(**kwargs):
        overrides = kwargs["atom_mapping_overrides"]
        seen["manual_A_B"] = overrides.get_b_to_a("A", "B")
        return {
            "A~B": {
                "image_data_uri": "data:image/png;base64,ZmFrZQ==",
                "mapper": kwargs["atom_mapper"],
                "n_mapped": 2,
            }
        }

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        _fake_mapping_artifacts,
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B", "C"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
            "C": str(tmp_path / "C.sdf"),
        },
        cfg,
        tmp_path,
    )

    assert payload["pairs"] == [["A", "B"], ["A", "C"]]
    assert payload["atom_mapping_file"] == str(atom_mapping_file)
    assert payload["n_atom_mapping_overrides"] == 1
    assert seen["manual_A_B"] == {0: 1, 2: 3}
    html_path = tmp_path / "rbfe_network.html"
    assert html_path.exists()
    html_text = html_path.read_text()
    assert "network-viewport" in html_text
    assert "plannedEdges" in html_text
    assert "A~B" in html_text
    assert "data:image/png;base64,ZmFrZQ==" in html_text
    assert payload["mapping_artifacts_dir"] == "rbfe_mappings"


def test_planned_rbfe_network_html_collapses_bidirectional_edges(
    tmp_path: Path,
) -> None:
    pytest.importorskip("networkx")
    from batter.analysis.network import write_planned_rbfe_network_html

    html_path = tmp_path / "rbfe_network.html"
    written = write_planned_rbfe_network_html(
        ligands=["A", "B", "C", "D"],
        pairs=[
            ["A", "B"],
            ["B", "A"],
            ["B", "C"],
            ["C", "A"],
            ["C", "D"],
        ],
        out_path=html_path,
        metadata={"both_directions": True},
        edge_assets={
            "A~B": {
                "mapping_score_rmsd": 0.91,
                "mapping_rmsd": 0.18,
                "pocket_shape_score": 0.84,
                "pocket_grid_score": 0.81,
                "pocket_grid_containment": 0.93,
                "pocket_grid_jaccard": 0.59,
            },
            "B~A": {
                "mapping_score_rmsd": 0.89,
                "mapping_rmsd": 0.21,
                "pocket_shape_score": 0.82,
                "pocket_grid_score": 0.79,
                "pocket_grid_containment": 0.91,
                "pocket_grid_jaccard": 0.57,
            },
            "B~C": {"mapping_score_ratio_mapped_atoms": 0.72},
        },
    )

    assert written
    html_text = html_path.read_text()
    assert html_text.count('class="edge-path"') == 4
    assert "displayed edges" in html_text
    assert '"pair_indexes": [1, 2]' in html_text
    assert "A &lt;-&gt; B" in html_text or "A <-> B" in html_text
    assert "connectivity_score" in html_text
    assert "edge-metric-select" in html_text
    assert "edgeMetricDefinitions" in html_text
    assert "Pocket similarity" in html_text
    assert "pocket_shape_score" in html_text
    assert 'const defaultEdgeMetric = "pocket_shape_score"' in html_text
    assert '"pocket_shape_score": 0.83' in html_text
    assert "Pocket containment" in html_text
    assert "Kartograf RMSD score" in html_text
    assert "mapping_score_rmsd" in html_text
    assert '"mapping_score_rmsd": 0.9' in html_text
    assert "#dc2626" in html_text
    assert "#f59e0b" in html_text


def test_build_rbfe_network_plan_adds_atom_mapping_edges_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"]]}))
    atom_mapping_file = tmp_path / "atom_mapping.json"
    atom_mapping_file.write_text(json.dumps({"B~C": {"0": 1}}))
    seen: dict[str, object] = {}

    def _fake_mapping_artifacts(**kwargs):
        seen["pairs"] = kwargs["pairs"]
        return {}

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        _fake_mapping_artifacts,
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B", "C"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
            "C": str(tmp_path / "C.sdf"),
        },
        RBFENetworkArgs(
            mapping_file=mapping_file,
            atom_mapping_file=atom_mapping_file,
            add_atom_mapping_edges=True,
        ),
        tmp_path,
    )

    assert payload["pairs"] == [["A", "B"], ["B", "C"]]
    assert payload["add_atom_mapping_edges"] is True
    assert payload["added_atom_mapping_edges"] == [["B", "C"]]
    assert seen["pairs"] == [["A", "B"], ["B", "C"]]


def test_build_rbfe_network_plan_defaults_septop_to_pocket_shape_scorer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    seen: dict[str, object] = {}

    def _fake_konnektor_pairs(*args, **kwargs):
        seen["network_scorer"] = kwargs["network_scorer"]
        seen["protocol"] = kwargs["protocol"]
        return [("A", "B")]

    def _fake_mapping_artifacts(**kwargs):
        return {}

    monkeypatch.setattr("batter.rbfe.konnektor_pairs", _fake_konnektor_pairs)
    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        _fake_mapping_artifacts,
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B"],
        {"A": str(tmp_path / "A.sdf"), "B": str(tmp_path / "B.sdf")},
        RBFENetworkArgs(mapping="default"),
        tmp_path,
        protocol="rbfe_septop",
    )

    assert payload["pairs"] == [["A", "B"]]
    assert payload["network_scorer"] == "pocket_shape"
    assert seen["network_scorer"] == "auto"
    assert seen["protocol"] == "rbfe_septop"


def test_build_rbfe_network_plan_skips_identical_duplicate_ligands(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    monkeypatch.setattr(
        "batter.rbfe.ligand_identity_key",
        lambda path: "same" if Path(path).stem in {"A", "B"} else Path(path).stem,
    )
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"], ["B", "C"]]}))
    seen: dict[str, object] = {}

    def _fake_mapping_artifacts(**kwargs):
        seen["pairs"] = kwargs["pairs"]
        seen["ligand_files"] = kwargs["ligand_files"]
        return {}

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        _fake_mapping_artifacts,
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B", "C"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
            "C": str(tmp_path / "C.sdf"),
        },
        RBFENetworkArgs(mapping_file=mapping_file),
        tmp_path,
    )

    assert payload["ligands"] == ["A", "C"]
    assert payload["pairs"] == [["A", "C"]]
    assert payload["skipped_identical_ligands"] == [
        {"ligand": "B", "kept": "A", "identity": "same"}
    ]
    assert payload["dropped_identical_ligand_pairs"] == [
        {"pair": ["A", "B"], "kept": "A", "reason": "identical_ligands"}
    ]
    assert payload["remapped_identical_ligand_pairs"] == [
        {"from": ["B", "C"], "to": ["A", "C"], "reason": "identical_ligands"}
    ]
    assert seen["pairs"] == [["A", "C"]]
    assert set(seen["ligand_files"]) == {"A", "C"}


def test_build_rbfe_network_plan_all_identical_ligands_writes_empty_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    monkeypatch.setattr("batter.rbfe.ligand_identity_key", lambda path: "same")

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
        },
        RBFENetworkArgs(mapping="default"),
        tmp_path,
    )

    assert payload["ligands"] == ["A"]
    assert payload["pairs"] == []
    assert payload["skipped_identical_ligands"] == [
        {"ligand": "B", "kept": "A", "identity": "same"}
    ]
    assert json.loads((tmp_path / "rbfe_network.json").read_text())["pairs"] == []


def test_build_rbfe_network_plan_skips_full_map_edges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    monkeypatch.setattr(
        "batter.rbfe.ligand_identity_key",
        lambda path: Path(path).stem,
    )
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"], ["A", "C"]]}))
    seen: dict[str, object] = {}

    def _fake_mapping_artifacts(**kwargs):
        seen["pairs"] = kwargs["pairs"]
        return {
            "A~B": {
                "full_atom_mapping": False,
                "full_heavy_atom_mapping": True,
                "n_mapped": 6,
                "n_ref_atoms": 8,
                "n_alt_atoms": 8,
                "n_ref_heavy_atoms": 6,
                "n_alt_heavy_atoms": 6,
            },
            "A~C": {
                "full_atom_mapping": False,
                "n_mapped": 4,
                "n_ref_atoms": 6,
                "n_alt_atoms": 7,
            },
        }

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        _fake_mapping_artifacts,
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B", "C"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
            "C": str(tmp_path / "C.sdf"),
        },
        RBFENetworkArgs(mapping_file=mapping_file),
        tmp_path,
    )

    assert seen["pairs"] == [["A", "B"], ["A", "C"]]
    assert payload["pairs"] == [["A", "C"]]
    assert payload["skipped_full_atom_map_edges"] == [
        {
            "pair": ["A", "B"],
            "n_mapped": 6,
            "n_ref_atoms": 8,
            "n_alt_atoms": 8,
            "n_ref_heavy_atoms": 6,
            "n_alt_heavy_atoms": 6,
            "reason": "full_heavy_atom_mapping",
        }
    ]


def test_build_rbfe_network_plan_treats_reverse_atom_mapping_edge_as_planned(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["B", "A"]]}))
    atom_mapping_file = tmp_path / "atom_mapping.json"
    atom_mapping_file.write_text(json.dumps({"A~B": {"0": 1}}))

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        lambda **kwargs: {},
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
        },
        RBFENetworkArgs(
            mapping_file=mapping_file,
            atom_mapping_file=atom_mapping_file,
        ),
        tmp_path,
    )

    assert payload["pairs"] == [["B", "A"]]
    assert "added_atom_mapping_edges" not in payload


def test_build_rbfe_network_plan_leaves_unplanned_atom_mapping_edges_unused_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from batter.config.run import RBFENetworkArgs

    monkeypatch.setitem(sys.modules, "konnektor", types.ModuleType("konnektor"))
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"]]}))
    atom_mapping_file = tmp_path / "atom_mapping.json"
    atom_mapping_file.write_text(json.dumps({"B~C": {"0": 1}}))

    monkeypatch.setattr(
        "batter.rbfe.write_planned_mapping_artifacts",
        lambda **kwargs: {},
    )

    payload = run_mod._build_rbfe_network_plan(
        ["A", "B", "C"],
        {
            "A": str(tmp_path / "A.sdf"),
            "B": str(tmp_path / "B.sdf"),
            "C": str(tmp_path / "C.sdf"),
        },
        RBFENetworkArgs(
            mapping_file=mapping_file,
            atom_mapping_file=atom_mapping_file,
        ),
        tmp_path,
    )

    assert payload["pairs"] == [["A", "B"]]
    assert payload["add_atom_mapping_edges"] is False
    assert payload["unused_atom_mapping_overrides"] == [["B", "C"]]


def test_rbfe_preflight_requires_atom_mapping_file(tmp_path: Path) -> None:
    rc = SimpleNamespace(
        protocol="rbfe",
        rbfe=SimpleNamespace(atom_mapping_file=tmp_path / "missing.json"),
    )

    with pytest.raises(FileNotFoundError, match="rbfe.atom_mapping_file"):
        run_mod._preflight_rbfe_mapping_files(rc, tmp_path / "run")


def test_rbfe_preflight_requires_network_file_when_prepare_marker_exists(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config_dir = run_dir / "artifacts" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "prepare_rbfe.ok").write_text("ok\n")
    rc = SimpleNamespace(protocol="rbfe", rbfe=None)

    with pytest.raises(FileNotFoundError, match="rbfe_network.json"):
        run_mod._preflight_rbfe_mapping_files(rc, run_dir)


def test_rbfe_network_pair_guard_rejects_empty_network(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "rbfe_network.json").write_text(
        json.dumps({"ligands": ["A", "B"], "pairs": []})
    )

    with pytest.raises(RuntimeError, match="no ligand pairs"):
        run_mod._require_rbfe_network_has_pairs(config_dir)


def test_prepare_rbfe_handler_writes_parent_stage_marker(
    monkeypatch, tmp_path: Path
) -> None:
    from batter.exec.handlers.prepare_rbfe import prepare_rbfe_handler
    from batter.orchestrate.state_registry import get_phase_state
    from batter.pipeline.payloads import StepPayload
    from batter.pipeline.step import Step

    called: dict[str, object] = {}

    def _fake_build(ligands, lig_map, rbfe_cfg, config_dir, **kwargs):
        called["ligands"] = list(ligands)
        called["lig_map"] = dict(lig_map)
        called["config_dir"] = config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "rbfe_network.json").write_text(
            json.dumps({"ligands": ligands, "pairs": [["A", "B"]]})
        )
        return {"pairs": [["A", "B"]]}

    monkeypatch.setattr(run_mod, "_build_rbfe_network_plan", _fake_build)
    payload = StepPayload(
        sys_params={
            "ligand_paths": {
                "A": tmp_path / "A.sdf",
                "B": tmp_path / "B.sdf",
            },
            "rbfe": {"mapping": "default"},
        }
    )
    system = SimSystem(name="sys:run1", root=tmp_path)

    prepare_rbfe_handler(
        Step(name="prepare_rbfe"),
        system,
        payload.model_dump(),
    )

    assert called["ligands"] == ["A", "B"]
    assert (tmp_path / "artifacts" / "config" / "prepare_rbfe.ok").exists()
    state = get_phase_state(tmp_path, "prepare_rbfe")
    assert state.success == [
        ["artifacts/config/rbfe_network.json", "artifacts/config/prepare_rbfe.ok"]
    ]


def test_rbfe_pipeline_prepares_network_before_equil() -> None:
    from batter.pipeline.factory import make_rbfe_pipeline

    pipeline = make_rbfe_pipeline(_make_sim_cfg(), sys_params={})
    names = [step.name for step in pipeline.ordered_steps()]

    assert names[:4] == [
        "system_prep",
        "param_ligands",
        "prepare_rbfe",
        "prepare_equil",
    ]
    assert pipeline.dependencies("prepare_rbfe") == ["param_ligands"]
    assert pipeline.dependencies("prepare_equil") == ["prepare_rbfe"]


def test_abfe_diff_pipeline_uses_pre_fe_equil_before_final_fe() -> None:
    from batter.orchestrate.pipeline_utils import select_pipeline

    pipeline = select_pipeline(
        "abfe_diff",
        _make_sim_cfg(),
        only_fe_prep=False,
        sys_params={},
    )
    names = [step.name for step in pipeline.ordered_steps()]

    assert "pre_prepare_fe" in names
    assert "pre_fe_equil" in names
    assert names.index("pre_fe_equil") < names.index("prepare_fe")
    assert pipeline.dependencies("pre_fe_equil") == ["pre_prepare_fe"]
    assert pipeline.dependencies("prepare_fe") == ["pre_fe_equil"]


def test_ligand_rest_pipeline_uses_normal_single_ligand_fe_flow() -> None:
    from batter.orchestrate.pipeline_utils import select_pipeline

    pipeline = select_pipeline(
        "ligand-rest",
        _make_sim_cfg().model_copy(
            update={
                "fe_type": "ligand_rest",
                "components": ["l"],
                "component_lambdas": {"l": [0.0, 1.0]},
                "dic_n_steps": {"l": 100_000},
                "lig_dihcf_force": 10.0,
            }
        ),
        only_fe_prep=False,
        sys_params={},
    )
    names = [step.name for step in pipeline.ordered_steps()]

    assert "pre_prepare_fe" not in names
    assert "pre_fe_equil" not in names
    assert names[names.index("equil_analysis") + 1] == "prepare_fe"
    assert pipeline.dependencies("prepare_fe") == ["equil_analysis"]


def test_save_fe_records_copies_rbfe_mapping_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run1"
    config_dir = run_dir / "artifacts" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "rbfe_network.png").write_text("png")

    child_root = run_dir / "simulations" / "pair1"
    results_dir = child_root / "fe" / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "Results.dat").write_text("Total\t-1.0\t0.1\n")

    mapping_dir = child_root / "fe" / "x" / "x-1"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "mapping.json").write_text('{"0": 0}')
    (mapping_dir / "mapping.png").write_text("png")

    sim_cfg = _make_sim_cfg()
    child = SimSystem(
        name="sys:pair1:run1",
        root=child_root,
        meta=SystemMeta(
            ligand="pair1",
            residue_name="lig1",
            mode="RBFE",
            extras={
                "ligand_ref": "A",
                "ligand_alt": "B",
                "residue_ref": "A",
                "residue_alt": "B",
            },
        ),
    )

    store = ArtifactStore(run_dir)
    repo = FEResultsRepository(store)

    failures = save_fe_records(
        run_dir=run_dir,
        run_id="run1",
        children_all=[child],
        sim_cfg_updated=sim_cfg,
        repo=repo,
        protocol="rbfe",
    )

    assert not failures
    out_results = run_dir / "results" / "run1" / "pair1" / "Results"
    assert (out_results / "mapping.json").exists()
    assert (out_results / "mapping.png").exists()
    assert not (out_results / "kartograf.json").exists()
    assert not (out_results / "kartograf_mapping.png").exists()


def test_compute_run_signature_excludes_run_section(tmp_path: Path) -> None:
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
run:
  output_folder: out
create:
  system_name: sys
fe_sim: {}
protocol: abfe
"""
    )
    sig, payload = run_mod._compute_run_signature(yaml_path, {"override": 1})
    assert isinstance(sig, str) and len(sig) == 64
    assert "run" not in payload["config"]
    assert set(payload["config"].keys()) <= {"create", "fe_sim", "fe"}
    assert payload["run_overrides"] == {}


def test_maybe_regenerate_rbfe_network_after_pruning_triggers_rebuild(
    monkeypatch, tmp_path: Path
) -> None:
    payload = {
        "ligands": ["A", "B", "C"],
        "pairs": [["A", "B"], ["A", "C"]],
        "mapping": "default",
    }
    called: dict[str, object] = {}

    def _fake_build(ligands, lig_map, rbfe_cfg, config_dir, **kwargs):
        called["ligands"] = list(ligands)
        called["lig_map"] = dict(lig_map)
        called["config_dir"] = config_dir
        return {"ligands": ["B", "C"], "pairs": [["B", "C"]], "mapping": "default"}

    monkeypatch.setattr(run_mod, "_build_rbfe_network_plan", _fake_build)

    out = run_mod._maybe_regenerate_rbfe_network_after_pruning(
        available_ligands=["B", "C"],
        lig_map={"A": "a.sdf", "B": "b.sdf", "C": "c.sdf"},
        payload=payload,
        rbfe_cfg=SimpleNamespace(mapping="default"),
        config_dir=tmp_path,
    )

    assert called["ligands"] == ["B", "C"]
    assert set(called["lig_map"]) == {"B", "C"}
    assert called["config_dir"] == tmp_path
    assert out["pairs"] == [["B", "C"]]


def test_maybe_regenerate_rbfe_network_after_pruning_noop_when_no_prune(
    monkeypatch, tmp_path: Path
) -> None:
    payload = {"ligands": ["A", "B"], "pairs": [["A", "B"]], "mapping": "default"}

    def _unexpected(*args, **kwargs):
        raise AssertionError("regeneration should not be called")

    monkeypatch.setattr(run_mod, "_build_rbfe_network_plan", _unexpected)

    out = run_mod._maybe_regenerate_rbfe_network_after_pruning(
        available_ligands=["A", "B"],
        lig_map={"A": "a.sdf", "B": "b.sdf"},
        payload=payload,
        rbfe_cfg=SimpleNamespace(mapping="default"),
        config_dir=tmp_path,
    )
    assert out is payload


def test_maybe_regenerate_rbfe_network_after_pruning_falls_back_on_error(
    monkeypatch, tmp_path: Path
) -> None:
    payload = {"ligands": ["A", "B", "C"], "pairs": [["A", "B"], ["A", "C"]]}

    def _raises(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_mod, "_build_rbfe_network_plan", _raises)

    out = run_mod._maybe_regenerate_rbfe_network_after_pruning(
        available_ligands=["B", "C"],
        lig_map={"A": "a.sdf", "B": "b.sdf", "C": "c.sdf"},
        payload=payload,
        rbfe_cfg=SimpleNamespace(mapping="default"),
        config_dir=tmp_path,
    )
    assert out is payload


def test_stored_payload_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "exec"
    path = run_mod._payload_path(run_dir)
    payload = {"config": {"a": 1}, "run_overrides": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    assert run_mod._stored_payload(run_dir) == payload


def test_resolve_signature_conflict_reports_diffs(tmp_path: Path, caplog) -> None:
    stored_payload = {"config": {"a": 1}}
    current_payload = {"config": {"a": 2}}
    keep = run_mod._resolve_signature_conflict(
        "aaa",
        "bbb",
        requested_run_id="auto",
        allow_run_id_mismatch=False,
        run_id="rid",
        run_dir=tmp_path,
        stored_payload=stored_payload,
        current_payload=current_payload,
    )
    assert keep is False


def test_normalize_for_hash_strips_output_folder_and_paths(tmp_path: Path) -> None:
    payload = {
        "create": {"output_folder": "/tmp/out", "protein": tmp_path / "pdb.pdb"},
        "extra": [Path("/a/b"), {"c": "d"}],
    }
    normalized = rs.normalize_for_hash(payload)
    assert "output_folder" not in normalized["create"]
    assert normalized["create"]["protein"] == str(tmp_path / "pdb.pdb")
    assert normalized["extra"][0] == "/a/b"


def test_resolve_signature_conflict_allows_mismatch_when_flag(
    tmp_path: Path, caplog
) -> None:
    keep = rs.resolve_signature_conflict(
        stored_sig="old",
        config_signature="new",
        requested_run_id="run1",
        allow_run_id_mismatch=True,
        run_id="run1",
        run_dir=tmp_path,
        stored_payload={"config": {"x": 1}},
        current_payload={"config": {"x": 2}},
    )
    assert keep is True


def test_resolve_signature_conflict_raises_on_mismatch(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        rs.resolve_signature_conflict(
            stored_sig="old",
            config_signature="new",
            requested_run_id="run1",
            allow_run_id_mismatch=False,
            run_id="run1",
            run_dir=tmp_path,
            stored_payload={"config": {"y": 1}},
            current_payload={"config": {"y": 2}},
        )


def test_select_system_builder_validates_system_type() -> None:
    builder = rs.select_system_builder("abfe", system_type=None)
    assert builder is not None
    abfe_diff_builder = rs.select_system_builder("ABFE-diff", system_type=None)
    assert abfe_diff_builder is not None
    ligand_rest_builder = rs.select_system_builder("ligand-rest", system_type=None)
    assert ligand_rest_builder is not None
    with pytest.raises(ValueError):
        rs.select_system_builder("abfe", system_type="MASFE")


def test_select_run_id_reuses_latest(tmp_path: Path) -> None:
    exec_dir = tmp_path / "executions"
    old = exec_dir / "old"
    new = exec_dir / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    run_id, run_dir = rs.select_run_id(tmp_path, "abfe", "sys", requested=None)
    assert run_id == "new"
    assert run_dir == new


def test_clear_failure_markers_removes_retry_counters_and_progress(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "executions" / "rep1"
    win_dir = run_dir / "simulations" / "LIG" / "fe" / "z" / "z00"
    progress_dir = run_dir / "simulations" / "LIG" / "progress"
    artifact_progress = run_dir / "artifacts" / "progress"

    win_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    artifact_progress.mkdir(parents=True, exist_ok=True)

    failed_marker = win_dir / "FAILED"
    attempt_failed_marker = win_dir / "ATTEMPT_FAILED"
    phase_failed_marker = run_dir / "simulations" / "LIG" / "equil" / "prepare_equil.failed"
    attempt_file = win_dir / "job_attempt.txt"
    progress_csv = progress_dir / "state.csv"
    keep_file = win_dir / "keep.txt"

    failed_marker.write_text("FAILED\n")
    attempt_failed_marker.write_text("FAILED\n")
    phase_failed_marker.parent.mkdir(parents=True, exist_ok=True)
    phase_failed_marker.write_text("FAILED\n")
    attempt_file.write_text("3\n")
    progress_csv.write_text("phase,status\n")
    (artifact_progress / "phase_state.json").write_text("{}")
    keep_file.write_text("keep\n")

    run_mod._clear_failure_markers(run_dir)

    assert not failed_marker.exists()
    assert not attempt_failed_marker.exists()
    assert not phase_failed_marker.exists()
    assert not attempt_file.exists()
    assert not progress_csv.exists()
    assert not artifact_progress.exists()
    assert keep_file.exists()


def test_run_phase_with_failure_policy_retries_once_then_prunes(
    monkeypatch, tmp_path: Path
) -> None:
    child = SimSystem(
        name="sys:LIG:run1",
        root=tmp_path / "simulations" / "LIG",
        meta=SystemMeta(ligand="LIG"),
    )
    phase = run_mod.Pipeline([])

    run_calls: list[str] = []
    handle_calls: list[str] = []
    status_calls = {"count": 0}

    def fake_run_phase(*args, **kwargs):
        run_calls.append(kwargs.get("on_failure") or "")
        return False

    def fake_partition(children, phase_name, **kwargs):
        status_calls["count"] += 1
        return ([], [child])

    def fake_handle(children, phase_name, mode, **kwargs):
        handle_calls.append(mode)
        if mode == "retry":
            return [child]
        if mode == "prune":
            return []
        raise AssertionError(f"unexpected mode: {mode}")

    monkeypatch.setattr(run_mod, "run_phase_skipping_done", fake_run_phase)
    monkeypatch.setattr(run_mod, "partition_children_by_status", fake_partition)
    monkeypatch.setattr(run_mod, "handle_phase_failures", fake_handle)

    out_children, should_exit = run_mod._run_phase_with_failure_policy(
        phase,
        [child],
        "fe",
        backend=object(),
        on_failure="retry",
    )

    assert should_exit is False
    assert out_children == []
    assert len(run_calls) == 2
    assert handle_calls == ["retry", "prune"]
    assert status_calls["count"] == 2


def test_build_run_summary_table_includes_success_and_failure_rows(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    repo = FEResultsRepository(store)
    repo.save(
        FERecord(
            run_id="run1",
            ligand="ligA",
            mol_name="LIGA",
            system_name="sys",
            fe_type="rest",
            temperature=300.0,
            total_dG=-7.125,
            total_se=0.222,
            original_name="Ligand A Original",
            protocol="abfe",
        )
    )
    repo.record_failure(
        run_id="run1",
        ligand="ligB",
        system_name="sys",
        temperature=300.0,
        status="failed",
        reason="no_totals_found",
        original_name="Ligand B Original",
        protocol="abfe",
    )

    table = run_mod._build_run_summary_table(repo, "run1")

    assert table is not None
    assert "Ligand A Original" in table
    assert "Ligand B Original" in table
    assert "ligA" in table
    assert "ligB" in table
    assert "-7.125" in table
    assert "0.222" in table
    assert "failed" in table
    assert "no_totals_found" in table


def _dummy_smtp(sent: dict[str, str | list[str]]):
    class DummySMTP:
        def __init__(self, host: str) -> None:
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def sendmail(self, sender: str, recipients: list[str], message: str) -> None:
            sent["sender"] = sender
            sent["recipients"] = recipients
            sent["message"] = message

    return DummySMTP


def _make_rc(tmp_path: Path, email_sender: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        protocol="abfe",
        create=SimpleNamespace(system_name="sys"),
        run=SimpleNamespace(
            email_on_completion="dest@example.com",
            email_sender=email_sender,
            output_folder=tmp_path,
        ),
    )


def test_notify_run_completion_prefers_config_sender(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}

    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender="config@example.com")

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent["sender"] == "config@example.com"
    assert sent["recipients"] == ["dest@example.com"]
    assert "From: batter <config@example.com>" in sent["message"]


def test_notify_run_completion_includes_summary_table(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}

    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender="config@example.com")
    summary_table = (
        "ligand mol_name total_dG total_se  status failure_reason\n"
        "  ligA     LIGA   -7.125    0.222 success"
    )

    run_mod._notify_run_completion(
        rc,
        "run1",
        tmp_path,
        [],
        summary_table=summary_table,
    )

    assert "Final FE summary:" in sent["message"]
    assert summary_table in sent["message"]


def test_notify_no_fe_record_completion_sends_only_equil_email(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}

    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender="config@example.com")
    child = SimSystem(
        name="sys:LIG:run1",
        root=tmp_path / "simulations" / "LIG",
        meta=SystemMeta(ligand="LIG"),
    )

    run_mod._notify_no_fe_record_completion(
        rc,
        "run1",
        tmp_path / "executions" / "run1",
        [child],
        "FE production skipped (--only-equil)",
    )

    assert sent["sender"] == "config@example.com"
    assert sent["recipients"] == ["dest@example.com"]
    assert "Subject: BATTER run 'run1' of sys completed" in sent["message"]
    assert "FE records were not exported." in sent["message"]
    assert "FE production skipped (--only-equil)" in sent["message"]
    assert "FE records stored under" not in sent["message"]
    assert "- LIG (unbound): UNBOUND detected during equilibration" in sent["message"]


def test_notify_run_completion_skips_when_sender_missing(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}
    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender=None)

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent == {}


def test_notify_run_completion_logs_when_sender_missing(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}
    warnings: list[str] = []
    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))
    monkeypatch.setattr(
        run_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg))
    )

    rc = _make_rc(tmp_path, email_sender=None)

    run_mod._notify_run_completion(rc, "run1", tmp_path, [])

    assert sent == {}
    assert any("no sender email configured" in w.lower() for w in warnings)


def test_notify_run_failure_includes_error_details(
    tmp_path: Path, monkeypatch
) -> None:
    sent: dict[str, str | list[str]] = {}

    monkeypatch.setattr(run_mod.smtplib, "SMTP", lambda host: _dummy_smtp(sent)(host))

    rc = _make_rc(tmp_path, email_sender="config@example.com")

    run_mod._notify_run_failure(
        rc,
        "run1",
        tmp_path / "executions" / "run1",
        RuntimeError("boom"),
    )

    assert sent["sender"] == "config@example.com"
    assert sent["recipients"] == ["dest@example.com"]
    assert "Subject: BATTER run 'run1' of sys failed" in sent["message"]
    assert "Error:\nboom" in sent["message"]
