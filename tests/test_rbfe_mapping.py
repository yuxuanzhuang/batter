from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from batter.rbfe import (
    _edge_asset_from_mapping_dir,
    _kartograf_mapper_kwargs,
    _pocket_grid_overlap_metrics,
    _pocket_grid_overlap_score,
    _write_pocket_shape_overlap_png,
    _wrap_atom_mapper_with_overrides,
    ManualAtomMappingOverrides,
    RBFENetwork,
    konnektor_pairs,
    load_atom_mapping_file,
    load_mapping_file,
    resolve_network_scorer_name,
    resolve_mapping_fn,
    write_pair_mapping_artifacts,
)


def test_load_mapping_file_text(tmp_path: Path) -> None:
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text(
        """
# comment
LIG1~LIG2
LIG2, LIG3
LIG3 LIG4
"""
    )
    pairs = load_mapping_file(mapping_file)
    assert pairs == [("LIG1", "LIG2"), ("LIG2", "LIG3"), ("LIG3", "LIG4")]


def test_load_mapping_file_json_pairs(tmp_path: Path) -> None:
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"pairs": [["A", "B"], ["A", "C"]]}))
    pairs = load_mapping_file(mapping_file)
    assert pairs == [("A", "B"), ("A", "C")]


def test_load_mapping_file_json_adjacency(tmp_path: Path) -> None:
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"A": ["B", "C"]}))
    pairs = load_mapping_file(mapping_file)
    assert pairs == [("A", "B"), ("A", "C")]


def test_load_atom_mapping_file_pair_dict_and_reverse(tmp_path: Path) -> None:
    mapping_file = tmp_path / "atom_mapping.json"
    mapping_file.write_text(json.dumps({"LIG1~LIG2": {"0": 4, "2": 5}}))

    overrides = load_atom_mapping_file(mapping_file)

    assert overrides.get_b_to_a("LIG1", "LIG2") == {0: 4, 2: 5}
    assert overrides.get_b_to_a("LIG2", "LIG1") == {4: 0, 5: 2}


def test_load_atom_mapping_file_inverts_reference_to_target(tmp_path: Path) -> None:
    mapping_file = tmp_path / "atom_mapping.json"
    mapping_file.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "ref": "A",
                        "alt": "B",
                        "componentA_to_componentB": {"0": 2, "1": 3},
                    }
                ]
            }
        )
    )

    overrides = load_atom_mapping_file(mapping_file)

    assert overrides.get_b_to_a("A", "B") == {2: 0, 3: 1}


def test_rbfe_network_default_mapping() -> None:
    network = RBFENetwork.from_ligands(["A", "B", "C"])
    assert network.pairs == (("A", "B"), ("A", "C"))


def test_kartograf_mapper_kwargs_defaults() -> None:
    kwargs = _kartograf_mapper_kwargs(None, atom_map_hydrogens_default=False)

    assert kwargs["atom_max_distance"] == 0.95
    assert kwargs["map_exact_ring_matches_only"] is True
    assert kwargs["allow_partial_fused_rings"] is True
    assert kwargs["allow_bond_breaks"] is False


def test_resolve_mapping_konnektor_requires_orchestrator() -> None:
    with pytest.raises(ValueError, match="konnektor"):
        resolve_mapping_fn("konnektor")


def test_manual_override_mapper_uses_manual_pair_and_falls_back() -> None:
    class Component:
        def __init__(self, name):
            self.name = name

    class Delegate:
        def __init__(self):
            self.calls = []

        def suggest_mappings(self, component_a, component_b):
            self.calls.append((component_a.name, component_b.name))
            yield object()

    delegate = Delegate()
    overrides = ManualAtomMappingOverrides({("A", "B"): {1: 0, 3: 2}})
    mapper = _wrap_atom_mapper_with_overrides(delegate, overrides)

    manual = next(mapper.suggest_mappings(Component("A"), Component("B")))
    reverse = next(mapper.suggest_mappings(Component("B"), Component("A")))
    fallback = next(mapper.suggest_mappings(Component("A"), Component("C")))

    assert manual.componentB_to_componentA == {1: 0, 3: 2}
    assert reverse.componentB_to_componentA == {0: 1, 2: 3}
    assert fallback is not None
    assert delegate.calls == [("A", "C")]


def test_manual_override_mapper_to_dict_is_json_compatible() -> None:
    overrides = ManualAtomMappingOverrides(
        {("A", "B"): {1: 0, 3: 2}},
        source=Path("atom_mapping.json"),
    )
    mapper = _wrap_atom_mapper_with_overrides("wrapped", overrides)

    data = mapper._to_dict()
    json.dumps(data)
    restored = type(mapper)._from_dict(data)

    assert restored.manual_overrides.get_b_to_a("A", "B") == {1: 0, 3: 2}
    assert restored.manual_overrides.source == Path("atom_mapping.json")


def test_write_pair_mapping_artifacts_uses_manual_override(tmp_path: Path) -> None:
    overrides = ManualAtomMappingOverrides({("A", "B"): {0: 1, 2: 3}})
    ligand_files = {
        "A": tmp_path / "A.sdf",
        "B": tmp_path / "B.sdf",
    }

    asset = write_pair_mapping_artifacts(
        ref="A",
        alt="B",
        ligand_files=ligand_files,
        out_dir=tmp_path / "mappings",
        atom_mapping_overrides=overrides,
    )

    pair_dir = tmp_path / "mappings" / "A~B"
    assert json.loads((pair_dir / "mapping.json").read_text()) == {"0": 1, "2": 3}
    status = json.loads((pair_dir / "mapping_status.json").read_text())
    assert status["mapper"] == "manual"
    assert status["mapping_override"] is True
    assert status["n_mapped"] == 2
    assert asset["mapper"] == "manual"


def test_konnektor_pairs_missing_dependency(tmp_path: Path) -> None:
    if importlib.util.find_spec("konnektor") is not None:
        pytest.skip("konnektor installed; dependency error test not applicable.")
    with pytest.raises(RuntimeError, match="konnektor"):
        konnektor_pairs(["A", "B"], {"A": tmp_path / "a.sdf", "B": tmp_path / "b.sdf"})


def _install_fake_konnektor(monkeypatch, generator_classes: dict[str, type]) -> None:
    konnektor_mod = types.ModuleType("konnektor")
    planners_mod = types.ModuleType("konnektor.network_planners")
    generators_mod = types.ModuleType("konnektor.network_planners.generators")

    for name, cls in generator_classes.items():
        setattr(generators_mod, name, cls)
        # rbfe imports `konnektor.network_planners` directly and inspects it
        setattr(planners_mod, name, cls)

    planners_mod.generators = generators_mod
    konnektor_mod.network_planners = planners_mod

    monkeypatch.setitem(sys.modules, "konnektor", konnektor_mod)
    monkeypatch.setitem(sys.modules, "konnektor.network_planners", planners_mod)
    monkeypatch.setitem(
        sys.modules, "konnektor.network_planners.generators", generators_mod
    )

    gufe_mod = types.ModuleType("gufe")

    class SmallMoleculeComponent:
        def __init__(self, mol, name=None):
            self.name = name or "lig"
            self.mol = mol

    class AtomMapper:
        pass

    class LigandAtomMapping:
        pass

    gufe_mod.SmallMoleculeComponent = SmallMoleculeComponent
    gufe_mod.AtomMapper = AtomMapper
    gufe_mod.LigandAtomMapping = LigandAtomMapping
    monkeypatch.setitem(sys.modules, "gufe", gufe_mod)

    kartograf_mod = types.ModuleType("kartograf")
    atom_mapper_mod = types.ModuleType("kartograf.atom_mapper")

    class KartografAtomMapper:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    atom_mapper_mod.KartografAtomMapper = KartografAtomMapper
    kartograf_mod.atom_mapper = atom_mapper_mod
    monkeypatch.setitem(sys.modules, "kartograf", kartograf_mod)
    monkeypatch.setitem(sys.modules, "kartograf.atom_mapper", atom_mapper_mod)

    lomap_mod = types.ModuleType("lomap")
    lomap_gufe_bindings_mod = types.ModuleType("lomap.gufe_bindings")
    lomap_scorers_mod = types.ModuleType("lomap.gufe_bindings.scorers")
    lomap_scorers_mod.default_lomap_score = object()

    class LomapAtomMapper:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    lomap_mod.LomapAtomMapper = LomapAtomMapper
    lomap_gufe_bindings_mod.LomapAtomMapper = LomapAtomMapper
    lomap_gufe_bindings_mod.scorers = lomap_scorers_mod
    lomap_mod.gufe_bindings = lomap_gufe_bindings_mod
    monkeypatch.setitem(sys.modules, "lomap", lomap_mod)
    monkeypatch.setitem(sys.modules, "lomap.gufe_bindings", lomap_gufe_bindings_mod)
    monkeypatch.setitem(
        sys.modules, "lomap.gufe_bindings.scorers", lomap_scorers_mod
    )


def test_konnektor_pairs_layout_resolution(monkeypatch, tmp_path: Path) -> None:
    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    pairs = konnektor_pairs(
        ["L1", "L2"],
        {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
        layout="star",
    )
    assert pairs == [("L1", "L2")]


def _make_pose_mol(points: list[tuple[float, float, float]]) -> Chem.Mol:
    editable = Chem.RWMol()
    for _point in points:
        editable.AddAtom(Chem.Atom(6))
    mol = editable.GetMol()
    conf = Chem.Conformer(len(points))
    for idx, (x, y, z) in enumerate(points):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def test_pocket_grid_overlap_prioritizes_shared_pocket_occupancy() -> None:
    a_xy = _make_pose_mol([(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
    b_x = _make_pose_mol([(0.0, 0.0, 0.0)])
    c_y = _make_pose_mol([(6.0, 0.0, 0.0)])
    d_xy = _make_pose_mol([(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)])

    assert _pocket_grid_overlap_score(a_xy, d_xy) > _pocket_grid_overlap_score(
        a_xy,
        b_x,
    )
    assert _pocket_grid_overlap_score(a_xy, b_x) > _pocket_grid_overlap_score(
        b_x,
        c_y,
    )
    assert _pocket_grid_overlap_score(c_y, d_xy) > _pocket_grid_overlap_score(
        b_x,
        c_y,
    )
    metrics = _pocket_grid_overlap_metrics(a_xy, b_x)
    assert metrics is not None
    assert metrics["pocket_grid_score"] > 0
    assert metrics["pocket_grid_containment"] > metrics["pocket_grid_jaccard"]
    assert metrics["pocket_grid_overlap_voxels"] > 0


def test_write_pocket_shape_overlap_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    a_xy = _make_pose_mol([(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
    b_x = _make_pose_mol([(0.0, 0.0, 0.0)])
    out = tmp_path / "pocket_shape_overlap.png"

    assert _write_pocket_shape_overlap_png(a_xy, b_x, out, pair_id="A~B")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_edge_asset_prefers_pocket_shape_overlap_image(tmp_path: Path) -> None:
    pair_dir = tmp_path / "A~B"
    pair_dir.mkdir()
    (pair_dir / "mapping.json").write_text("{}")
    (pair_dir / "mapping.png").write_bytes(b"atom")
    (pair_dir / "pocket_shape_overlap.png").write_bytes(b"shape")
    (pair_dir / "mapping_status.json").write_text(
        json.dumps({"pocket_shape_score": 0.75})
    )

    asset = _edge_asset_from_mapping_dir("A~B", pair_dir)

    assert asset["image_kind"] == "pocket_shape_overlap"
    assert asset["image_alt"] == "Pocket shape overlap for A~B"
    assert asset["image_data_uri"].endswith("c2hhcGU=")
    assert asset["atom_mapping_image_data_uri"].endswith("YXRvbQ==")
    assert asset["pocket_shape_score"] == 0.75


def test_network_scorer_auto_defaults_to_pocket_shape_for_septop() -> None:
    assert resolve_network_scorer_name("auto", protocol="rbfe") == "lomap"
    assert (
        resolve_network_scorer_name("auto", protocol="rbfe_septop")
        == "pocket_shape"
    )
    assert (
        resolve_network_scorer_name("shape-mismatch", protocol="rbfe")
        == "shape_difference"
    )
    assert resolve_network_scorer_name("grid-shape", protocol="rbfe") == "pocket_shape"


def test_konnektor_pairs_septop_auto_uses_pocket_shape_scorer(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            seen["scorer"] = kwargs.get("scorer")

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    pairs = konnektor_pairs(
        ["L1", "L2"],
        {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
        layout="star",
        protocol="rbfe_septop",
    )

    assert pairs == [("L1", "L2")]
    assert callable(seen["scorer"])
    assert getattr(seen["scorer"], "__name__", "") == "_pocket_shape_network_score"


def test_konnektor_pairs_unknown_layout(monkeypatch, tmp_path: Path) -> None:
    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    with pytest.raises(ValueError, match="Unknown Konnektor layout"):
        konnektor_pairs(
            ["L1", "L2"],
            {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
            layout="unknown",
        )


def test_konnektor_pairs_explicit_requires_edges(monkeypatch, tmp_path: Path) -> None:
    class ExplicitNetworkGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(
        monkeypatch, {"ExplicitNetworkGenerator": ExplicitNetworkGenerator}
    )
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    with pytest.raises(ValueError, match="explicit"):
        konnektor_pairs(
            ["L1", "L2"],
            {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
            layout="explicit",
        )


def test_konnektor_pairs_uses_lomap_mapper(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            seen["mapper"] = kwargs.get("mappers")

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    pairs = konnektor_pairs(
        ["L1", "L2"],
        {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
        layout="star",
        atom_mapper="lomap",
    )
    from lomap import LomapAtomMapper

    assert pairs == [("L1", "L2")]
    assert isinstance(seen["mapper"], LomapAtomMapper)


def test_konnektor_pairs_forwards_lomap_options(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            seen["mapper"] = kwargs.get("mappers")

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    konnektor_pairs(
        ["L1", "L2"],
        {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
        layout="star",
        atom_mapper="lomap",
        lomap_options={"time": 7, "max3d": 2.0, "shift": False},
    )

    assert seen["mapper"].kwargs == {
        "time": 7,
        "threed": True,
        "max3d": 2.0,
        "element_change": False,
        "shift": False,
    }


def test_konnektor_pairs_forwards_kartograf_options(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            seen["mapper"] = kwargs.get("mappers")

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    konnektor_pairs(
        ["L1", "L2"],
        {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
        layout="star",
        atom_mapper="kartograf",
        kartograf_options={
            "atom_max_distance": 1.23,
            "allow_bond_breaks": True,
            "filter_element_changes": False,
            "atom_map_hydrogens": True,
            "map_hydrogens_on_hydrogens_only": False,
        },
    )

    kwargs = seen["mapper"].kwargs
    assert kwargs["atom_max_distance"] == 1.23
    assert kwargs["allow_bond_breaks"] is True
    assert kwargs["atom_map_hydrogens"] is False
    assert kwargs["map_hydrogens_on_hydrogens_only"] is True
    assert kwargs["additional_mapping_filter_functions"] == []


def test_konnektor_pairs_rejects_unknown_atom_mapper(monkeypatch, tmp_path: Path) -> None:
    class StarNetworkGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate_ligand_network(self, components):
            class Network:
                def __init__(self, comps):
                    self.edges = [(comps[0], comps[1])]

            return Network(components)

    _install_fake_konnektor(monkeypatch, {"StarNetworkGenerator": StarNetworkGenerator})
    monkeypatch.setattr("batter.rbfe._load_rdkit_mol", lambda path: object())

    with pytest.raises(ValueError, match="Unknown atom mapper"):
        konnektor_pairs(
            ["L1", "L2"],
            {"L1": tmp_path / "l1.sdf", "L2": tmp_path / "l2.sdf"},
            layout="star",
            atom_mapper="bad_mapper",
        )
