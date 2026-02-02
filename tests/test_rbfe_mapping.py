from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from batter.rbfe import (
    RBFENetwork,
    konnektor_pairs,
    load_mapping_file,
    resolve_mapping_fn,
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


def test_rbfe_network_default_mapping() -> None:
    network = RBFENetwork.from_ligands(["A", "B", "C"])
    assert network.pairs == (("A", "B"), ("A", "C"))


def test_resolve_mapping_konnektor_requires_orchestrator() -> None:
    with pytest.raises(ValueError, match="konnektor"):
        resolve_mapping_fn("konnektor")


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

    gufe_mod.SmallMoleculeComponent = SmallMoleculeComponent
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
