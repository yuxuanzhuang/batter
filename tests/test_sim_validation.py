from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import MDAnalysis as mda

# sim_validation imports networkx unconditionally but does not use it in this test.
try:
    import networkx  # noqa: F401
except Exception:
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))

from batter.analysis.sim_validation import (
    STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
    SimValidator,
)


def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {name:<4}{resname:>4} {chain}{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def _make_test_universe(tmp_path: Path) -> mda.Universe:
    pdb = tmp_path / "full.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 92, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "ALA", "A", 61, 100.0, 0.0, 0.0, "C"),
        _atom_line(3, "CA", "ALA", "A", 257, 0.0, 100.0, 0.0, "C"),
        _atom_line(4, "C1", "LIG", "A", 300, 3.0, 0.0, 0.0, "C"),
        _atom_line(5, "C2", "LIG", "A", 300, 50.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    return mda.Universe(str(pdb))


def _make_validator(
    u: mda.Universe, workdir: Path, protein_anchor_masks: list[str] | None = None
) -> SimValidator:
    validator = SimValidator.__new__(SimValidator)
    validator.universe = u
    validator.workdir = workdir
    validator.ligand = "LIG"
    validator.protein_anchor_masks = protein_anchor_masks or []
    validator.results = {}
    return validator


class _DummyTrajectory:
    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_frames


class _DummyUniverse:
    def __init__(self, n_frames: int) -> None:
        self.trajectory = _DummyTrajectory(n_frames)


def test_ligand_bs_uses_initial_binding_site_atoms(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    validator = _make_validator(u, tmp_path, [":61@CA", ":257@CA", ":61@CA"])
    validator._ligand_bs()

    assert np.allclose(validator.results["ligand_bs"], np.array([3.0]))


def test_ligand_bs_requires_site_or_anchor_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "far.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 92, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "C1", "LIG", "A", 300, 30.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    validator = _make_validator(u, tmp_path)

    with pytest.raises(ValueError, match="binding-site atoms"):
        validator._ligand_bs()


def test_representative_snapshot_uses_last_quarter_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _make_validator(_DummyUniverse(8), tmp_path)
    called = {}

    def fake_ligand_dihedral(start_frame: int = 0) -> None:
        called["start_frame"] = start_frame
        validator.results["ligand_dihedrals"] = np.array(
            [
                [90.0, -90.0],
                [90.0, -90.0],
            ]
        )
        validator.results["ligand_dihedral_frame_indices"] = np.array([6, 7])

    monkeypatch.setattr(validator, "_ligand_dihedral", fake_ligand_dihedral)

    rep_idx = validator.find_representative_snapshot(
        savefig=True, output_filename="tail_dihed_hist.png"
    )

    assert called["start_frame"] == 6
    assert rep_idx == 6
    assert validator.results["representative_frame_index"] == 6
    assert validator.results["representative_analysis_start_frame"] == 6


def test_representative_snapshot_reads_disang_without_assign_in(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    coords = np.repeat(u.atoms.positions[None, :, :], 4, axis=0)
    coords[:, 0, :] = np.array([0.0, 0.0, 0.0])
    coords[:, 1, :] = np.array([1.0, 0.0, 0.0])
    coords[:, 3, :] = np.array([1.0, 1.0, 0.0])
    coords[:, 4, :] = np.array([2.0, 1.0, 1.0])
    coords[3, 4, :] = np.array([2.0, 1.2, 0.8])
    u.load_new(coords, order="fac")

    (tmp_path / "disang.rest").write_text(
        "&rst iat=1,2,4,5, r1=-180.0, r2=0.0, r3=0.0, "
        "r4=180.0, rk2=0.0, rk3=0.0, &end #Lig_D\n"
    )
    validator = _make_validator(u, tmp_path)

    rep_idx = validator.find_representative_snapshot(
        savefig=True, output_filename="disang_dihed_hist.png"
    )

    assert rep_idx == 3
    assert validator.results["representative_frame_index"] == 3
    assert validator.results["ligand_dihedrals"].shape == (1, 1)
    assert validator.results["ligand_dihedral_frame_indices"].tolist() == [3]
    assert validator.results["ligand_dihedral_source"] == str(tmp_path / "disang.rest")
    assert not (tmp_path / "assign.in").exists()
    assert (tmp_path / "disang_dihed_hist.png").stat().st_size > 0


def test_simulation_analysis_plot_records_frame_and_time_axes(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    coords = np.repeat(u.atoms.positions[None, :, :], 3, axis=0)
    u.load_new(coords, order="fac", dt=2000.0)
    validator = _make_validator(u, tmp_path)
    validator.results = {
        "ligand_bs": np.array([3.0, 3.2, 3.1]),
        "protein_rmsd": np.array([0.0, 0.5, 0.7]),
        "ligand_rmsd": np.array([0.0, 0.8, 0.6]),
        "representative_frame_index": 1,
    }

    validator.plot_analysis(savefig=True)

    assert np.array_equal(validator.results["frame_indices"], np.array([0, 1, 2]))
    assert np.allclose(validator.results["simulation_time_ns"], np.array([0.0, 2.0, 4.0]))
    assert (tmp_path / "simulation_analysis.png").stat().st_size > 0


def test_stable_boresch_distance_uses_tail_candidate_stability(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "stable.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "ALA", "A", 20, 12.0, 0.0, 0.0, "C"),
        _atom_line(3, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        _atom_line(4, "C2", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))

    coords = np.repeat(u.atoms.positions[None, :, :], 8, axis=0)
    # Only the final 25% (frames 6 and 7) should be considered.
    coords[:6, 2, :] = np.array([9.0, 0.0, 0.0])
    coords[:6, 3, :] = np.array([9.0, 0.0, 0.0])
    coords[6, 2, :] = np.array([4.0, 0.0, 0.0])
    coords[7, 2, :] = np.array([4.1, 0.0, 0.0])
    coords[6, 3, :] = np.array([3.2, 0.0, 0.0])
    coords[7, 3, :] = np.array([6.8, 0.0, 0.0])
    u.load_new(coords, order="fac")

    validator = _make_validator(u, tmp_path)
    record = validator.find_stable_boresch_distance(
        tail_fraction=0.25,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1", "C2"],
    )

    assert record["analysis_start_frame"] == 6
    assert record["schema_version"] == STABLE_BORESCH_DISTANCE_SCHEMA_VERSION
    assert record["frame_indices"] == [6, 7]
    assert record["protein"]["mask"] == ":10@CA"
    assert record["ligand"]["name"] == "C1"
    assert record["ranked_pairs"][0]["protein"]["mask"] == ":10@CA"
    assert record["ranked_pairs"][0]["ligand"]["name"] == "C1"
    assert record["ranked_pairs"][0]["rank"] == 1
    assert "rank_score" in record["ranked_pairs"][0]
    assert np.isclose(record["distance"]["mean"], 4.05)
    assert np.isclose(record["distance"]["std"], 0.05)
    assert np.allclose(validator.results["stable_boresch_distance"], [4.0, 4.1])


def test_stable_boresch_distance_can_filter_to_protein_residues(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "stable_residue_filter.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "SER", "A", 20, 1.0, 0.0, 0.0, "C"),
        _atom_line(3, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    validator = _make_validator(u, tmp_path)

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1"],
        protein_residue_ids=[20],
    )

    assert record["protein"]["mask"] == ":20@CA"
    assert record["criteria"]["protein_atom_names"] == ["CA"]
    assert record["criteria"]["protein_residue_ids"] == [20]


def test_stable_boresch_distance_prioritizes_interaction_priority(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "stable_residue_priority.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "SER", "A", 20, 20.0, 0.0, 0.0, "C"),
        _atom_line(3, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        _atom_line(4, "C2", "LIG", "A", 300, 24.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    coords = np.repeat(u.atoms.positions[None, :, :], 2, axis=0)
    coords[0, 2, :] = np.array([4.0, 0.0, 0.0])
    coords[1, 2, :] = np.array([4.8, 0.0, 0.0])
    coords[0, 3, :] = np.array([24.0, 0.0, 0.0])
    coords[1, 3, :] = np.array([24.1, 0.0, 0.0])
    u.load_new(coords, order="fac")
    validator = _make_validator(u, tmp_path)

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1", "C2"],
        protein_residue_priorities={10: 0, 20: 2},
    )

    assert record["protein"]["mask"] == ":10@CA"
    assert record["ligand"]["name"] == "C1"
    assert record["rank_score"]["protein_interaction_priority"] == 0
    assert record["criteria"]["protein_residue_priorities"] == [
        {"resid": 10, "priority": 0},
        {"resid": 20, "priority": 2},
    ]


def test_stable_boresch_distance_prioritizes_salt_bridge_ligand_atom(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "stable_ligand_priority.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        _atom_line(3, "N1", "LIG", "A", 300, 5.0, 0.0, 0.0, "N"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    coords = np.repeat(u.atoms.positions[None, :, :], 2, axis=0)
    coords[0, 1, :] = np.array([4.0, 0.0, 0.0])
    coords[1, 1, :] = np.array([4.1, 0.0, 0.0])
    coords[0, 2, :] = np.array([5.0, 0.0, 0.0])
    coords[1, 2, :] = np.array([6.0, 0.0, 0.0])
    u.load_new(coords, order="fac")
    validator = _make_validator(u, tmp_path)

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1", "N1"],
        ligand_atom_priorities={"N1": 0},
    )

    assert record["ligand"]["name"] == "N1"
    assert record["rank_score"]["ligand_atom_priority"] == 0
    assert record["criteria"]["ligand_atom_priorities"] == [
        {"name": "N1", "priority": 0}
    ]


def test_stable_boresch_distance_rejects_collinear_first_anchor(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "angle.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "ALA", "A", 11, 4.0, 4.0, 0.0, "C"),
        _atom_line(3, "CA", "ALA", "A", 20, 8.0, 0.0, 0.0, "C"),
        _atom_line(4, "CA", "ALA", "A", 30, 8.0, 8.0, 0.0, "C"),
        _atom_line(5, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    validator = _make_validator(u, tmp_path, [":10@CA", ":20@CA", ":30@CA"])

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1"],
    )

    assert record["protein"]["mask"] == ":11@CA"
    assert record["ligand"]["name"] == "C1"
    assert record["angle"] is not None
    assert np.isclose(record["angle"]["mean"], 45.0)
    assert np.allclose(validator.results["stable_boresch_angle"], [45.0])


def test_stable_boresch_distance_stays_on_anchor_chain(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "chain.pdb"
    lines = [
        _atom_line(1, "CA", "ALA", "B", 40, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "ALA", "A", 10, 20.0, 20.0, 0.0, "C"),
        _atom_line(3, "CA", "ALA", "A", 11, 4.0, 4.0, 0.0, "C"),
        _atom_line(4, "CA", "ALA", "A", 20, 8.0, 0.0, 0.0, "C"),
        _atom_line(5, "CA", "ALA", "A", 30, 8.0, 8.0, 0.0, "C"),
        _atom_line(6, "C1", "LIG", "A", 300, 4.0, 0.0, 0.0, "C"),
        "TER\n",
        "END\n",
    ]
    pdb.write_text("".join(lines))
    u = mda.Universe(str(pdb))
    validator = _make_validator(u, tmp_path, [":10@CA", ":20@CA", ":30@CA"])

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1"],
    )

    assert record["protein"]["mask"] == ":11@CA"
    assert record["protein"]["segid"] == "A"
    assert record["protein"]["chainID"] == "A"


def test_stable_boresch_distance_supports_single_frame(tmp_path: Path) -> None:
    u = _make_test_universe(tmp_path)
    validator = _make_validator(u, tmp_path)

    record = validator.find_stable_boresch_distance(
        tail_fraction=1.0,
        min_distance=3.0,
        max_distance=7.0,
        ligand_atom_names=["C1"],
    )

    assert record["analysis_start_frame"] == 0
    assert record["n_frames"] == 1
    assert record["ligand"]["name"] == "C1"
    assert np.isclose(record["distance"]["mean"], 3.0)
    assert np.isclose(record["distance"]["std"], 0.0)
