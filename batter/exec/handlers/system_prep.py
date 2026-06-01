"""Prepare complex systems (protein/ligand/membrane) for simulations."""

from __future__ import annotations

from collections import Counter
import contextlib
import itertools
import json
import os
import shutil
import string
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import align
from MDAnalysis.analysis.dssp import DSSP
from loguru import logger

from batter.config.utils import is_apo_ligand_path
from batter._internal.templates import BUILD_FILES_DIR as build_files_orig
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils.builder_utils import (
    find_anchor_atoms,
    select_apo_receptor_anchor_atoms,
    select_receptor_anchor_atoms,
)

_PROTEIN_BREAK_CA_DISTANCE_CUTOFF_A = 10.0
_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
_XY_ROTATION_REFINE_DEGREES = (45.0, 15.0, 5.0, 1.0)


def _as_abs(p: str | Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def _select_anchor_reference_ligand(
    ligand_order: Sequence[str],
    ligand_paths: Mapping[str, Any],
) -> tuple[str, bool]:
    """
    Return the ligand to use for receptor-anchor and L1 geometry setup.

    Mixed MD runs can include an apo dummy plus one or more real ligands. The
    dummy keeps downstream ligand-oriented setup code wired, but it should not
    drive ligand-pose receptor-anchor selection when a real ligand pose exists.
    """
    if not ligand_paths:
        raise ValueError("No ligands available for anchor reference selection.")

    ordered_names = [name for name in ligand_order if name in ligand_paths]
    seen_names = set(ordered_names)
    ordered_names.extend(
        name for name in sorted(ligand_paths, key=str) if name not in seen_names
    )
    for name in ordered_names:
        if not is_apo_ligand_path(ligand_paths[name]):
            return name, False
    return ordered_names[0], True


def _ligand_sdf_reference(path: Any, *, is_apo: bool) -> str | None:
    if is_apo:
        return None
    p = Path(path)
    if p.suffix.lower() != ".sdf":
        return None
    return str(p)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _chain_id_from_index(index: int) -> str:
    if index >= len(_CHAIN_ID_ALPHABET):
        raise ValueError(
            "Too many protein fragments to encode in single-character PDB chain IDs. "
            f"Found fragment index {index + 1}, but only {len(_CHAIN_ID_ALPHABET)} IDs are available."
        )
    return _CHAIN_ID_ALPHABET[index]


def _get_single_ca_position(residue) -> np.ndarray | None:
    ca_atoms = residue.atoms.select_atoms("name CA")
    if ca_atoms.n_atoms != 1:
        return None
    return np.asarray(ca_atoms.positions[0], dtype=float)


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=float,
    )


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _apply_rotation(coords: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a column-vector rotation matrix to row-vector coordinates."""
    return np.asarray(coords, dtype=float) @ rotation.T


def _xy_box_score(coords: np.ndarray) -> tuple[float, float, float]:
    spans = np.ptp(coords, axis=0)
    return (
        float(spans[0] * spans[1]),
        float(spans[2]),
        float(spans[0] + spans[1]),
    )


def _score_lt(
    candidate: tuple[float, float, float],
    current: tuple[float, float, float],
    *,
    tol: float = 1e-6,
) -> bool:
    for cand_val, curr_val in zip(candidate, current):
        if cand_val < curr_val - tol:
            return True
        if cand_val > curr_val + tol:
            return False
    return False


def _principal_axis_rotations(coords: np.ndarray) -> list[np.ndarray]:
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0

    rotations: list[np.ndarray] = [np.eye(3, dtype=float)]
    for perm in itertools.permutations(range(3)):
        permuted = axes[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            basis = permuted * np.asarray(signs, dtype=float)
            if np.linalg.det(basis) <= 0.0:
                continue
            rotations.append(basis.T)
    return rotations


def _refine_xy_box_rotation(
    coords: np.ndarray,
    initial_rotation: np.ndarray,
    *,
    step_degrees: tuple[float, ...] = _XY_ROTATION_REFINE_DEGREES,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    best_rotation = np.asarray(initial_rotation, dtype=float)
    best_score = _xy_box_score(_apply_rotation(coords, best_rotation))

    for step_deg in step_degrees:
        while True:
            improved = False
            local_best_rotation = best_rotation
            local_best_score = best_score
            delta_values = (-step_deg, 0.0, step_deg)

            for dx, dy, dz in itertools.product(delta_values, repeat=3):
                if dx == dy == dz == 0.0:
                    continue
                delta_rotation = (
                    _rotation_matrix_z(dz)
                    @ _rotation_matrix_y(dy)
                    @ _rotation_matrix_x(dx)
                )
                candidate_rotation = delta_rotation @ best_rotation
                candidate_score = _xy_box_score(
                    _apply_rotation(coords, candidate_rotation)
                )
                if _score_lt(candidate_score, local_best_score):
                    local_best_rotation = candidate_rotation
                    local_best_score = candidate_score
                    improved = True

            if not improved:
                break
            best_rotation = local_best_rotation
            best_score = local_best_score

    return best_rotation, best_score


def _find_min_xy_box_rotation(
    coords: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Expected an (N, 3) coordinate array for XY box optimization, got {coords.shape}."
        )
    if coords.shape[0] < 2:
        score = _xy_box_score(coords if len(coords) else np.zeros((1, 3), dtype=float))
        return np.eye(3, dtype=float), score, score

    centered = coords - coords.mean(axis=0, keepdims=True)
    before_score = _xy_box_score(centered)
    best_rotation = np.eye(3, dtype=float)
    best_score = before_score

    for rotation in _principal_axis_rotations(centered):
        refined_rotation, refined_score = _refine_xy_box_rotation(centered, rotation)
        if _score_lt(refined_score, best_score):
            best_rotation = refined_rotation
            best_score = refined_score

    return best_rotation, before_score, best_score


def _split_residues_on_breaks(
    residues,
    *,
    segid: str,
    chain_id: str,
    ca_distance_cutoff: float = _PROTEIN_BREAK_CA_DISTANCE_CUTOFF_A,
) -> tuple[list[list[Any]], list[str]]:
    residue_list = list(residues)
    if not residue_list:
        return [], []

    fragments: list[list[Any]] = [[residue_list[0]]]
    warnings: list[str] = []

    for prev_residue, curr_residue in zip(residue_list, residue_list[1:]):
        reasons: list[str] = []
        prev_resid = int(prev_residue.resid)
        curr_resid = int(curr_residue.resid)

        if curr_resid != prev_resid + 1:
            reasons.append(f"resid discontinuity ({prev_resid} -> {curr_resid})")

        prev_ca = _get_single_ca_position(prev_residue)
        curr_ca = _get_single_ca_position(curr_residue)
        if prev_ca is not None and curr_ca is not None:
            ca_distance = float(np.linalg.norm(curr_ca - prev_ca))
            if ca_distance > ca_distance_cutoff:
                reasons.append(
                    f"C-alpha distance {ca_distance:.1f} A > {ca_distance_cutoff:.1f} A"
                )

        if reasons:
            warnings.append(
                "Detected a protein break in system_prep "
                f"(segid={segid or '?'}, chain={chain_id or '?'}) "
                f"between residues {prev_resid} and {curr_resid}: "
                + "; ".join(reasons)
                + ". BATTER will split these residues into separate segments/chains."
            )
            fragments.append([])

        fragments[-1].append(curr_residue)

    return fragments, warnings


def _group_residues_by_source_identity(residues) -> list[list[Any]]:
    residue_list = list(residues)
    if not residue_list:
        return []

    groups: list[list[Any]] = [[residue_list[0]]]
    prev_chain_id = str(residue_list[0].atoms.chainIDs[0]).strip() if len(residue_list[0].atoms) else ""
    prev_segid = str(residue_list[0].segid).strip()

    for residue in residue_list[1:]:
        chain_id = str(residue.atoms.chainIDs[0]).strip() if len(residue.atoms) else ""
        segid = str(residue.segid).strip()
        if chain_id != prev_chain_id or segid != prev_segid:
            groups.append([])
        groups[-1].append(residue)
        prev_chain_id = chain_id
        prev_segid = segid

    return groups


def _protein_segid_overrides(universe: mda.Universe) -> tuple[dict[int, str], int]:
    """
    Build per-atom segid overrides to canonicalize segids within each protein residue.

    Some input PDBs carry a segid on heavy atoms but leave hydrogens blank.
    MDAnalysis then parses those atoms as separate residues/segments on reload.
    Compute a residue-level canonical segid so aligned intermediates can be
    rewritten with consistent per-residue segids before they are reloaded.
    """
    try:
        universe.atoms.segids
    except AttributeError:
        return {}, 0

    protein_atoms = universe.select_atoms("protein")
    if protein_atoms.n_atoms == 0:
        return {}, 0

    residue_atom_indices: dict[tuple[str, int, str], list[int]] = {}
    for atom in protein_atoms:
        chain_id = str(getattr(atom, "chainID", "")).strip()
        residue_key = (chain_id, int(atom.resid), str(atom.resname).strip())
        residue_atom_indices.setdefault(residue_key, []).append(int(atom.index))

    segid_overrides: dict[int, str] = {}
    normalized_count = 0
    for atom_indices in residue_atom_indices.values():
        atom_group = universe.atoms[atom_indices]
        segids = [str(segid).strip() for segid in atom_group.segids]
        unique_segids = set(segids)
        if len(unique_segids) <= 1:
            continue

        nonempty_segids = [segid for segid in segids if segid]
        if nonempty_segids:
            canonical_segid = Counter(nonempty_segids).most_common(1)[0][0]
        else:
            canonical_segid = segids[0]

        for atom_index in atom_indices:
            segid_overrides[atom_index] = canonical_segid
        normalized_count += 1

    return segid_overrides, normalized_count


def _write_pdb_with_normalized_protein_segids(
    universe: mda.Universe,
    output_path: Path,
) -> int:
    """
    Write a PDB while normalizing mixed per-atom protein segids per residue.
    """
    segid_overrides, normalized_count = _protein_segid_overrides(universe)
    universe.atoms.write(output_path.as_posix())
    if not segid_overrides:
        return normalized_count

    rewritten_lines: list[str] = []
    atom_counter = 0
    for line in output_path.read_text().splitlines(True):
        if line.startswith(("ATOM", "HETATM")):
            atom = universe.atoms[atom_counter]
            atom_counter += 1
            canonical_segid = segid_overrides.get(int(atom.index))
            if canonical_segid is not None:
                stripped = line.rstrip("\n")
                if len(stripped) < 76:
                    stripped = stripped.ljust(76)
                line = f"{stripped[:72]}{canonical_segid:<4}{stripped[76:]}\n"
        rewritten_lines.append(line)

    output_path.write_text("".join(rewritten_lines))
    return normalized_count


def _select_fragment_atoms(
    universe: mda.Universe,
    residues: list[Any],
    *,
    chain_id: str,
    segid: str,
):
    resid_seq = " ".join(str(int(residue.resid)) for residue in residues)
    selectors: list[str] = []
    if chain_id:
        selectors.append(f"protein and chainID {chain_id} and resid {resid_seq}")
    if segid:
        selectors.append(f"protein and segid {segid} and resid {resid_seq}")
    selectors.append(f"protein and resid {resid_seq}")

    for selector in selectors:
        selection = universe.select_atoms(selector)
        if selection.n_residues == len(residues):
            return selection

    raise ValueError(
        "Could not match a protein fragment back to the aligned protein using "
        f"segid={segid!r}, chainID={chain_id!r}, residues={[int(r.resid) for r in residues]}."
    )


def _ensure_pdb(lig_path: Path, out_dir: Path) -> Path:
    """
    Ensure a PDB exists for ligand file; if not PDB, convert via RDKit.
    Returns the path to a PDB file.
    """
    if lig_path.suffix.lower() == ".pdb":
        return lig_path

    try:
        from rdkit import Chem
    except Exception as e:
        raise RuntimeError(
            f"Ligand {lig_path} is not PDB; RDKit is required to convert SDF/MOL2 → PDB."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"{lig_path.stem}.pdb"

    if lig_path.suffix.lower() == ".sdf":
        suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            raise ValueError(f"RDKit could not read any molecule from {lig_path}")
        Chem.MolToPDBFile(mols[0], str(out_pdb))
    elif lig_path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(lig_path), removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not read {lig_path}")
        Chem.MolToPDBFile(mol, str(out_pdb))
    elif lig_path.suffix.lower() == "pdb":
        _copy(lig_path, out_pdb)
    else:
        raise ValueError(f"Unsupported ligand format: {lig_path.suffix} for {lig_path}")
    return out_pdb


class _SystemPrepRunner:
    def __init__(self, system: SimSystem, yaml_dir: Path) -> None:
        self.system = system
        self.yaml_dir = yaml_dir

        self.output_dir = system.root
        self.ligands_folder = self.output_dir / "all-ligands"
        self.ligandff_folder = self.output_dir / "artifacts" / "ligands"
        self.ligandff_folder.mkdir(parents=True, exist_ok=True)

        # state
        self._system_name: str = ""
        self._protein_input: str = ""
        self._system_topology: str | None = None
        self._system_coordinate: str | None = None

        self.receptor_segment: str | None = None
        self.protein_align: str = "name CA and resid 60 to 250"
        self.receptor_ff: str = "protein.ff14SB"
        self.retain_lig_prot: bool = True
        self.ligand_ph: float = 7.4
        self.lipid_mol: List[str] = []
        self.membrane_simulation: bool = False
        self.lipid_ff: str = "lipid21"
        self.overwrite: bool = False
        self.verbose: bool = False

        self.ligand_dict: Dict[str, str] = {}
        self.ligand_order: List[str] = []
        self.unique_mol_names: List[str] = []
        self.system_dimensions = np.zeros(3)

        # alignment intermediates
        self._protein_aligned_pdb: str | None = None
        self._system_aligned_pdb: str | None = None
        self.mobile_coord: np.ndarray | None = None
        self.ref_coord: np.ndarray | None = None
        self.mobile_com: np.ndarray | None = None
        self.ref_com: np.ndarray | None = None
        self.box_rotation_matrix: np.ndarray = np.eye(3)

        # anchors
        self.anchor_atoms: List[str] = []
        self.ligand_anchor_atom: str | None = None
        self.l1_x = self.l1_y = self.l1_z = None
        self.l1_range = None
        self.p1 = self.p2 = self.p3 = None

    @property
    def system_name(self) -> str:
        return self._system_name

    def _resolve_input_path(self, p: str) -> str:
        ap = _as_abs(p, self.yaml_dir)
        if ap is None:
            raise ValueError("unexpected None path")
        return str(ap)

    @contextlib.contextmanager
    def _change_dir(self, path: Path):
        cwd = Path.cwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(cwd)

    def _prepare_membrane(self):
        """
        Convert input lipid names to lipid21 set (PC/PA/OL for POPC) via lookup CSV.
        """
        logger.debug("Input: membrane system")

        # read charmmlipid2amber file
        charmm_csv_path = resources.files("batter") / "data/charmmlipid2amber.csv"
        charmm_amber_lipid_df = pd.read_csv(charmm_csv_path, header=1, sep=",")

        lipid_mol = list(self.lipid_mol)
        logger.debug(f"Converting lipid input: {lipid_mol}")
        amber_lipid_mol = charmm_amber_lipid_df.query("residue in @lipid_mol")[
            "replace"
        ]
        amber_lipid_mol = (
            amber_lipid_mol.apply(lambda x: x.split()[1]).unique().tolist()
        )

        # extend instead of replacing so that we can have both
        lipid_mol.extend(amber_lipid_mol)
        self.lipid_mol = lipid_mol
        logger.debug(f"New lipid_mol list: {self.lipid_mol}")

    def _run_input_protein_dssp(self) -> Dict[str, Any]:
        """
        Run DSSP on the input protein structure and persist the assignments.
        """
        dssp_npy = self.ligands_folder / "protein_input_dssp.npy"
        dssp_json = self.ligands_folder / "protein_input_dssp.json"
        try:
            u_prot = mda.Universe(self._protein_input)
            dssp_ana = DSSP(u_prot.select_atoms('protein and not resname NMA ACE')).run()
            dssp_array = np.asarray(dssp_ana.results["dssp"])
        except Exception as exc:
            try:
                logger.warning(f"Failed to run DSSP on full protein input {self._protein_input}, trying with last residue removed")
                dssp_ana = DSSP(u_prot.select_atoms('protein and not resname NMA ACE').residues[:-1].atoms).run()
                dssp_array = np.asarray(dssp_ana.results["dssp"])
            except Exception as exc:
                logger.warning(
                    f"Failed to run DSSP on protein input {self._protein_input}: {exc}. No secondary-structure conditioned restraints. "
                    "If you want to debug, please run `DSSP` in MDAnalysis on the input protein file."
                )
                dssp_array = np.array([])

        np.save(dssp_npy, dssp_array)
        dssp_json.write_text(json.dumps(dssp_array.tolist()))
        return {
            "npy": str(dssp_npy),
            "json": str(dssp_json),
            "shape": list(dssp_array.shape),
            "results": dssp_array.tolist(),
        }

    def _get_alignment(self):
        """
        Prepare for the alignment of the protein and ligand to the system.
        """
        logger.debug("Getting the alignment of the protein and ligand to the system")

        # translate the cog of protein to the origin
        #
        u_prot = mda.Universe(self._protein_input)

        u_sys = mda.Universe(self._system_input_pdb, format="XPDB")
        cog_prot = u_sys.select_atoms("protein and name CA C N O").center_of_geometry()
        u_sys.atoms.positions -= cog_prot

        # get translation-rotation matrix
        mobile = u_prot.select_atoms(self.protein_align).select_atoms(
            "name CA and not resname NMA ACE"
        )
        ref = u_sys.select_atoms(self.protein_align).select_atoms(
            "name CA and not resname NMA ACE"
        )

        if mobile.n_atoms != ref.n_atoms:
            raise ValueError(
                f"Number of atoms in the alignment selection is different: protein_input: "
                f"{mobile.n_atoms} and system_input {ref.n_atoms} \n"
                f"The selection string is {self.protein_align} and name CA and not resname NMA ACE\n"
                f"protein selected resids: {mobile.residues.resids}\n"
                f"system selected resids: {ref.residues.resids}\n"
                "set `protein_align` to a selection string that has the same number of atoms in both files"
                "when running `create_system`."
            )
        mobile_com = mobile.center(weights=None)
        ref_com = ref.center(weights=None)
        mobile_coord = mobile.positions - mobile_com
        ref_coord = ref.positions - ref_com

        _ = align._fit_to(
            mobile_coordinates=mobile_coord,
            ref_coordinates=ref_coord,
            mobile_atoms=u_prot.atoms,
            mobile_com=mobile_com,
            ref_com=ref_com,
        )

        cog_prot = u_prot.select_atoms("protein and name CA C N O").center_of_geometry()
        u_prot.atoms.positions -= cog_prot

        self.box_rotation_matrix = np.eye(3)
        if self._system_topology is None:
            protein_atoms = u_prot.select_atoms("protein and not resname NMA ACE")
            if protein_atoms.n_atoms >= 2:
                rotation_matrix, score_before, score_after = _find_min_xy_box_rotation(
                    protein_atoms.positions
                )
                if _score_lt(score_after, score_before):
                    u_prot.atoms.positions = _apply_rotation(
                        u_prot.atoms.positions, rotation_matrix
                    )
                    u_sys.atoms.positions = _apply_rotation(
                        u_sys.atoms.positions, rotation_matrix
                    )
                    self.box_rotation_matrix = rotation_matrix
                    logger.info(
                        "Optimized protein orientation for smaller XY box area without system_input: "
                        f"{score_before[0]:.2f} -> {score_after[0]:.2f} A^2 "
                        f"(z span {score_before[1]:.2f} -> {score_after[1]:.2f} A)."
                    )

        final_ref = u_prot.select_atoms(self.protein_align).select_atoms(
            "name CA and not resname NMA ACE"
        )
        final_ref_com = final_ref.center(weights=None)
        final_ref_coord = final_ref.positions - final_ref_com

        protein_aligned_path = self.ligands_folder / "protein_aligned.pdb"
        system_aligned_path = self.ligands_folder / "system_aligned.pdb"
        normalized_prot_residues = _write_pdb_with_normalized_protein_segids(
            u_prot, protein_aligned_path
        )
        normalized_sys_residues = _write_pdb_with_normalized_protein_segids(
            u_sys, system_aligned_path
        )
        if normalized_prot_residues or normalized_sys_residues:
            logger.warning(
                "Detected mixed per-atom protein segid assignments; normalized segids "
                f"for {normalized_prot_residues} residue(s) in the aligned protein and "
                f"{normalized_sys_residues} residue(s) in the aligned system before grouping."
            )

        self._protein_aligned_pdb = str(protein_aligned_path)
        self._system_aligned_pdb = str(system_aligned_path)

        # store these for ligand alignment
        self.mobile_com = mobile_com
        self.mobile_coord = mobile_coord
        self.ref_com = final_ref_com
        self.ref_coord = final_ref_coord

    def _process_system(self):
        """
        Generate the protein, reference, and lipid (if applicable) files.
        We will align the protein_input to the system_topology because
        the system_topology is generated by dabble and may be shifted;
        we want to align the protein to the system so the membrane is
        properly positioned.
        """
        logger.debug("Processing the system")

        if not self._protein_aligned_pdb or not self._system_aligned_pdb:
            raise RuntimeError("Alignment not computed. Call _get_alignment() first.")

        u_prot = mda.Universe(self._protein_aligned_pdb)
        u_sys = mda.Universe(self._system_aligned_pdb, format="XPDB")
        try:
            u_sys.atoms.chainIDs
        except AttributeError:
            u_sys.add_TopologyAttr("chainIDs")
        try:
            u_prot.atoms.chainIDs
        except AttributeError:
            u_prot.add_TopologyAttr("chainIDs")

        memb_seg = u_sys.add_Segment(segid="MEMB")
        water_seg = u_sys.add_Segment(segid="WATR")

        protein_fragment_groups: list[tuple[Any, str]] = []
        fragment_chain_index = 0
        protein_source_groups = _group_residues_by_source_identity(
            u_prot.select_atoms("protein").residues
        )

        for source_group in protein_source_groups:
            chain_id = (
                str(source_group[0].atoms.chainIDs[0]).strip() if len(source_group[0].atoms) else ""
            )
            segid = str(source_group[0].segid).strip()
            residue_groups, split_warnings = _split_residues_on_breaks(
                source_group,
                segid=segid,
                chain_id=chain_id,
            )
            for warning_message in split_warnings:
                logger.warning(warning_message)

            for residues in residue_groups:
                new_chain_id = _chain_id_from_index(fragment_chain_index)
                fragment_chain_index += 1

                prot_selection = _select_fragment_atoms(
                    u_prot,
                    residues,
                    chain_id=chain_id,
                    segid=segid,
                )

                prot_selection.atoms.chainIDs = new_chain_id
                protein_fragment_groups.append((prot_selection, new_chain_id))

        comp_2_combined = []

        if self.receptor_segment:
            protein_anchor = u_prot.select_atoms(
                f"segid {self.receptor_segment} and protein"
            )
            other_protein = u_prot.select_atoms(
                f"not segid {self.receptor_segment} and protein"
            )
            comp_2_combined.append(protein_anchor)
            comp_2_combined.append(other_protein)
        else:
            comp_2_combined.append(u_prot.select_atoms("protein"))

        for prot_selection, new_chain_id in protein_fragment_groups:
            prot_selection.residues.segments = u_prot.add_Segment(segid=new_chain_id)

        if self.membrane_simulation:
            membrane_ag = u_sys.select_atoms(f'resname {" ".join(self.lipid_mol)}')
            if len(membrane_ag) == 0:
                logger.warning(
                    f"No membrane atoms found with resname {self.lipid_mol}. Available resnames are {list(np.unique(u_sys.atoms.resnames))}. "
                    "Please check the lipid_mol parameter.",
                )
            else:
                with open(f"{build_files_orig}/memb_opls2charmm.json", "r") as f:
                    MEMB_OPLS_2_CHARMM_DICT = json.load(f)
                if np.any(membrane_ag.names == "O1"):
                    if np.any(membrane_ag.residues.resnames != "POPC"):
                        raise ValueError(
                            f"Found OPLS lipid name {membrane_ag.residues.resnames}, only 'POPC' is supported."
                        )
                    # convert the lipid names to CHARMM names
                    membrane_ag.names = [
                        MEMB_OPLS_2_CHARMM_DICT.get(name, name)
                        for name in membrane_ag.names
                    ]
                    logger.info("Converting OPLS lipid names to CHARMM names.")
                membrane_ag.chainIDs = "M"
                membrane_ag.residues.segments = memb_seg
                logger.debug(f"Number of lipid molecules: {membrane_ag.n_residues}")
                comp_2_combined.append(membrane_ag)
        else:
            membrane_ag = u_sys.atoms[[]]  # empty selection

        # gather water (and ions) around protein/membrane
        water_ag = u_sys.select_atoms(
            "byres (((resname SPC and name O) or water) and around 15 (protein or group memb))",
            memb=membrane_ag,
        )
        logger.debug(f"Number of water molecules: {water_ag.n_residues}")
        ion_ag = u_sys.select_atoms(
            "byres (resname SOD POT CLA NA CL and around 5 (protein))"
        )
        logger.debug(f"Number of ion molecules: {ion_ag.n_residues}")
        # normalize ion names
        ion_ag.select_atoms("resname SOD").names = "Na+"
        ion_ag.select_atoms("resname SOD").residues.resnames = "Na+"
        ion_ag.select_atoms("resname NA").names = "Na+"
        ion_ag.select_atoms("resname NA").residues.resnames = "Na+"
        ion_ag.select_atoms("resname POT").names = "K+"
        ion_ag.select_atoms("resname POT").residues.resnames = "K+"
        ion_ag.select_atoms("resname CLA").names = "Cl-"
        ion_ag.select_atoms("resname CLA").residues.resnames = "Cl-"
        ion_ag.select_atoms("resname CL").names = "Cl-"
        ion_ag.select_atoms("resname CL").residues.resnames = "Cl-"

        water_ag = water_ag + ion_ag
        water_ag.chainIDs = "W"
        water_ag.residues.segments = water_seg
        if len(water_ag) == 0:
            logger.warning(
                f"No water molecules found in the system. Available resnames are {np.unique(u_sys.atoms.resnames)}. "
                "Please check the system_topology and system_coordinate files.",
            )
        else:
            comp_2_combined.append(water_ag)

        u_merged = mda.Merge(*comp_2_combined)

        water = u_merged.select_atoms("water or resname SPC")
        if len(water) != 0:
            logger.debug(
                f"Number of water molecules in merged system: {water.n_residues}"
            )
            logger.debug(f"Water atom names: {water.residues[0].atoms.names}")

        # Normalize water O names for tleap
        water.select_atoms("name OW").names = "O"
        water.select_atoms("name OH2").names = "O"

        box_dim = np.zeros(6)
        if len(self.system_dimensions) == 3:
            box_dim[:3] = self.system_dimensions
            box_dim[3:] = 90.0
        elif len(self.system_dimensions) == 6:
            box_dim = self.system_dimensions
        else:
            raise ValueError(f"Invalid system_dimensions: {self.system_dimensions}")
        u_merged.dimensions = box_dim

        charmm_2_std_resname_map = {
            "HIS": "HIE",   # generic HIS → HIE
            "HSD": "HID",   # δ-protonated
            "HSE": "HIE",   # ε-protonated
            "HIP": "HIP",   # doubly protonated
        }
        def infer_histidine_resname(res) -> str:
            """
            Infer HID/HIE/HIP from explicit hydrogens, if present.
            Falls back to HIE when ambiguous or hydrogens absent.
            """
            # Atom names are the most informative for histidine protonation
            atom_names = {a.name.upper() for a in res.atoms}

            # Common naming across force fields: HD1 on ND1, HE2 on NE2
            has_hd1 = "HD1" in atom_names
            has_he2 = "HE2" in atom_names

            if has_hd1 and has_he2:
                logger.warning(f"Found both HD1 and HE2 in residue {res.resname} {res.resid}; setting to HIP")
                return "HIP"
            if has_hd1:
                return "HID"
            if has_he2:
                return "HIE"

            # If hydrogens exist but aren't named HD1/HE2, we can't reliably infer
            # (or hydrogens are absent entirely). Default to HIE.
            return "HIE"

        # replace CHARMM specific resname
        for res in u_merged.residues:
            # if the protein contains hydrogen and use a generic HIS name, get the correct resname based on protonation
            if res.resname == "HIS":
                new_name = infer_histidine_resname(res)
            else:
                new_name = charmm_2_std_resname_map.get(res.resname, res.resname)
            res.resname = new_name

        charmm_2_std_resname_map = {
            ("ILE", "CD"): "CD1",
        }
        # replace CHARMM specific atom name
        for atom in u_merged.atoms:
            new_name = charmm_2_std_resname_map.get((atom.resname, atom.name), atom.name)
            atom.name = new_name

        u_merged.atoms.write(f"{self.ligands_folder}/{self.system_name}.pdb")
        protein_ref = u_prot.select_atoms("protein")
        protein_ref.write(f"{self.ligands_folder}/reference.pdb")

    def _align_2_system(self, mobile_atoms):
        """
        Apply the stored rigid-body transform to bring a ligand into system frame.
        """
        _ = align._fit_to(
            mobile_coordinates=self.mobile_coord,
            ref_coordinates=self.ref_coord,
            mobile_atoms=mobile_atoms,
            mobile_com=self.mobile_com,
            ref_com=self.ref_com,
        )

    def _prepare_all_ligands(self):
        """
        Prepare ligand ligands for the system from input ligand files (PDB/SDF/MOL2).
        """
        logger.debug("prepare ligands")
        new_ligand_dict: Dict[str, str] = {}
        # name order is deterministic
        for i, (name, ligand_path) in enumerate(sorted(self.ligand_dict.items())):
            name_up = name.upper()
            ligand_file = _ensure_pdb(Path(ligand_path), self.ligandff_folder)

            u = mda.Universe(str(ligand_file))
            try:
                u.atoms.chainIDs
            except AttributeError:
                u.add_TopologyAttr("chainIDs")
            lig_seg = u.add_Segment(segid="LIG")
            u.atoms.chainIDs = "L"
            u.atoms.residues.segments = lig_seg
            u.atoms.residues.resnames = "lig"

            logger.debug(f"Processing ligand {i}: {ligand_path}")
            self._align_2_system(u.atoms)
            out_ligand = f"{self.ligands_folder}/{name}.pdb"
            u.atoms.write(out_ligand)

            new_ligand_dict[name] = out_ligand
        self.ligand_dict = new_ligand_dict

    # -----------------------
    # Orchestrated entry
    # -----------------------
    def run(
        self,
        *,
        system_name: str,
        protein_input: str,
        ligand_paths: Dict[str, str],
        anchor_atoms: List[str],
        system_topology: str | None = None,
        ligand_anchor_atom: str | None = None,
        receptor_segment: str | None = None,
        system_coordinate: str | None = None,
        protein_align: str = "name CA and resid 60 to 250",
        receptor_ff: str = "protein.ff14SB",
        retain_lig_prot: bool = True,
        ligand_ph: float = 7.4,
        lipid_mol: List[str] = [],
        lipid_ff: str = "lipid21",
        unbound_threshold: float | None = None,
        min_adis: float = 3.0,
        max_adis: float = 7.0,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        self._system_name = system_name
        self._protein_input = self._resolve_input_path(protein_input)
        self._system_topology = self._resolve_input_path(system_topology) if system_topology else None
        self._system_coordinate = (
            self._resolve_input_path(system_coordinate)
            if system_coordinate
            else None
        )

        self.ligand_dict = {
            k: self._resolve_input_path(v) for k, v in ligand_paths.items()
        }
        self.ligand_order = list(ligand_paths.keys())
        # prefer the provided keys for naming
        self.unique_mol_names = [k.upper() for k in ligand_paths.keys()]

        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.receptor_ff = receptor_ff
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.overwrite = overwrite

        self.lipid_mol = lipid_mol or []
        self.membrane_simulation = bool(self.lipid_mol)
        self.lipid_ff = lipid_ff

        # sanity checks
        if not Path(self._protein_input).exists():
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        for p in self.ligand_dict.values():
            if not Path(p).exists():
                raise FileNotFoundError(f"Ligand file not found: {p}")
        if self._system_coordinate and not Path(self._system_coordinate).exists():
            raise FileNotFoundError(
                f"System coordinate file not found: {system_coordinate}"
            )

        # Directories
        self.ligands_folder.mkdir(parents=True, exist_ok=True)
        dssp_result = self._run_input_protein_dssp()

        # Box dimensions
        if self.membrane_simulation or self._system_topology is not None:
            u_sys = mda.Universe(self._system_topology, format="XPDB")
            if self._system_coordinate:
                with open(self._system_coordinate) as f:
                    lines = f.readlines()
                    box = np.array([float(x) for x in lines[-1].split()])
                self.system_dimensions = box
                u_sys.load_new(self._system_coordinate, format="INPCRD")
            else:
                try:
                    self.system_dimensions = u_sys.dimensions[:3]
                except TypeError:
                    if self.membrane_simulation:
                        raise ValueError(
                            "No box dimensions found in system_topology; required for membrane systems."
                        )
                    protein = u_sys.select_atoms("protein")
                    padding = 10.0
                    box_x = (
                        protein.positions[:, 0].max()
                        - protein.positions[:, 0].min()
                        + 2 * padding
                    )
                    box_y = (
                        protein.positions[:, 1].max()
                        - protein.positions[:, 1].min()
                        + 2 * padding
                    )
                    box_z = (
                        protein.positions[:, 2].max()
                        - protein.positions[:, 2].min()
                        + 2 * padding
                    )
                    self.system_dimensions = np.array([box_x, box_y, box_z])
                    logger.warning(
                        "No box dimensions in system_topology. Using default 10 Å padding around protein. "
                        f"Box dimensions: {self.system_dimensions}"
                    )
            u_sys.atoms.write(f"{self.ligands_folder}/system_input.pdb")
            self._system_input_pdb = f"{self.ligands_folder}/system_input.pdb"
        else:
            self._system_input_pdb = self._protein_input
        if (
            self.membrane_simulation
            and (u_sys.atoms.dimensions is None or not u_sys.atoms.dimensions.any())
            and self._system_coordinate is None
        ):
            raise ValueError(
                "No box dimensions found in system_topology or system_coordinate when lipid system is on."
            )


        # membrane remapping (if any)
        if self.membrane_simulation:
            self._prepare_membrane()

        # Align protein to system, save aligned files, compute translation
        self._get_alignment()

        # Build reference & docked PDBs
        self._process_system()

        # Make <ligand>.pdb for each ligand by translation-only
        self._prepare_all_ligands()

        # Anchors from a real ligand + protein when available, otherwise apo protein geometry.
        u_prot = mda.Universe(f"{self.output_dir}/all-ligands/reference.pdb")
        anchor_ligand_name, anchor_ligand_is_apo = _select_anchor_reference_ligand(
            self.ligand_order,
            ligand_paths,
        )
        if (
            self.ligand_order
            and anchor_ligand_name != self.ligand_order[0]
            and not anchor_ligand_is_apo
        ):
            logger.info(
                "[system_prep] Using real ligand '{}' for anchor reference instead of apo dummy '{}'.",
                anchor_ligand_name,
                self.ligand_order[0],
            )
        anchor_ligand_path = self.ligand_dict[anchor_ligand_name]
        u_lig = mda.Universe(anchor_ligand_path)
        lig_sdf = _ligand_sdf_reference(
            ligand_paths[anchor_ligand_name],
            is_apo=anchor_ligand_is_apo,
        )
        resolved_anchor_atoms = list(anchor_atoms or [])
        if not resolved_anchor_atoms:
            if anchor_ligand_is_apo:
                resolved_anchor_atoms = select_apo_receptor_anchor_atoms(
                    u_prot,
                    protein_dssp=dssp_result.get("results"),
                )
            else:
                resolved_anchor_atoms = select_receptor_anchor_atoms(
                    u_prot,
                    u_lig,
                    lig_sdf,
                    protein_dssp=dssp_result.get("results"),
                )

        l1_x, l1_y, l1_z, p1, p2, p3, l1_range = find_anchor_atoms(
            u_prot,
            u_lig,
            lig_sdf,
            resolved_anchor_atoms,
            ligand_anchor_atom,
            unbound_threshold=unbound_threshold,
            protein_dssp=dssp_result.get("results"),
            apo_ligand=anchor_ligand_is_apo,
            apo_ligand_distance=(float(min_adis) + float(max_adis)) / 2.0,
        )
        self.anchor_atoms = resolved_anchor_atoms
        self.ligand_anchor_atom = ligand_anchor_atom
        self.l1_x, self.l1_y, self.l1_z = l1_x, l1_y, l1_z
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.l1_range = l1_range

        # manifest for downstream steps
        manifest = {
            "system_name": self._system_name,
            "reference": str(self.ligands_folder / "reference.pdb"),
            "docked": str(self.ligands_folder / f"{self._system_name}.pdb"),
            "ligands": dict(self.ligand_dict),
            "dssp": dssp_result,
            "anchors": {"p1": self.p1, "p2": self.p2, "p3": self.p3},
            "anchor_atom_selections": list(self.anchor_atoms),
            "l1": {
                "x": self.l1_x,
                "y": self.l1_y,
                "z": self.l1_z,
                "range": self.l1_range,
            },
            "membrane": (
                {"lipid_mol": self.lipid_mol, "lipid_ff": self.lipid_ff}
                if self.membrane_simulation
                else None
            ),
        }
        (self.ligands_folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        logger.debug("System loaded and prepared.")
        return manifest


def system_prep(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Prepare a system by aligning components and generating reference structures.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Paths to generated reference structures and a metadata dictionary with
        anchor and membrane information.
    """
    logger.info(f"[system_prep] Preparing system in {system.root}")
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams()
    yaml_dir = Path(sys_params["yaml_dir"]).resolve()
    threshold_val = sys_params.get(
        "unbound_threshold",
        getattr(payload.sim, "unbound_threshold", 8.0),
    )
    unbound_threshold = (
        float(threshold_val) if threshold_val is not None else None
    )
    ligand_paths = dict(sys_params["ligand_paths"])
    apo_only = bool(ligand_paths) and all(
        is_apo_ligand_path(path) for path in ligand_paths.values()
    )
    if apo_only:
        logger.info("[system_prep] Apo dummy ligand detected; skipping unbound check.")
        unbound_threshold = None

    runner = _SystemPrepRunner(system, yaml_dir)
    manifest = runner.run(
        system_name=sys_params["system_name"],
        protein_input=sys_params["protein_input"],
        system_topology=sys_params.get("system_input", None),
        ligand_paths=ligand_paths,
        anchor_atoms=list(sys_params.get("anchor_atoms", [])),
        ligand_anchor_atom=sys_params.get("ligand_anchor_atom"),
        receptor_segment=sys_params.get("receptor_segment"),
        system_coordinate=sys_params.get("system_coordinate"),
        protein_align=sys_params.get("protein_align", "name CA and resid 60 to 250"),
        receptor_ff=sys_params.get("receptor_ff", "protein.ff14SB"),
        retain_lig_prot=bool(sys_params.get("retain_lig_prot", True)),
        ligand_ph=float(sys_params.get("ligand_ph", 7.4)),
        lipid_mol=list(sys_params.get("lipid_mol", [])),
        lipid_ff=sys_params.get("lipid_ff", "lipid21"),
        unbound_threshold=unbound_threshold,
        min_adis=float(sys_params.get("min_adis", 3.0)),
        max_adis=float(sys_params.get("max_adis", 7.0)),
        overwrite=bool(sys_params.get("overwrite", False)),
        verbose=bool(sys_params.get("verbose", False)),
    )

    outputs = [
        system.root / "all-ligands" / "reference.pdb",
        system.root / "all-ligands" / f"{sys_params['system_name']}.pdb",
    ]
    updates = {
        "p1": manifest["anchors"]["p1"],
        "p2": manifest["anchors"]["p2"],
        "p3": manifest["anchors"]["p3"],
        "l1_x": manifest["l1"]["x"],
        "l1_y": manifest["l1"]["y"],
        "l1_z": manifest["l1"]["z"],
        "l1_range": manifest["l1"]["range"],
        "lipid_mol": manifest["membrane"]["lipid_mol"] if manifest["membrane"] else [],
    }
    (manifest_dir := (system.root / "artifacts" / "config")).mkdir(
        parents=True, exist_ok=True
    )
    overrides_path = system.root / "artifacts" / "config" / "sim_overrides.json"
    overrides_path.write_text(json.dumps(updates, indent=2))

    marker_rel = overrides_path.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "system_prep",
        required=[[marker_rel]],
        success=[[marker_rel]],
    )

    logger.info(f"[system_prep] System preparation complete.")
    info = {"system_prep_ok": True, **manifest, "sim_updates": updates}
    return ExecResult(outputs, info)
