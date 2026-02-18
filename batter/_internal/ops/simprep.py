from __future__ import annotations

from pathlib import Path
import json
import pickle
import os
from typing import Iterable, List, Tuple, Optional, Set, Sequence
import shutil
import re
import random

from batter.param import ligand
from batter.utils.builder_utils import get_buffer_z
from loguru import logger

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align

from batter._internal.builders.fe_registry import register_create_simulation
from batter._internal.builders.interfaces import BuildContext
from batter._internal.ops.helpers import (
    load_anchors,
    save_anchors,
    Anchors,
    get_sdr_dist,
    copy_if_exists as _copy_if_exists,
    is_atom_line as _is_atom_line,
    field_slice as _field,
)

from batter.utils import run_with_log
from batter.config.simulation import SimulationConfig
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign, AllChem
from kartograf.atom_mapper import KartografAtomMapper
from kartograf import SmallMoleculeComponent
from kartograf.atom_aligner import align_mol_shape

ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}


# ---------------------- small utils ----------------------
def _rel_symlink(target: Path, link_path: Path) -> None:
    """Create/replace a relative symlink at link_path → target."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)
def _read_nonblank_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]


def _safe_resid(resid: int) -> int:
    """Clamp resid into PDB 1..9999 domain."""
    r = resid % 10000
    return r if r != 0 else 1


def _fmt_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
) -> str:
    name4 = f"{name:>4s}"[:4]
    res3 = f"{resname:>3s}"[:3]
    chain1 = (chain or " ")[:1]
    resid4 = _safe_resid(resid)
    return (
        f"ATOM  {serial:5d} {name4} {res3} {chain1}{resid4:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{0.00:6.2f}{0.00:6.2f}"
    )


def _append_ligand_to_build(build_pdb: Path, lig_pdb: Path, *, resname: str) -> None:
    """Append ligand atoms from lig_pdb to build_pdb as a new residue."""
    if not build_pdb.exists():
        raise FileNotFoundError(f"Missing build file: {build_pdb}")
    if not lig_pdb.exists():
        raise FileNotFoundError(f"Missing ligand PDB: {lig_pdb}")

    lines = build_pdb.read_text().splitlines()
    # strip END lines
    while lines and lines[-1].strip().startswith("END"):
        lines.pop()

    last_serial = 0
    last_resid = 0
    for ln in lines:
        if _is_atom_line(ln):
            try:
                last_serial = max(last_serial, int(_field(ln, 6, 11) or 0))
            except Exception:
                pass
            try:
                last_resid = max(last_resid, int(_field(ln, 22, 26) or 0))
            except Exception:
                pass

    new_resid = last_resid + 1 if last_resid else 1
    serial = last_serial + 1
    appended: List[str] = []
    for ln in lig_pdb.read_text().splitlines():
        if not _is_atom_line(ln):
            continue
        name = _field(ln, 12, 16) or "C"
        chain = _field(ln, 21, 22) or "A"
        try:
            x = float(_field(ln, 30, 38) or 0.0)
            y = float(_field(ln, 38, 46) or 0.0)
            z = float(_field(ln, 46, 54) or 0.0)
        except Exception:
            x, y, z = 0.0, 0.0, 0.0
        appended.append(
            _fmt_atom_line(serial, name, resname, chain, new_resid, x, y, z)
        )
        serial += 1

    if not appended:
        raise RuntimeError(f"No atoms found in ligand PDB: {lig_pdb}")

    lines.append("TER")
    lines.extend(appended)
    lines.append("TER")
    lines.append("END")
    build_pdb.write_text("\n".join(lines) + "\n")


def _set_dummy_position(
    build_pdb: Path, pos: np.ndarray, target_resid: int | None = None
) -> bool:
    """Update DUM atom coordinates in build_pdb. Returns True if updated."""
    if not build_pdb.exists():
        raise FileNotFoundError(f"Missing build file: {build_pdb}")

    lines = build_pdb.read_text().splitlines()
    out: List[str] = []
    updated = False
    for ln in lines:
        if _is_atom_line(ln):
            resname = _field(ln, 17, 20)
            if resname == "DUM":
                try:
                    serial = int(_field(ln, 6, 11) or 1)
                except Exception:
                    serial = 1
                name = _field(ln, 12, 16) or "Pb"
                chain = _field(ln, 21, 22) or "Z"
                try:
                    resid = int(_field(ln, 22, 26) or 1)
                except Exception:
                    resid = 1
                if target_resid is not None and resid != target_resid:
                    out.append(ln)
                    continue
                out.append(
                    _fmt_atom_line(
                        serial,
                        name,
                        "DUM",
                        chain,
                        resid,
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]),
                    )
                )
                updated = True
                continue
        out.append(ln)

    if updated:
        build_pdb.write_text("\n".join(out) + "\n")
    return updated


def _append_dummy_atom(
    build_pdb: Path, pos: np.ndarray, target_resid: int | None = None
) -> None:
    """Append a new DUM atom to build_pdb (best-effort fallback)."""
    lines = build_pdb.read_text().splitlines()
    # strip END lines
    while lines and lines[-1].strip().startswith("END"):
        lines.pop()
    last_serial = 0
    last_resid = 0
    for ln in lines:
        if _is_atom_line(ln):
            try:
                last_serial = max(last_serial, int(_field(ln, 6, 11) or 0))
            except Exception:
                pass
            try:
                last_resid = max(last_resid, int(_field(ln, 22, 26) or 0))
            except Exception:
                pass
    serial = last_serial + 1
    resid = target_resid
    if resid is None:
        resid = last_resid + 1 if last_resid else 1
    lines.append(
        _fmt_atom_line(
            serial,
            "Pb",
            "DUM",
            "Z",
            resid,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
    )
    lines.append("TER")
    lines.append("END")
    build_pdb.write_text("\n".join(lines) + "\n")


def _write_build_dry_no_water(build_pdb: Path, out_dry: Path) -> None:
    """Write build-dry.pdb by filtering out water residues (WAT)."""
    with build_pdb.open("rt") as fin, out_dry.open("wt") as fout:
        for ln in fin:
            if _is_atom_line(ln) and len(ln) >= 20 and ln[17:20] == "WAT":
                continue
            fout.write(ln)


def filter_element_changes(
    molA: Chem.Mol, molB: Chem.Mol, mapping: dict[int, int]
) -> dict[int, int]:
    """Forces a mapping to exclude any alchemical element changes in the core"""
    filtered_mapping = {}

    for i, j in mapping.items():
        if (
            molA.GetAtomWithIdx(i).GetAtomicNum()
            != molB.GetAtomWithIdx(j).GetAtomicNum()
        ):
            continue
        filtered_mapping[i] = j

    return filtered_mapping


def set_mol_positions(mol: Chem.Mol, xyz: np.ndarray, conf_id: int = -1) -> Chem.Mol:
    """
    Set atomic coordinates for mol from xyz (shape: (n_atoms, 3)).
    Creates a conformer if needed; otherwise overwrites existing.
    Returns the same mol (mutated).
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(f"xyz must be shape {(mol.GetNumAtoms(), 3)}, got {xyz.shape}")

    # Ensure we have a conformer
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetId(0)
        mol.AddConformer(conf, assignId=True)
        conf_id = 0

    conf = mol.GetConformer(conf_id)

    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    return mol

def set_alt_coords_from_ref_mapping(
    ref: Chem.Mol,
    alt: Chem.Mol,
    ref_to_alt: Dict[int, int],
) -> Chem.Mol:
    """
    Return a copy of `alt` where coordinates of mapped atoms are overwritten
    to match those atoms' coordinates in `ref`.

    Assumptions:
      - `ref` and `alt` each have exactly one conformer (3D coords exist).
      - `ref_to_alt` maps 0-based atom indices: {ref_idx: alt_idx}.
    """
    ref_conf = ref.GetConformer()  # only one
    alt_out = Chem.Mol(alt)        # copy
    alt_conf = alt_out.GetConformer()

    for ref_idx, alt_idx in ref_to_alt.items():
        p = ref_conf.GetAtomPosition(ref_idx)
        alt_conf.SetAtomPosition(alt_idx, Point3D(p.x, p.y, p.z))

    return alt_out



def force_mapped_coords_and_minimize(
    lig1: Chem.Mol,
    lig2: Chem.Mol,
    atom_map_1to2: list[tuple[int, int]],
    ff: str = "MMFF",
    maxIters: int = 500,
    restrain_instead_of_freeze: bool = False,
    k: float = 1e6,
):
    """
    lig1, lig2 must have at least one conformer each.
    atom_map_1to2: list of (idx_in_lig1, idx_in_lig2), 0-based indices.
    Returns lig2 (modified) with minimized geometry.
    """
    if lig1.GetNumConformers() == 0 or lig2.GetNumConformers() == 0:
        raise ValueError("Both ligands must already have 3D conformers.")

    conf1 = lig1.GetConformer()
    conf2 = lig2.GetConformer()

    # 1) Align lig2 -> lig1 based on mapped atoms (moves lig2 conformer)
    # rdMolAlign uses (probeAtomIdx, refAtomIdx) pairs for atomMap
    atomMap_probe_to_ref = [(j, i) for (i, j) in atom_map_1to2]
    rdMolAlign.AlignMol(lig2, lig1, atomMap=atomMap_probe_to_ref)

    # 2) Overwrite mapped atom coordinates in lig2 to EXACTLY match lig1
    for i1, i2 in atom_map_1to2:
        p = conf1.GetAtomPosition(i1)
        conf2.SetAtomPosition(i2, p)

    # 3) Build force field
    lig2_ffmol = Chem.Mol(lig2)  # keep same object; just being explicit
    if ff.upper() == "MMFF":
        props = AllChem.MMFFGetMoleculeProperties(lig2_ffmol, mmffVariant="MMFF94s")
        if props is None:
            raise ValueError("MMFF properties failed (missing params?). Try ff='UFF'.")
        ffobj = AllChem.MMFFGetMoleculeForceField(lig2_ffmol, props)
    elif ff.upper() == "UFF":
        ffobj = AllChem.UFFGetMoleculeForceField(lig2_ffmol)
    else:
        raise ValueError("ff must be 'MMFF' or 'UFF'")

    # 4) Freeze or strongly restrain mapped atoms, then minimize
    mapped2 = [i2 for _, i2 in atom_map_1to2]
    if restrain_instead_of_freeze:
        # Keep atoms near their exact target coords; still allows tiny relaxation
        # (maxDispl=0.0 means hard constraint in practice; you can set small >0 if needed)
        for a in mapped2:
            ffobj.AddPositionConstraint(a, maxDispl=0.0, forceConstant=k)
    else:
        # Truly freeze atoms at their current coordinates
        for a in mapped2:
            ffobj.AddFixedPoint(a)

    ffobj.Initialize()
    ffobj.Minimize(maxIts=maxIters)

    return lig2_ffmol

# ---------------------- unified writer ----------------------


def write_build_from_aligned(
    *,
    lig: str,
    window_dir: Path,
    build_dir: Path,
    aligned_pdb: Path,
    other_mol: Iterable[str],
    lipid_mol: Iterable[str],
    ion_mol: Iterable[str],
    extra_ligand_shift: List = [],
    sdr_dist: float = 0.0,
    start_off_set: int = 0,
    use_ter_markers: bool = False,
    ter_atoms: Optional[Set[int]] = None,
) -> int:
    """
    Write build.pdb and build-dry.pdb from an aligned system PDB file,
    adding dummy atoms and categorizing residues.

    Returns the last receptor residue id.
    """
    ter_atoms = ter_atoms or set()

    # ---- read aligned system
    lines = [ln for ln in aligned_pdb.read_text().splitlines() if ln.strip()]

    # ---- load ALL dumN.pdb. If none, synthesize one at origin.
    coords_dum: List[Tuple[float, float, float]] = []
    atom_dum: List[Tuple[str, str, int, str]] = []  # (name, resname, resid, chain)
    for dfile in sorted(build_dir.glob("dum[0-9]*.pdb")):
        dlines = [ln for ln in dfile.read_text().splitlines() if ln.strip()]
        # convention: coordinates on the 2nd line (index 1)
        if len(dlines) >= 2 and _is_atom_line(dlines[1]):
            x = float(_field(dlines[1], 30, 38) or 0.0)
            y = float(_field(dlines[1], 38, 46) or 0.0)
            z = float(_field(dlines[1], 46, 54) or 0.0)
            name = _field(dlines[1], 12, 16) or "DU"
            resname = _field(dlines[1], 17, 20) or "DUM"
            resid = int(float(_field(dlines[1], 22, 26) or 1))
            chain = _field(dlines[1], 21, 22) or "A"
            coords_dum.append((x, y, z))
            atom_dum.append((name, resname, resid, chain))
    if not coords_dum:
        coords_dum.append((0.0, 0.0, 0.0))
        atom_dum.append(("DU", "DUM", 1, "Z"))

    dum_count = len(coords_dum)

    # ---- categorize atoms
    om = set(other_mol or [])
    lm = set(lipid_mol or [])
    im = set(ion_mol or [])
    recep_block: List[Tuple[str, str, int, str, float, float, float]] = []
    lig_block: List[Tuple[str, str, int, str, float, float, float]] = []
    oth_block: List[Tuple[str, str, int, str, float, float, float]] = []
    recep_last_resid = 0

    for ln in lines:
        if not _is_atom_line(ln):
            continue
        resname = _field(ln, 17, 21)
        chain = _field(ln, 21, 22)
        resid = int(_field(ln, 22, 26) or 0)
        name = _field(ln, 12, 16)
        x = float(_field(ln, 30, 38) or 0.0)
        y = float(_field(ln, 38, 46) or 0.0)
        z = float(_field(ln, 46, 54) or 0.0)

        if (
            resname not in {lig, "DUM", "WAT"}
            and resname not in om
            and resname not in lm
            and resname not in im
        ):
            recep_block.append((name, resname, resid - start_off_set, chain, x, y, z))
            recep_last_resid = max(recep_last_resid, resid - start_off_set)
        elif resname == lig:
            lig_block.append((name, resname, resid - start_off_set, chain, x, y, z))
        else:
            oth_block.append((name, resname, resid - start_off_set, chain, x, y, z))

    # ---- write build.pdb
    out_build = window_dir / "build.pdb"
    out_build.parent.mkdir(parents=True, exist_ok=True)
    with out_build.open("w") as fout:
        serial = 1

        # Dummy blocks (all)
        for (name, resname, resid, chain), (x, y, z) in zip(atom_dum, coords_dum):
            fout.write(
                _fmt_atom_line(serial, name, resname, chain, resid, x, y, z) + "\n"
            )
            fout.write("TER\n")
            serial += 1
            recep_last_resid += 1
        fout.write("TER\n")

        # Receptor (+dum_count)
        prev_chain = None
        for name, resname, resid, chain, x, y, z in recep_block:
            if (
                prev_chain is not None
                and chain != prev_chain
                and resname not in om
                and resname != "WAT"
            ):
                fout.write("TER\n")
            prev_chain = chain
            fout.write(
                _fmt_atom_line(serial, name, resname, chain, resid + dum_count, x, y, z)
                + "\n"
            )
            serial += 1
            if use_ter_markers:
                leg_idx = resid + 2 - dum_count
                if leg_idx in (ter_atoms or set()):
                    fout.write("TER\n")
        fout.write("TER\n")

        # Ligand at lig_resid
        for name, resname, resid, chain, x, y, z in lig_block:
            fout.write(
                _fmt_atom_line(serial, name, resname, chain, resid + dum_count, x, y, z)
                + "\n"
            )
            serial += 1
        fout.write("TER\n")

        # Optional shifted ligand copy (+sdr_dist along z) for z/v/o with SDR/EXCHANGE
        # extra_ligand_shift is a list of whether to shift the ligand or not
        for i, shift in enumerate(extra_ligand_shift, start=1):
            # read dum{i}.pdb for the x,y shift of the extra ligand copy
            lig_x_y_shift = (0.0, 0.0)
            if shift:
                dum_pdb = build_dir / f"dum{i}.pdb"
                if dum_pdb.exists():
                    dlines = [ln for ln in dum_pdb.read_text().splitlines() if ln.strip()]
                    if len(dlines) >= 2 and _is_atom_line(dlines[1]):
                        x = float(_field(dlines[1], 30, 38) or 0.0)
                        y = float(_field(dlines[1], 38, 46) or 0.0)
                        lig_x_y_shift = (x, y)
                    else:
                        logger.warning(f"[simprep] {dum_pdb} is malformed; using no x/y shift for extra ligand copy.")
                else:
                    logger.warning(f"[simprep] {dum_pdb} not found; using no x/y shift for extra ligand copy.")
            shift_sdr_dist = sdr_dist if shift else 0.0
            for name, _, __, chain, x, y, z in lig_block:
                fout.write(
                    _fmt_atom_line(
                        serial,
                        name,
                        lig,
                        chain,
                        resid + i,
                        x - float(lig_x_y_shift[0]),
                        y - float(lig_x_y_shift[1]),
                        z + float(shift_sdr_dist),
                    )
                    + "\n"
                )
                serial += 1
            fout.write("TER\n")

        # Others (+dum_count plus optional ligand offset)
        last_resid = None
        for name, resname, resid, chain, x, y, z in oth_block:
            out_resid = resid + dum_count + len(extra_ligand_shift)
            if last_resid is not None and out_resid != last_resid:
                fout.write("TER\n")
            last_resid = out_resid
            fout.write(
                _fmt_atom_line(serial, name, resname, chain, out_resid, x, y, z) + "\n"
            )
            serial += 1

        fout.write("TER\nEND\n")

    # ---- build-dry.pdb (until first water)
    out_dry = window_dir / "build-dry.pdb"
    with out_build.open() as fin, out_dry.open("w") as fout:
        for ln in fin:
            if len(ln) >= 20 and ln[17:20] == "WAT":
                break
            fout.write(ln)
    return recep_last_resid


# ---------------------- create_simulation_dir: EQUIL ----------------------


def _read_protein_anchors(txt: Path) -> Tuple[str, str, str]:
    """protein_anchors.txt is expected to have 3 lines: P1, P2, P3."""
    if not txt.exists():
        raise FileNotFoundError(f"[simprep] Missing required protein anchors: {txt}")
    lines = _read_nonblank_lines(txt)
    if len(lines) < 3:
        raise ValueError(f"[simprep] protein_anchors.txt malformed: {txt}")
    return lines[0], lines[1], lines[2]


def _read_ligand_anchor_names(
    txt: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not txt.exists():
        logger.warning(f"[simprep] anchors file not found: {txt}")
        return None, None, None
    line = _read_nonblank_lines(txt)[0]
    parts = line.split()
    if len(parts) < 3:
        logger.warning(f"[simprep] anchors file malformed: '{line}'")
        return None, None, None
    return parts[0], parts[1], parts[2]


def _mask(resid: int, atom: Optional[str]) -> Optional[str]:
    return f":{resid}@{atom}" if atom else None


def create_simulation_dir_eq(ctx: BuildContext) -> None:
    """
    Create the simulation directory layout and utility files for equil.
    """
    work = ctx.working_dir
    comp = ctx.comp
    win = ctx.win
    if win != -1:
        raise ValueError("create_simulation_dir_eq should only be called for win == -1")
    lig = ctx.ligand
    mol = ctx.residue_name

    # folders
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    # sources
    src_equil = build_dir / f"equil-{mol}.pdb"
    src_lig_noh = build_dir / f"{mol}-noh.pdb"
    src_anchors = build_dir / f"anchors-{lig}.txt"
    src_prot_anchors = build_dir / "protein_anchors.txt"
    src_dum_prmtop = build_dir / "dum.prmtop"
    src_dum_inpcrd = build_dir / "dum.inpcrd"
    src_dum_frcmod = build_dir / "dum.frcmod"
    src_dum_mol2 = build_dir / "dum.mol2"

    # destinations
    dst_equil = window_dir / f"equil-{mol}.pdb"
    dst_ref = window_dir / "equil-reference.pdb"
    dst_lig = window_dir / f"{mol}.pdb"
    dst_anchors = window_dir / "anchors.txt"
    dst_dum_prmtop = window_dir / "dum.prmtop"
    dst_dum_inpcrd = window_dir / "dum.inpcrd"
    dst_dum_frcmod = window_dir / "dum.frcmod"
    dst_dum_mol2 = window_dir / "dum.mol2"

    # copy inputs
    for s, d in [
        (src_equil, dst_equil),
        (src_equil, dst_ref),
        (src_lig_noh, dst_lig),
        (src_anchors, dst_anchors),
        (src_dum_prmtop, dst_dum_prmtop),
        (src_dum_inpcrd, dst_dum_inpcrd),
        (src_dum_frcmod, dst_dum_frcmod),
        (src_dum_mol2, dst_dum_mol2),
    ]:
        _copy_if_exists(s, d)

    if not dst_equil.exists():
        raise FileNotFoundError(f"[simprep] Missing required input {src_equil}")

    # anchors: protein + ligand atom names
    P1, P2, P3 = _read_protein_anchors(src_prot_anchors)
    l1_name, l2_name, l3_name = _read_ligand_anchor_names(src_anchors)

    # write build.pdb / build-dry.pdb
    recep_last = write_build_from_aligned(
        lig=mol,
        window_dir=window_dir,
        build_dir=build_dir,
        aligned_pdb=dst_equil,
        other_mol=ctx.sim.other_mol,
        lipid_mol=ctx.sim.lipid_mol,
        ion_mol=ION_NAMES,
        extra_ligand_shift=[],
        sdr_dist=0.0,
        start_off_set=0,
        use_ter_markers=False,
    )

    # compute residue numbers for REMARK and anchors.json
    lig_resid = recep_last + 1
    L1 = _mask(lig_resid, l1_name)
    L2 = _mask(lig_resid, l2_name)
    L3 = _mask(lig_resid, l3_name)

    # prepend REMARK A to equil-<lig>.pdb
    first_res = 1
    remark = f"REMARK A  {P1:6s}  {P2:6s}  {P3:6s}  {L1 or 'NA':6s}  {L2 or 'NA':6s}  {L3 or 'NA':6s}  {first_res:4d}  {recep_last:4d}\n"
    orig = dst_equil.read_text().splitlines(True)
    # keep original first line content after remark
    dst_equil.write_text(remark + "".join(orig[1:]))

    # persist anchors.json
    anchors = Anchors(P1=P1, P2=P2, P3=P3, L1=L1, L2=L2, L3=L3, lig_res=str(lig_resid))
    save_anchors(ctx.working_dir, anchors)
    logger.debug("[simprep:equil] wrote build files in {}", window_dir)


# ---------------------- create_simulation_dir: Z ----------------------
@register_create_simulation("z")
def create_simulation_dir_z(ctx: BuildContext) -> None:
    """
    Create the initial simulation directory for component 'z' at window `-1`
    (the FE-equil bootstrap). No chdir; all paths handled explicitly.
    """
    comp = ctx.comp.lower()
    ligand = ctx.ligand
    mol = ctx.residue_name
    sim = ctx.sim
    membrane_builder = sim.membrane_simulation
    buffer_z = sim.buffer_z

    # paths
    sys_root = ctx.system_root
    build_dir = ctx.build_dir
    amber_dir = ctx.amber_dir
    dest_dir = ctx.equil_dir
    ff_dir = sys_root / "simulations" / ligand / "params"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # link amber files
    _rel_symlink(amber_dir, dest_dir / "amber_files")

    # bring build outputs
    for p in build_dir.glob("vac_ligand*"):
        _copy_if_exists(p, dest_dir / p.name)

    for s, d in [
        (build_dir / f"{mol}.pdb", dest_dir / f"{mol}.pdb"),
        (build_dir / f"fe-{mol}.pdb", dest_dir / "build-ini.pdb"),
        (build_dir / f"fe-{mol}.pdb", dest_dir / f"fe-{mol}.pdb"),
        (build_dir / f"anchors-{ligand}.txt", dest_dir / f"anchors-{ligand}.txt"),
        (build_dir / "sdr_info.txt", dest_dir / "sdr_info.txt"),
        (build_dir / "equil-reference.pdb", dest_dir / "equil-reference.pdb"),
        (build_dir / "dum.inpcrd", dest_dir / "dum.inpcrd"),
        (build_dir / "dum.prmtop", dest_dir / "dum.prmtop"),
        (build_dir / "rec_file.pdb", dest_dir / "rec_file.pdb"),
    ]:
        _copy_if_exists(s, d)

    # copy ff files (ligand + dum)
    for p in ff_dir.glob(f"{mol}.*"):
        _copy_if_exists(p, dest_dir / p.name)
    for p in build_dir.glob("dum.*"):
        _copy_if_exists(p, dest_dir / p.name)

    # derive TER list from rec_file (no waters)
    rec_clean = dest_dir / "rec_file-clean.pdb"
    rec_amber = dest_dir / "rec_amber.pdb"
    with (dest_dir / "rec_file.pdb").open() as fin, rec_clean.open("w") as fout:
        for ln in fin:
            if len(ln) >= 22 and ln[17:20] != "WAT":
                fout.write(ln)

    run_with_log(f"pdb4amber -i {rec_clean} -o {rec_amber} -y")
    ter_atoms: List[int] = []
    with rec_amber.open() as f:
        for ln in f:
            if ln.startswith("TER"):
                try:
                    ter_atoms.append(int(ln[6:11].strip()))
                except Exception:
                    pass

    # open sdr_info to read SDR distance
    sdr_dist, abs_z, buffer_z_left = map(float, open(dest_dir / "sdr_info.txt").read().split())

    # write build files for z
    write_build_from_aligned(
        lig=mol,
        window_dir=dest_dir,
        build_dir=build_dir,
        aligned_pdb=dest_dir / "build-ini.pdb",
        other_mol=ctx.sim.other_mol,
        lipid_mol=ctx.sim.lipid_mol,
        ion_mol=ION_NAMES,
        extra_ligand_shift=[True],  # SDR copy
        sdr_dist=sdr_dist,
        start_off_set=1,  # equil offset
        use_ter_markers=True,
        ter_atoms=set(ter_atoms),
    )

    logger.debug(f"[simprep:z] simulation directory created → {dest_dir}")


@register_create_simulation("x")
def create_simulation_dir_x(ctx: BuildContext) -> None:
    """
    RBFE (x-component) simulation-dir builder.
    """
    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    res_ref = extra.get("residue_ref") or ctx.residue_name
    res_alt = extra.get("residue_alt")
    logger.debug(f"[simprep:x] {lig_ref} → {lig_alt} ({res_ref} → {res_alt})")

    if not lig_ref or not lig_alt or not res_ref or not res_alt:
        raise ValueError(
            "RBFE component 'x' requires pair metadata "
            "(ligand_ref/ligand_alt/residue_ref/residue_alt)."
        )

    sys_root = ctx.system_root
    build_dir = ctx.build_dir
    amber_dir = ctx.amber_dir
    dest_dir = ctx.equil_dir
    sim = ctx.sim
    buffer_z = sim.buffer_z
    ion_def = sim.ion_def

    membrane_builder = sim.membrane_simulation
    mol = ctx.residue_name
    # ligand is ligand pair
    ligand = ctx.ligand
    protein_align = sim.protein_align
    ref_equil_dir = sys_root / "simulations" / str(lig_ref) / "equil"
    alt_equil_dir = sys_root / "simulations" / str(lig_alt) / "equil"
    ref_pre_fe = sys_root / "simulations" / str(lig_ref) / "fe" / "z" / "z-1"
    alt_pre_fe = sys_root / "simulations" / str(lig_alt) / "fe" / "z" / "z-1"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # link amber files
    _rel_symlink(amber_dir, dest_dir / "amber_files")

    # bring build outputs from reference ligand build (fe-<ref>.pdb, anchors, etc.)
    for s, d in [
        (build_dir / f"{res_ref}.pdb", dest_dir / f"{res_ref}.pdb"),
        (build_dir / f"fe-{res_ref}.pdb", dest_dir / f"fe-{res_ref}.pdb"),
        (build_dir / f"anchors-{lig_ref}.txt", dest_dir / f"anchors-{lig_ref}.txt"),
        (build_dir / "equil-reference.pdb", dest_dir / "equil-reference.pdb"),
        (build_dir / "rec_file.pdb", dest_dir / "rec_file.pdb"),
        (build_dir / "dum.inpcrd", dest_dir / "dum.inpcrd"),
        (build_dir / "dum.prmtop", dest_dir / "dum.prmtop"),
        (build_dir / "dum.frcmod", dest_dir / "dum.frcmod"),
        (build_dir / "dum.mol2", dest_dir / "dum.mol2"),
        (ref_pre_fe / "full.prmtop", dest_dir / "ref_full.prmtop"),
        (ref_pre_fe / "full.pdb", dest_dir / "ref_full.pdb"),
        (ref_pre_fe / "other_parts.prmtop", dest_dir / "other_parts.prmtop"),
        (ref_pre_fe / "other_parts.pdb", dest_dir / "other_parts.pdb"),
        (ref_pre_fe / "vac.prmtop", dest_dir / "ref_vac.prmtop"),
        (ref_pre_fe / "vac.pdb", dest_dir / "ref_vac.pdb"),
        (ref_pre_fe / "eq_output.pdb", dest_dir / "ref_eq_output.pdb"),
        (ref_pre_fe / "build_amber_renum.txt", dest_dir / "build_amber_renum.txt"),
        (alt_equil_dir / "representative.pdb", dest_dir / "alter_representative.pdb"),
        (alt_pre_fe / "solvate_ligands.prmtop", dest_dir / "alter_ligand.prmtop"),
    ]:
        _copy_if_exists(s, d)

    # copy reference ligand FF files
    ff_ref = sys_root / "simulations" / str(lig_ref) / "params"
    for p in ff_ref.glob(f"{res_ref}.*"):
        _copy_if_exists(p, dest_dir / p.name)

    # copy alternate ligand FF files
    ff_alt = sys_root / "simulations" / str(lig_alt) / "params"
    for p in ff_alt.glob(f"{res_alt}.*"):
        _copy_if_exists(p, dest_dir / p.name)

    # Prefer pre_fe_equil eqnpt04 coordinates for the reference complex
    build_ini = dest_dir / "build-ini.pdb"
    # only full.pdb have the correct resid information
    ref_pdb = dest_dir / "ref_full.pdb"
    ref_pdb_coord = dest_dir / "ref_eq_output.pdb"
    if ref_pdb.exists() and ref_pdb_coord.exists():
        try:
            u_ref = mda.Universe(ref_pdb.as_posix(), ref_pdb_coord.as_posix())
            ion_names = " ".join(sorted(ION_NAMES))
            try:
                sel = u_ref.select_atoms(f"not resname WAT {ion_names} DUM")
            except Exception:
                sel = u_ref.select_atoms("not resname WAT DUM")
            if sel.n_atoms == 0:
                sel = u_ref.atoms
            sel.write(build_ini.as_posix())
        except Exception as exc:
            raise RuntimeError(f"Failed to build eqnpt04 reference PDB: {exc}")
    else:
        raise FileNotFoundError(
            f"[simprep:x] Missing reference pdb or coordinate: {ref_pdb}, {ref_pdb_coord}"
        )
    
    # get alt and ref mapping and create new coordinates for alt
    sdf_ref = dest_dir / f"{res_ref}.sdf"
    sdf_alt = dest_dir / f"{res_alt}.sdf"

    rdmol_ref = Chem.SDMolSupplier(sdf_ref.as_posix(), removeHs=False)[0]
    rdmol_alt = Chem.SDMolSupplier(sdf_alt.as_posix(), removeHs=False)[0]
    mol_ref = SmallMoleculeComponent.from_rdkit(rdmol_ref)
    mol_alt = SmallMoleculeComponent.from_rdkit(rdmol_alt)

    mol_alt_aligned = align_mol_shape(mol_alt, ref_mol=mol_ref)

    # get mapper based on inital poses
    mapper = KartografAtomMapper(atom_max_distance=0.95, map_hydrogens_on_hydrogens_only=True, atom_map_hydrogens=False,
                                map_exact_ring_matches_only=True, allow_partial_fused_rings=True, allow_bond_breaks=False,
                                additional_mapping_filter_functions=[filter_element_changes]
    )
    # mapper = KartografAtomMapper(additional_mapping_filter_functions=[filter_element_changes])

    # Get Mapping
    kartograf_mapping = next(mapper.suggest_mappings(mol_ref, mol_alt_aligned))
    logger.debug(f"mapping: {kartograf_mapping.componentA_to_componentB}")
    try:
        kartograf_mapping.draw_to_file(fname=dest_dir / "kartograf_mapping.png")
    except RuntimeError:
        pass
    atomMap = [(probe, ref) for ref, probe in sorted(kartograf_mapping.componentB_to_componentA.items())]
    if not atomMap:
        raise ValueError(f"No atom mapping found between {res_ref} and {res_alt}.")

    # align representative_complex.pdb to u_ref
    u_alter = mda.Universe(dest_dir / "alter_representative.pdb")
    align.alignto(
        mobile=u_alter.atoms,
        reference=u_ref.atoms,
        select=f"({protein_align}) and name CA and not resname NMA ACE",
    )
    u_alter_lig = u_alter.select_atoms(f"resname {res_alt}")
    u_alter_lig.write(dest_dir / f"alter_representative_ligand.pdb")
    u_alter_lig_pocket = mda.Universe(dest_dir / f"alter_representative_ligand.pdb")
    u_lig = u_ref.select_atoms(f"resname {res_ref}")
    u_alter_lig.positions = (
        u_alter_lig.positions
        - u_alter_lig.center_of_mass()
        + u_lig.residues[1].atoms.center_of_mass()
    )
    u_alter_lig_merged = mda.Merge(u_alter_lig_pocket.atoms, u_alter_lig)
    u_alter_lig_merged.atoms.write(dest_dir / "alter_ligand.pdb")

    u_lig_alter = mda.Universe(dest_dir / "alter_ligand.prmtop", dest_dir / "alter_ligand.pdb")
    # only one present in the system
    q = u_lig_alter.atoms.charges.sum() / 2.0
    u_lig_charge = int(np.rint(q))
    if u_lig_charge != 0:
        # add ion to the system
        ion = mda.Universe.empty(
            n_atoms=np.abs(u_lig_charge),
            n_residues=np.abs(u_lig_charge),
            atom_resindex=list(range(np.abs(u_lig_charge))),
            trajectory=True,
        )

        # topology attrs (minimal but useful)
        ion_name = ion_def[0] if u_lig_charge < 0 else ion_def[1]
        ion.add_TopologyAttr("name", [ion_name] * np.abs(u_lig_charge))
        ion.add_TopologyAttr("type", [ion_name] * np.abs(u_lig_charge))
        ion.add_TopologyAttr("resname", [ion_name] * np.abs(u_lig_charge))
        ion.add_TopologyAttr("resids", list(range(1, np.abs(u_lig_charge) + 1)))

        # coordinates by adding random number to a random water
        water = u_ref.select_atoms(f"water and not around 10 (protein or resname {res_ref})")

        pos = np.asarray(
            [
                random.choice(water).position + np.random.rand(3)
                for i in range(np.abs(u_lig_charge))
            ]
        ).reshape(np.abs(u_lig_charge), 3)
        ion.atoms.positions = pos
        ion.atoms.write(dest_dir / "ions.pdb")

    # update ref_vac positions
    ref_vac = mda.Universe(dest_dir / "ref_vac.pdb")
    ref_vac.atoms.positions = u_ref.atoms.positions[: ref_vac.atoms.n_atoms]
    ref_vac.dimensions = u_ref.dimensions

    # update DUM protein position
    dum_p = ref_vac.select_atoms('resname DUM')[0]
    dum_p.position = ref_vac.select_atoms('protein and name CA N C O').center_of_mass()
    dum_l = ref_vac.select_atoms('resname DUM')[1]
    ref_res_atoms = ref_vac.select_atoms(f"resname {res_ref}").residues[1].atoms
    mapped_ref_indices = sorted({ref_idx for ref_idx, _ in atomMap})
    dum_l.position = ref_res_atoms[mapped_ref_indices].center_of_mass()

    ref_vac.atoms.write(dest_dir / "ref_vac.pdb")

    # update other_parts positions
    ref_other_parts = mda.Universe(dest_dir / "other_parts.pdb")
    ref_other_parts.atoms.positions = u_ref.atoms.positions[ref_vac.atoms.n_atoms :]
    ref_other_parts.atoms.write(dest_dir / "other_parts.pdb")

    # use reference ligand, steer alt ligand to the atom mapped position.
    ref_pos = u_lig.residues[0].atoms.positions
    mol_ref._rdkit = set_mol_positions(mol_ref._rdkit, ref_pos)

    # set mol_alt_aligned common core positions to be exactly the same as mol_ref
    mol_alt_aligned = align_mol_shape(mol_alt, ref_mol=mol_ref)
    mol_alt_aligned._rdkit = force_mapped_coords_and_minimize(
                mol_ref._rdkit, mol_alt_aligned._rdkit, atom_map_1to2=atomMap, 
    )
    # save mol_alt_aligned as PDB
    alter_site_pdb = dest_dir / "alter_ligand_aligned_site.pdb"
    alter_solvent_pdb = dest_dir / "alter_ligand_aligned_solvent.pdb"
    alter_merged_pdb = dest_dir / "alter_ligand_aligned.pdb"
    pdb_block_m = Chem.MolToPDBBlock(mol_alt_aligned._rdkit)
    with alter_site_pdb.open("w") as f:
        f.write(pdb_block_m)

    ref_pos = u_lig.residues[1].atoms.positions
    mol_ref._rdkit = set_mol_positions(mol_ref._rdkit, ref_pos)
    mol_alt_aligned = align_mol_shape(mol_alt, ref_mol=mol_ref)

    mol_alt_aligned._rdkit = force_mapped_coords_and_minimize(
                mol_ref._rdkit, mol_alt_aligned._rdkit, atom_map_1to2=atomMap, 
    )

    # save mol_alt_aligned as PDB
    pdb_block_m = Chem.MolToPDBBlock(mol_alt_aligned._rdkit)
    with alter_solvent_pdb.open("w") as f:
        f.write(pdb_block_m)

    u_alt_site = mda.Universe(alter_site_pdb.as_posix())
    u_alt_solvent = mda.Universe(alter_solvent_pdb.as_posix())
    u_alt = mda.Merge(u_alt_site.atoms, u_alt_solvent.atoms)
    u_alt.atoms.write(alter_merged_pdb.as_posix())

    with open(dest_dir / "kartograf.json", "w") as f:
        json.dump(kartograf_mapping.componentA_to_componentB, f)
    
    # serialize kartograf
    with open(dest_dir / "kartograf.pkl", "wb") as f:
        pickle.dump(kartograf_mapping, f)

    logger.debug(f"[simprep:x] simulation directory created → {dest_dir}")


@register_create_simulation("y")
@register_create_simulation("m")
def create_simulation_dir_lig(ctx: BuildContext) -> None:
    mol = ctx.residue_name
    ligand = ctx.ligand

    # paths
    sys_root = ctx.system_root
    build_dir = ctx.build_dir
    amber_dir = ctx.amber_dir
    dest_dir = ctx.equil_dir
    ff_dir = sys_root / "simulations" / ligand / "params"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # link amber files
    _rel_symlink(amber_dir, dest_dir / "amber_files")

    # bring build outputs
    for p in build_dir.glob("vac_ligand*"):
        _copy_if_exists(p, dest_dir / p.name)

    for s, d in [
        (build_dir / f"{mol}.pdb", dest_dir / f"{mol}.pdb"),
        (build_dir / "dum.inpcrd", dest_dir / "dum.inpcrd"),
        (build_dir / "dum.prmtop", dest_dir / "dum.prmtop"),
    ]:
        _copy_if_exists(s, d)

    # copy ff files (ligand + dum)
    for p in ff_dir.glob(f"{mol}.*"):
        _copy_if_exists(p, dest_dir / p.name)
    for p in build_dir.glob("dum.*"):
        _copy_if_exists(p, dest_dir / p.name)

    # write build.pdb with dum atom + ligand
    # the position of the DUM atom is the center of mass of the ligand
    u_lig = mda.Universe(dest_dir / f"{mol}.pdb")
    com = u_lig.atoms.center_of_mass()
    u_dum = mda.Universe.empty(
        1, n_residues=1, atom_resindex=[0], residue_segindex=[0], trajectory=True
    )
    u_dum.add_TopologyAttr("name", ["Pb"])
    u_dum.add_TopologyAttr("resname", ["DUM"])
    u_dum.atoms.positions = np.array([com])
    with mda.Writer(dest_dir / "build.pdb", n_atoms=u_lig.atoms.n_atoms + 1) as W:
        W.write(u_dum)
        W.write(u_lig)


# ---------------------- window copier ----------------------
def copy_simulation_dir(source: Path, dest: Path, sim: SimulationConfig) -> None:
    """Symlink (using relative links) or copy only the needed files from the source simulation dir."""
    needed = [
        "full.prmtop",
        "full.inpcrd",
        "full.pdb",
        "vac.pdb",
        "vac_ligand.pdb",
        "vac.prmtop",
        "vac_ligand.prmtop",
        "fe-lig.pdb",
        "lig.mol2",
        "disang.rest",
        "cv.in",
    ]
    if not hasattr(sim, "hmr"):
        raise AttributeError("SimulationConfig missing 'hmr'.")
    if sim.hmr == "yes":
        needed.append("full.hmr.prmtop")

    dest.mkdir(parents=True, exist_ok=True)

    for name in needed:
        src = source / name
        if not src.exists():
            continue

        dst = dest / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        # Always copy files to avoid issues with transferring between computers
        shutil.copy2(src, dst)
