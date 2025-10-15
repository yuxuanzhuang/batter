from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import shutil
import json
from dataclasses import asdict
from dataclasses import dataclass

from loguru import logger

from batter._internal.builders.interfaces import BuildContext
from batter._internal.ops.helpers import load_anchors, save_anchors, Anchors

ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}  # NA/CL appear in some pdbs too


def _read_nonblank_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]


def _is_atom_line(line: str) -> bool:
    tag = line[0:6].strip()
    return tag == "ATOM" or tag == "HETATM"


def _field(line: str, start: int, end: int) -> str:
    # 0-based, end is exclusive (python slice semantics)
    return line[start:end].strip()


def _safe_resid(resid: int) -> int:
    # keep in 1..9999 (0 becomes 1)
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
    """
    Format one ATOM line (PDB v3.3 compliant, 80-char width).

    Columns:
    1–6   Record name ("ATOM  ")
    7–11  Atom serial number
    13–16 Atom name (right-justified if length < 4)
    17    AltLoc (blank)
    18–20 Residue name (right-justified)
    22    Chain ID
    23–26 Residue sequence number
    31–38 x
    39–46 y
    47–54 z
    55–60 occupancy
    61–66 B-factor
    """
    name4 = f"{name:>4s}"[:4]            # right-align atom name into 4 cols
    res3  = f"{resname:>3s}"[:3]         # 3-char residue name
    chain1 = (chain or " ")[:1]          # 1-char chain
    resid4 = _safe_resid(resid)          # clamp to 1..9999

    return (
        f"ATOM  {serial:5d} {name4} {res3} {chain1}{resid4:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{0.00:6.2f}{0.00:6.2f}"
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
    else:
        logger.warning(f"[simprep] expected file not found: {src.name} (continuing)")


def create_simulation_dir(ctx: BuildContext) -> None:
    """
    Create the per-window simulation directory layout and utility files.

    Inputs expected under:
        build_dir = <ctx.working_dir>/<comp>_build_files
        files:
          - equil-<lig>.pdb
          - <lig>-noh.pdb
          - anchors-<pose>.txt  (3 ligand atom names on one line)
          - dum1.pdb, dum.prmtop, dum.inpcrd, dum.frcmod, dum.mol2

    Outputs written into:
        window_dir = <ctx.working_dir>/<comp>-<win or 1>
          - equil-<lig>.pdb            (with REMARK A prepended)
          - equil-reference.pdb        (copy of equil-<lig>.pdb)
          - <lig>.pdb                  (copy of <lig>-noh.pdb)
          - anchors.txt                (copy of anchors-<pose>.txt)
          - dum.prmtop, dum.inpcrd, dum.frcmod, dum.mol2
          - build.pdb, build-dry.pdb
        
    Return
    """
    work = ctx.working_dir
    comp = ctx.comp
    win = ctx.win
    if win != -1:
        raise ValueError("create_simulation_dir should only be called for win == -1")
    lig = ctx.ligand  # canonical ligand residue/name string you used during build_complex
    mol = ctx.residue_name  # residue name in the PDB for the ligand
    lig3 = mol.lower()

    # folders
    build_dir = work / f"{comp}_build_files"
    if comp != "q":
        window_dir = work / f"{comp}-1"
    else:
        window_dir = work
    window_dir.mkdir(parents=True, exist_ok=True)

    # source files in build_dir
    src_equil = build_dir / f"equil-{lig3}.pdb"
    src_lig_noh = build_dir / f"{lig3}-noh.pdb"
    
    src_anchors = build_dir / f"anchors-{lig}.txt"
    if not src_anchors.exists():
        raise FileNotFoundError(f"[simprep] Missing required input {src_anchors}")
    src_dum1 = build_dir / "dum1.pdb"
    src_dum_prmtop = build_dir / "dum.prmtop"
    src_dum_inpcrd = build_dir / "dum.inpcrd"
    src_dum_frcmod = build_dir / "dum.frcmod"
    src_dum_mol2 = build_dir / "dum.mol2"

    # destination paths
    dst_equil = window_dir / f"equil-{lig3}.pdb"
    dst_ref   = window_dir / "equil-reference.pdb"
    dst_lig   = window_dir / f"{lig3}.pdb"
    dst_anchors = window_dir / "anchors.txt"
    dst_dum_prmtop = window_dir / "dum.prmtop"
    dst_dum_inpcrd = window_dir / "dum.inpcrd"
    dst_dum_frcmod = window_dir / "dum.frcmod"
    dst_dum_mol2 = window_dir / "dum.mol2"

    # Copy minimal required inputs
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

    # Read anchor ligand atom names
    l1_name = l2_name = l3_name = None
    if dst_anchors.exists():
        line = _read_nonblank_lines(dst_anchors)[0]
        parts = line.split()
        if len(parts) >= 3:
            l1_name, l2_name, l3_name = parts[0], parts[1], parts[2]
        else:
            logger.warning(f"[simprep] anchors.txt malformed: '{line}'")

    # Parse the aligned system (equil-<lig>.pdb) for categorization and residue info
    lines = _read_nonblank_lines(dst_equil)
    other_mol = set(ctx.sim.other_mol or [])
    lipid_mol = set(ctx.sim.lipid_mol or [])
    ion_mol = set(ION_NAMES)

    # First pass: gather receptor last resid and separate lists
    dum_atom_count = 1  # we will place one dummy atom block at the top
    recep_last_resid = 0
    coords_dum: List[Tuple[float, float, float]] = []
    atom_dum: List[Tuple[str, str, int, str]] = []  # (name, resname, resid, chain)

    # Seed dummy from dum1.pdb if present; otherwise synthesize at origin
    if src_dum1.exists():
        d1 = _read_nonblank_lines(src_dum1)
        if len(d1) >= 2 and _is_atom_line(d1[1]):
            x = float(_field(d1[1], 30, 38))
            y = float(_field(d1[1], 38, 46))
            z = float(_field(d1[1], 46, 54))
            name = _field(d1[1], 12, 16)
            resname = _field(d1[1], 17, 20) or "DUM"
            resid = int(float(_field(d1[1], 22, 26)))
            chain = _field(d1[1], 21, 22) or "A"
            coords_dum.append((x, y, z))
            atom_dum.append((name, resname, resid, chain))
    if not coords_dum:
        coords_dum.append((0.0, 0.0, 0.0))
        atom_dum.append(("DU", "DUM", 1, "Z"))

    recep_block: List[Tuple[str, str, int, str, float, float, float]] = []
    lig_block:   List[Tuple[str, str, int, str, float, float, float]] = []
    oth_block:   List[Tuple[str, str, int, str, float, float, float]] = []

    for ln in lines:
        if not _is_atom_line(ln):
            continue
        resname = _field(ln, 17, 21)
        chain   = _field(ln, 21, 22)
        resid   = int(_field(ln, 22, 26)) if _field(ln, 22, 26) else 0
        name    = _field(ln, 12, 16)
        x = float(_field(ln, 30, 38) or "0")
        y = float(_field(ln, 38, 46) or "0")
        z = float(_field(ln, 46, 54) or "0")

        if resname not in {lig, "DUM", "WAT"} and resname not in other_mol and resname not in lipid_mol and resname not in ion_mol:
            recep_block.append((name, resname, resid, chain, x, y, z))
            recep_last_resid = max(recep_last_resid, resid)
        elif resname == lig:
            lig_block.append((name, resname, resid, chain, x, y, z))
        else:
            # waters, ions, lipids, and other molecules
            oth_block.append((name, resname, resid, chain, x, y, z))

    lig_resid = recep_last_resid + dum_atom_count
    # Build L1/L2/L3 fully-qualified masks if we have names
    def _lig_mask(atom_name: str | None) -> str | None:
        return f":{lig_resid}@{atom_name}" if atom_name else None

    L1 = _lig_mask(l1_name)
    L2 = _lig_mask(l2_name)
    L3 = _lig_mask(l3_name)

    # P1/P2/P3 come from (build_dir / "protein_anchors.txt")
    with build_dir.joinpath("protein_anchors.txt").open() as f:
        P1 = f.readline().strip()
        P2 = f.readline().strip()
        P3 = f.readline().strip()
    if not all([P1, P2, P3, L1, L2, L3]):
        raise ValueError(f"[simprep] Missing anchors; got P1={P1}, P2={P2}, P3={P3}, L1={L1}, L2={L2}, L3={L3}")

    # -------- write build.pdb --------
    out_build = window_dir / "build.pdb"
    with out_build.open("w") as fout:
        serial = 1

        # Dummy block
        for (name, resname, resid, chain), (x, y, z) in zip(atom_dum, coords_dum):
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid, x, y, z) + "\n")
            serial += 1
        fout.write("TER\n")

        # Receptor (resids shifted by + dum_atom_count)
        prev_chain = None
        for name, resname, resid, chain, x, y, z in recep_block:
            if prev_chain is not None and chain != prev_chain and resname not in other_mol and resname != "WAT":
                fout.write("TER\n")
            prev_chain = chain
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid + dum_atom_count, x, y, z) + "\n")
            serial += 1
        fout.write("TER\n")

        # Ligand (shifted)
        for name, resname, resid, chain, x, y, z in lig_block:
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid + dum_atom_count, x, y, z) + "\n")
            serial += 1
        fout.write("TER\n")

        # Others (waters, ions, lipids, others)
        prev_resid = None
        for name, resname, resid, chain, x, y, z in oth_block:
            if prev_resid is not None and int(str(resid).strip()) != prev_resid:
                fout.write("TER\n")
            prev_resid = int(str(resid).strip())
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid + dum_atom_count, x, y, z) + "\n")
            serial += 1
        fout.write("TER\nEND\n")

    # -------- write build-dry.pdb (up to first water) --------
    out_dry = window_dir / "build-dry.pdb"
    with out_build.open() as fin, out_dry.open("w") as fout:
        for ln in fin:
            if len(ln) >= 20 and ln[17:20] == "WAT":
                break
            fout.write(ln)

    # -------- prepend REMARK A line to equil-<lig>.pdb --------
    # first protein residue id (from ctx or infer)
    first_res = getattr(ctx.sim, "first_res", 1)
    recep_last = recep_last_resid
    remark = f"REMARK A  {P1 or 'NA':6s}  {P2 or 'NA':6s}  {P3 or 'NA':6s}  {L1 or 'NA':6s}  {L2 or 'NA':6s}  {L3 or 'NA':6s}  {first_res:4d}  {recep_last:4d}\n"
    orig = dst_equil.read_text()
    dst_equil.write_text(remark + "".join(orig.splitlines(True)[1:]))

    anchors = Anchors(
        P1=P1, P2=P2, P3=P3,
        L1=L1, L2=L2, L3=L3,
        lig_res=str(lig_resid),
    )
    save_anchors(ctx.working_dir, anchors)
    logger.debug("[simprep] wrote build files in {}", window_dir)
    return


def copy_simulation_dir(source: Path, dest: Path, sim: SimulationConfig) -> None:
    """
    Symlink/copy only the needed files from first window dir.
    """
    needed = [
        "full.prmtop", "full.inpcrd", "full.pdb",
        "vac.pdb", "vac_ligand.pdb",
        "vac.prmtop", "vac_ligand.prmtop",
        # You may want to resolve ligand name dynamically; placeholder here:
        "fe-lig.pdb", "lig.mol2",
        "disang.rest", "cv.in",
    ]
    if getattr(sim, "hmr", "no") == "yes":
        needed.append("full.hmr.prmtop")

    dest.mkdir(parents=True, exist_ok=True)

    for name in needed:
        src = source / name
        if not src.exists():
            continue
        dst = dest / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    
    anchors = load_anchors(source)
    (dest / "anchors.json").write_text((source / "anchors.json").read_text())