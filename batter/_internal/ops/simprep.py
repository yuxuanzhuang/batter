from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, List, Tuple, Optional, Set
import shutil
import re

from loguru import logger

from batter._internal.builders.fe_registry import register_create_simulation
from batter._internal.builders.interfaces import BuildContext
from batter._internal.ops.helpers import load_anchors, save_anchors, Anchors, get_sdr_dist

from batter.utils import run_with_log
from batter.config.simulation import SimulationConfig

ION_NAMES = {"Na+", "K+", "Cl-", "NA", "CL", "K"}  # NA/CL appear in some pdbs too


# ---------------------- small utils ----------------------
def _rel_symlink(target: Path, link_path: Path) -> None:
    """Create/replace a relative symlink at link_path → target."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)

def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        logger.warning(f"[simprep] expected file not found: {src} (continuing)")

def _read_nonblank_lines(p: Path) -> List[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]

def _is_atom_line(line: str) -> bool:
    tag = line[0:6].strip()
    return tag == "ATOM" or tag == "HETATM"

def _field(line: str, start: int, end: int) -> str:
    # 0-based, end exclusive
    return line[start:end].strip()

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
    res3  = f"{resname:>3s}"[:3]
    chain1 = (chain or " ")[:1]
    resid4 = _safe_resid(resid)
    return (
        f"ATOM  {serial:5d} {name4} {res3} {chain1}{resid4:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{0.00:6.2f}{0.00:6.2f}"
    )


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
    offset_other_by_ligand: int = 1,
    start_off_set: int = 0,
    use_ter_markers: bool = False,
    ter_atoms: Optional[Set[int]] = None,
) -> None:
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
            name    = _field(dlines[1], 12, 16) or "DU"
            resname = _field(dlines[1], 17, 20) or "DUM"
            resid   = int(float(_field(dlines[1], 22, 26) or 1))
            chain   = _field(dlines[1], 21, 22) or "A"
            coords_dum.append((x, y, z))
            atom_dum.append((name, resname, resid, chain))
    if not coords_dum:
        coords_dum.append((0.0, 0.0, 0.0))
        atom_dum.append(("DU", "DUM", 1, "Z"))

    dum_count = len(coords_dum)

    # ---- categorize atoms
    om = set(other_mol or []); lm = set(lipid_mol or []); im = set(ion_mol or [])
    recep_block: List[Tuple[str, str, int, str, float, float, float]] = []
    lig_block:   List[Tuple[str, str, int, str, float, float, float]] = []
    oth_block:   List[Tuple[str, str, int, str, float, float, float]] = []
    recep_last_resid = 0

    for ln in lines:
        if not _is_atom_line(ln):
            continue
        resname = _field(ln, 17, 21)
        chain   = _field(ln, 21, 22)
        resid   = int(_field(ln, 22, 26) or 0)
        name    = _field(ln, 12, 16)
        x = float(_field(ln, 30, 38) or 0.0)
        y = float(_field(ln, 38, 46) or 0.0)
        z = float(_field(ln, 46, 54) or 0.0)

        if resname not in {lig, "DUM", "WAT"} and resname not in om and resname not in lm and resname not in im:
            recep_block.append((name, resname, resid - start_off_set, chain, x, y, z))
            recep_last_resid = max(recep_last_resid, resid)
        elif resname == lig:
            lig_block.append((name, resname, resid - start_off_set, chain, x, y, z))
        else:
            oth_block.append((name, resname, resid - start_off_set, chain, x, y, z))

    lig_resid = recep_last_resid + dum_count

    # ---- write build.pdb
    out_build = window_dir / "build.pdb"
    out_build.parent.mkdir(parents=True, exist_ok=True)
    with out_build.open("w") as fout:
        serial = 1

        # Dummy blocks (all)
        for (name, resname, resid, chain), (x, y, z) in zip(atom_dum, coords_dum):
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid, x, y, z) + "\n")
            fout.write("TER\n")
            serial += 1
        fout.write("TER\n")

        # Receptor (+dum_count)
        prev_chain = None
        for name, resname, resid, chain, x, y, z in recep_block:
            if prev_chain is not None and chain != prev_chain and resname not in om and resname != "WAT":
                fout.write("TER\n")
            prev_chain = chain
            fout.write(_fmt_atom_line(serial, name, resname, chain, resid + dum_count, x, y, z) + "\n")
            serial += 1
            if use_ter_markers:
                leg_idx = (resid + 2 - dum_count)  # legacy mapping heuristic
                if leg_idx in (ter_atoms or set()):
                    fout.write("TER\n")
        fout.write("TER\n")

        # Ligand at lig_resid
        for name, resname, resid, chain, x, y, z in lig_block:
            fout.write(_fmt_atom_line(serial, name, resname, chain, lig_resid, x, y, z) + "\n")
            serial += 1
        fout.write("TER\n")

        # Optional shifted ligand copy (+sdr_dist along z) for z/v/o with SDR/EXCHANGE
        # extra_ligand_shift is a list of whether to shift the ligand or not
        for shift in extra_ligand_shift:
            shift_sdr_dist = sdr_dist if shift else 0.0
            for name, _, __, chain, x, y, z in lig_block:
                fout.write(_fmt_atom_line(serial, name, lig, chain, lig_resid + 1, x, y, z + float(shift_sdr_dist)) + "\n")
                serial += 1
            fout.write("TER\n")

        # Others (+dum_count plus optional ligand offset)
        last_resid = None
        for name, resname, resid, chain, x, y, z in oth_block:
            out_resid = resid + dum_count + (offset_other_by_ligand - 1)
            if last_resid is not None and out_resid != last_resid:
                fout.write("TER\n")
            last_resid = out_resid
            fout.write(_fmt_atom_line(serial, name, resname, chain, out_resid, x, y, z) + "\n")
            serial += 1

        fout.write("TER\nEND\n")

    # ---- build-dry.pdb (until first water)
    out_dry = window_dir / "build-dry.pdb"
    with out_build.open() as fin, out_dry.open("w") as fout:
        for ln in fin:
            if len(ln) >= 20 and ln[17:20] == "WAT":
                break
            fout.write(ln)


# ---------------------- create_simulation_dir: EQUIL ----------------------
def _parse_recep_last_resid(pdb_path: Path, lig: str, other_mol: set, lipid_mol: set) -> int:
    """Find the last receptor residue id (not water/lig/other/lipid/ion)."""
    lines = _read_nonblank_lines(pdb_path)
    recep_last = 0
    for ln in lines:
        if not _is_atom_line(ln):
            continue
        resn = _field(ln, 17, 21)
        resid = int(_field(ln, 22, 26) or 0)
        if resn not in {lig, "DUM", "WAT"} and resn not in other_mol and resn not in lipid_mol and resn not in ION_NAMES:
            recep_last = max(recep_last, resid)
    return recep_last

def _read_protein_anchors(txt: Path) -> Tuple[str, str, str]:
    """protein_anchors.txt is expected to have 3 lines: P1, P2, P3."""
    if not txt.exists():
        raise FileNotFoundError(f"[simprep] Missing required protein anchors: {txt}")
    lines = _read_nonblank_lines(txt)
    if len(lines) < 3:
        raise ValueError(f"[simprep] protein_anchors.txt malformed: {txt}")
    return lines[0], lines[1], lines[2]

def _read_ligand_anchor_names(txt: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    build_dir = work / f"{comp}_build_files"
    window_dir = work / (f"{comp}-1" if comp != "q" else "")
    if str(window_dir) == "":
        window_dir = work
    window_dir.mkdir(parents=True, exist_ok=True)

    # sources
    src_equil = build_dir / f"equil-{mol}.pdb"
    src_lig_noh = build_dir / f"{mol}-noh.pdb"
    src_anchors = build_dir / f"anchors-{lig}.txt"
    src_prot_anchors = build_dir / "protein_anchors.txt"
    src_dum_prmtop = build_dir / "dum.prmtop"
    src_dum_inpcrd = build_dir / "dum.inpcrd"
    src_dum_frcmod = build_dir / "dum.frcmod"
    src_dum_mol2   = build_dir / "dum.mol2"

    # destinations
    dst_equil = window_dir / f"equil-{mol}.pdb"
    dst_ref   = window_dir / "equil-reference.pdb"
    dst_lig   = window_dir / f"{mol}.pdb"
    dst_anchors = window_dir / "anchors.txt"
    dst_dum_prmtop = window_dir / "dum.prmtop"
    dst_dum_inpcrd = window_dir / "dum.inpcrd"
    dst_dum_frcmod = window_dir / "dum.frcmod"
    dst_dum_mol2   = window_dir / "dum.mol2"

    # copy inputs
    for s, d in [
        (src_equil, dst_equil),
        (src_equil, dst_ref),
        (src_lig_noh, dst_lig),
        (src_anchors, dst_anchors),
        (src_dum_prmtop, dst_dum_prmtop),
        (src_dum_inpcrd, dst_dum_inpcrd),
        (src_dum_frcmod, dst_dum_frcmod),
        (src_dum_mol2,   dst_dum_mol2),
    ]:
        _copy_if_exists(s, d)

    if not dst_equil.exists():
        raise FileNotFoundError(f"[simprep] Missing required input {src_equil}")

    # anchors: protein + ligand atom names
    P1, P2, P3 = _read_protein_anchors(src_prot_anchors)
    l1_name, l2_name, l3_name = _read_ligand_anchor_names(src_anchors)

    # write build.pdb / build-dry.pdb
    write_build_from_aligned(
        lig=mol,
        window_dir=window_dir,
        build_dir=build_dir,
        aligned_pdb=dst_equil,
        other_mol=ctx.sim.other_mol,
        lipid_mol=ctx.sim.lipid_mol,
        ion_mol=ION_NAMES,
        extra_ligand_shift=[],
        sdr_dist=0.0,
        offset_other_by_ligand=1,
        start_off_set=0,
        use_ter_markers=False,
    )

    # compute residue numbers for REMARK and anchors.json
    recep_last = _parse_recep_last_resid(dst_equil, mol, set(ctx.sim.other_mol or []), set(ctx.sim.lipid_mol or []))
    lig_resid  = recep_last + 1
    L1 = _mask(lig_resid, l1_name)
    L2 = _mask(lig_resid, l2_name)
    L3 = _mask(lig_resid, l3_name)

    # prepend REMARK A to equil-<lig>.pdb
    first_res = int(getattr(ctx.sim, "first_res", 1))
    remark = f"REMARK A  {P1:6s}  {P2:6s}  {P3:6s}  {L1 or 'NA':6s}  {L2 or 'NA':6s}  {L3 or 'NA':6s}  {first_res:4d}  {recep_last:4d}\n"
    orig = dst_equil.read_text().splitlines(True)
    # keep original first line content after remark
    dst_equil.write_text(remark + "".join(orig[1:]))

    # persist anchors.json
    anchors = Anchors(P1=P1, P2=P2, P3=P3, L1=L1 or "", L2=L2 or "", L3=L3 or "", lig_res=str(lig_resid))
    save_anchors(ctx.working_dir, anchors)
    logger.debug("[simprep:equil] wrote build files in {}", window_dir)


# ---------------------- create_simulation_dir: Z ----------------------
@register_create_simulation('z')
def create_simulation_dir_z(ctx: BuildContext) -> None:
    """
    Create the initial simulation directory for component 'z' at window `-1`
    (the FE-equil bootstrap). No chdir; all paths handled explicitly.
    """
    comp = ctx.comp.lower()
    ligand = ctx.ligand
    mol = ctx.residue_name
    buffer_z    = ctx.sim.buffer_z

    # paths
    sys_root = ctx.system_root
    build_dir = ctx.working_dir / f"{comp}_build_files"
    amber_dir = ctx.working_dir / "amber_files"
    dest_dir  = ctx.working_dir / f"{comp}-1"   # z-1 is fe_equil folder
    ff_dir    = sys_root / "simulations" / ligand / "params"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # link amber files
    _rel_symlink(amber_dir, dest_dir / "amber_files")

    # bring build outputs
    for p in build_dir.glob("vac_ligand*"):
        _copy_if_exists(p, dest_dir / p.name)

    for s, d in [
        (build_dir / f"{mol}.pdb",                 dest_dir / f"{mol}.pdb"),
        (build_dir / f"fe-{mol}.pdb",              dest_dir / "build-ini.pdb"),
        (build_dir / f"fe-{mol}.pdb",              dest_dir / f"fe-{mol}.pdb"),
        (build_dir / f"anchors-{ligand}.txt",      dest_dir / f"anchors-{ligand}.txt"),
        (build_dir / "equil-reference.pdb",        dest_dir / "equil-reference.pdb"),
        (build_dir / "dum.inpcrd",                 dest_dir / "dum.inpcrd"),
        (build_dir / "dum.prmtop",                 dest_dir / "dum.prmtop"),
        (build_dir / "rec_file.pdb",               dest_dir / "rec_file.pdb"),
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
    
    if not buffer_z: buffer_z = 25
    sdr_dist = get_sdr_dist(build_dir / "complex.pdb",
        lig_resname=mol, buffer_z=buffer_z, extra_buffer=5)


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
        offset_other_by_ligand=2,   # v/o/z legacy offset
        start_off_set=1, # equil offset
        use_ter_markers=True,
        ter_atoms=set(ter_atoms),
    )

    logger.debug(f"[simprep:z] simulation directory created → {dest_dir}")


# ---------------------- window copier ----------------------
def copy_simulation_dir(source: Path, dest: Path, sim: SimulationConfig) -> None:
    """Symlink (using relative links) or copy only the needed files from the source simulation dir."""
    needed = [
        "full.prmtop", "full.inpcrd", "full.pdb",
        "vac.pdb", "vac_ligand.pdb",
        "vac.prmtop", "vac_ligand.prmtop",
        "fe-lig.pdb", "lig.mol2",
        "disang.rest", "cv.in",
    ]
    if getattr(sim, "hmr", None) == "yes":
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
            # Compute relative path from destination directory to source file
            rel_src = os.path.relpath(src, start=dest)
            dst.symlink_to(rel_src)
        except Exception as e:
            # Fall back to copying if symlink fails (e.g., on Windows or cross-device)
            shutil.copy2(src, dst)