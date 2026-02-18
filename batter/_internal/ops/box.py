from __future__ import annotations

import os
import glob
import json
import shutil
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger
import parmed as pmd

from batter.utils import run_with_log, tleap
from batter.utils.builder_utils import get_buffer_z
from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_create_box
from batter._internal.ops.helpers import run_parmed_hmr_if_enabled


def _merge_consecutive(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Merge sorted indices into inclusive consecutive ranges.

    Parameters
    ----------
    indices : Sequence[int]
        Integer indices. Duplicates are allowed but will be removed.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) inclusive ranges. If start == end, it's a singleton.
    """
    uniq = sorted(set(indices))
    if not uniq:
        return []

    ranges: List[Tuple[int, int]] = []
    start = prev = uniq[0]
    for x in uniq[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = prev = x
    ranges.append((start, prev))
    return ranges


def _ranges_to_str(ranges: Sequence[Tuple[int, int]]) -> str:
    """Convert ranges to selection segments like '5-8,10,12-14'."""
    parts: List[str] = []
    for a, b in ranges:
        parts.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(parts)


def indices_to_selection(
    include: Iterable[int],
    exclude: Iterable[int] = (),
    *,
    prefix: str = "@",
    negate_op: str = "!",
    and_op: str = "&",
) -> str:
    """Build a selection string from include/exclude indices with merged ranges.

    Parameters
    ----------
    include : Iterable[int]
        Indices to include.
    exclude : Iterable[int], optional
        Indices to exclude. Indices not present in `include` are ignored.
    prefix : str, optional
        Prefix for the include expression (default '@', e.g., AMBER-style atom selection).
    negate_op : str, optional
        Negation operator (default '!').
    and_op : str, optional
        Conjunction operator (default '&').

    Returns
    -------
    str
        Selection string, e.g. '@1-10 & ! (@3-4,7)'.

    Raises
    ------
    ValueError
        If `include` is empty.
    """
    inc = sorted(set(include))
    exc = sorted(set(exclude))
    if not inc:
        raise ValueError("include must be non-empty")

    inc_ranges = _merge_consecutive(inc)
    inc_str = _ranges_to_str(inc_ranges)

    # Only exclude indices that are actually in include
    inc_set = set(inc)
    exc_in_inc = [i for i in set(exc) if i in inc_set]
    if not exc_in_inc:
        return f"{prefix}{inc_str}"

    exc_ranges = _merge_consecutive(exc_in_inc)
    exc_str = _ranges_to_str(exc_ranges)
    return f"{prefix}{inc_str} {and_op} {negate_op} ({prefix}{exc_str})"


def _cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_res_blocks(selection, out_pdb: Path) -> None:
    lines = []
    if len(selection.residues) != 0:
        prev = selection.residues.resids[0]
        for res in selection.residues:
            if res.resid != prev:
                lines.append("TER\n")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            res.atoms.write(tmp.name)
            tmp.close()
            with open(tmp.name) as f:
                lines += [ln for ln in f if ln.startswith("ATOM")]
            prev = res.resid
    out_pdb.write_text("".join(lines))


def _ligand_charge_from_metadata(meta_path: Path) -> int | None:
    """Return the integer ligand charge recorded during parametrization."""
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        charge_val = data.get("ligand_charge")
        if charge_val is None:
            return None
        return int(round(float(charge_val)))
    except Exception as exc:
        logger.debug(f"Failed to read ligand charge from {meta_path}: {exc}")
        return None

@register_create_box("z")
def create_box_z(ctx: BuildContext) -> None:
    """
    Create the solvated box for the given component and window.
    """
    work = ctx.working_dir
    comp = ctx.comp
    param_dir = work.parent.parent / "params"
    sim = ctx.sim
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    membrane_builder = sim.membrane_simulation
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol

    ligand = ctx.ligand
    mol = ctx.residue_name

    for attr in ("buffer_x", "buffer_y", "buffer_z"):
        if not hasattr(sim, attr):
            raise AttributeError(
                f"SimulationConfig missing '{attr}'. Please specify this buffer in the YAML."
            )
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if (not membrane_builder) and (buffer_x < 5 or buffer_y < 5 or buffer_z < 5):
        raise ValueError("For water systems, buffer_x/y/z must be ≥ 5 Å.")

    if membrane_builder:
        targeted_buffer_z = max([float(sim.buffer_z), 25.0])
        buffer_x = 0.0
        buffer_y = 0.0
    else:
        # for non-membrane systems,
        # reduce the buffer by existing solvation shell
        solv_shell = sim.solv_shell
        buffer_x = max(0.0, buffer_x - solv_shell)
        buffer_y = max(0.0, buffer_y - solv_shell)

    sdr_dist, abs_z, buffer_z_left = map(float, open(window_dir / "sdr_info.txt").read().split())

    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut)

    if not hasattr(sim, "dec_method"):
        raise AttributeError("SimulationConfig missing 'dec_method'.")
    dec_method = str(sim.dec_method)

    # ---- copy FF artifacts (resolve ff/ relative to window_dir: ../../param) ----
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "json"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / src.name)

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / f"vac_ligand.{ext}")

    shutil.copy2(build_dir / f"{ligand}.pdb", window_dir / f"{ligand}.pdb")

    # other_mol
    if other_mol:
        raise NotImplementedError("Other molecules not supported now.")

    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    # --- tleap solvate pre ---
    tleap_solv_pre = window_dir / "tleap_solvate_pre.in"
    _cp(window_dir / "tleap.in", tleap_solv_pre)
    with tleap_solv_pre.open("a") as f:
        f.write("# Load the necessary parameters\n")
        for om in other_mol:
            f.write(f"loadamberparams {om.lower()}.frcmod\n")
            f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        f.write("model = loadpdb build.pdb\n\n")
        f.write(
            f"solvatebox model {water_box} {{ {buffer_x} {buffer_y} {buffer_z_left} }} 1\n\n"
        )
        f.write("desc model\n")
        f.write("savepdb model full_pre.pdb\n")
        f.write("quit\n")
    run_with_log(
        f"{tleap} -s -f {tleap_solv_pre.name} > tleap_solvate_pre.log",
        working_dir=window_dir,
    )

    # Count waters in build.pdb
    num_waters = sum(
        1 for ln in (window_dir / "build.pdb").read_text().splitlines() if "WAT" in ln
    )

    # pdb4amber
    run_with_log("pdb4amber -i build.pdb -o build_amber.pdb -y", working_dir=window_dir)
    renum_df = pd.read_csv(
        window_dir / "build_amber_renum.txt",
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df["old_resname"] = renum_df["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    revised_resids = []
    resid_counter = 1
    prev_resid = 0
    for _, row in renum_df.iterrows():
        if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
            revised_resids.append(resid_counter)
            resid_counter += 1
        else:
            revised_resids.append(resid_counter - 1)
        prev_resid = row["old_resid"]

    # MDAnalysis universes
    u = mda.Universe(str(window_dir / "full_pre.pdb"))
    final_system = u.atoms
    system_dimensions = u.dimensions[:3]

    if membrane_builder:
        u_ref = mda.Universe(str(window_dir / "equil-reference.pdb"))
        u.dimensions[0] = u_ref.dimensions[0]
        u.dimensions[1] = u_ref.dimensions[1]
        u.dimensions[2] = u.dimensions[2] - 3
        u.atoms.positions[:, 2] -= 3

        membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
        memb_z_max = membrane_region.select_atoms("type P").positions[:, 2].max() - 10
        memb_z_min = membrane_region.select_atoms("type P").positions[:, 2].min() + 10
        water_in_mem = u.select_atoms(
            f"byres (resname WAT and prop z > {memb_z_min} and prop z < {memb_z_max})"
        )
        final_system = final_system - water_in_mem

    box_xy = [u.dimensions[0], u.dimensions[1]]
    water_around_prot = u.select_atoms("resname WAT").residues[:num_waters].atoms
    final_system = final_system | water_around_prot

    if membrane_builder:
        outside_wat = final_system.select_atoms(
            "byres (resname WAT and "
            f"((prop x > {box_xy[0]/2}) or (prop x < -{box_xy[0]/2}) or "
            f"(prop y > {box_xy[1]/2}) or (prop y < -{box_xy[1]/2})))"
        )
        final_system = final_system - outside_wat

    system_dimensions[2] = abs_z

    # renumber residues
    revised_resids = np.array(revised_resids)
    total_residues = final_system.residues.n_residues
    final_resids = np.zeros(total_residues, dtype=int)
    final_resids[: len(revised_resids)] = revised_resids
    next_resnum = revised_resids[-1] + 1
    final_resids[len(revised_resids) :] = np.arange(
        next_resnum, total_residues - len(revised_resids) + next_resnum
    )
    final_system.residues.resids = final_resids

    # partitions
    final_system_dum = final_system.select_atoms("resname DUM")
    final_system_dum[0].position = final_system.select_atoms("protein and name CA N C O").center_of_mass()
    if comp == 'z':
        final_system_dum[1].position = final_system.select_atoms(f"resname {mol}").residues[1].atoms.center_of_mass()
    final_system_prot = final_system.select_atoms("protein")
    final_system_others = final_system - final_system_prot - final_system_dum
    final_system_ligs = final_system.select_atoms(f"resname {mol}")
    final_system_other_mol = (
        final_system_others.select_atoms("not resname WAT") - final_system_ligs
    )
    final_system_water = final_system_others.select_atoms("resname WAT")
    final_system_water_notaround = final_system.select_atoms(
        "byres (resname WAT and not (around 6 protein))"
    )
    final_system_water_around = final_system_water - final_system_water_notaround

    # write parts
    _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")

    # set chainIDs using renum_df and write protein by chains
    for residue in u.select_atoms("protein").residues:
        resid_str = residue.resid
        resid_resname = (
            "HIS"
            if residue.resname in ["HIS", "HIE", "HIP", "HID"]
            else residue.resname
        )
        residue.atoms.chainIDs = renum_df.query(
            "old_resid == @resid_str and old_resname == @resid_resname"
        ).old_chain.values[0]
    prot_lines = []
    for chain_name in np.unique(final_system_prot.atoms.chainIDs):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        final_system.select_atoms(f"chainID {chain_name}").write(tmp.name)
        tmp.close()
        with open(tmp.name) as f:
            prot_lines += [ln for ln in f if ln.startswith("ATOM")]
        prot_lines.append("TER\n")
    (window_dir / "solvate_pre_prot.pdb").write_text("".join(prot_lines))

    _write_res_blocks(final_system_ligs, window_dir / "solvate_pre_ligands.pdb")

    other_lines_exist = len(final_system_other_mol.residues) != 0
    if other_lines_exist:
        _write_res_blocks(final_system_other_mol, window_dir / "solvate_pre_others.pdb")

    outside_wat_exist = len(final_system_water_notaround.residues) != 0
    if outside_wat_exist:
        _write_res_blocks(
            final_system_water_notaround, window_dir / "solvate_pre_outside_wat.pdb"
        )

    around_wat_exist = len(final_system_water_around.residues) != 0
    if around_wat_exist:
        _write_res_blocks(
            final_system_water_around, window_dir / "solvate_pre_around_water.pdb"
        )

    # --- tleap parts (all with working_dir=window_dir) ---

    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # prot
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_prot.in")
    with (window_dir / "tleap_solvate_prot.in").open("a") as f:
        f.write("prot = loadpdb solvate_pre_prot.pdb\n\n")
        f.write(
            f"set prot box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb prot solvate_prot.pdb\n")
        f.write("saveamberparm prot solvate_prot.prmtop solvate_prot.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_prot.in > tleap_prot.log", working_dir=window_dir
    )

    # ligands
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_ligands.in")
    with (window_dir / "tleap_solvate_ligands.in").open("a") as f:
        f.write("# Load the necessary parameters\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if comp == "x":
            f.write(f"loadamberparams {mol}.frcmod\n")
            f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write("ligands = loadpdb solvate_pre_ligands.pdb\n\n")
        f.write(
            f"set ligands box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb ligands solvate_ligands.pdb\n")
        f.write(
            "saveamberparm ligands solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_ligands.in > tleap_ligands.log",
        working_dir=window_dir,
    )

    # others
    if other_lines_exist:
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
        with (window_dir / "tleap_solvate_others.in").open("a") as f:
            for om in other_mol:
                f.write(f"loadamberparams {om.lower()}.frcmod\n")
                f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("others = loadpdb solvate_pre_others.pdb\n\n")
            f.write(
                f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb others solvate_others.pdb\n")
            f.write(
                "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
            working_dir=window_dir,
        )

    # charge accounting
    def _sum_unit_charge_from_log(logfile: Path) -> Tuple[int, int]:
        neu_cat = neu_ani = 0
        if not logfile.exists():
            return 0, 0
        for line in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in line:
                q = float(line.split()[6].strip("'\",.:;#()]["))
                if q < 0:
                    neu_cat += round(float(re.sub(r"[+-]", "", str(q))))
                elif q > 0:
                    neu_ani += round(float(re.sub(r"[+-]", "", str(q))))
        return neu_cat, neu_ani

    neu_cat, neu_ani = _sum_unit_charge_from_log(window_dir / "tleap_prot.log")
    if (window_dir / "tleap_others.log").exists():
        nc2, na2 = _sum_unit_charge_from_log(window_dir / "tleap_others.log")
        neu_cat += nc2
        neu_ani += na2
    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    lig_cat = max(0, -lig_charge)
    lig_ani = max(0, lig_charge)

    charge_neut = neu_cat - neu_ani + lig_cat - lig_ani
    neu_cat = max(0, charge_neut)
    neu_ani = max(0, -charge_neut)

    box_volume = system_dimensions[0] * system_dimensions[1] * system_dimensions[2]
    num_ions = round(ion_def[2] * 6.02e23 * box_volume * 1e-27)
    # put a minimum of 5 ions
    num_ions = max(5, num_ions)
    if membrane_builder:
        num_ions //= 2
    num_cat = num_ions
    num_ani = num_ions - neu_cat + neu_ani
    if num_ani < 0:
        num_cat = neu_cat
        num_ions = neu_cat
        num_ani = 0

    # outside water — ionization
    if (window_dir / "solvate_pre_outside_wat.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_outside_wat.in")
        with (window_dir / "tleap_solvate_outside_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("outside_wat = loadpdb solvate_pre_outside_wat.pdb\n\n")
            if neut == "no":
                f.write(f"addionsrand outside_wat {ion_def[0]} {num_cat}\n")
                f.write(f"addionsrand outside_wat {ion_def[1]} {num_ani}\n")
            elif neut == "yes":
                if neu_cat:
                    f.write(f"addionsrand outside_wat {ion_def[0]} {neu_cat}\n")
                if neu_ani:
                    f.write(f"addionsrand outside_wat {ion_def[1]} {neu_ani}\n")
            f.write(
                f"set outside_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb outside_wat solvate_outside_wat.pdb\n")
            f.write(
                "saveamberparm outside_wat solvate_outside_wat.prmtop solvate_outside_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_outside_wat.in > tleap_outside_wat.log",
            working_dir=window_dir,
        )

    # around water
    if (window_dir / "solvate_pre_around_water.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_around_wat.in")
        with (window_dir / "tleap_solvate_around_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("around_wat = loadpdb solvate_pre_around_water.pdb\n\n")
            f.write(
                f"set around_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb around_wat solvate_around_wat.pdb\n")
            f.write(
                "saveamberparm around_wat solvate_around_wat.prmtop solvate_around_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_around_wat.in > tleap_around_wat.log",
            working_dir=window_dir,
        )

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    prot_p = pmd.load_file(
        str(window_dir / "solvate_prot.prmtop"), str(window_dir / "solvate_prot.inpcrd")
    )
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p_1.residues[0].name = mol
    ligand_p_1.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))

    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    if dec_method == "dd" or comp == "q":
        ligands_p = ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["z", "o", "s", "v"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["e"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1 + ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    else:
        raise ValueError(
            f"Unsupported comp={comp} with dec={dec_method} for custom ligand params."
        )

    combined = dum_p + prot_p + ligands_p
    vac = dum_p + prot_p + ligands_p
    other_parts = []

    if (window_dir / "solvate_others.prmtop").exists():
        others_p = pmd.load_file(
            str(window_dir / "solvate_others.prmtop"),
            str(window_dir / "solvate_others.inpcrd"),
        )
        combined += others_p
        other_parts.append(others_p)
    if (window_dir / "solvate_outside_wat.prmtop").exists():
        out_wat_pmd =  pmd.load_file(
            str(window_dir / "solvate_outside_wat.prmtop"),
            str(window_dir / "solvate_outside_wat.inpcrd"),
        )
        combined += out_wat_pmd
        other_parts.append(out_wat_pmd)
    if (window_dir / "solvate_around_wat.prmtop").exists():
        around_wat_pmd = pmd.load_file(
            str(window_dir / "solvate_around_wat.prmtop"),
            str(window_dir / "solvate_around_wat.inpcrd"),
        )
        combined += around_wat_pmd
        other_parts.append(around_wat_pmd)

    if len(other_parts) == 1:
        other_parts_pmd = other_parts[0]
    elif len(other_parts) == 2:
        other_parts_pmd = other_parts[0] + other_parts[1]
    elif len(other_parts) == 3:
        other_parts_pmd = other_parts[0] + other_parts[1] + other_parts[2]
    else:
        raise ValueError(f"Unsupported number of other_parts: {len(other_parts)}")

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    other_parts_pmd.save(str(window_dir / "other_parts.prmtop"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.inpcrd"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))
    u_vac = mda.Universe(str(window_dir / "vac.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    u_full.select_atoms("protein").residues.resids = renum_df2["old_resid"].values
    u_vac.select_atoms("protein").residues.resids = renum_df2["old_resid"].values

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))
    u_vac.atoms.write(str(window_dir / "vac_orig.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    return


@register_create_box("x")
def create_box_x(ctx: BuildContext) -> None:
    """
    Create the box for RBFE (x-component) ligand-pair systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir

    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    res_ref = extra.get("residue_ref") or ctx.residue_name
    res_alt = extra.get("residue_alt")

    if not res_alt:
        raise ValueError(
            "RBFE component 'x' requires residue_alt in BuildContext.extra."
        )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        param_dir = work.parent.parent / "params"
        src = param_dir / f"{res_ref}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")
        param_dir = work.parent.parent.parent / lig_alt / "params"
        src = param_dir / f"{res_alt}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")

    membrane_builder = sim.membrane_simulation
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    
    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    water_model = str(sim.water_model).upper()

    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    if water_model != "TIP3PF":
        water_line = f"source leaprc.water.{water_model.lower()}\n\n"
    else:
        water_line = "source leaprc.water.fb3\n\n"


    # combine with ParmEd
    vac_p = pmd.load_file(
        str(window_dir / "ref_vac.prmtop"), str(window_dir / "ref_vac.pdb")
    )
    other_part_p = pmd.load_file(
        str(window_dir / "other_parts.prmtop"),
        str(window_dir / "other_parts.pdb"),
    )
    alter_ligands_p = pmd.load_file(
        str(window_dir / "alter_ligand.prmtop"),
        str(window_dir / "alter_ligand_aligned.pdb"),
    )

    combined = vac_p + alter_ligands_p + other_part_p

    # build the ion prmtop if exists
    if os.path.exists(window_dir / "ions.pdb"):
        tleap_ion_txt = (window_dir / "tleap.in").read_text().splitlines()
        tleap_ion_txt += [
            "# ion topology",
            water_line,
            f"ions = loadpdb ions.pdb",
            "saveamberparm ions ions.prmtop ions.inpcrd",
            "quit",
        ]
        _write(window_dir / "tleap_ions.in", "\n".join(tleap_ion_txt) + "\n")
        run_with_log(
            f"{tleap} -s -f tleap_ions.in > tleap_ions.log", working_dir=window_dir
        )
        ion_p = pmd.load_file(
        str(window_dir / "ions.prmtop"),
        str(window_dir / "ions.inpcrd"),
        )
        combined += ion_p

    vac = vac_p + alter_ligands_p

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    u_full.select_atoms("protein").residues.resids = renum_df2["old_resid"].values

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)

    # get mapping file

    kartograf_mapping = json.load(open(window_dir / "kartograf.json"))
    ref_site = u_full.select_atoms(f"resname {res_ref}").residues[0]
    ref_solvent = u_full.select_atoms(f"resname {res_ref}").residues[1]
    alt_site = u_full.select_atoms(f"resname {res_alt}").residues[0]
    alt_solvent = u_full.select_atoms(f"resname {res_alt}").residues[1]

    # select cc parts
    ref_index_list = [int(i) for i in kartograf_mapping.keys()]
    alt_index_list = [int(i) for i in kartograf_mapping.values()]
    cc_indices_t0 = (
        np.concatenate(
            (
                ref_site.atoms[ref_index_list].indices,
                alt_solvent.atoms[alt_index_list].indices,
            )
        )
        + 1
    )
    cc_indices_t1 = (
        np.concatenate(
            (
                ref_solvent.atoms[ref_index_list].indices,
                alt_site.atoms[alt_index_list].indices,
            )
        )
        + 1
    )
    all_indices_t0 = (
        np.concatenate((ref_site.atoms.indices, alt_solvent.atoms.indices)) + 1
    )
    all_indices_t1 = (
        np.concatenate((ref_solvent.atoms.indices, alt_site.atoms.indices)) + 1
    )

    dict_sc_mask = {
        "scmk1": indices_to_selection(all_indices_t0, cc_indices_t0),
        "scmk2": indices_to_selection(all_indices_t1, cc_indices_t1),
    }
    logger.debug(f"scmk1: {dict_sc_mask['scmk1']}")
    logger.debug(f"scmk2: {dict_sc_mask['scmk2']}")

    with open(window_dir / "scmask.json", "w") as f:
        json.dump(dict_sc_mask, f)

    return


@register_create_box("y")
def create_box_y(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (solvation FE) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if buffer_x < 15 or buffer_y < 15 or buffer_z < 15:
        raise ValueError(f"For water systems, buffer_x/y/z must be ≥ 15 Å; got {buffer_x}/{buffer_y}/{buffer_z}.")
    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def
    if len(ion_def) < 3:
        raise ValueError("`ion_def` must contain [cation, anion, concentration].")

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut).lower()

    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    build_pdb = window_dir / "build.pdb"
    if not build_pdb.exists():
        fallback = build_dir / "build.pdb"
        if fallback.exists():
            _cp(fallback, build_pdb)
        else:
            raise FileNotFoundError(
                f"[create_box_y] build.pdb missing in {window_dir} (fallback: {fallback})."
            )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_y] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # --- determine water box keyword ---
    if water_model == "TIP3PF":
        water_box = "FB3BOX"  # leaprc.water.fb3
        water_leaprc = "leaprc.water.fb3"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
        water_leaprc = "leaprc.water.spce"
    else:
        water_box = f"{water_model}BOX"
        water_leaprc = f"leaprc.water.{water_model.lower()}"

    # --- read ligand net charge from tleap log (unperturbed unit charge line) ---
    def _unit_charge_from_log(logfile: Path) -> int:
        if not logfile.exists():
            return 0
        q = 0.0
        for ln in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in ln:
                try:
                    q = float(ln.split()[6].strip("'\",.:;#()[]"))
                except Exception:
                    pass
        return int(round(q))

    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    # put a minimum of 5 ions
    box_volume_A3 = 2 * buffer_x * 2 * buffer_y * 2 * buffer_z
    num_ions = max(
        5,
        round(ion_def[2] * 6.02e23 * box_volume_A3 * 1e-27),
    )

    add_neu_cat = max(0, -lig_charge)
    add_neu_ani = max(0, lig_charge)

    tleap_solv_lines = (window_dir / "tleap.in").read_text().splitlines()
    tleap_solv_lines += [
        "# ligand-only solvation",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f"source {water_leaprc}",
        f'set {{{mol}.1}} name "{mol}"',
        f"model = loadpdb {build_pdb.name}",
        "",
        f"solvatebox model {water_box} {{ {buffer_x:.3f} {buffer_y:.3f} {buffer_z:.3f} }} 1",
        "",
        "# ions",
    ]
    if neut == "no":
        if num_ions > 0 or add_neu_cat > 0 or add_neu_ani > 0:
            tleap_solv_lines += [
                f"addionsrand model {ion_def[0]} {num_ions + add_neu_cat}",
                f"addionsrand model {ion_def[1]} {num_ions + add_neu_ani}",
            ]
    else:
        if add_neu_cat:
            tleap_solv_lines.append(f"addionsrand model {ion_def[0]} {add_neu_cat}")
        if add_neu_ani:
            tleap_solv_lines.append(f"addionsrand model {ion_def[1]} {add_neu_ani}")

    tleap_solv_lines += [
        "desc model",
        "savepdb model full_pre.pdb",
        "quit",
        "",
    ]
    _write(window_dir / "tleap_solvate.in", "\n".join(tleap_solv_lines))
    run_with_log(
        f"{tleap} -s -f tleap_solvate.in > tleap_solvate.log", working_dir=window_dir
    )

    # --- process full_pre.pdb into final full.{prmtop,inpcrd,pdb} ---
    #
    u = mda.Universe(str(window_dir / "full_pre.pdb"))
    final_system = u.atoms
    system_dimensions = u.dimensions[:3]
    final_system_dum = final_system.select_atoms("resname DUM")
    final_system_lig = final_system.select_atoms(f"resname {mol}")
    final_system_others = final_system - final_system_dum - final_system_lig

    _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")
    _write_res_blocks(final_system_lig, window_dir / "solvate_pre_lig.pdb")
    _write_res_blocks(final_system_others, window_dir / "solvate_pre_others.pdb")

    # tleap parts
    # dum
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # ligand
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_lig.in")
    with (window_dir / "tleap_solvate_lig.in").open("a") as f:
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        f.write("lig = loadpdb solvate_pre_lig.pdb\n\n")
        f.write(
            f"set lig box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb lig solvate_ligands.pdb\n")
        f.write(
            "saveamberparm lig solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_lig.in > tleap_lig.log", working_dir=window_dir
    )

    # others
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
    with (window_dir / "tleap_solvate_others.in").open("a") as f:
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        f.write("others = loadpdb solvate_pre_others.pdb\n\n")
        f.write(
            f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb others solvate_others.pdb\n")
        f.write(
            "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
        working_dir=window_dir,
    )

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    ligand_p = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p.residues[0].name = mol
    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    ligand_p.coordinates = lig_inp
    ligand_p.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)

    others = pmd.load_file(
        str(window_dir / "solvate_others.prmtop"),
        str(window_dir / "solvate_others.inpcrd"),
    )
    combined = dum_p + ligand_p + others
    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac = dum_p + ligand_p
    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    return


@register_create_box("m")
def create_box_m(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (vacuum) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    
    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_m] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        # set box to 40
        "set lig box {40.000000 40.000000 40.000000}",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # copy ligand_p to vac.prmtop
    ligand_p_file = window_dir / f"{mol}.prmtop"
    _cp(ligand_p_file, window_dir / "vac.prmtop")

    # copy vac to full
    _cp(window_dir / "vac.pdb", window_dir / "full.pdb")
    _cp(window_dir / "vac.prmtop", window_dir / "full.prmtop")
    _cp(window_dir / "vac.inpcrd", window_dir / "full.inpcrd")
    
    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    return
