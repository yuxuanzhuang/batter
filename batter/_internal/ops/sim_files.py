# sim_files.py — drop-in replacement
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger
import os
import json
import shutil


from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_sim_files
from batter._internal.ops.helpers import format_ranges


# ----------------------------- helpers ----------------------------- #


def _patch_restraint_block(
    text: str, new_mask_component: str, force_const: float
) -> str:
    """
    Idempotently enable ntr=1, merge/append restraintmask with new_mask_component,
    and set restraint_wt. If mask already present, replace the appended part.
    """
    lines = text.splitlines(True)
    out = []
    seen_mask = False
    for line in lines:
        if re.search(r"\bntr\s*=", line):
            line = re.sub(r"\bntr\s*=\s*\d+", "  ntr = 1", line)
        elif re.search(r"\brestraintmask\s*=", line):
            m = re.search(r'restraintmask\s*=\s*["\']([^"\']*)["\']', line)
            base_mask = m.group(1).strip() if m else ""
            # drop any previously appended “| ((:... ) & @CA)” chunk to stay idempotent
            base_mask = re.sub(
                r"\|\s*\(\s*\(:[^)]*\)\s*&\s*@CA\s*\)\s*", "", base_mask
            ).strip()
            mask = (
                f"({base_mask}) | ({new_mask_component})"
                if base_mask
                else new_mask_component
            )
            if len(mask) > 256:
                raise ValueError(f"Restraint mask too long (>256 chars): {mask}")
            line = f'  restraintmask = "{mask}",\n'
            seen_mask = True
        elif re.search(r"\brestraint_wt\s*=", line):
            line = re.sub(
                r"\brestraint_wt\s*=\s*[\d.]+", f" restraint_wt = {force_const}", line
            )
        out.append(line)

    if not seen_mask:
        out.append(f'\n  restraintmask = "{new_mask_component}",\n')
        out.append(f"  restraint_wt   = {force_const},\n")

    return "".join(out)


def _maybe_extra_mask(ctx: BuildContext, work: Path) -> tuple[Optional[str], float]:
    """
    Build '(:a-b,c-... ) & @CA' + force constant from ctx.extra.
    Returns (mask or None, force_const).
    """
    extra = ctx.extra or {}
    extra_sel = extra.get("extra_restraints")
    if not extra_sel:
        return None, 0.0

    if ctx.win != -1:
        # load from window -1 dir
        res_json = ctx.equil_dir / "extra_restraints.json"
        if not os.path.exists(res_json):
            raise FileNotFoundError(
                f"Missing extra_restraints.json in equil dir: {res_json}"
            )
        with open(res_json, "rt") as f:
            data = json.load(f)
        return data.get("mask"), data.get("force_const", 10.0)

    force_const = float(extra.get("extra_restraints_fc", 10.0))

    ref_pdb = work / "full.pdb"
    renum_txt1 = work / "build_files" / "protein_renum.txt"
    renum_txt2 = ctx.build_dir / "protein_renum.txt"
    renum_txt = renum_txt1 if renum_txt1.exists() else renum_txt2

    if not ref_pdb.exists():
        logger.warning(f"[extra_restraints] Missing reference PDB: {ref_pdb}; skip.")
        return None, force_const
    if not renum_txt.exists():
        logger.warning(
            f"[extra_restraints] Missing renumber map: {renum_txt1} / {renum_txt2}; skip."
        )
        return None, force_const

    u = mda.Universe(str(ref_pdb))
    sel = u.select_atoms(f"({extra_sel}) and name CA")
    if len(sel) == 0:
        logger.warning(
            f"[extra_restraints] 0 atoms selected for '({extra_sel}) and name CA'; skip."
        )
        return None, force_const

    ren = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    amber_resids = ren.loc[ren["old_resid"].isin(sel.residues.resids), "new_resid"]
    mask_ranges = format_ranges(amber_resids)
    if not mask_ranges:
        logger.warning("[extra_restraints] No mapped residues after renumber; skip.")
        return None, force_const

    mask = f"(:{mask_ranges}) & @CA"
    # save as json
    json.dump(
        {"mask": mask, "force_const": force_const},
        (work / "extra_restraints.json").open("wt"),
    )
    logger.debug(f"[extra_restraints] Mask: {mask} (wt={force_const})")
    return mask, force_const


# ------------------------- generic equil files ------------------------- #


def write_sim_files(ctx: BuildContext, *, infe: bool) -> None:
    """
    Writes minimization/NVT/NPT inputs and mdin-XX files based on
    release schedule; fills in temperature, restraint file names, etc.
    Also (optionally) injects extra CA restraints via ctx.extra['extra_restraints']
    **only** into mdin-XX files (NOT eqnpt.in).
    """
    sim = ctx.sim
    work = ctx.working_dir
    amber_dir = ctx.amber_dir

    temperature = sim.temperature
    mol = ctx.residue_name
    infe_flag = "1" if infe else "0"

    # disang anchor triplet (L1/L2/L3)
    with open(work / "disang.rest", "r") as f:
        parts = f.readline().split()
        L1 = parts[6].strip()
        L2 = parts[7].strip()
        L3 = parts[8].strip()

    def _sub_write(src: Path, dst: Path, repl: dict[str, str]) -> None:
        text = Path(src).read_text()
        for k, v in repl.items():
            text = text.replace(k, v)
        dst.write_text(text)

    # mini.in
    _sub_write(amber_dir / "mini.in", work / "mini.in", {"_lig_name_": mol})

    # eqnvt.in
    _sub_write(
        amber_dir / "eqnvt.in",
        work / "eqnvt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt0.in (membrane vs water variant)
    eqnpt0_src = amber_dir / (
        "eqnpt0.in" if sim.membrane_simulation else "eqnpt0-water.in"
    )
    _sub_write(
        eqnpt0_src,
        work / "eqnpt0.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt.in  (no extra restraints here)
    eqnpt_src = amber_dir / (
        "eqnpt.in" if sim.membrane_simulation else "eqnpt-water.in"
    )
    _sub_write(
        eqnpt_src,
        work / "eqnpt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # Additional equilibration inputs for disappear/appear stages
    _sub_write(
        amber_dir / "eqnpt-disappear.in",
        work / "eqnpt_disappear.in",
        {
            "_temperature_": f"{temperature}",
            "_lig_name_": mol,
            "_enable_infe_": infe_flag,
            "disang_file": "disang",
        },
    )
    _sub_write(
        amber_dir / "eqnpt-appear.in",
        work / "eqnpt_appear.in",
        {
            "_temperature_": f"{temperature}",
            "_lig_name_": mol,
            "_enable_infe_": infe_flag,
            "disang_file": "disang",
        },
    )

    # mdin-template for runtime chunking (total_steps is the total target)
    mdin_src = amber_dir / "mdin-equil"
    base_text = mdin_src.read_text()
    total_steps = int(getattr(sim, "eq_steps", 0) or 0)
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0 to write equilibration templates.")

    # compute extra mask once for equil (applied to template)
    extra_mask, extra_fc = _maybe_extra_mask(ctx, work)

    text = (
        base_text.replace("_temperature_", f"{temperature}")
        .replace("_enable_infe_", infe_flag)
        .replace("_lig_name_", mol)
        .replace("_num-steps_", f"{total_steps}")
        .replace("disang_file", "disang")
    )

    if extra_mask:
        try:
            text = _patch_restraint_block(text, extra_mask, extra_fc)
        except Exception as e:
            logger.warning(f"[extra_restraints] Could not patch mdin-template: {e}")

    # Prepend total eq steps marker for runtime scripts (comment starts with '!')
    text = f"! total_steps={total_steps}\n{text}"
    (work / "mdin-template").write_text(text)

    logger.debug(f"[Equil] Simulation input files written under {work}")


# ------------------------- FE component: z ------------------------- #


@register_sim_files("z")
def sim_files_z(ctx: BuildContext, lambdas: Sequence[float]) -> None:
    """
    Create per-window MD input files for component 'z' (UNO-REST style),
    supporting decoupling methods 'sdr' and 'dd'. Optionally applies
    extra CA restraints via ctx.extra['extra_restraints'] to mdin-XX only.
    """
    work: Path = ctx.working_dir
    sim = ctx.sim
    comp = ctx.comp
    mol = ctx.residue_name
    win = ctx.win
    windows_dir = ctx.window_dir
    all_atoms = sim.all_atoms

    if not hasattr(sim, "dec_method"):
        raise AttributeError(
            "SimulationConfig is missing 'dec_method'. "
            "Set 'dec_method' to 'sdr' or 'dd' in the YAML."
        )
    dec_method = sim.dec_method
    if dec_method not in {"sdr", "dd"}:
        raise ValueError(
            f"Decoupling method '{dec_method}' not recognized. Use 'sdr' or 'dd'."
        )

    temperature = sim.temperature
    steps2 = sim.dic_n_steps[comp]
    ntwx = sim.ntwx

    weight = lambdas[win if win != -1 else 0]

    # Count atoms
    if all_atoms.lower() == "no":
        vac_pdb = windows_dir / "vac.pdb"
        if not vac_pdb.exists():
            raise FileNotFoundError(f"Missing required file: {vac_pdb}")
        vac_atoms = mda.Universe(vac_pdb.as_posix()).atoms.n_atoms
    else:
        full_pdb = windows_dir / "full.pdb"
        vac_atoms = mda.Universe(full_pdb.as_posix()).atoms.n_atoms
        vac_pdb = windows_dir / "vac.pdb"  # still needed below to find last_lig

    # find *last* residue index of this ligand in vac.pdb
    last_lig: Optional[str] = None
    with vac_pdb.open("rt") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip().lower() == mol.lower():
                last_lig = line[22:26].strip()
    if last_lig is None:
        raise ValueError(f"No ligand residue matching '{mol}' found in {vac_pdb.name}")

    amber_dir = ctx.amber_dir

    # compute extra mask once for this window root; applied to mdin-XX only
    extra_mask, extra_fc = _maybe_extra_mask(ctx, windows_dir)

    if dec_method == "sdr":
        mk2 = int(last_lig)
        mk1 = mk2 - 1
        template_mdin = amber_dir / "mdin-unorest"
        template_mini = amber_dir / "mini-unorest"

        # first write eq.in
        n_steps_run = 5000
        out_path = windows_dir / "eq.in"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            for line in fin:
                if "ntx = 5" in line:
                    line = "ntx = 1,\n"
                elif "irest" in line:
                    line = "irest = 0,\n"
                elif "dt = " in line:
                    line = "dt = 0.001,\n"
                elif "restraintmask" in line:
                    rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                    if rm == "":
                        line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                    else:
                        line = f"restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", str(n_steps_run))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("mk2", str(mk2))
                )
                fout.write(line)

        with out_path.open("a") as mdin:
            mdin.write(f"  mbar_states = {len(lambdas):02d}\n")
            mdin.write("  mbar_lambda =")
            for lam in lambdas:
                mdin.write(f" {lam:6.5f},")
            mdin.write("\n")
            mdin.write("  infe = 1,\n")
            mdin.write(" /\n")
            mdin.write(" &pmd \n")
            mdin.write("  output_file = 'cmass.txt'\n")
            mdin.write(f"  output_freq = {int(ntwx):02d}\n")
            mdin.write("  cv_file = 'cv.in'\n")
            mdin.write(" /\n")
            mdin.write(" &wt type = 'END' , /\n")
            mdin.write("DISANG=disang.rest\n")
            mdin.write("LISTOUT=POUT\n")

        # end eq.in

        # write mdin-template
        n_steps_run = str(steps2)
        out_path = windows_dir / f"mdin-template"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            fout.write(f"! total_steps={steps2}\n")
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", n_steps_run)
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("mk2", str(mk2))
                )
                fout.write(line)

        with out_path.open("a") as mdin:
            mdin.write(f"  mbar_states = {len(lambdas):02d}\n")
            mdin.write("  mbar_lambda =")
            for lam in lambdas:
                mdin.write(f" {lam:6.5f},")
            mdin.write("\n")
            mdin.write("  infe = 1,\n")
            mdin.write(" /\n")
            mdin.write(" &pmd \n")
            mdin.write("  output_file = 'cmass.txt'\n")
            mdin.write(f"  output_freq = {int(ntwx):02d}\n")
            mdin.write("  cv_file = 'cv.in'\n")
            mdin.write(" /\n")
            mdin.write(" &wt type = 'END' , /\n")
            mdin.write("DISANG=disang.rest\n")
            mdin.write("LISTOUT=POUT\n")

        # Patch mdin with extra restraints (only mdin-XX)
        if extra_mask:
            try:
                content = out_path.read_text()
                content = _patch_restraint_block(content, extra_mask, extra_fc)
                out_path.write_text(content)
            except Exception as e:
                logger.warning(
                    f"[extra_restraints] Could not patch {out_path.name}: {e}"
                )

        # end mdin-template

        # mini.in
        with (
            template_mini.open("rt") as fin,
            (windows_dir / "mini.in").open("wt") as fout,
        ):
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("mk2", str(mk2))
                    .replace("_lig_name_", mol)
                )
                fout.write(line)
        # end mini.in

    else:  # dd
        extra_ctx = ctx.extra or {}
        if "infe" not in extra_ctx:
            raise KeyError(
                "BuildContext.extra missing 'infe'. Ensure BaseBuilder sets this flag."
            )
        infe_flag = 1 if extra_ctx["infe"] else 0
        mk1 = int(last_lig)
        template_mdin = amber_dir / "mdin-unorest-dd"
        template_mini = amber_dir / "mini-unorest-dd"

        # optional short equilibration input
        eq_path = windows_dir / "eq.in"
        with template_mdin.open("rt") as fin, eq_path.open("wt") as fout:
            for line in fin:
                if "ntx = 5" in line:
                    line = "ntx = 1,\n"
                elif "irest" in line:
                    line = "irest = 0,\n"
                elif "dt = " in line:
                    line = "dt = 0.001,\n"
                elif "restraintmask" in line:
                    rm = (
                        line.split("=", 1)[1]
                        .strip()
                        .rstrip(",")
                        .replace("'", "")
                    )
                    if rm == "":
                        line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                    else:
                        line = (
                            f"restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"
                        )
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", "5000")
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                )
                fout.write(line)
        with eq_path.open("a") as mdin:
            mdin.write(f"  mbar_states = {len(lambdas)}\n")
            mdin.write("  mbar_lambda =")
            for lbd in lambdas:
                mdin.write(f" {lbd:6.5f},")
            mdin.write("\n")
            mdin.write(f"  infe = {infe_flag},\n")
            mdin.write(" /\n")
            mdin.write(" &pmd \n")
            mdin.write("  output_file = 'cmass.txt'\n")
            mdin.write(f"  output_freq = {int(ntwx):02d}\n")
            mdin.write("  cv_file = 'cv.in'\n")
            mdin.write(" /\n")
            mdin.write(" &wt type = 'END' , /\n")
            mdin.write("DISANG=disang.rest\n")
            mdin.write("LISTOUT=POUT\n")

        # production template
        n_steps_run = str(steps2)
        out_path = windows_dir / "mdin-template"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            fout.write(f"! total_steps={steps2}\n")
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", n_steps_run)
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                )
                fout.write(line)

        with out_path.open("a") as mdin:
            mdin.write(f"  mbar_states = {len(lambdas)}\n")
            mdin.write("  mbar_lambda =")
            for lbd in lambdas:
                mdin.write(f" {lbd:6.5f},")
            mdin.write("\n")
            mdin.write(f"  infe = {infe_flag},\n")
            mdin.write(" /\n")
            mdin.write(" &pmd \n")
            mdin.write("  output_file = 'cmass.txt'\n")
            mdin.write(f"  output_freq = {int(ntwx):02d}\n")
            mdin.write("  cv_file = 'cv.in'\n")
            mdin.write(" /\n")
            mdin.write(" &wt type = 'END' , /\n")
            mdin.write("DISANG=disang.rest\n")
            mdin.write("LISTOUT=POUT\n")

        # Patch mdin with extra restraints (only mdin-template)
        if extra_mask:
            try:
                content = out_path.read_text()
                content = _patch_restraint_block(content, extra_mask, extra_fc)
                out_path.write_text(content)
            except Exception as e:
                logger.warning(
                    f"[extra_restraints] Could not patch {out_path.name}: {e}"
                )

        with (
            template_mini.open("rt") as fin,
            (windows_dir / "mini.in").open("wt") as fout,
        ):
            for line in fin:
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                    .replace("_lig_name_", mol)
                )
                fout.write(line)

    # Always emit mini_eq.in, eqnpt0.in, eqnpt.in from UNO templates (no extra restraints here)
    with (
        (amber_dir / "mini.in").open("rt") as fin,
        (windows_dir / "mini_eq.in").open("wt") as fout,
    ):
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    with (
        (amber_dir / "eqnpt0-uno.in").open("rt") as fin,
        (windows_dir / "eqnpt0.in").open("wt") as fout,
    ):
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(
                    line.replace("_temperature_", str(temperature)).replace(
                        "_lig_name_", mol
                    )
                )

    with (
        (amber_dir / "eqnpt-uno.in").open("rt") as fin,
        (windows_dir / "eqnpt.in").open("wt") as fout,
    ):
        for line in fin:
            if "mcwat" in line:
                fout.write("  mcwat = 0,\n")
            else:
                fout.write(
                    line.replace("_temperature_", str(temperature)).replace(
                        "_lig_name_", mol
                    )
                )

    (windows_dir / "lambda.sch").write_text(
        "TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n"
    )

    logger.debug(
        f"[sim_files_z] wrote mdin/mini/eq inputs in {windows_dir} for comp='z', win={win}, weight={weight:0.5f}"
    )


# ------------------------- FE component: y ------------------------- #


@register_sim_files("y")
def sim_files_y(ctx: BuildContext, lambdas: Sequence[float]) -> None:
    """
    Generate MD input files for ligand-only component 'y'.
    (No extra CA restraints apply to ligand-only eq inputs.)
    """
    sim = ctx.sim
    mol = ctx.residue_name
    windows_dir = ctx.window_dir

    temperature = sim.temperature
    n_steps = sim.dic_n_steps["y"]
    ntwx = sim.ntwx

    weight = lambdas[ctx.win if ctx.win != -1 else 0]
    mk1 = 2  # ligand-only marker convention

    amber_dir = ctx.amber_dir

    # mini.in from ligand template
    with (
        (amber_dir / "mini-unorest-lig").open("rt") as fin,
        (windows_dir / "mini.in").open("wt") as fout,
    ):
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    # mini_eq.in from generic mini template
    with (
        (amber_dir / "mini.in").open("rt") as fin,
        (windows_dir / "mini_eq.in").open("wt") as fout,
    ):
        for line in fin:
            fout.write(line.replace("_lig_name_", mol))

    # eqnpt.in / eqnpt0.in from ligand templates
    with (
        (amber_dir / "eqnpt-lig.in").open("rt") as fin,
        (windows_dir / "eqnpt.in").open("wt") as fout,
    ):
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace(
                    "_lig_name_", mol
                )
            )
    with (
        (amber_dir / "eqnpt0-lig.in").open("rt") as fin,
        (windows_dir / "eqnpt0.in").open("wt") as fout,
    ):
        for line in fin:
            fout.write(
                line.replace("_temperature_", str(temperature)).replace(
                    "_lig_name_", mol
                )
            )

    template = amber_dir / "mdin-unorest-lig"

    # short equilibration input
    eq_path = windows_dir / "eq.in"
    with template.open("rt") as fin, eq_path.open("wt") as fout:
        for line in fin:
            if "ntx = 5" in line:
                line = "  ntx = 1,\n"
            elif "irest" in line:
                line = "  irest = 0,\n"
            elif "dt = " in line:
                line = "  dt = 0.001,\n"
            elif "restraintmask" in line:
                rm = (
                    line.split("=", 1)[1]
                    .strip()
                    .rstrip(",")
                    .replace("'", "")
                )
                if rm == "":
                    line = f"  restraintmask = '(@CA | :{mol}) & !@H='\n"
                else:
                    line = f"  restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-steps_", "5000")
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("disang_file", "disang")
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    with eq_path.open("a") as mdin:
        mdin.write(f"  mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 1,\n")
        mdin.write(" /\n")
        mdin.write(" &pmd \n")
        mdin.write("  output_file = 'cmass.txt'\n")
        mdin.write(f"  output_freq = {int(ntwx):02d}\n")
        mdin.write("  cv_file = 'cv.in'\n")
        mdin.write(" /\n")
        mdin.write(" &wt type = 'END' , /\n")
        mdin.write("DISANG=disang.rest\n")
        mdin.write("LISTOUT=POUT\n")

    # production template (single long segment)
    out_path = windows_dir / "mdin-template"
    with template.open("rt") as fin, out_path.open("wt") as fout:
        fout.write(f"! total_steps={n_steps}\n")
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-steps_", str(n_steps))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("disang_file", "disang")
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    with out_path.open("a") as mdin:
        mdin.write(f"  mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 1,\n")
        mdin.write(" /\n")
        mdin.write(" &pmd \n")
        mdin.write("  output_file = 'cmass.txt'\n")
        mdin.write(f"  output_freq = {int(ntwx):02d}\n")
        mdin.write("  cv_file = 'cv.in'\n")
        mdin.write(" /\n")
        mdin.write(" &wt type = 'END' , /\n")
        mdin.write("DISANG=disang.rest\n")
        mdin.write("LISTOUT=POUT\n")

    logger.debug(
        f"[sim_files_y] wrote mdin/mini/eq inputs in {windows_dir} for comp='y', weight={weight:0.5f}"
    )


@register_sim_files("m")
def sim_files_m(ctx: BuildContext, lambdas: Sequence[float]) -> None:
    """
    Generate MD input files for vaccum ligand-only component 'm'.
    """
    sim = ctx.sim
    mol = ctx.residue_name
    windows_dir = ctx.window_dir

    temperature = sim.temperature
    n_steps = sim.dic_n_steps["m"]
    ntwx = sim.ntwx

    weight = lambdas[ctx.win if ctx.win != -1 else 0]
    mk1 = 2  # ligand-only marker convention

    amber_dir = ctx.amber_dir

    # mini.in from ligand template
    with (
        (amber_dir / "mini-unorest-vacuum").open("rt") as fin,
        (windows_dir / "mini_eq.in").open("wt") as fout,
    ):
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    template = amber_dir / "mdin-unorest-vacuum"

    # short equilibration input
    eq_path = windows_dir / "eq.in"
    with template.open("rt") as fin, eq_path.open("wt") as fout:
        for line in fin:
            if "ntx = 5" in line:
                line = "  ntx = 1,\n"
            elif "irest" in line:
                line = "  irest = 0,\n"
            elif "dt = " in line:
                line = "  dt = 0.001,\n"
            elif "restraintmask" in line:
                rm = (
                    line.split("=", 1)[1]
                    .strip()
                    .rstrip(",")
                    .replace("'", "")
                )
                if rm == "":
                    line = f"  restraintmask = '(@CA | :{mol}) & !@H='\n"
                else:
                    line = f"  restraintmask = '(@CA | :{mol} | {rm}) & !@H='\n"
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-steps_", "5000")
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("disang_file", "disang")
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    with eq_path.open("a") as mdin:
        mdin.write(f"  mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        mdin.write(" &pmd \n")
        mdin.write("  output_file = 'cmass.txt'\n")
        mdin.write(f"  output_freq = {int(ntwx):02d}\n")
        mdin.write("  cv_file = 'cv.in'\n")
        mdin.write(" /\n")
        mdin.write(" &wt type = 'END' , /\n")
        mdin.write("DISANG=disang.rest\n")
        mdin.write("LISTOUT=POUT\n")

    # production template (single long segment)
    out_path = windows_dir / "mdin-template"
    with template.open("rt") as fin, out_path.open("wt") as fout:
        fout.write(f"! total_steps={n_steps}\n")
        for line in fin:
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-steps_", str(n_steps))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("mk1", str(mk1))
                .replace("disang_file", "disang")
                .replace("_lig_name_", mol)
            )
            fout.write(line)

    with out_path.open("a") as mdin:
        mdin.write(f"  mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        mdin.write(" &pmd \n")
        mdin.write("  output_file = 'cmass.txt'\n")
        mdin.write(f"  output_freq = {int(ntwx):02d}\n")
        mdin.write("  cv_file = 'cv.in'\n")
        mdin.write(" /\n")
        mdin.write(" &wt type = 'END' , /\n")
        mdin.write("DISANG=disang.rest\n")
        mdin.write("LISTOUT=POUT\n")

    logger.debug(
        f"[sim_files_m] wrote mdin/mini/eq inputs in {windows_dir} for comp='m', weight={weight:0.5f}"
    )
