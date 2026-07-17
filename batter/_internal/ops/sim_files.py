# sim_files.py — drop-in replacement
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Optional, Tuple, Iterable, List
import hashlib

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger
import os
import json
import shutil


from batter._internal.parmed_compat import import_parmed
from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_sim_files
from batter._internal.ops.helpers import format_ranges
from batter._internal.ops.remd import patch_mdin_file

pmd = import_parmed()
from parmed.amber.mask import AmberMask


# ----------------------------- helpers ----------------------------- #


def _non_loop_mask_from_dssp_assignments(
    assignments: Sequence[str], *, min_len: int = 4, shift: int = 0
) -> str:
    """
    Convert DSSP assignments to a compact AMBER residue range string.

    Keeps contiguous non-loop segments (assignment != '-') with length >= min_len.
    Default shift-based residue indices.
    """
    if min_len < 1:
        raise ValueError("min_len must be >= 1")

    keep: list[int] = []
    run_start: int | None = None
    seq = [str(x).strip() for x in assignments]

    for idx, ss in enumerate(seq, start=shift):
        if ss and ss != "-":
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            run_len = idx - run_start
            if run_len >= min_len:
                keep.extend(range(run_start, idx))
            run_start = None

    if run_start is not None:
        run_len = len(seq) + 1 - run_start
        if run_len >= min_len:
            keep.extend(range(run_start, len(seq) + 1))

    return format_ranges(keep)


def _fallback_non_loop_mask_from_renum(build_dir: Path, shift: int) -> str:
    """
    Fallback to all mapped protein residues when DSSP-derived mask is unavailable.
    """
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        logger.warning(
            f"[dssp] Missing renumber map for fallback mask: {renum_txt}; using ':1'."
        )
        return f"@CA"

    renum_data = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    if renum_data.empty:
        logger.warning(
            f"[dssp] Empty renumber map for fallback mask: {renum_txt}; using ':1'."
        )
        return f"@CA"

    # one-based
    ranges = renum_data["new_resid"].astype(int) + shift - 1
    ranges = format_ranges(ranges.tolist())
    if not ranges:
        logger.warning(
            f"[dssp] Failed to build fallback residue ranges from {renum_txt}; using ':1'."
        )
        return f"@CA"
    return f':{ranges}'


def _resolve_non_loop_mask(ctx: BuildContext, shift: int) -> str:
    """
    Build the `_non_loop_` replacement mask from system_prep DSSP artifacts.
    """
    manifest_path = ctx.system_root / "all-ligands" / "manifest.json"
    if not manifest_path.exists():
        logger.warning(
            f"[dssp] Missing system_prep manifest: {manifest_path}; using fallback mask."
        )
        return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        logger.warning(
            f"[dssp] Could not parse manifest {manifest_path}: {exc}; using fallback mask."
        )
        return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    dssp_info = manifest.get("dssp") or {}
    dssp_results = dssp_info.get("results")
    if dssp_results is None:
        dssp_json = dssp_info.get("json")
        if dssp_json:
            try:
                dssp_results = json.loads(Path(dssp_json).read_text())
            except Exception as exc:
                logger.warning(
                    f"[dssp] Could not read DSSP JSON {dssp_json}: {exc}; using fallback mask."
                )
                return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    if dssp_results is None:
        logger.warning(
            "[dssp] No DSSP results found in manifest; using fallback mask."
        )
        return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    dssp_arr = np.asarray(dssp_results)
    if dssp_arr.size == 0:
        logger.debug("[dssp] Empty DSSP results; using fallback mask.")
        return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    if dssp_arr.ndim == 1:
        assignments = dssp_arr.tolist()
    else:
        assignments = dssp_arr[0].tolist()

    mask_ranges = _non_loop_mask_from_dssp_assignments(assignments, min_len=4, shift=shift)
    if not mask_ranges:
        logger.warning(
            "[dssp] No non-loop DSSP stretches with len>=4; using fallback mask."
        )
        return _fallback_non_loop_mask_from_renum(ctx.build_dir, shift=shift)

    logger.debug(f"[dssp] Non-loop restraint mask ranges: {mask_ranges}")
    return f':{mask_ranges}'


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
                logger.debug(
                    "[restraintmask] Mask exceeds 256 chars; will attempt legacy-group conversion."
                )
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


def _format_restraint_weight(value: str | float) -> str:
    if isinstance(value, str):
        value = value.strip().rstrip(",")
    try:
        num = float(value)
    except Exception:
        return str(value)
    text = f"{num:.6f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _convert_restraintmask_to_legacy_group_block(
    prmtop_path: Path, maskstr: str, restraint_wt: str | float, title: str
) -> list[str]:
    parm = pmd.load_file(prmtop_path.as_posix())
    sel = AmberMask(parm, maskstr).Selection()
    indices = [i + 1 for i, flag in enumerate(sel) if flag > 0]
    ranges = _merge_consecutive(indices)
    selected_count = len(indices)
    range_count = sum(end - start + 1 for start, end in ranges)
    if selected_count != range_count:
        raise ValueError(
            f"Selection size mismatch: selected={selected_count} vs ranges={range_count}"
        )
    out = [title, _format_restraint_weight(restraint_wt)]
    for i in range(0, len(ranges), 7):
        chunk = ranges[i : i + 7]
        parts: List[str] = []
        for start, end in chunk:
            parts.append(str(start))
            parts.append(str(end))
        out.append("ATOM " + " ".join(parts))
    out.append("END")
    out.append("END")
    return out


def _find_prmtop_for_masks(work_dir: Path) -> Optional[Path]:
    candidates = [
        "full_merged.prmtop",
        "full.hmr.prmtop",
        "full.prmtop",
    ]
    for base in [work_dir, *work_dir.parents]:
        for name in candidates:
            path = base / name
            if path.exists():
                return path
    return None


def _apply_restraintmask_length_limit(
    mdin_path: Path,
    prmtop_path: Optional[Path],
    *,
    title: str = "Converted from restraintmask",
    cache_dir: Optional[Path] = None,
    cache_tag: Optional[str] = None,
    cache_master: bool = False,
    max_mask_chars: Optional[int] = None,
) -> None:
    """Convert restraintmask input to legacy AMBER GROUP input when possible.

    By default any restraintmask is converted. Callers can pass max_mask_chars
    to keep shorter masks in restraintmask form and only convert long masks.
    """
    if not mdin_path.exists():
        return
    text = mdin_path.read_text()
    mask = None
    mask_lines_removed = False
    lines = text.splitlines(True)
    out_lines: List[str] = []
    for line in lines:
        m = re.search(r'restraintmask\s*=\s*["\']([^"\']*)["\']', line)
        if m:
            mask = m.group(1).strip()
            mask_lines_removed = True
            continue
        out_lines.append(line)

    if not mask:
        return

    if max_mask_chars is not None and len(mask) <= max_mask_chars:
        return

    cache_path = None
    mask_hash = hashlib.sha1(mask.encode("utf-8")).hexdigest()
    if cache_dir is not None and cache_tag:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_tag}.legacy_restraint"

    if cache_path is not None and cache_path.exists() and not cache_master:
        cached = cache_path.read_text().splitlines()
        if cached and cached[0].startswith("# mask_sha1="):
            cached_hash = cached[0].split("=", 1)[1].strip()
            if cached_hash != mask_hash:
                logger.debug(
                    f"[restraintmask] Cache hash mismatch for {mdin_path.name}; reusing cached block anyway."
                )
            block = cached[1:]
            new_text = "".join(out_lines)
            if not new_text.endswith("\n"):
                new_text += "\n"
            new_text += "&end\n"
            new_text += "\n".join(block) + "\n"
            mdin_path.write_text(new_text)
            logger.debug(
                f"[restraintmask] Reused cached legacy block for {mdin_path.name}."
            )
            return

    if prmtop_path is None or not prmtop_path.exists():
        logger.warning(
            f"[restraintmask] No prmtop found for {mdin_path.name}; leaving restraintmask as-is."
        )
        return

    wt_match = re.search(
        r"\brestraint_wt\s*=\s*([0-9.+-eEdD]+)", text, flags=re.IGNORECASE
    )
    restraint_wt = wt_match.group(1) if wt_match else "0.0"

    try:
        block = _convert_restraintmask_to_legacy_group_block(
            prmtop_path, mask, restraint_wt, title
        )
    except Exception as exc:
        logger.warning(
            f"[restraintmask] Failed to convert mask for {mdin_path.name}: {exc}"
        )
        return

    if not mask_lines_removed:
        return

    new_text = "".join(out_lines)
    if not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "&end\n"
    new_text += "\n".join(block) + "\n"
    mdin_path.write_text(new_text)
    if cache_path is not None and cache_master:
        cache_path.write_text("# mask_sha1=" + mask_hash + "\n" + "\n".join(block) + "\n")
    logger.debug(
        f"[restraintmask] Converted restraintmask in {mdin_path.name} to legacy group block."
    )


def _first_residue_atom_mask(
    pdb_path: Path,
    *,
    resid: int | None = None,
    resname: str | None = None,
) -> str:
    """Return an AMBER atom-index mask for the first atom in one residue."""
    if resid is None and resname is None:
        raise ValueError("Either resid or resname must be provided")

    u = mda.Universe(pdb_path.as_posix())
    if resid is not None:
        atoms = u.select_atoms(f"resid {int(resid)}")
    else:
        atoms = u.select_atoms(f"resname {resname}")

    if atoms.n_atoms == 0:
        target = f"resid {resid}" if resid is not None else f"resname {resname}"
        raise ValueError(f"No atoms matched {target!r} in {pdb_path}")

    atom = atoms[0]
    return f"@{int(atom.index) + 1}"


def _solvent_ligand_restraint_mask(pdb_path: Path, *, resid: int, comp: str) -> str:
    """Return the non-lambda restraintmask component for the solvent ligand."""
    if comp == "d":
        return f":{int(resid)}"
    return _first_residue_atom_mask(pdb_path, resid=int(resid))


def _first_absolute_atom_mask(atom_indices: Sequence[int | str]) -> str:
    """Return an AMBER mask for the smallest 1-based absolute atom index."""
    tokens = sorted(
        int(str(atom).strip())
        for atom in atom_indices
        if str(atom).strip() and int(str(atom).strip()) > 0
    )
    if not tokens:
        raise ValueError("atom_indices must contain at least one positive atom index")
    return f"@{tokens[0]}"


def _write_batch_mdin_template(window_dir: Path, comp_dir: Path) -> None:
    base_template = window_dir / "mdin-template"
    if not base_template.exists():
        return
    batch_template = window_dir / "mdin-batch-template"
    batch_template.write_text(base_template.read_text())
    prefix = window_dir.relative_to(comp_dir).as_posix()
    patch_mdin_file(batch_template, prefix, add_numexchg=False)


def _mask_with_added_component(base_mask: str, new_mask_component: str) -> str:
    """Append one atom/group to a template restraintmask."""
    base = re.sub(r"\s*&\s*!@H=\s*$", "", base_mask.strip()).strip()
    if not base:
        return f"({new_mask_component}) & !@H="
    return f"({base} | {new_mask_component}) & !@H="


def _maybe_extra_mask(
    ctx: BuildContext, work: Path, *, resid_shift: int = 0
) -> tuple[Optional[str], float]:
    """
    Build an absolute atom-index mask from ctx.extra["extra_restraints"].

    The selection is evaluated against full.pdb and converted to AMBER's
    1-based @atom-index mask syntax. resid_shift is accepted for compatibility
    with existing call sites but is not used for absolute atom masks.
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

    force_const = float(extra.get("extra_restraint_fc", 10.0))

    ref_pdb = work / "full.pdb"

    if not ref_pdb.exists():
        logger.warning(f"[extra_restraints] Missing reference PDB: {ref_pdb}; skip.")
        return None, force_const

    u = mda.Universe(str(ref_pdb))
    sel = u.select_atoms(f"({extra_sel})")
    if len(sel) == 0:
        logger.warning(
            f"[extra_restraints] 0 atoms selected for '({extra_sel})'; skip."
        )
        return None, force_const

    atom_indices = [int(idx) + 1 for idx in sel.indices]
    mask_ranges = format_ranges(atom_indices)
    if not mask_ranges:
        logger.warning("[extra_restraints] No atom indices in selected atoms; skip.")
        return None, force_const

    mask = f"@{mask_ranges}"
    # save as json
    json.dump(
        {
            "mask": mask,
            "force_const": force_const,
            "selection": str(extra_sel),
            "source": str(ref_pdb),
        },
        (work / "extra_restraints.json").open("wt"),
    )
    logger.debug(f"[extra_restraints] Mask: {mask} (wt={force_const})")
    return mask, force_const

def build_dyna_steps_run_per_lambda(n_steps_run_per_lambda = 10000, n_lambdas = 5):
    dynlmb = 1 / (n_lambdas-1)
    n_steps_run = int(n_steps_run_per_lambda * n_lambdas)
    return n_steps_run_per_lambda, n_lambdas, dynlmb, n_steps_run


def _replace_d_sdr_tokens(
    text: str,
    *,
    mk1: int,
    mk2: int,
    mk3: int,
    weight: float,
) -> str:
    if re.search(r"\btimask1\s*=", text):
        return f"  timask1 = ':{mk1},{mk3}',\n"
    if re.search(r"\btimask2\s*=", text):
        return f"  timask2 = ':{mk2}',\n"
    if re.search(r"\bscmask1\s*=", text):
        return f"  scmask1=':{mk1}',\n"
    if re.search(r"\bscmask2\s*=", text):
        return "  scmask2='',\n"
    if re.search(r"\bcrgmask\s*=", text):
        return f"  crgmask = ':{mk3}',\n"
    return (
        text.replace("lbd_val", f"{float(weight):6.5f}")
        .replace("mk1", str(mk1))
        .replace("mk2", str(mk2))
        .replace("mk3", str(mk3))
    )


def _d_sdr_ti_block(
    *,
    mk1: int,
    mk2: int,
    mk3: int,
    weight: float,
) -> str:
    return (
        f"  icfe = 1, clambda = {float(weight):6.5f},\n"
        f"  timask1 = ':{mk1},{mk3}',\n"
        f"  timask2 = ':{mk2}',\n"
        "  ifsc=1,\n"
        f"  scmask1=':{mk1}',\n"
        "  scmask2='',\n"
        f"  crgmask = ':{mk3}',\n"
        "  gti_cut   = 1,\n"
        "  gti_output = 1,\n"
        "  gti_add_sc = 25,\n"
        "  gti_scale_beta  = 1,\n"
        "  gti_lam_sch = 1,\n"
        "  gti_ele_sc  = 1,\n"
        "  gti_vdw_sc  = 1,\n"
        "  gti_cut_sc  = 2,\n"
        "  scalpha = 0.5,\n"
        "  scbeta = 1.0,\n"
        "  gti_cut_sc_on   = 7,\n"
        "  gti_cut_sc_off  = 9,\n"
        "  gti_ele_exp     = 2,\n"
        "  gti_vdw_exp     = 2,\n"
        "  gti_chg_keep    = 1,\n"
        "  gti_bat_sc      = 1,\n"
    )


def _write_d_sdr_equil_input(
    *,
    src: Path,
    dst: Path,
    replacements: dict[str, str],
    mk1: int,
    mk2: int,
    mk3: int,
    weight: float,
    restraint_mask: str,
) -> None:
    inserted_ti = False
    with src.open("rt") as fin, dst.open("wt") as fout:
        for line in fin:
            if line.lstrip().startswith("/") and not inserted_ti:
                fout.write(
                    _d_sdr_ti_block(
                        mk1=mk1,
                        mk2=mk2,
                        mk3=mk3,
                        weight=weight,
                    )
                )
                inserted_ti = True
            if "mcwat" in line:
                line = "  mcwat = 0,\n"
            elif re.search(r"\bnmropt\s*=", line):
                line = "  nmropt = 1,\n"
            elif re.search(r"\brestraintmask\s*=", line):
                line = f"  restraintmask = '{restraint_mask}',\n"
            elif re.search(r"\bntp\s*=", line):
                line = "  ntp = 1,\n"
            elif re.search(r"\bcsurften\s*=", line):
                line = "  csurften = 0,\n"
            for key, value in replacements.items():
                line = line.replace(key, value)
            line = _replace_d_sdr_tokens(
                line,
                mk1=mk1,
                mk2=mk2,
                mk3=mk3,
                weight=weight,
            )
            fout.write(line)


def _sub_write(src: Path, dst: Path, repl: dict[str, str]) -> None:
    text = Path(src).read_text()
    for k, v in repl.items():
        text = text.replace(k, v)
    dst.write_text(text)


def _force_fe_mini_constraints(line: str) -> str:
    if re.search(r"\bntf\s*=", line):
        return "  ntf = 2,\n"
    if re.search(r"\bntc\s*=", line):
        return "  ntc = 2,\n"
    return line


def _force_softcore_mini_constraints(line: str) -> str:
    if re.search(r"\bntf\s*=", line):
        return "  ntf = 1,\n"
    if re.search(r"\bntc\s*=", line):
        return "  ntc = 2,\n"
    return line


def _force_x_mini_constraints(line: str) -> str:
    if re.search(r"\bntf\s*=", line):
        return "  ntf = 1,\n"
    if re.search(r"\bntc\s*=", line):
        return "  ntc = 2,\n"
    return line


def _sub_write_fe_mini(src: Path, dst: Path, repl: dict[str, str]) -> None:
    text = Path(src).read_text()
    text = "".join(_force_fe_mini_constraints(line) for line in text.splitlines(True))
    for k, v in repl.items():
        text = text.replace(k, v)
    dst.write_text(text)


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
    indices: Iterable[int],
) -> str:
    """Build a selection string from include or exclude indices
    """
    inc = sorted(set(indices))
    if not inc:
        raise ValueError("indices must be non-empty")

    inc_ranges = _merge_consecutive(inc)
    inc_str = _ranges_to_str(inc_ranges)

    return f"@{inc_str}"


def _write_cmass_dump_block(handle, *, istep1: int | str, disang: str = "disang.rest") -> None:
    """Write the AMBER wt/DUMPAVE footer used for cv/disang-driven runs."""
    handle.write(f" &wt type='DUMPFREQ', istep1={istep1}, /\n")
    handle.write(" &wt type='END', /\n")
    handle.write(f"DISANG={disang}\n")
    handle.write("DUMPAVE=cmass.txt\n")
    handle.write("LISTIN=POUT\n")
    handle.write("LISTOUT=POUT\n")


def _component_l_cmass_dumpfreq(ntwx: int) -> int:
    """Use denser restraint-energy traces for component l than trajectory output."""
    return max(1, min(int(ntwx), 1000))


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
    # Keep infe disabled while nmropt/disang mirrors the cv restraints.
    infe_flag = "0"

    # disang anchor triplet (L1/L2/L3)
    with open(work / "disang.rest", "r") as f:
        parts = f.readline().split()
        L1 = parts[6].strip()
        L2 = parts[7].strip()
        L3 = parts[8].strip()



    # mini.in
    _sub_write_fe_mini(amber_dir / "mini.in", work / "mini.in", {"_lig_name_": mol})

    # eqnvt.in
    _sub_write(
        amber_dir / "eqnvt.in",
        work / "eqnvt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt0.in
    eqnpt0_src = amber_dir / (
        "eqnpt0.in" if sim.membrane_simulation else "eqnpt0-water.in"
    )
    _sub_write(
        eqnpt0_src,
        work / "eqnpt0.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt.in
    eqnpt_src = amber_dir / (
        "eqnpt.in" if sim.membrane_simulation else "eqnpt-water.in"
    )
    _sub_write(
        eqnpt_src,
        work / "eqnpt.in",
        {"_temperature_": f"{temperature}", "_lig_name_": mol},
    )

    # eqnpt-eq.in (longer equil with restraints on non-loop regions)
    non_loop_mask = _resolve_non_loop_mask(ctx, shift=2)
    eqnpt_src = amber_dir / (
        "eqnpt-eq.in" if sim.membrane_simulation else "eqnpt-water-eq.in"
    )
    _sub_write(
        eqnpt_src,
        work / "eqnpt_eq.in",
        {
            "_temperature_": f"{temperature}",
            "_lig_name_": mol,
            "_non_loop_": non_loop_mask,
        },
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
            "_non_loop_": non_loop_mask,
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
            "_non_loop_": non_loop_mask,
        },
    )

    # mdin-template for runtime chunking (total_steps is the total target)
    mdin_src = amber_dir / "mdin-equil"
    base_text = mdin_src.read_text()
    total_steps = int(getattr(sim, "eq_steps", 0) or 0)
    if total_steps <= 0:
        total_steps = 0

    # compute extra mask once for equil (applied to template)
    extra_mask, extra_fc = _maybe_extra_mask(ctx, work, resid_shift=1)

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

    prmtop_for_masks = _find_prmtop_for_masks(work)
    _apply_restraintmask_length_limit(work / "mdin-template", prmtop_for_masks)
    _apply_restraintmask_length_limit(work / "eqnpt0.in", prmtop_for_masks)
    _apply_restraintmask_length_limit(work / "eqnpt.in", prmtop_for_masks)
    _apply_restraintmask_length_limit(work / "eqnpt_eq.in", prmtop_for_masks)
    _apply_restraintmask_length_limit(work / "eqnpt_disappear.in", prmtop_for_masks)
    _apply_restraintmask_length_limit(work / "eqnpt_appear.in", prmtop_for_masks)

    logger.debug(f"[Equil] Simulation input files written under {work}")


# ------------------------- FE components: z / d ------------------------- #


def _sim_files_d_sdr_charge_transfer(
    ctx: BuildContext,
    lambdas: Sequence[float],
    *,
    vac_atoms: int,
    vac_pdb: Path,
    ligand_resids: Sequence[int],
    non_loop_mask: str,
    prmtop_for_masks: Optional[Path],
    cache_dir: Path,
    cache_master: bool,
    extra_mask: Optional[str],
    extra_fc: float,
) -> None:
    """Write ABFE_diff d-component inputs with charge-balanced SDR bookkeeping."""
    if len(ligand_resids) < 3:
        raise ValueError(
            "ABFE_diff d-component SDR charge-transfer requires three ligand "
            f"residues in {vac_pdb}: bound ligand, solvent alchemical "
            "charge-transfer copy, and solvent neutral charge-mask copy."
        )

    sim = ctx.sim
    comp = ctx.comp
    mol = ctx.residue_name
    win = ctx.win
    windows_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    temperature = sim.temperature
    steps2 = sim.dic_n_steps[comp]
    ntwx = sim.ntwx
    weight = lambdas[win if win != -1 else 0]

    mk1, mk2, mk3 = [int(resid) for resid in ligand_resids[:3]]
    solvent_ligand_restraint_mask = f":{mk2},{mk3}"
    receptor_solvent_restraint_mask = (
        f"((@CA & {non_loop_mask}) | {solvent_ligand_restraint_mask}) & !@H="
    )
    production_restraint_mask = f"({solvent_ligand_restraint_mask}) & !@H="
    initial_equil_restraint_mask = (
        f"(@CA,C,N,P31 | {solvent_ligand_restraint_mask}) & !@H="
    )
    template_mdin = amber_dir / "mdin-diff-sdr"
    template_mini = amber_dir / "mini-diff-sdr"

    if not template_mdin.exists() or not template_mini.exists():
        missing = [
            path.name
            for path in (template_mdin, template_mini)
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Missing ABFE_diff d-component template(s): " + ", ".join(missing)
        )

    n_steps_run_per_lambda, _n_lambdas, dynlmb, n_steps_run = (
        build_dyna_steps_run_per_lambda(
            n_lambdas=len(lambdas) if len(lambdas) > 1 else 5
        )
    )
    if win != -1:
        n_steps_run = 10000
        n_steps_run_per_lambda = 10000

    eq_path = windows_dir / "eq.in"
    with template_mdin.open("rt") as fin, eq_path.open("wt") as fout:
        for line in fin:
            if "ntx = 5" in line:
                line = "ntx = 1,\n"
            elif "ntwx = " in line:
                line = f"ntwx = {n_steps_run_per_lambda},\n"
            elif "ntwprt = " in line:
                line = "\n"
            elif "irest" in line:
                line = "irest = 0,\n"
            elif "dt = " in line:
                line = "dt = 0.002,\n"
            elif "nmropt = " in line:
                line = "nmropt = 1,\n"
            elif "restraint_wt = " in line:
                line = "restraint_wt = 10,\n"
            elif "restraintmask" in line:
                line = f"restraintmask = '{receptor_solvent_restraint_mask}',\n"
            elif "gti_bat_sc" in line:
                line = "  gti_bat_sc      = 1,\n"

            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-atoms_", str(vac_atoms))
                .replace("_num-steps_", str(n_steps_run))
            )
            line = _replace_d_sdr_tokens(
                line,
                mk1=mk1,
                mk2=mk2,
                mk3=mk3,
                weight=weight,
            )
            fout.write(line)

    with eq_path.open("a") as mdin:
        mdin.write(" ntwv = -1,\n")
        if win == -1:
            mdin.write(f" dynlmb = {dynlmb},\n")
            mdin.write(f" ntave = {n_steps_run_per_lambda},\n")
        mdin.write(f" \n mbar_states = {len(lambdas):02d}\n")
        mdin.write("  mbar_lambda =")
        for lam in lambdas:
            mdin.write(f" {lam:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        eq_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag=f"{comp}-eq.in",
        cache_master=cache_master,
    )

    mdin_template = windows_dir / "mdin-template"
    with template_mdin.open("rt") as fin, mdin_template.open("wt") as fout:
        fout.write(f"! total_steps={steps2}\n")
        for line in fin:
            if "restraintmask" in line:
                line = f"restraintmask = '{production_restraint_mask}',\n"
            elif "gti_bat_sc" in line:
                line = "  gti_bat_sc      = 1,\n"
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-atoms_", str(vac_atoms))
                .replace("_num-steps_", str(steps2))
            )
            line = _replace_d_sdr_tokens(
                line,
                mk1=mk1,
                mk2=mk2,
                mk3=mk3,
                weight=weight,
            )
            fout.write(line)

    with mdin_template.open("a") as mdin:
        mdin.write(f" \n mbar_states = {len(lambdas):02d}\n")
        mdin.write("  mbar_lambda =")
        for lam in lambdas:
            mdin.write(f" {lam:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))

    if extra_mask:
        try:
            content = mdin_template.read_text()
            content = _patch_restraint_block(content, extra_mask, extra_fc)
            mdin_template.write_text(content)
        except Exception as e:
            logger.warning(f"[extra_restraints] Could not patch mdin-template: {e}")
    _apply_restraintmask_length_limit(
        mdin_template,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag=f"{comp}-mdin-template",
        cache_master=cache_master,
    )

    for out_name in ("mini.in", "mini_eq.in"):
        with template_mini.open("rt") as fin, (windows_dir / out_name).open("wt") as fout:
            for line in fin:
                line = _force_softcore_mini_constraints(line)
                if "restraintmask" in line:
                    line = f"  restraintmask = '{initial_equil_restraint_mask}',\n"
                elif "gti_bat_sc" in line:
                    line = "  gti_bat_sc      = 1,\n"
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("_lig_name_", mol)
                )
                line = _replace_d_sdr_tokens(
                    line,
                    mk1=mk1,
                    mk2=mk2,
                    mk3=mk3,
                    weight=weight,
                )
                fout.write(line)

    _write_d_sdr_equil_input(
        src=amber_dir / "eqnpt0-uno.in",
        dst=windows_dir / "eqnpt0.in",
        replacements={"_temperature_": str(temperature), "_lig_name_": mol},
        mk1=mk1,
        mk2=mk2,
        mk3=mk3,
        weight=weight,
        restraint_mask=initial_equil_restraint_mask,
    )
    _write_d_sdr_equil_input(
        src=amber_dir / "eqnpt-uno.in",
        dst=windows_dir / "eqnpt.in",
        replacements={"_temperature_": str(temperature), "_lig_name_": mol},
        mk1=mk1,
        mk2=mk2,
        mk3=mk3,
        weight=weight,
        restraint_mask=initial_equil_restraint_mask,
    )
    _write_d_sdr_equil_input(
        src=amber_dir / "eqnpt-uno-eq.in",
        dst=windows_dir / "eqnpt_eq.in",
        replacements={
            "_temperature_": str(temperature),
            "_lig_name_": mol,
            "_non_loop_": non_loop_mask,
        },
        mk1=mk1,
        mk2=mk2,
        mk3=mk3,
        weight=weight,
        restraint_mask=receptor_solvent_restraint_mask,
    )

    (windows_dir / "lambda.sch").write_text(
        "TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n"
    )

    logger.debug(
        f"[sim_files_d] wrote charge-balanced SDR inputs in {windows_dir} "
        f"for win={win}, weight={weight:0.5f}, masks=:{mk1}/{mk2}/{mk3}"
    )


@register_sim_files("d")
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
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1
    all_atoms = sim.all_atoms
    non_loop_mask = _resolve_non_loop_mask(ctx, shift=3)
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = win == -1


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
        vac_pdb = windows_dir / "vac.pdb"

    u = mda.Universe(vac_pdb.as_posix())
    mol_ref_ag = u.select_atoms(f'resname {mol}')
    ligand_resids_ordered = [int(res.resid) for res in mol_ref_ag.residues]
    ligand_resids = sorted(set(ligand_resids_ordered))
    if not ligand_resids:
        raise ValueError(f"No residues with resname {mol!r} found in {vac_pdb}")
    ref_resid = ligand_resids[0]
    bulk_resid = ligand_resids[1] if len(ligand_resids) > 1 else ref_resid
    ref_lig_in_site_mask = f':{int(ref_resid)}'
    solvent_ligand_restraint_mask = _solvent_ligand_restraint_mask(
        vac_pdb,
        resid=int(bulk_resid),
        comp=comp,
    )

    amber_dir = ctx.amber_dir
    prmtop_for_masks = _find_prmtop_for_masks(windows_dir)
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1

    # compute extra mask once for this window root; applied to mdin-XX only
    extra_mask, extra_fc = _maybe_extra_mask(ctx, windows_dir, resid_shift=2)

    if comp == "d" and dec_method == "sdr":
        _sim_files_d_sdr_charge_transfer(
            ctx,
            lambdas,
            vac_atoms=vac_atoms,
            vac_pdb=vac_pdb,
            ligand_resids=ligand_resids_ordered,
            non_loop_mask=non_loop_mask,
            prmtop_for_masks=prmtop_for_masks,
            cache_dir=cache_dir,
            cache_master=cache_master,
            extra_mask=extra_mask,
            extra_fc=extra_fc,
        )
        return

    if dec_method == "sdr":
        mk1 = ref_resid
        mk2 = mk1 + 1
        template_mdin = amber_dir / "mdin-unorest"
        template_mini = amber_dir / "mini-unorest"

        # first write eq.in
        # it will gradually increase lambda value
        n_steps_run_per_lambda, n_lambdas, dynlmb, n_steps_run = build_dyna_steps_run_per_lambda()
        if win != -1:
            n_steps_run = 10000
            n_steps_run_per_lambda = 10000
        out_path = windows_dir / "eq.in"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            for line in fin:
                if "ntx = 5" in line:
                    line = "ntx = 1,\n"
                # save every lambda value
                elif "ntwx = " in line:
                    line = f"ntwx = {n_steps_run_per_lambda},\n"
                # write all atoms
                elif "ntwprt = " in line:
                    line = "\n"
                elif "irest" in line:
                    line = "irest = 0,\n"
                elif "dt = " in line:
                    line = "dt = 0.002,\n"
                elif "nmropt = " in line:
                    line = "nmropt = 1,\n"
                elif "restraint_wt = " in line:
                    line = "restraint_wt = 10,\n"
                elif "restraintmask" in line:
                    rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                    if rm == "":
                        line = f"restraintmask = '((@CA & {non_loop_mask}) | :{mol}) & !@H='\n"
                        #line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                    else:
                        #line = f"restraintmask = '(@CA | :{mol} | {rm} ) & !@H='\n"
                        line = f"restraintmask = '((@CA & {non_loop_mask}) | :{mol} | {rm} ) & !@H='\n"

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
            # also save velocity info
            mdin.write(f" ntwv = -1,\n")
            # run dynlmb
            # if window is -1
            if win == -1:
                mdin.write(f" dynlmb = {dynlmb},\n")
                mdin.write(f" ntave = {n_steps_run_per_lambda},\n")
            # run mcwat
            mdin.write("  mcwat = 1,\n")
            mdin.write("  nmd = 1000,\n")
            mdin.write("  nmc = 1000,\n")
            mdin.write(f"  mcwatmask = \"{ref_lig_in_site_mask}\",\n")
            mdin.write("  mcligshift = 10,\n")
            mdin.write("  mcwatretry = 3000,\n")
            mdin.write("  mcresstr = \"WAT\",\n")
            mdin.write(f" \n mbar_states = {len(lambdas):02d}\n")
            mdin.write("  mbar_lambda =")
            for lam in lambdas:
                mdin.write(f" {lam:6.5f},")
            mdin.write("\n")
            # no need to run infe as everything is restrainted
            mdin.write("  infe = 0,\n")
            mdin.write(" /\n")
            _write_cmass_dump_block(mdin, istep1=int(ntwx))
        _apply_restraintmask_length_limit(
            out_path,
            prmtop_for_masks,
            cache_dir=cache_dir,
            cache_tag=f"{comp}-eq.in",
            cache_master=cache_master,
        )

        # end eq.in

        # write mdin-template
        n_steps_run = str(steps2)
        out_path = windows_dir / f"mdin-template"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            fout.write(f"! total_steps={steps2}\n")
            for line in fin:
                if "restraintmask" in line:
                    rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                    line = (
                        "restraintmask = "
                        f"'{_mask_with_added_component(rm, solvent_ligand_restraint_mask)}',\n"
                    )
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
            mdin.write(f" \n mbar_states = {len(lambdas):02d}\n")
            mdin.write("  mbar_lambda =")
            for lam in lambdas:
                mdin.write(f" {lam:6.5f},")
            mdin.write("\n")
            mdin.write("  infe = 0,\n")
            mdin.write(" /\n")
            _write_cmass_dump_block(mdin, istep1=int(ntwx))

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
        _apply_restraintmask_length_limit(
            out_path,
            prmtop_for_masks,
            cache_dir=cache_dir,
            cache_tag=f"{comp}-mdin-template",
            cache_master=cache_master,
        )

        # end mdin-template

        # mini.in
        with (
            template_mini.open("rt") as fin,
            (windows_dir / "mini.in").open("wt") as fout,
        ):
            for line in fin:
                line = _force_softcore_mini_constraints(line)
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
        # Keep infe disabled while nmropt/disang mirrors the cv restraints.
        infe_flag = 0
        mk1 = ref_resid
        template_mdin = amber_dir / "mdin-unorest-dd"
        template_mini = amber_dir / "mini-unorest-dd"

        n_steps_run = 20000
        eq_path = windows_dir / "eq.in"
        with template_mdin.open("rt") as fin, eq_path.open("wt") as fout:
            for line in fin:
                if "ntx = 5" in line:
                    line = "ntx = 1,\n"
                elif "irest" in line:
                    line = "irest = 0,\n"
                elif "dt = " in line:
                    line = "dt = 0.002,\n"
                elif "restraint_wt = " in line:
                    line = "restraint_wt = 0.2,\n"
                elif "restraintmask" in line:
                    rm = (
                        line.split("=", 1)[1]
                        .strip()
                        .rstrip(",")
                        .replace("'", "")
                    )
                    if rm == "":
                        line = f"restraintmask = '((@CA & {non_loop_mask}) | :{mol}) & !@H='\n"
                        #line = f"restraintmask = '(@CA | :{mol}) & !@H='\n"
                    else:
                        line = f"restraintmask = '((@CA & {non_loop_mask}) | :{mol} | {rm} ) & !@H='\n"
                        #line = f"restraintmask = '(@CA | :{mol}) | {rm} ) & !@H='\n"
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", n_steps_run)
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                )
                fout.write(line)
        with eq_path.open("a") as mdin:
            mdin.write(f" \n mbar_states = {len(lambdas)}\n")
            mdin.write("  mbar_lambda =")
            for lbd in lambdas:
                mdin.write(f" {lbd:6.5f},")
            mdin.write("\n")
            mdin.write(f"  infe = {infe_flag},\n")
            mdin.write(" /\n")
            _write_cmass_dump_block(mdin, istep1=int(ntwx))
        _apply_restraintmask_length_limit(
            eq_path,
            prmtop_for_masks,
            cache_dir=cache_dir,
            cache_tag=f"{comp}-eq.in",
            cache_master=cache_master,
        )

        # production template
        n_steps_run = str(steps2)
        out_path = windows_dir / "mdin-template"
        with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
            fout.write(f"! total_steps={steps2}\n")
            for line in fin:
                if "restraintmask" in line:
                    rm = (
                        line.split("=", 1)[1]
                        .strip()
                        .rstrip(",")
                        .replace("'", "")
                    )
                    line = (
                        "restraintmask = "
                        f"'{_mask_with_added_component(rm, solvent_ligand_restraint_mask)}',\n"
                    )
                line = (
                    line.replace("_temperature_", str(temperature))
                    .replace("_num-atoms_", str(vac_atoms))
                    .replace("_num-steps_", n_steps_run)
                    .replace("lbd_val", f"{float(weight):6.5f}")
                    .replace("mk1", str(mk1))
                )
                fout.write(line)

        with out_path.open("a") as mdin:
            mdin.write(f" \n mbar_states = {len(lambdas)}\n")
            mdin.write("  mbar_lambda =")
            for lbd in lambdas:
                mdin.write(f" {lbd:6.5f},")
            mdin.write("\n")
            mdin.write(f"  infe = {infe_flag},\n")
            mdin.write(" /\n")
            _write_cmass_dump_block(mdin, istep1=int(ntwx))
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
        _apply_restraintmask_length_limit(
            out_path,
            prmtop_for_masks,
            cache_dir=cache_dir,
            cache_tag=f"{comp}-mdin-template",
            cache_master=cache_master,
        )

        with (
            template_mini.open("rt") as fin,
            (windows_dir / "mini.in").open("wt") as fout,
        ):
            for line in fin:
                line = _force_softcore_mini_constraints(line)
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
            line = _force_fe_mini_constraints(line)
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
    # eqnpt-eq.in (longer equil with restraints on non-loop regions)
    non_loop_mask = _resolve_non_loop_mask(ctx, shift=2)
    eqnpt_src = amber_dir / (
        "eqnpt-uno-eq.in"
    )
    _sub_write(
        amber_dir / ("eqnpt-uno-eq.in"),
        windows_dir / "eqnpt_eq.in",
        {
            "_temperature_": str(temperature),
            "_lig_name_": mol,
            "_non_loop_": non_loop_mask,
        },
    )

    (windows_dir / "lambda.sch").write_text(
        "TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n"
    )

    logger.debug(
        f"[sim_files_z] wrote mdin/mini/eq inputs in {windows_dir} for comp='{comp}', win={win}, weight={weight:0.5f}"
    )


# ------------------------- FE component: l ------------------------- #


def _write_l_mdin_from_equil_template(
    *,
    src: Path,
    dst: Path,
    mol: str,
    replacements: dict[str, str],
    total_steps: int,
    ntwx: int,
    eq_seed: bool,
    rest_ramp: tuple[float, float] | None = None,
    cmass_dumpfreq: int | None = None,
) -> None:
    inserted_rest_weight = False
    with src.open("rt") as fin, dst.open("wt") as fout:
        if not eq_seed:
            fout.write(f"! total_steps={total_steps}\n")
        for line in fin:
            if eq_seed:
                if re.search(r"\bntx\s*=", line):
                    line = "  ntx = 1,\n"
                elif re.search(r"\birest\s*=", line):
                    line = "  irest = 0,\n"
                elif re.search(r"\bntwx\s*=", line):
                    line = f"  ntwx = {int(ntwx)},\n"
                elif re.search(r"\bntwr\s*=", line):
                    line = f"  ntwr = {int(ntwx)},\n"
                elif re.search(r"\bdt\s*=", line):
                    line = "  dt = 0.002,\n"
            if re.search(r"\bmcwat\s*=", line):
                line = "  mcwat = 0,\n"
            elif re.search(r"\bnstlim\s*=", line):
                line = f"  nstlim = {int(total_steps)},\n"
            elif re.search(r"\binfe\s*=", line):
                line = "  infe = 0,\n"
            if rest_ramp is not None and "type='DUMPFREQ'" in line and not inserted_rest_weight:
                fout.write(
                    " &wt type='REST', istep1=0, "
                    f"istep2={int(total_steps)}, value1={float(rest_ramp[0]):.8g}, "
                    f"value2={float(rest_ramp[1]):.8g}, /\n"
                )
                inserted_rest_weight = True
            if cmass_dumpfreq is not None and "type='DUMPFREQ'" in line:
                line = re.sub(
                    r"istep1\s*=\s*[^,/\s]+",
                    f"istep1={_component_l_cmass_dumpfreq(int(cmass_dumpfreq))}",
                    line,
                )
            line = (
                line.replace("_num-steps_", str(int(total_steps)))
                .replace("_lig_name_", mol)
                .replace("disang_file", "disang")
            )
            for key, value in replacements.items():
                line = line.replace(key, value)
            fout.write(line)


@register_sim_files("l")
def sim_files_l(ctx: BuildContext, lambdas: Sequence[float]) -> None:
    """
    Ligand conformational-restraint component.

    No alchemical masks are used. AMBER reads ligand torsion restraints through
    ``nmropt=1``/``DISANG``; RESTMBARAnalysis later evaluates the fixed-window
    restraint Hamiltonians from the generated cpptraj traces.
    """
    sim = ctx.sim
    mol = ctx.residue_name
    comp = ctx.comp
    win = ctx.win
    windows_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    temperature = sim.temperature
    n_steps = int(sim.dic_n_steps[comp])
    ntwx = int(sim.ntwx)
    lambdas = list(lambdas)
    if not lambdas:
        raise ValueError("[sim_files:l] component l requires a lambda schedule.")
    weight = float(lambdas[win if win != -1 else 0])
    mdin_replacements = {
        "_temperature_": str(temperature),
        "_cutoff_": str(sim.cut),
        "_gamma_ln_": str(sim.gamma_ln),
        "_p_coupling_": "3" if sim.membrane_simulation else "1",
        "_c_surften_": "3" if sim.membrane_simulation else "0",
        "_barostat_": str(sim.barostat),
        "_step_": str(sim.dt),
        "_ntpr_": str(sim.ntpr),
        "_ntwr_": str(sim.ntwr),
        "_ntwe_": str(sim.ntwe),
        "_ntwx_": str(ntwx),
        "_enable_mcwat_": "0",
        "_enable_infe_": "0",
    }

    prmtop_for_masks = _find_prmtop_for_masks(windows_dir)
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = win == -1

    # Generic equilibration/minimization stages keep nmropt=0; the seed ramp is
    # applied only in eq.in after the usual density/relaxation stages.
    _sub_write_fe_mini(amber_dir / "mini.in", windows_dir / "mini_eq.in", {"_lig_name_": mol})
    _sub_write_fe_mini(amber_dir / "mini.in", windows_dir / "mini.in", {"_lig_name_": mol})

    eqnpt0_src = amber_dir / (
        "eqnpt0.in" if sim.membrane_simulation else "eqnpt0-water.in"
    )
    eqnpt_src = amber_dir / (
        "eqnpt.in" if sim.membrane_simulation else "eqnpt-water.in"
    )
    eqnpt_eq_src = amber_dir / (
        "eqnpt-eq.in" if sim.membrane_simulation else "eqnpt-water-eq.in"
    )
    non_loop_mask = _resolve_non_loop_mask(ctx, shift=2)
    _sub_write(
        eqnpt0_src,
        windows_dir / "eqnpt0.in",
        {"_temperature_": str(temperature), "_lig_name_": mol},
    )
    _sub_write(
        eqnpt_src,
        windows_dir / "eqnpt.in",
        {"_temperature_": str(temperature), "_lig_name_": mol},
    )
    _sub_write(
        eqnpt_eq_src,
        windows_dir / "eqnpt_eq.in",
        {
            "_temperature_": str(temperature),
            "_lig_name_": mol,
            "_non_loop_": non_loop_mask,
        },
    )

    n_lambdas = max(2, len(lambdas))
    n_steps_run_per_lambda, _, _, n_steps_run = build_dyna_steps_run_per_lambda(
        n_lambdas=n_lambdas
    )
    if win != -1:
        n_steps_run_per_lambda = 10000
        n_steps_run = 10000

    _write_l_mdin_from_equil_template(
        src=amber_dir / "mdin-equil",
        dst=windows_dir / "eq.in",
        mol=mol,
        replacements=mdin_replacements,
        total_steps=n_steps_run,
        ntwx=n_steps_run_per_lambda,
        eq_seed=True,
        rest_ramp=(0.0, 1.0) if win == -1 else None,
        cmass_dumpfreq=n_steps_run_per_lambda,
    )
    _apply_restraintmask_length_limit(
        windows_dir / "eq.in",
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="l-eq.in",
        cache_master=cache_master,
    )

    _write_l_mdin_from_equil_template(
        src=amber_dir / "mdin-equil",
        dst=windows_dir / "mdin-template",
        mol=mol,
        replacements=mdin_replacements,
        total_steps=n_steps,
        ntwx=ntwx,
        eq_seed=False,
        cmass_dumpfreq=ntwx,
    )
    _apply_restraintmask_length_limit(
        windows_dir / "mdin-template",
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="l-mdin-template",
        cache_master=cache_master,
    )

    (windows_dir / "ligand_dihedral_schedule.json").write_text(
        json.dumps(
            {
                "component": "l",
                "window": win,
                "lambda": weight,
                "lambdas": [float(x) for x in lambdas],
                "lig_dihcf_force": float(getattr(sim, "lig_dihcf_force", 0.0) or 0.0),
                "amber_mbar": False,
                "analysis": "RESTMBARAnalysis evaluates restraint energies from restraints.in/cpptraj traces.",
            },
            indent=2,
        )
        + "\n"
    )

    logger.debug(
        f"[sim_files_l] wrote ligand-dihedral restraint inputs in {windows_dir} "
        f"for win={win}, weight={weight:0.5f}"
    )


# ------------------------- FE component: x ------------------------- #


@register_sim_files("x")
def sim_files_x(ctx: BuildContext, lambdas: Sequence[float]) -> None:
    """
    RBFE (x-component) sim_files.

    Uses mdin-ex / eqnpt-ex.in templates to build per-window inputs for
    relative transformations (ligand pair).
    """
    sim = ctx.sim
    comp = ctx.comp
    mol_ref = ctx.residue_name
    septop = str(getattr(sim, "fe_type", "")).lower() == "relative_septop"
    extra = ctx.extra or {}
    mol_alt = extra.get("residue_alt")
    non_loop_mask = _resolve_non_loop_mask(ctx, shift=3)


    if not mol_alt:
        raise ValueError(
            "RBFE component 'x' requires residue_alt in BuildContext.extra."
        )

    windows_dir = ctx.window_dir
    win = ctx.win
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = win == -1
    
    temperature = sim.temperature
    steps2 = sim.dic_n_steps[comp]
    ntwx = sim.ntwx

    weight = lambdas[ctx.win if ctx.win != -1 else 0]

    # Count atoms (vac or full)
    all_atoms = sim.all_atoms
    if all_atoms.lower() == "no":
        vac_pdb = windows_dir / "vac.pdb"
        if not vac_pdb.exists():
            raise FileNotFoundError(f"Missing required file: {vac_pdb}")
        vac_atoms = mda.Universe(vac_pdb.as_posix()).atoms.n_atoms
    else:
        full_pdb = windows_dir / "full.pdb"
        if not full_pdb.exists():
            raise FileNotFoundError(f"Missing required file: {full_pdb}")
        vac_atoms = mda.Universe(full_pdb.as_posix()).atoms.n_atoms
        vac_pdb = windows_dir / "vac.pdb"

    u = mda.Universe(vac_pdb.as_posix())
    mol_ref_ag = u.select_atoms(f'resname {mol_ref}')
    ref_resid = mol_ref_ag.resids[0]

    lig_in_site_mask = f':{int(ref_resid)},{int(ref_resid)+2}'
    mk1 = f':{int(ref_resid)},{int(ref_resid)+3}'
    mk2 = f':{int(ref_resid)+1},{int(ref_resid)+2}'
    # load scmask.json for scmk1, scmk2
    scmk_dict = json.loads((windows_dir.parent / "x-1" / "scmask.json").read_text())
    scmk1_cc_solvent_indices = scmk_dict["scmk1_cc_solvent_indices"]
    scmk2_cc_solvent_indices = scmk_dict["scmk2_cc_solvent_indices"]
    if septop:
        scmk1_cc_solvent_first_atom_mask = _first_residue_atom_mask(
            vac_pdb, resid=int(ref_resid) + 3
        )
        scmk2_cc_solvent_first_atom_mask = _first_residue_atom_mask(
            vac_pdb, resid=int(ref_resid) + 1
        )
    else:
        if scmk1_cc_solvent_indices:
            scmk1_cc_solvent_first_atom_mask = _first_absolute_atom_mask(
                scmk1_cc_solvent_indices
            )
        else:
            scmk1_cc_solvent_first_atom_mask = _first_residue_atom_mask(
                vac_pdb, resid=int(ref_resid) + 3
            )
        if scmk2_cc_solvent_indices:
            scmk2_cc_solvent_first_atom_mask = _first_absolute_atom_mask(
                scmk2_cc_solvent_indices
            )
        else:
            scmk2_cc_solvent_first_atom_mask = _first_residue_atom_mask(
                vac_pdb, resid=int(ref_resid) + 1
            )
    ligand_cc_solvent_first_atom_mask = (
        f"{scmk1_cc_solvent_first_atom_mask} | {scmk2_cc_solvent_first_atom_mask}"
    )
    scmk1_all_indice = scmk_dict['scmk1_all_indices']
    scmk2_all_indice = scmk_dict['scmk2_all_indices']
    scmk1_all_indice = indices_to_selection(scmk1_all_indice)
    scmk2_all_indice = indices_to_selection(scmk2_all_indice)

    #scmk1 = f'{scmk1_all_indice} & (!{scmk1_exclude_indice} | @H=)'
    #scmk2 = f'{scmk2_all_indice} & (!{scmk2_exclude_indice} | @H=)'
    if septop:
        scmk1 = scmk1_all_indice
        scmk2 = scmk2_all_indice
        # During pre-window equilibration, keep all four ligand copies close to
        # their staged poses. Production uses the lighter solvent-anchor mask
        # plus lambda-dependent Boresch restraints for the bound ligands.
        eq_ligand_restraint_mask = (
            f":{int(ref_resid)},{int(ref_resid)+1},"
            f"{int(ref_resid)+2},{int(ref_resid)+3}"
        )
    else:
        scmk1_exclude_indice = np.concatenate([
                        scmk1_cc_solvent_indices,
                        scmk_dict['scmk1_cc_site_indices']
        ])
        scmk2_exclude_indice = np.concatenate([
            scmk2_cc_solvent_indices,
            scmk_dict['scmk2_cc_site_indices']
        ])
        scmk_common_core_indice = np.concatenate([
            scmk1_exclude_indice,
            scmk2_exclude_indice,
        ])
        scmk1_exclude_indice = indices_to_selection(scmk1_exclude_indice)
        scmk2_exclude_indice = indices_to_selection(scmk2_exclude_indice)
        scmk_common_core_indice = indices_to_selection(scmk_common_core_indice)
        scmk1 = f'{scmk1_all_indice} & !{scmk1_exclude_indice}'
        scmk2 = f'{scmk2_all_indice} & !{scmk2_exclude_indice}'
        eq_ligand_restraint_mask = scmk_common_core_indice

    noshakemk = f':{int(ref_resid)},{int(ref_resid)+1},{int(ref_resid)+2},{int(ref_resid)+3}'

    amber_dir = ctx.amber_dir
    prmtop_for_masks = _find_prmtop_for_masks(windows_dir)
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1

    # optional extra restraints (only applied to mdin-template)
    extra_mask, extra_fc = _maybe_extra_mask(ctx, windows_dir, resid_shift=2)

    template_mdin = amber_dir / "mdin-ex"
    if not template_mdin.exists():
        raise FileNotFoundError(f"Missing RBFE mdin template: {template_mdin}")

    eq_path = windows_dir / "eq.in"
    n_steps_run_per_lambda, n_lambdas, dynlmb, n_steps_run = build_dyna_steps_run_per_lambda()
    if win != -1:
        n_steps_run = 10000
        n_steps_run_per_lambda = 10000

    with template_mdin.open("rt") as fin, eq_path.open("wt") as fout:
        for line in fin:
            if "ntx = 5" in line:
                line = "  ntx = 1,\n"
            elif "irest" in line:
                line = "  irest = 0,\n"
            # save every lambda value
            elif "ntwx = " in line:
                line = f"ntwx = {n_steps_run_per_lambda},\n"
            # write all atoms
            elif "ntwprt = " in line:
                line = f"\n"
            elif "dt = " in line:
                line = "  dt = 0.002,\n"
            elif "restraint_wt = " in line:
                line = f"  restraint_wt = 5,\n"
            elif "nmropt = " in line:
                line = "  nmropt = 1,\n"
            elif "restraintmask" in line:
                rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                if rm == "":
                    # restraining 1) non-loop C-alpha 2) ligand stabilizing masks/anchors
                    line = f"restraintmask = '((@CA & {non_loop_mask}) | ({eq_ligand_restraint_mask}) ) & !@H='\n"
                    #line = (
                    #    "restraintmask = "
                    #    f"'(@CA | ({scmk1_exclude_indice}) ) & !@H=',\n"
                    #)
                else:
                    line = f"restraintmask = '((@CA & {non_loop_mask}) | ({eq_ligand_restraint_mask}) | {rm} ) & !@H='\n"
                    #line = (
                    #    "restraintmask = "
                    #    f"'(@CA | ({scmk1_exclude_indice}) | {rm} ) & !@H=',\n"
                    #)
                if len(line) > 256:
                    logger.debug(
                        f"[restraintmask] Mask exceeds 256 chars in eq.in; conversion will be applied after write."
                    )
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-atoms_", str(vac_atoms))
                .replace("_num-steps_", str(n_steps_run))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("timk1", mk1)
                .replace("timk2", mk2)
                .replace("scmk1", scmk1)
                .replace("scmk2", scmk2)
                .replace("noshakemk", noshakemk)
            )
            fout.write(line)
            if septop and re.search(r"\bgti_vdw_exp\b", line):
                fout.write("  gti_bat_sc      = 1\n")
    with eq_path.open("a") as mdin:
        # also write velocity info
        mdin.write(f" ntwv = -1,\n")
        # run dynlmb
        if win == -1:
            mdin.write(f" dynlmb = {dynlmb},\n")
            mdin.write(f" ntave = {n_steps_run_per_lambda},\n")
        # run mcwat
        mdin.write(f"  mcwat = 1,\n")
        mdin.write(f"  nmd = 1000,\n")
        mdin.write(f"  nmc = 1000,\n")
        mdin.write(f"  mcwatmask = \"{lig_in_site_mask}\",\n")
        mdin.write(f"  mcligshift = 10,\n")
        mdin.write(f"  mcwatretry = 3000,\n")
        mdin.write(f"  mcresstr = \"WAT\",\n")
        mdin.write(f" \n mbar_states = {len(lambdas):02d}\n")
        mdin.write("  mbar_lambda =")
        for lam in lambdas:
            mdin.write(f" {lam:6.5f},")
        mdin.write("\n")
        # no need to run infe as everything is restrainted
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        eq_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="x-eq.in",
        cache_master=cache_master,
    )

    # --- mdin-template (production) ---
    out_path = windows_dir / "mdin-template"
    with template_mdin.open("rt") as fin, out_path.open("wt") as fout:
        fout.write(f"! total_steps={steps2}\n")
        for line in fin:
            if "restraintmask" in line:
                rm = line.split("=", 1)[1].strip().rstrip(",").replace("'", "")
                line = (
                    "restraintmask = "
                    f"'{_mask_with_added_component(rm, ligand_cc_solvent_first_atom_mask)}',\n"
                )
            line = (
                line.replace("_temperature_", str(temperature))
                .replace("_num-atoms_", str(vac_atoms))
                .replace("_num-steps_", str(steps2))
                .replace("lbd_val", f"{float(weight):6.5f}")
                .replace("timk1", str(mk1))
                .replace("timk2", str(mk2))
                .replace("scmk1", scmk1)
                .replace("scmk2", scmk2)
                .replace("noshakemk", noshakemk)
            )
            fout.write(line)
            if septop and re.search(r"\bgti_vdw_exp\b", line):
                fout.write("  gti_bat_sc      = 1\n")
    with out_path.open("a") as mdin:
        mdin.write(f" \n  mbar_states = {len(lambdas):02d}\n")
        mdin.write("  mbar_lambda =")
        for lam in lambdas:
            mdin.write(f" {lam:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
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
    _apply_restraintmask_length_limit(
        out_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="x-mdin-template",
        cache_master=cache_master,
    )

    # --- mini.in / mini_eq.in ---
    with (amber_dir / "mini-ex").open("rt") as fin, (windows_dir / "mini.in").open(
        "wt"
    ) as fout:
        for line in fin:
            line = _force_x_mini_constraints(line)
            line = (
                line.replace("_lig1_name_", mol_ref)
                .replace("_lig2_name_", mol_alt)
                .replace("timk1", mk1)
                .replace("timk2", mk2)
                .replace("scmk1", scmk1)
                .replace("scmk2", scmk2)
                .replace("noshakemk", noshakemk)
                .replace('lbd_val', f"{float(weight):6.5f}")
            )
            fout.write(line)
            if septop and re.search(r"\bgti_vdw_exp\b", line):
                fout.write("  gti_bat_sc      = 1\n")

    with (amber_dir / "mini-ex").open("rt") as fin, (windows_dir / "mini_eq.in").open(
        "wt"
    ) as fout:
        for line in fin:
            line = _force_fe_mini_constraints(line)
            line = (
                line.replace("_lig1_name_", mol_ref)
                .replace("_lig2_name_", mol_alt)
                .replace("timk1", str(mk1))
                .replace("timk2", str(mk2))
                .replace("scmk1", scmk1)
                .replace("scmk2", scmk2)
                .replace("noshakemk", noshakemk)
                .replace('lbd_val', f"{float(weight):6.5f}")
            )
            fout.write(line)
            if septop and re.search(r"\bgti_vdw_exp\b", line):
                fout.write("  gti_bat_sc      = 1\n")

    if septop:
        (windows_dir / "lambda.sch").write_text(
            "TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n"
        )

    logger.debug(
        f"[sim_files_x] wrote mdin/mini/eq inputs in {windows_dir} "
        f"for comp='x', win={ctx.win}, weight={weight:0.5f}"
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
    cache_dir = windows_dir.parent / ".restraintmask_cache"
    cache_master = ctx.win == -1

    temperature = sim.temperature
    n_steps = sim.dic_n_steps["y"]
    ntwx = sim.ntwx

    weight = lambdas[ctx.win if ctx.win != -1 else 0]
    mk1 = 2  # ligand-only marker convention
    vac_pdb = windows_dir / "vac.pdb"
    if not vac_pdb.exists():
        raise FileNotFoundError(f"Missing required file: {vac_pdb}")
    ligand_first_atom_mask = _first_residue_atom_mask(vac_pdb, resname=mol)

    amber_dir = ctx.amber_dir
    prmtop_for_masks = _find_prmtop_for_masks(windows_dir)

    # mini.in from ligand template
    with (
        (amber_dir / "mini-unorest-lig").open("rt") as fin,
        (windows_dir / "mini.in").open("wt") as fout,
    ):
        for line in fin:
            line = _force_softcore_mini_constraints(line)
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
            line = _force_fe_mini_constraints(line)
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
        mdin.write(f" \n  mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        eq_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="y-eq.in",
        cache_master=cache_master,
    )

    # production template (single long segment)
    out_path = windows_dir / "mdin-template"
    with template.open("rt") as fin, out_path.open("wt") as fout:
        fout.write(f"! total_steps={n_steps}\n")
        for line in fin:
            if "nmropt = " in line:
                line = "  nmropt = 0,\n"
            elif "restraintmask" in line:
                rm = (
                    line.split("=", 1)[1]
                    .strip()
                    .rstrip(",")
                    .replace("'", "")
                )
                line = (
                    "  restraintmask = "
                    f"'{_mask_with_added_component(rm, ligand_first_atom_mask)}',\n"
                )
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
        mdin.write(f" \n mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        out_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="y-mdin-template",
        cache_master=cache_master,
    )

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
    prmtop_for_masks = _find_prmtop_for_masks(windows_dir)

    # mini.in from ligand template
    with (
        (amber_dir / "mini-unorest-vacuum").open("rt") as fin,
        (windows_dir / "mini_eq.in").open("wt") as fout,
    ):
        for line in fin:
            line = _force_softcore_mini_constraints(line)
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
        mdin.write(f" \n mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        eq_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="m-eq.in",
        cache_master=cache_master,
    )

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
        mdin.write(f" \n mbar_states = {len(lambdas)}\n")
        mdin.write("  mbar_lambda =")
        for lbd in lambdas:
            mdin.write(f" {lbd:6.5f},")
        mdin.write("\n")
        mdin.write("  infe = 0,\n")
        mdin.write(" /\n")
        _write_cmass_dump_block(mdin, istep1=int(ntwx))
    _apply_restraintmask_length_limit(
        out_path,
        prmtop_for_masks,
        cache_dir=cache_dir,
        cache_tag="m-mdin-template",
        cache_master=cache_master,
    )

    logger.debug(
        f"[sim_files_m] wrote mdin/mini/eq inputs in {windows_dir} for comp='m', weight={weight:0.5f}"
    )
