from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from loguru import logger

from batter.config.simulation import SimulationConfig
from batter.utils.components import COMPONENTS_DICT
from batter.utils.slurm_templates import render_slurm_with_header_body

# Default REMD exchange settings to mirror legacy batching behaviour (overridden by config).
NUMEXCHG_DEFAULT = 3000
BAR_INTERVAL_DEFAULT = 100


def _prefix_path(value: str, prefix: str) -> str:
    """
    Prefix ``value`` with ``prefix`` unless it is already absolute or prefixed.
    """
    clean = value.strip()
    if not clean:
        return clean
    if clean.startswith(("/", "./", "../", f"{prefix}/")):
        return clean
    return f"{prefix}/{clean}"


def _rewrite_path_line(line: str, key: str, prefix: str) -> tuple[str, bool]:
    """
    Rewrite file path parameters (cv_file/output_file/DISANG) to live under ``prefix``.
    """
    lower = line.lower()
    if key not in lower:
        return line, False

    before, sep, after = line.partition("=")
    if not sep:
        return line, False

    leading_space = " " if after.startswith(" ") else ""
    tail = after.strip()
    suffix = "," if tail.endswith(",") else ""
    raw_value = tail.rstrip(",").strip()
    quoted = raw_value.startswith("'") or raw_value.startswith('"')
    value = raw_value.strip("'\"")

    new_value = _prefix_path(value, prefix)
    if new_value == value:
        return line, False

    val_text = f"'{new_value}'" if quoted else new_value
    new_line = f"{before}{sep}{leading_space}{val_text}{suffix}\n"
    return new_line, True


def _inject_numexchg(lines: List[str], numexchg: int | None) -> tuple[List[str], bool]:
    """
    Insert numexchg/bar_intervall into the &cntrl block if missing.
    """
    out: List[str] = []
    in_cntrl = False
    inserted = False
    num_val = numexchg or NUMEXCHG_DEFAULT

    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("&cntrl"):
            in_cntrl = True
        if "numexchg" in lower:
            inserted = True

        if in_cntrl and lower == "/":
            if not inserted:
                out.append(f"  numexchg = {num_val},\n")
                out.append(f"  bar_intervall = {BAR_INTERVAL_DEFAULT},\n")
                inserted = True
            in_cntrl = False

        out.append(line)

    return out, inserted


def patch_mdin_file(
    mdin_path: Path,
    prefix: str,
    *,
    add_numexchg: bool,
    remd_nstlim: int | None = None,
    remd_numexchg: int | None = None,
) -> bool:
    """
    Update mdin-like files so embedded file paths are relative to ``prefix``.

    When ``add_numexchg`` is True, numexchg/bar_intervall are injected into
    the &cntrl block if not already present. When ``remd_nstlim`` is set,
    override nstlim to this value.
    """
    try:
        lines = mdin_path.read_text().splitlines(keepends=True)
    except FileNotFoundError:
        return False

    changed = False
    new_lines: List[str] = []

    for line in lines:
        lower = line.lower().strip()
        if "cv_file" in lower:
            line, did_change = _rewrite_path_line(line, "cv_file", prefix)
            changed = changed or did_change
        elif "output_file" in lower:
            line, did_change = _rewrite_path_line(line, "output_file", prefix)
            changed = changed or did_change
        elif lower.startswith("disang"):
            line, did_change = _rewrite_path_line(line, "disang", prefix)
            changed = changed or did_change
        elif remd_numexchg is not None and "numexchg" in lower:
            line = f"  numexchg = {remd_numexchg},\n"
            changed = True
        elif remd_numexchg is not None and "bar_intervall" in lower:
            line = f"  bar_intervall = {BAR_INTERVAL_DEFAULT},\n"
            changed = True
        elif remd_nstlim is not None and "nstlim" in lower:
            line = f"  nstlim = {remd_nstlim},\n"
            changed = True
        new_lines.append(line)

    if add_numexchg:
        new_lines, inserted = _inject_numexchg(new_lines, remd_numexchg)
        changed = changed or inserted

    if changed:
        mdin_path.write_text("".join(new_lines))
    return changed


def _component_window_dirs(comp_dir: Path, comp: str) -> Iterable[Path]:
    """
    Yield window directories under a component folder (comp-1, comp00, comp01, ...).
    """
    if not comp_dir.exists():
        return []
    for p in sorted(comp_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name == f"{comp}-1":
            yield p
            continue
        if not name.startswith(comp):
            continue
        tail = name[len(comp) :].lstrip("-")
        if tail.isdigit():
            yield p


def patch_component_inputs(
    comp_dir: Path,
    comp: str,
    sim: SimulationConfig,
    *,
    add_numexchg: bool,
) -> List[Path]:
    """
    Prepare REMD-specific mdin copies:
      - mdin-00      → mdin-00-remd  (first stage)
      - mdin-01      → mdin-remd     (re-used for all subsequent stages)

    Only production windows (comp00, comp01, ...) are touched; the scaffold
    window (comp-1) is left intact.
    """
    patched: List[Path] = []
    for window_dir in _component_window_dirs(comp_dir, comp):
        if window_dir.name == f"{comp}-1":
            continue
        prefix = window_dir.relative_to(comp_dir).as_posix()
        nstlim = getattr(sim, "remd_nstlim", None)
        nstlim_val = int(nstlim) if nstlim else None
        numexchg_val = getattr(sim, "remd_numexchg", None)
        numexchg_val = int(numexchg_val) if numexchg_val else None
        src00 = window_dir / "mdin-00"
        dst00 = window_dir / "mdin-00-remd"
        if src00.exists():
            dst00.write_text(src00.read_text())
            if patch_mdin_file(
                dst00,
                prefix,
                add_numexchg=add_numexchg,
                remd_nstlim=nstlim_val,
                remd_numexchg=numexchg_val,
            ):
                patched.append(dst00)

        src01 = window_dir / "mdin-01"
        dst01 = window_dir / "mdin-remd"
        if src01.exists():
            dst01.write_text(src01.read_text())
            if patch_mdin_file(
                dst01,
                prefix,
                add_numexchg=add_numexchg,
                remd_nstlim=nstlim_val,
                remd_numexchg=numexchg_val,
            ):
                patched.append(dst01)

    if patched:
        logger.debug(f"[remd] Patched {len(patched)} mdin files under {comp_dir}")
    return patched


def _write_groupfile(path: Path, n_windows: int, builder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for idx in range(n_windows):
            f.write(f"# remd {path.stem} window {idx}\n")
            f.write(builder(idx))


def write_remd_groupfiles(
    comp_dir: Path, comp: str, sim: SimulationConfig, n_windows: int
) -> List[Path]:
    """
    Emit REMD groupfiles for minimisation and FE production under ``comp_dir``.
    """
    if n_windows <= 0:
        return []

    group_dir = comp_dir / "remd"
    prmtop = "full.hmr.prmtop" if str(sim.hmr).lower() == "yes" else "full.prmtop"
    prmtop_path = f"{comp}-1/{prmtop}"
    eq_restart = f"{comp}-1/eqnpt04.rst7"
    num_extends = int(getattr(sim, "num_fe_extends", 0))
    allow_small_box = " -AllowSmallBox" if comp == "m" else ""

    def _window_name(idx: int) -> str:
        return f"{comp}{idx:02d}"

    def mini_line(idx: int) -> str:
        win = _window_name(idx)
        return (
            f"-O -i {win}/mini.in -p {prmtop_path} -c {eq_restart} "
            f"-o {win}/mini.in.out -r {win}/mini.in.rst7 -x {win}/mini.in.nc "
            f"-ref {eq_restart} -inf {win}/mini.in.mdinfo -l {win}/mini.in.log "
            f"-e {win}/mini.in.mden{allow_small_box}\n"
        )

    def prod_line(idx: int, stage: int) -> str:
        win = _window_name(idx)
        if stage == 0:
            inp = "mdin-00-remd"
            curr = "mdin-00"
            prev = "mini.in"
        else:
            inp = "mdin-remd"
            curr = f"mdin-{stage:02d}"
            prev = f"mdin-{stage - 1:02d}"
        return (
            f"-O -i {win}/{inp} -p {prmtop_path} -c {win}/{prev}.rst7 "
            f"-o {win}/{curr}.out -r {win}/{curr}.rst7 -x {win}/{curr}.nc "
            f"-ref {win}/mini.in.rst7 -inf {win}/mdinfo -l {win}/{curr}.log "
            f"-e {win}/{curr}.mden{allow_small_box}\n"
        )

    out_files: List[Path] = []
    mini_path = group_dir / "mini.in.remd.groupfile"
    _write_groupfile(mini_path, n_windows, mini_line)
    out_files.append(mini_path)

    prod0_path = group_dir / "mdin.in.remd.groupfile"
    _write_groupfile(prod0_path, n_windows, lambda idx: prod_line(idx, 0))
    out_files.append(prod0_path)

    for stage in range(1, num_extends + 1):
        path = group_dir / f"mdin.in.stage{stage:02d}.remd.groupfile"
        _write_groupfile(path, n_windows, lambda idx, st=stage: prod_line(idx, st))
        out_files.append(path)

    logger.debug(f"[remd] Wrote {len(out_files)} groupfiles under {group_dir}")
    return out_files


def write_remd_run_scripts(
    comp_dir: Path,
    comp: str,
    sim: SimulationConfig,
    n_windows: int,
    *,
    partition: str | None = None,
) -> List[Path]:
    """
    Write REMD run helper scripts (local + SLURM) into the component folder.
    """
    out: List[Path] = []
    comp_dir.mkdir(parents=True, exist_ok=True)

    num_extends = int(getattr(sim, "num_fe_extends", 0))
    gpus = n_windows if n_windows > 0 else 1

    def _copy_template(src: Path, dst: Path, repl: dict[str, str], override_text: str | None = None) -> None:
        text = override_text if override_text is not None else src.read_text()
        for k, v in repl.items():
            text = text.replace(k, v)
        dst.write_text(text)

    run_local_tpl = RUN_TPL["local"]
    run_local = comp_dir / "run-local-remd.bash"
    _copy_template(
        run_local_tpl,
        run_local,
        {"COMPONENT": comp, "NWINDOWS": str(n_windows), "FERANGE": str(num_extends)},
    )
    run_local.chmod(0o755)
    out.append(run_local)

    slurm_body = comp_dir / "SLURMM-BATCH-remd.body"
    body_text = render_slurm_body(
        TEMPLATE_DIR / "SLURMM-BATCH-remd.body",
        {
            "COMPONENT": comp,
            "NWINDOWS": str(gpus),
            "FERANGE": str(num_extends),
        },
    )
    slurm_body.write_text(body_text)
    slurm_body.chmod(0o644)
    out.append(slurm_body)

    # copy check_run.bash alongside for failure checks
    check_src = Path(__file__).resolve().parent.parent / "templates" / "run_files_orig" / "check_run.bash"
    check_dst = comp_dir / "check_run.bash"
    if check_src.exists():
        check_dst.write_text(check_src.read_text())
        out.append(check_dst)

    return out


def prepare_remd_component(
    comp_dir: Path,
    comp: str,
    sim: SimulationConfig,
    n_windows: int,
    *,
    partition: str | None = None,
) -> List[Path]:
    """
    Patch mdin files and emit groupfiles so REMD can run from ``comp_dir``.
    """
    if str(sim.remd).lower() != "yes":
        return []

    add_numexchg = comp in COMPONENTS_DICT["dd"]
    patch_component_inputs(comp_dir, comp, sim, add_numexchg=add_numexchg)

    # lambda schedule needed for amber REMD runs
    lambda_tpl = RUN_TPL["lambda"]
    lambda_dst = comp_dir / "lambda.sch"
    if lambda_tpl.exists():
        lambda_dst.write_text(lambda_tpl.read_text())
    else:
        lambda_dst.write_text("TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n")

    scripts = write_remd_groupfiles(comp_dir, comp, sim, n_windows)
    scripts.extend(
        write_remd_run_scripts(
            comp_dir, comp, sim, n_windows, partition=partition
        )
    )
    return scripts


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "remd_run_files"
RUN_TPL = {
    "local": TEMPLATE_DIR / "run-local-remd.bash",
    "slurm": TEMPLATE_DIR / "SLURMM-BATCH-remd",
    "lambda": TEMPLATE_DIR / "lambda.sch",
}
