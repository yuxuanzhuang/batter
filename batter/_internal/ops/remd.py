from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from loguru import logger

from batter.config.simulation import SimulationConfig
from batter.utils.components import COMPONENTS_DICT

# Default REMD exchange settings to mirror legacy batching behaviour.
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


def _inject_numexchg(lines: List[str]) -> tuple[List[str], bool]:
    """
    Insert numexchg/bar_intervall into the &cntrl block if missing.
    """
    out: List[str] = []
    in_cntrl = False
    inserted = False

    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("&cntrl"):
            in_cntrl = True
        if "numexchg" in lower:
            inserted = True

        if in_cntrl and lower == "/":
            if not inserted:
                out.append(f"  numexchg = {NUMEXCHG_DEFAULT},\n")
                out.append(f"  bar_intervall = {BAR_INTERVAL_DEFAULT},\n")
                inserted = True
            in_cntrl = False

        out.append(line)

    return out, inserted


def patch_mdin_file(mdin_path: Path, prefix: str, *, add_numexchg: bool) -> bool:
    """
    Update mdin-like files so embedded file paths are relative to ``prefix``.

    When ``add_numexchg`` is True, numexchg/bar_intervall are injected into
    the &cntrl block if not already present.
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
        new_lines.append(line)

    if add_numexchg:
        new_lines, inserted = _inject_numexchg(new_lines)
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


def patch_component_inputs(comp_dir: Path, comp: str, *, add_numexchg: bool) -> List[Path]:
    """
    Patch all mdin-like files for REMD execution under ``comp_dir``.
    """
    patched: List[Path] = []
    for window_dir in _component_window_dirs(comp_dir, comp):
        prefix = window_dir.relative_to(comp_dir).as_posix()
        for mdin in sorted(window_dir.glob("*.in")):
            is_production = mdin.name.startswith("mdin-")
            if patch_mdin_file(mdin, prefix, add_numexchg=add_numexchg and is_production):
                patched.append(mdin)
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

    group_dir = comp_dir / "groupfiles"
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
        curr = f"mdin-{stage:02d}"
        prev = "mini.in" if stage == 0 else f"mdin-{stage - 1:02d}"
        return (
            f"-O -i {win}/{curr} -p {prmtop_path} -c {win}/{prev}.rst7 "
            f"-o {win}/{curr}.out -r {win}/{curr}.rst7 -x {win}/{curr}.nc "
            f"-ref {win}/mini.in.rst7 -inf {win}/mdinfo -l {win}/{curr}.log "
            f"-e {win}/{curr}.mden{allow_small_box}\n"
        )

    out_files: List[Path] = []
    mini_path = group_dir / f"{comp}_mini.in.groupfile"
    _write_groupfile(mini_path, n_windows, mini_line)
    out_files.append(mini_path)

    prod0_path = group_dir / f"{comp}_mdin.in.groupfile"
    _write_groupfile(prod0_path, n_windows, lambda idx: prod_line(idx, 0))
    out_files.append(prod0_path)

    for stage in range(1, num_extends + 1):
        path = group_dir / f"{comp}_mdin.in.stage{stage:02d}.groupfile"
        _write_groupfile(path, n_windows, lambda idx, st=stage: prod_line(idx, st))
        out_files.append(path)

    logger.debug(f"[remd] Wrote {len(out_files)} groupfiles under {group_dir}")
    return out_files


def prepare_remd_component(
    comp_dir: Path, comp: str, sim: SimulationConfig, n_windows: int
) -> List[Path]:
    """
    Patch mdin files and emit groupfiles so REMD can run from ``comp_dir``.
    """
    if str(sim.remd).lower() != "yes":
        return []

    add_numexchg = comp in COMPONENTS_DICT["dd"]
    patch_component_inputs(comp_dir, comp, add_numexchg=add_numexchg)
    return write_remd_groupfiles(comp_dir, comp, sim, n_windows)
