from __future__ import annotations

import shutil
from pathlib import Path
from loguru import logger

from batter.config.simulation import SimulationConfig
from batter._internal.builders.interfaces import BuildContext
from batter._internal.templates import RUN_FILES_DIR as run_files_orig

def write_equil_run_files(ctx: BuildContext, stage: str) -> None:
    """
    Port of `_run_files` for equilibration.

    Copies run scripts and substitutes variables
    like RANGE, STAGE, POSE, SYSTEMNAME, PARTITIONNAME, etc.
    """
    sim = ctx.sim
    ligand_name = ctx.ligand
    work = Path(ctx.working_dir)
    hmr = ctx.sim.hmr

    logger.debug(f"[Equil] Creating run scripts in {work}")

    # Copy templates from internal RUN_FILES_DIR
    for template_name in ["check_run.bash", "check_penetration.py", "run-equil.bash", "SLURMM-Am"]:
        src = run_files_orig / template_name
        dst = work / (
            "run-local.bash" if template_name == "run-equil.bash" else
            "SLURMM-run" if template_name == "SLURMM-Am" else
            template_name
        )
        if not src.exists():
            logger.warning(f"[Equil] Missing run template {src.name}; creating placeholder.")
            dst.write_text(f"# Missing template: {src}\n")
        else:
            shutil.copy2(src, dst)

        text = dst.read_text()
        text = (
            text.replace("RANGE", str(sim.rng))
                .replace("STAGE", stage)
                .replace("POSE", ligand_name)
                .replace("SYSTEMNAME", sim.system_name)
                .replace("PARTITIONNAME", sim.partition)
        )

        if hmr:
            text = text.replace("full.prmtop", "full.hmr.prmtop")
        else:
            text = text.replace("full.prmtop", "full.prmtop")
        dst.write_text(text)

        try:
            dst.chmod(0o755)
        except Exception as e:
            logger.debug(f"chmod +x failed for {dst}: {e}")

    logger.debug(f"[Equil] Run scripts ready at {work}")

def _dst_window_dir(ctx: BuildContext) -> Path:
    """Return the per-window destination directory (e.g. .../fe-1 or fe-03)."""
    tag = f"{ctx.comp}-{ctx.win if ctx.win != -1 else 1}"
    return ctx.working_dir / tag


def write_fe_run_file(
    ctx: BuildContext,
    num_sim: int,
    lambdas: Sequence[float],
    *,
    partition_override: Optional[str] = None,
) -> None:
    """
    Materialize run scripts for a given component/window:

    Reads templates from:   <working_dir>/<comp>_run_files/
      - check_run.bash
      - run-local.bash
      - SLURMM-Am

    Writes to the window dir: <working_dir>/<comp>-<win or 1>/
      - check_run.bash
      - run-local.bash     (FERANGE/NWINDOWS/COMPONENT replaced)
      - SLURMM-run         (STAGE/POSE/SYSTEMNAME/PARTITIONNAME replaced)

    Makes outputs executable.
    """
    # --- source/dest paths
    src_dir = ctx.working_dir / f"{ctx.comp}_run_files"
    dst_dir = _dst_window_dir(ctx)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # templates (fail clearly if missing)
    tpl_check = src_dir / "check_run.bash"
    tpl_local = src_dir / "run-local.bash"
    tpl_slurm = src_dir / "SLURMM-Am"

    for p in (tpl_check, tpl_local, tpl_slurm):
        if not p.exists():
            raise FileNotFoundError(f"Missing run template: {p}")

    # replacements
    pose = ctx.ligand
    comp = ctx.comp
    win_idx = ctx.win if ctx.win != -1 else 0
    n_windows = len(lambdas)
    system_name = getattr(ctx.sim, "system_name", "system")
    partition = (
        partition_override
        or getattr(ctx.sim, "partition", None)
        or getattr(ctx.sim, "queue", None)
        or "normal"
    )

    # -------- check_run.bash (verbatim copy)
    out_check = dst_dir / "check_run.bash"
    out_check.write_text(tpl_check.read_text())
    os.chmod(out_check, 0o755)

    # -------- run-local.bash (replace FERANGE/NWINDOWS/COMPONENT)
    out_local = dst_dir / "run-local.bash"
    txt = tpl_local.read_text()
    txt = (
        txt.replace("FERANGE", str(num_sim))
           .replace("NWINDOWS", str(n_windows))
           .replace("COMPONENT", comp)
    )
    out_local.write_text(txt)
    os.chmod(out_local, 0o755)

    # -------- SLURMM-run (from SLURMM-Am template)
    out_slurm = dst_dir / "SLURMM-run"
    stxt = tpl_slurm.read_text()
    stxt = (
        stxt.replace("STAGE", pose)
            .replace("POSE", f"{comp}{int(win_idx):02d}")
            .replace("SYSTEMNAME", system_name)
            .replace("PARTITIONNAME", partition)
    )
    out_slurm.write_text(stxt)
    os.chmod(out_slurm, 0o755)

    logger.debug(
        f"[runfiles] wrote run scripts → {dst_dir} "
        f"(FERANGE={num_sim}, NWINDOWS={n_windows}, COMPONENT={comp}, PARTITION={partition})"
    )