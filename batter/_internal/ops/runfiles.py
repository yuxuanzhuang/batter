from __future__ import annotations

import shutil
from pathlib import Path
from loguru import logger

from batter.config.simulation import SimulationConfig
from batter._internal.builders.interfaces import BuildContext
from batter._internal.templates import RUN_FILES_DIR as run_files_orig


def write_run_files(
    window_dir: Path,
    *,
    sim: SimulationConfig,
    comp: str,
    win: int,
) -> None:
    """
    Create run files (SLURM or local) for this window.

    Parameters
    ----------
    window_dir : Path
        Directory for this simulation window (e.g. work/.../fe/windows/001/).
    sim : SimulationConfig
        The simulation configuration.
    comp : str
        Component name (e.g. 'q', 'n', 'm', 'e').
    win : int
        Window index.
    """
    sh = window_dir / "run.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# Component: {comp}  Window: {win}",
        f"# Partition: {sim.partition}",
        f"# Temperature: {sim.temperature}",
        "",
        "# Launch command (placeholder)",
        f"pmemd.cuda -O -i mdin -p full.hmr.prmtop -c inpcrd -r out.rst7 -x traj.nc -o mdout.log",
    ]
    sh.write_text("\n".join(lines) + "\n")
    try:
        sh.chmod(0o755)
    except Exception as e:
        logger.debug(f"chmod +x failed for {sh}: {e}")
    logger.info(f"[RunFiles] Created run script for {comp} window {win}: {sh}")


def write_equil_run_files(ctx: BuildContext, stage: str) -> None:
    """
    Port of `_run_files` for equilibration.

    Copies run scripts and substitutes variables
    like RANGE, STAGE, POSE, SYSTEMNAME, PARTITIONNAME, etc.
    """
    sim = ctx.sim
    ligand_name = ctx.ligand
    work = Path(ctx.working_dir)
    run_dir = work / "run_files"
    run_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"[Equil] Creating run scripts in {run_dir}")

    # Copy templates from internal RUN_FILES_DIR
    for template_name in ["check_run.bash", "check_penetration.py", "run-equil.bash", "SLURMM-Am"]:
        src = run_files_orig / template_name
        dst = run_dir / (
            "run-local.bash" if template_name == "run-equil.bash" else
            "SLURMM-run" if template_name == "SLURMM-Am" else
            template_name
        )
        if not src.exists():
            logger.warning(f"[Equil] Missing run template {src.name}; creating placeholder.")
            dst.write_text(f"# Missing template: {src}\n")
        else:
            shutil.copy2(src, dst)

        # Perform substitutions
        text = dst.read_text()
        text = (
            text.replace("RANGE", str(sim.rng))
            .replace("STAGE", stage)
            .replace("POSE", ligand_name)
            .replace("SYSTEMNAME", sim.system_name)
            .replace("PARTITIONNAME", sim.partition)
        )
        dst.write_text(text)

        try:
            dst.chmod(0o755)
        except Exception as e:
            logger.debug(f"chmod +x failed for {dst}: {e}")

    logger.info(f"[Equil] Run scripts ready at {run_dir}")