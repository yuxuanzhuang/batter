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
            text = text.replace("full.prmtop', 'full.hmr.prmtop")
        else:
            text = text.replace("full.prmtop', 'full.prmtop")
        dst.write_text(text)

        try:
            dst.chmod(0o755)
        except Exception as e:
            logger.debug(f"chmod +x failed for {dst}: {e}")

    logger.debug(f"[Equil] Run scripts ready at {work}")