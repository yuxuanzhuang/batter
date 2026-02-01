"""Batch-mode helpers for FE production runs."""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from batter._internal.ops.remd import patch_batch_component_inputs

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "remd_run_files"
RUN_TEMPLATE = TEMPLATE_DIR / "run-local-batch.bash"
CHECK_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "templates" / "run_files_orig" / "check_run.bash"
)


def prepare_batch_component(
    comp_dir: Path, comp: str, n_windows: int, *, hmr: bool = True
) -> List[Path]:
    """
    Patch batch mdin templates and write helper scripts under ``comp_dir``.

    This mirrors the REMD preparation step, but targets normal batch runs.
    """
    out: List[Path] = []
    comp_dir.mkdir(parents=True, exist_ok=True)

    if RUN_TEMPLATE.exists():
        text = RUN_TEMPLATE.read_text()
        text = text.replace("COMPONENT", comp).replace("NWINDOWS", str(n_windows))
        if hmr:
            text = text.replace("full.prmtop", "full.hmr.prmtop")
        else:
            text = text.replace("full.hmr.prmtop", "full.prmtop")
        run_local = comp_dir / "run-local-batch.bash"
        run_local.write_text(text)
        try:
            run_local.chmod(0o755)
        except Exception:
            pass
        out.append(run_local)
    else:
        logger.warning(f"[batch] Missing run-local template at {RUN_TEMPLATE}")

    if CHECK_TEMPLATE.exists():
        check_dst = comp_dir / "check_run.bash"
        check_dst.write_text(CHECK_TEMPLATE.read_text())
        try:
            check_dst.chmod(0o755)
        except Exception:
            pass
        out.append(check_dst)
    else:
        logger.warning(f"[batch] Missing check_run template at {CHECK_TEMPLATE}")

    out.extend(patch_batch_component_inputs(comp_dir, comp))
    return out
