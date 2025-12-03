from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Dict, Optional

from batter.utils.slurm_templates import render_slurm_body


def _tpl_path(filename: str) -> Path:
    pkg = "batter._internal.templates.batch_run"
    with resources.as_file(resources.files(pkg) / filename) as p:
        return Path(p)


def render_batch_slurm_script(
    *,
    batch_root: Path,
    target_dir: Path,
    run_script: str,
    env: Optional[Dict[str, str]],
    system_name: str,
    stage: str,
    pose: str,
    header_root: Optional[Path],
) -> Path:
    """Render a batch-mode SLURM script that enters ``target_dir`` and runs ``run_script``."""
    batch_root.mkdir(parents=True, exist_ok=True)
    safe_name = pose.replace("/", "_")
    out = batch_root / f"{stage}_{safe_name}_batch.sh"

    env_lines = []
    for k, v in (env or {}).items():
        env_lines.append(f"export {k}={v}")
    env_block = "\n".join(env_lines) if env_lines else ":"

    out_body = out.with_suffix(out.suffix + ".body")
    body_text = render_slurm_body(
        _tpl_path("SLURMM-BATCH.body"),
        {
            "SYSTEMNAME": system_name,
            "STAGE": stage,
            "POSE": safe_name,
            "TARGET_DIR": str(target_dir.resolve()),
            "ENV_EXPORT": env_block,
            "RUN_SCRIPT": run_script,
        },
    )
    out_body.write_text(body_text)
    try:
        out_body.chmod(0o644)
    except Exception:
        pass
    return out
