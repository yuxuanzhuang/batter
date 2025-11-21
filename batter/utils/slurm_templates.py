from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from loguru import logger


def render_slurm_with_header_body(
    name: str,
    header_path: Path,
    body_path: Path,
    replacements: Dict[str, str],
    *,
    header_root: Optional[Path] = None,
) -> str:
    """
    Concatenate a user (or default) Slurm header with a packaged body.

    The default header is copied to ~/.batter/<name> if no user file exists.
    """
    root = header_root or (Path.home() / ".batter")
    user_header = root / name
    header_text: str

    if user_header.exists():
        try:
            header_text = user_header.read_text()
        except Exception as e:
            logger.warning(
                f"[slurm] Failed to read {user_header}: {e}; using packaged header."
            )
            header_text = header_path.read_text()
    else:
        # seed ~/.batter with the packaged header for easy customization
        logger.info(f"[slurm] Adding {user_header}.")
        try:
            user_header.parent.mkdir(parents=True, exist_ok=True)
            user_header.write_text(header_path.read_text())
        except Exception:
            pass
        header_text = header_path.read_text()

    body_text = body_path.read_text()
    text = header_text
    if not text.endswith("\n"):
        text += "\n"
    text += body_text

    for k, v in replacements.items():
        text = text.replace(k, v)
    return text
