from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Dict, Mapping, Optional

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

    Parameters
    ----------
    name : str
        Header filename to look for (e.g., ``SLURMM-Am.header``).
    header_path : Path
        Packaged header template path.
    body_path : Path
        Packaged body template path.
    replacements : dict
        Token substitutions applied to the concatenated text.
    header_root : Path, optional
        Root directory containing user headers; defaults to ``~/.batter``.

    Returns
    -------
    str
        Rendered Slurm script text.
    """
    root = header_root or (Path.home() / ".batter")
    user_header = root / name

    def _read_header() -> str:
        try:
            return user_header.read_text()
        except Exception as exc:
            logger.warning(f"[slurm] Failed to read {user_header}: {exc}; using packaged header.")
            return header_path.read_text()

    if user_header.exists():
        header_text = _read_header()
    else:
        try:
            user_header.parent.mkdir(parents=True, exist_ok=True)
            user_header.write_text(header_path.read_text())
        except Exception as exc:
            logger.debug(f"[slurm] Could not seed header {user_header}: {exc}")
        header_text = header_path.read_text()

    body_text = body_path.read_text()
    text = header_text
    if not text.endswith("\n"):
        text += "\n"
    text += body_text

    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def seed_default_headers(
    header_root: Optional[Path] = None,
    resource_map: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Copy packaged Slurm header templates into ``header_root`` (default: ~/.batter).

    Parameters
    ----------
    header_root : Path, optional
        Destination directory for headers; defaults to ``~/.batter``.
    resource_map : Mapping[str, str], optional
        Map of header name → resource ref (``pkg.module/path``) or absolute path.
    overwrite : bool, default False
        When True, replace existing headers in ``header_root``.

    Returns
    -------
    list[Path]
        List of header paths copied/overwritten.
    """
    root = header_root or (Path.home() / ".batter")
    root.mkdir(parents=True, exist_ok=True)

    targets = resource_map or {
        "SLURMM-Am.header": "batter._internal.templates.run_files_orig/SLURMM-Am.header",
        "SLURMM-BATCH-remd.header": "batter._internal.templates.remd_run_files/SLURMM-BATCH-remd.header",
        "job_manager.header": "batter.data/job_manager.header",
        "SLURMM-BATCH.header": "batter._internal.templates.batch_run/SLURMM-BATCH.header",
    }

    copied: list[Path] = []
    for name, ref in targets.items():
        dst = root / name
        if dst.exists() and not overwrite:
            continue
        try:
            ref_path = Path(ref)
            if ref_path.exists():
                dst.write_text(ref_path.read_text())
                copied.append(dst)
                continue
            pkg_name, rel = ref.split("/", 1)
            with resources.as_file(resources.files(pkg_name) / rel) as src_path:
                dst.write_text(src_path.read_text())
                copied.append(dst)
        except Exception as e:
            logger.warning(f"[slurm] Failed to seed header {dst}: {e}")
    return copied
