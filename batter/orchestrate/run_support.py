from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import yaml
from loguru import logger

from batter.systems.core import SystemBuilder
from batter.systems.mabfe import MABFEBuilder
from batter.systems.masfe import MASFEBuilder


# -------------------- hashing / signatures -------------------- #
def normalize_for_hash(obj: Any) -> Any:
    """Recursively normalize a payload for stable hashing."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        drop_keys = {"output_folder"}
        return {k: normalize_for_hash(v) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, (list, tuple, set)):
        return [normalize_for_hash(v) for v in obj]
    return obj


def compute_run_signature(
    yaml_path: Path,
    run_overrides: Dict[str, Any] | None,
) -> tuple[str, Dict[str, Any]]:
    """Hash the user-facing config (create/fe_sim/fe) to detect run_id reuse conflicts."""
    raw = Path(yaml_path).read_text()
    yaml_data = yaml.safe_load(raw) or {}
    sim_only = {k: v for k, v in yaml_data.items() if k in {"create", "fe_sim", "fe"}}
    payload = {
        "config": normalize_for_hash(sim_only),
        "run_overrides": {},  # execution knobs intentionally excluded
    }
    frozen = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(frozen).hexdigest(), payload


def stored_signature(run_dir: Path) -> tuple[str | None, Path]:
    sig_path = run_dir / "artifacts" / "config" / "run_config.hash"
    if sig_path.exists():
        return sig_path.read_text().strip(), sig_path
    return None, sig_path


def payload_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "config" / "run_config.normalized.json"


def stored_payload(run_dir: Path) -> Dict[str, Any] | None:
    path = payload_path(run_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load stored run payload from %s: %s", path, exc)
        return None


# -------------------- ligand bookkeeping -------------------- #
def ligand_names_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "ligand_names.json"


def load_stored_ligand_names(run_dir: Path) -> Dict[str, str]:
    path = ligand_names_path(run_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to load ligand names from %s: %s", path, exc)
    return {}


def store_ligand_names(run_dir: Path, mapping: Dict[str, str]) -> None:
    path = ligand_names_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, sort_keys=True))


# -------------------- run_id selection -------------------- #
def _diff_dicts(left: Any, right: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    if type(left) != type(right):
        return [(prefix or ".", left, right)]
    if isinstance(left, dict):
        out: list[tuple[str, Any, Any]] = []
        keys = set(left) | set(right)
        for k in sorted(keys):
            lp = prefix + ("." if prefix else "") + str(k)
            if k not in left:
                out.append((lp, None, right[k]))
            elif k not in right:
                out.append((lp, left[k], None))
            else:
                out.extend(_diff_dicts(left[k], right[k], lp))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        out: list[tuple[str, Any, Any]] = []
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            lp = prefix + f"[{idx}]"
            lval = left[idx] if idx < len(left) else None
            rval = right[idx] if idx < len(right) else None
            out.extend(_diff_dicts(lval, rval, lp))
        return out
    return [] if left == right else [(prefix or ".", left, right)]


def resolve_signature_conflict(
    stored_sig: str | None,
    config_signature: str,
    requested_run_id: str | None,
    allow_run_id_mismatch: bool,
    *,
    stored_payload: Dict[str, Any] | None,
    current_payload: Dict[str, Any] | None,
    run_id: str,
    run_dir: Path,
) -> bool:
    """Decide whether to reuse an existing run_dir given the stored signature."""
    if stored_sig is None or stored_sig == config_signature:
        return True

    diffs: list[tuple[str, Any, Any]] = []
    if stored_payload and current_payload:
        diffs = _diff_dicts(stored_payload, current_payload, prefix="config")
    diff_str = "; ".join(f"{p}: stored={l!r}, current={r!r}" for p, l, r in diffs[:8])

    if requested_run_id == "auto":
        if diffs:
            logger.info(
                "Existing execution %s does not match current configuration; differences: %s",
                run_dir,
                diff_str,
            )
        return False

    if allow_run_id_mismatch:
        msg = (
            f"Execution '{run_id}' already exists with configuration hash {stored_sig[:12]} "
            f"(current {config_signature[:12]}); continuing because --allow-run-id-mismatch is enabled."
        )
        if diffs:
            msg += f" Differences: {diff_str}"
        logger.warning(msg)
        return True

    logger.error(
        f"Execution '{run_id}' already exists with configuration hash {stored_sig[:12]} "
        f"(current {config_signature[:12]}); use a different --run-id, enable "
        f"--allow-run-id-mismatch, or update the existing run. Differences: {diff_str}"
    )
    raise RuntimeError("Run ID configuration mismatch detected.")


def _builder_info_for_protocol(protocol: str) -> tuple[Type[SystemBuilder], str]:
    name = (protocol or "abfe").lower()
    mapping: Dict[str, tuple[Type[SystemBuilder], str]] = {
        "abfe": (MABFEBuilder, "MABFE"),
        "md": (MABFEBuilder, "MABFE"),
        "asfe": (MASFEBuilder, "MASFE"),
    }
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(
            f"Unsupported protocol '{protocol}' for system builder selection."
        )


def select_system_builder(protocol: str, system_type: str | None) -> SystemBuilder:
    builder_cls, expected_type = _builder_info_for_protocol(protocol)
    if system_type and system_type != expected_type:
        raise ValueError(
            f"run.system_type={system_type!r} is incompatible with protocol '{protocol}'. "
            f"Expected '{expected_type}'. Remove or update 'run.system_type'."
        )
    return builder_cls()


def select_run_id(
    sys_root: Path | str, protocol: str, system_name: str, requested: str | None
) -> Tuple[str, Path]:
    """Resolve the execution run identifier and backing directory."""
    runs_dir = Path(sys_root) / "executions"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if requested and requested != "auto":
        run_dir = runs_dir / requested
        run_dir.mkdir(parents=True, exist_ok=True)
        return requested, run_dir

    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name, latest

    rid = generate_run_id(protocol, system_name)
    run_dir = runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return rid, run_dir


def generate_run_id(protocol: str, system_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{protocol}-{system_name}-{ts}"
