from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REGISTRY_FILENAME = "phase_state.json"
SCHEMA_VERSION = 1

LEGACY_DEFAULTS: Dict[str, Dict[str, List[List[str]]]] = {
    "system_prep": {
        "required": [["artifacts/config/sim_overrides.json"]],
        "success": [["artifacts/config/sim_overrides.json"]],
        "failure": [],
    },
    "system_prep_asfe": {
        "required": [["artifacts/config/sim_overrides.json"]],
        "success": [["artifacts/config/sim_overrides.json"]],
        "failure": [],
    },
    "param_ligands": {
        "required": [["artifacts/ligand_params/index.json"]],
        "success": [["artifacts/ligand_params/index.json"]],
        "failure": [],
    },
    "prepare_equil": {
        "required": [["equil/full.prmtop", "equil/artifacts/prepare_equil.ok"]],
        "success": [["equil/full.prmtop", "equil/artifacts/prepare_equil.ok"]],
        "failure": [],
    },
    "equil": {
        "required": [["equil/FINISHED"], ["equil/FAILED"]],
        "success": [["equil/FINISHED"]],
        "failure": [["equil/FAILED"]],
    },
    "equil_analysis": {
        "required": [["equil/representative.pdb"], ["equil/UNBOUND"]],
        "success": [["equil/representative.pdb"], ["equil/UNBOUND"]],
        "failure": [],
    },
    "prepare_fe": {
        "required": [["fe/artifacts/prepare_fe.ok", "fe/artifacts/prepare_fe_windows.ok"]],
        "success": [["fe/artifacts/prepare_fe.ok", "fe/artifacts/prepare_fe_windows.ok"]],
        "failure": [],
    },
    "fe_equil": {
        "required": [["fe/{comp}/{comp}-1/EQ_FINISHED"]],
        "success": [["fe/{comp}/{comp}-1/EQ_FINISHED"]],
        "failure": [["fe/{comp}/{comp}-1/FAILED"]],
    },
    "fe": {
        "required": [["fe/{comp}/{comp}{win:02d}/FINISHED"]],
        "success": [["fe/{comp}/{comp}{win:02d}/FINISHED"]],
        "failure": [["fe/{comp}/{comp}{win:02d}/FAILED"]],
    },
    "analyze": {
        "required": [["fe/artifacts/analyze.ok", "fe/Results/Results.dat"]],
        "success": [["fe/artifacts/analyze.ok", "fe/Results/Results.dat"]],
        "failure": [],
    },
}


def _registry_path(root: Path) -> Path:
    return Path(root) / "artifacts" / REGISTRY_FILENAME


def _as_dnf(spec: Sequence[Sequence[str] | str] | Sequence[str] | str | None) -> List[List[str]]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [[spec]]
    groups: List[List[str]] = []
    for group in spec:  # type: ignore[assignment]
        if isinstance(group, str):
            groups.append([group])
        else:
            groups.append([str(item) for item in group])
    return groups


@dataclass(slots=True)
class PhaseState:
    phase: str
    required: List[List[str]] = field(default_factory=list)
    success: List[List[str]] = field(default_factory=list)
    failure: List[List[str]] = field(default_factory=list)


def _load_raw(root: Path) -> Dict[str, Dict[str, List[List[str]]]]:
    path = _registry_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if payload.get("version") != SCHEMA_VERSION:
        return {}
    return payload.get("phases", {})


def _dump_raw(root: Path, payload: Dict[str, Dict[str, List[List[str]]]]) -> None:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": SCHEMA_VERSION,
        "phases": payload,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def register_phase_state(
    root: Path | str,
    phase: str,
    *,
    required: Sequence[Sequence[str] | str] | Sequence[str] | str | None = None,
    success: Sequence[Sequence[str] | str] | Sequence[str] | str | None = None,
    failure: Sequence[Sequence[str] | str] | Sequence[str] | str | None = None,
) -> PhaseState:
    """
    Record or update the on-disk phase state specification for a system root.
    """
    root_path = Path(root)
    registry = _load_raw(root_path)
    entry = registry.get(phase, {})
    if required is not None:
        entry["required"] = _as_dnf(required)
    if success is not None:
        entry["success"] = _as_dnf(success)
    if failure is not None:
        entry["failure"] = _as_dnf(failure)
    registry[phase] = entry
    _dump_raw(root_path, registry)
    return PhaseState(
        phase=phase,
        required=entry.get("required", []),
        success=entry.get("success", []),
        failure=entry.get("failure", []),
    )


def read_phase_states(root: Path | str) -> Dict[str, PhaseState]:
    """
    Return all recorded phase state specifications for ``root``.
    """
    root_path = Path(root)
    raw = _load_raw(root_path)
    out: Dict[str, PhaseState] = {}
    for phase, entry in raw.items():
        out[phase] = PhaseState(
            phase=phase,
            required=entry.get("required", []),
            success=entry.get("success", []),
            failure=entry.get("failure", []),
        )
    return out


def get_phase_state(root: Path | str, phase: str) -> PhaseState:
    """
    Convenience accessor for a single phase specification.
    """
    states = read_phase_states(root)
    if phase in states:
        return states[phase]
    legacy = LEGACY_DEFAULTS.get(phase)
    if legacy:
        return PhaseState(
            phase=phase,
            required=legacy.get("required", []),
            success=legacy.get("success", []),
            failure=legacy.get("failure", []),
        )
    return PhaseState(phase=phase)
