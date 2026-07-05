"""Prepare alchemical FE inputs for a ligand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from batter._internal.builders.fe_alchemical import AlchemicalFEBuilder
from batter.orchestrate.state_registry import register_phase_state
from batter._internal.ops import remd as remd_ops
from batter._internal.ops import batch as batch_ops
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem


_PREPARE_FE_MARKER = "fe/prepare_fe.ok"
_PREPARE_FE_WINDOWS_MARKER = "fe/prepare_fe_windows.ok"


def _atomic_write_marker(marker: Path, text: str = "ok\n") -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(marker)


def _unlink_marker(marker: Path) -> None:
    try:
        marker.unlink(missing_ok=True)
    except TypeError:
        if marker.exists():
            marker.unlink()


def _system_root_for(child_root: Path) -> Path:
    """work/<sys>/simulations/<lig> → work/<sys>"""
    # Prefer the parent of the "simulations" directory so this works for
    # both per-ligand (simulations/<lig>) and RBFE transformations
    # (simulations/transformations/<pair>).
    try:
        parts = child_root.parts
        if "simulations" in parts:
            idx = parts.index("simulations")
            return Path(*parts[:idx])
        return child_root.parents[1]
    except Exception:
        return child_root


def _load_param_dir_dict(system_root: Path) -> Dict[str, str]:
    """
    Read artifacts/ligand_params/index.json → {residue_name: store_dir}
    """
    index_json = system_root / "artifacts" / "ligand_params" / "index.json"
    data = json.loads(index_json.read_text())
    out: Dict[str, str] = {}
    for entry in data.get("ligands", []):
        resn = entry.get("residue_name")
        store_dir = entry.get("store_dir")
        if resn and store_dir:
            out[resn] = store_dir
    if not out:
        raise RuntimeError(f"No ligand param entries found in {index_json}")
    return out


def _required_component_files(comp_dir: Path, n_windows: int) -> list[Path]:
    paths = [
        comp_dir / "run-local-batch.bash",
        comp_dir / "run-local-remd.bash",
        comp_dir / "SLURMM-BATCH-remd",
        comp_dir / "check_run.bash",
        comp_dir / "lambda.sch",
    ]
    if n_windows > 0:
        paths.append(comp_dir / "remd" / "mini.in.remd.groupfile")
    return paths


def _required_window_files(window_dir: Path, *, hmr: bool) -> list[Path]:
    paths = [
        window_dir / "run-local.bash",
        window_dir / "check_run.bash",
        window_dir / "mdin-template",
        window_dir / "mdin-batch-template",
        window_dir / "mdin-remd-template",
        window_dir / "full.prmtop",
        window_dir / "full.inpcrd",
        window_dir / "full_merged.prmtop",
        window_dir / "eq.rst7",
    ]
    if hmr:
        paths.append(window_dir / "full.hmr.prmtop")
    return paths


def _find_missing_prepare_fe_window_files(
    child_root: Path,
    components: Sequence[str],
    comp_windows: dict,
    *,
    hmr: bool,
) -> list[Path]:
    missing: list[Path] = []
    for comp in components:
        lambdas = comp_windows.get(comp)
        if lambdas is None:
            missing.append(child_root / "fe" / comp / "<lambda schedule>")
            continue

        comp_dir = child_root / "fe" / comp
        for path in _required_component_files(comp_dir, len(lambdas)):
            if not path.exists():
                missing.append(path)

        for win_idx in range(len(lambdas)):
            window_dir = comp_dir / f"{comp}{win_idx:02d}"
            if not window_dir.is_dir():
                missing.append(window_dir)
                continue
            for path in _required_window_files(window_dir, hmr=hmr):
                if not path.exists():
                    missing.append(path)
    return missing


def _raise_for_missing_prepare_fe_window_files(
    child_root: Path,
    missing: Sequence[Path],
) -> None:
    if not missing:
        return

    preview = []
    for path in missing[:20]:
        try:
            preview.append(path.relative_to(child_root).as_posix())
        except ValueError:
            preview.append(str(path))
    extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
    raise RuntimeError(
        "Incomplete prepare_fe_windows output; not marking prepare_fe complete. "
        "Missing required file(s):\n"
        + "\n".join(f"- {item}" for item in preview)
        + extra
    )


def prepare_fe_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """Construct the initial FE directory layout for a ligand.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Metadata describing the generated directories.
    """
    payload = StepPayload.model_validate(params)
    if payload.sim is None:
        raise ValueError("[prepare_fe] Missing simulation configuration in payload.")
    sim = payload.sim
    partition = payload.get("partition") or payload.get("queue") or "normal"
    phase_name = payload.get("phase_name") or "prepare_fe"
    components = list(payload.get("components") or getattr(sim, "components", []) or [])
    if not components:
        raise ValueError("No components specified in sim config.")

    ligand = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)
    param_dir_dict = _load_param_dir_dict(system_root)

    comp_windows: dict = payload.get("component_lambdas") or sim.component_lambdas  # type: ignore[attr-defined]
    sys_params = payload.sys_params or SystemParams()
    user_anchor_atoms = list(sys_params.get("anchor_atoms", []) or [])
    extra_restraints: Optional[str] = sys_params.get("extra_restraints", None)
    extra_restraint_fc = float(sys_params.get("extra_restraint_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = sys_params.get(
        "extra_conformation_restraints", None
    )
    pair_meta = {
        key: system.meta.get(key)
        for key in (
            "pair_id",
            "ligand_ref",
            "ligand_alt",
            "residue_ref",
            "residue_alt",
            "input_ref",
            "input_alt",
            "prepared_mapping_dir",
            "atom_mapper",
            "atom_mapper_options",
            "kartograf_options",
            "lomap_options",
        )
        if system.meta.get(key) is not None
    }

    infe = bool(sim.infe)
    fe_root = child_root / ("pre_fe" if phase_name == "pre_prepare_fe" else "fe")
    marker = fe_root / f"{phase_name}.ok"
    _unlink_marker(marker)

    artifacts: Dict[str, Any] = {}
    logger.debug(
        f"[{phase_name}] start ligand={ligand} residue={residue_name} components={components}"
    )

    # Patch sim for pre-prepare cases (e.g., RBFE pre_prepare_fe uses z)
    if any(comp == "z" for comp in components):
        if getattr(sim, "dec_method", None) not in {"sdr", "dd"}:
            sim = sim.model_copy(deep=True)
            sim.dec_method = "sdr"
        if sim.dic_n_steps.get("z", 0) <= 0:
            sim = sim.model_copy(deep=True)
            fallback = sim.dic_n_steps.get("x", 0) or sim.n_steps_dict.get("x_n_steps", 0)
            if fallback <= 0:
                fallback = int(getattr(sim, "eq_steps", 0) or 0)
            if fallback <= 0:
                fallback = 100000
            sim.dic_n_steps["z"] = int(fallback)

    # Build per component (scaffold / templates only; win=-1)
    for comp in components:
        workdir = fe_root / comp
        workdir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[{phase_name}] building component '{comp}' in {workdir}")
        builder = AlchemicalFEBuilder(
            ligand=ligand,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            sim_config=sim,
            component=comp,
            component_windows=comp_windows[comp],
            working_dir=workdir,
            system_root=system_root,
            infe=infe,
            win=-1,
            extra={
                "extra_restraints": extra_restraints,
                "extra_restraint_fc": extra_restraint_fc,
                "extra_conformation_restraints": extra_conformation_restraints,
                "user_anchor_atoms": user_anchor_atoms,
                "partition": partition,
                "phase_name": phase_name,
                **pair_meta,
            },
        )
        builder.build()  # will create <comp>-1, amber templates, run files, etc.

        artifacts[f"{comp}_workdir"] = str(workdir)

    # emit the common OK marker used by the orchestrator
    _atomic_write_marker(marker)

    logger.debug(f"[{phase_name}] finished ligand={ligand} → {marker}")
    marker_rel = marker.relative_to(system.root).as_posix()
    required = [[marker_rel]]
    if phase_name == "prepare_fe":
        required = [[_PREPARE_FE_MARKER, _PREPARE_FE_WINDOWS_MARKER]]
    register_phase_state(
        system.root,
        phase_name,
        required=required,
        success=required,
    )
    return ExecResult(job_ids=[], artifacts={"prepare_fe_ok": marker, **artifacts})


# -----------------------------
# prepare_fe_windows (expand per-lambda windows)
# -----------------------------
def prepare_fe_windows_handler(
    step: Step, system: SimSystem, params: Dict[str, Any]
) -> ExecResult:
    """
    Expand FE windows for each requested component:
      - copies <comp>-1 to <comp>-2, <comp>-3, ... (depending on lambda schedule)
      - keeps run scripts consistent in each window (builders call write_run_file)
      - writes artifacts/fe/windows.json summarizing windows

    Builders re-use the same interface; here we just iterate components and request
    per-window builds by calling with win >= 1.
    """
    payload = StepPayload.model_validate(params)
    if payload.sim is None:
        raise ValueError(
            "[prepare_fe_windows] Missing simulation configuration in payload."
        )
    sim = payload.sim
    partition = payload.get("partition") or payload.get("queue") or "normal"
    components = list(getattr(sim, "components", []) or [])
    if not components:
        raise RuntimeError(
            "No components specified in sim config for FE window preparation."
        )

    ligand = system.meta.get("ligand")
    residue_name = system.meta.get("residue_name")
    if not ligand or not residue_name:
        raise ValueError("System meta must include 'ligand' and 'residue_name'.")

    child_root = system.root
    system_root = _system_root_for(child_root)

    param_dir_dict = _load_param_dir_dict(system_root)

    comp_windows: dict = payload.get("component_lambdas") or sim.component_lambdas  # type: ignore[attr-defined]
    prepare_finished = child_root / "fe" / "prepare_fe_windows.ok"
    _unlink_marker(prepare_finished)

    sys_params = payload.sys_params or SystemParams()
    user_anchor_atoms = list(sys_params.get("anchor_atoms", []) or [])
    extra_restraints: Optional[str] = sys_params.get("extra_restraints", None)
    extra_restraint_fc = float(sys_params.get("extra_restraint_fc", 10.0))
    extra_conformation_restraints: Optional[Path] = sys_params.get(
        "extra_conformation_restraints", None
    )
    pair_meta = {
        key: system.meta.get(key)
        for key in (
            "pair_id",
            "ligand_ref",
            "ligand_alt",
            "residue_ref",
            "residue_alt",
            "input_ref",
            "input_alt",
            "prepared_mapping_dir",
            "atom_mapper",
            "atom_mapper_options",
            "kartograf_options",
            "lomap_options",
        )
        if system.meta.get(key) is not None
    }

    infe = False
    if extra_restraints is not None:
        infe = False
        sim.barostat = "1"
    if extra_conformation_restraints is not None:
        infe = True
        # cannot do NFE with barostat 1 (Berendsen)
        sim.barostat = "2"

    windows_summary: Dict[str, Any] = {}
    logger.debug(
        f"[prepare_fe_windows] start ligand={ligand} residue={residue_name} components={components}"
    )

    for comp in components:
        workdir = child_root / "fe" / comp
        lambdas = comp_windows[comp]

        for win_idx, _ in enumerate(lambdas):
            logger.debug(
                f"[prepare_fe_windows] {comp} → creating window {win_idx} in {workdir}"
            )
            builder = AlchemicalFEBuilder(
                ligand=ligand,
                residue_name=residue_name,
                param_dir_dict=param_dir_dict,
                sim_config=sim,
                component=comp,
                infe=infe,
                component_windows=lambdas,
                working_dir=workdir,
                system_root=system_root,
                win=win_idx,
                extra={
                    "extra_restraints": extra_restraints,
                    "extra_restraint_fc": extra_restraint_fc,
                    "extra_conformation_restraints": extra_conformation_restraints,
                    "user_anchor_atoms": user_anchor_atoms,
                    "partition": partition,
                    **pair_meta,
                },
            )
            builder.build()

        windows_summary[comp] = {"n_windows": len(lambdas), "lambdas": lambdas}

        # Always write REMD inputs; run.remd controls whether they are submitted.
        batch_ops.prepare_batch_component(
            workdir,
            comp=comp,
            n_windows=len(lambdas),
            hmr=str(sim.hmr).lower() == "yes",
        )
        remd_ops.prepare_remd_component(
            workdir,
            comp=comp,
            sim=sim,
            n_windows=len(lambdas),
            partition=partition,
        )

    missing = _find_missing_prepare_fe_window_files(
        child_root,
        components,
        comp_windows,
        hmr=str(sim.hmr).lower() == "yes",
    )
    if missing:
        _unlink_marker(prepare_finished)
        _raise_for_missing_prepare_fe_window_files(child_root, missing)

    # write a canonical windows.json under artifacts/fe/
    windows_json = child_root / "fe" / "artifacts" / "windows.json"
    windows_json.parent.mkdir(parents=True, exist_ok=True)
    windows_json.write_text(json.dumps(windows_summary, indent=2) + "\n")

    _atomic_write_marker(prepare_finished)

    windows_rel = prepare_finished.relative_to(system.root).as_posix()
    prepare_rel = (
        (child_root / "fe" / "prepare_fe.ok")
        .relative_to(system.root)
        .as_posix()
    )
    register_phase_state(
        system.root,
        "prepare_fe",
        required=[[prepare_rel, windows_rel]],
        success=[[prepare_rel, windows_rel]],
    )

    logger.debug(f"[prepare_fe_windows] finished ligand={ligand} → {windows_json}")
    return ExecResult(job_ids=[], artifacts={"windows_json": windows_json})
