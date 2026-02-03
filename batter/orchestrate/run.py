"""
batter.orchestrate.run
======================

Top-level orchestration entry for BATTER runs.

This module wires:
YAML (RunConfig) → shared system build → bulk ligand staging →
single param job ("param_ligands") → per-ligand pipelines → FE record save.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import smtplib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
from smtplib import SMTPException
import yaml

from loguru import logger
from pprint import pprint

from batter.config.run import RunConfig
from batter.systems.core import SimSystem
from batter.exec.local import LocalBackend
from batter.exec.slurm import SlurmBackend

from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step
from batter.pipeline.payloads import StepPayload

from batter.runtime.portable import ArtifactStore
from batter.runtime.fe_repo import FEResultsRepository, FERecord

from batter.exec.slurm_mgr import SlurmJobManager

from batter.orchestrate.backend import register_local_handlers
from batter.orchestrate.ligands import (
    discover_staged_ligands,
    resolve_ligand_map,
)
from batter.orchestrate.markers import (
    handle_phase_failures,
    run_phase_skipping_done,
    is_done,
)
from batter.orchestrate.pipeline_utils import select_pipeline
from batter.orchestrate.results_io import (
    extract_ligand_metadata,
    fallback_totals_from_json,
    parse_results_dat,
    save_fe_records,
)
from batter.orchestrate.run_support import (
    compute_run_signature as _compute_run_signature,
    generate_run_id,
    ligand_names_path as _ligand_names_path,
    load_stored_ligand_names as _load_stored_ligand_names,
    payload_path as _payload_path,
    resolve_signature_conflict as _resolve_signature_conflict,
    select_run_id,
    select_system_builder as _select_system_builder,
    stored_payload as _stored_payload,
    stored_signature as _stored_signature,
    store_ligand_names as _store_ligand_names,
)


def _slurm_registry_path(run_dir: Path) -> Path:
    """Return the registry path under artifacts/slurm, migrating legacy .slurm if present."""
    new_path = run_dir / "artifacts" / "slurm" / "queue.jsonl"
    old_path = run_dir / ".slurm" / "queue.jsonl"
    if old_path.exists() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            old_path.replace(new_path)
        except Exception:
            shutil.copy2(old_path, new_path)
    return new_path


def _store_run_yaml_copy(run_dir: Path, yaml_path: Path) -> None:
    """Persist a copy of the user YAML under artifacts/config for future reuse."""
    cfg_dir = run_dir / "artifacts" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    dst = cfg_dir / "run_config.yaml"
    if dst.exists():
        return
    try:
        shutil.copy2(yaml_path, dst)
    except Exception as exc:
        logger.warning(f"Could not store run YAML copy at {dst}: {exc}")


def _clear_failure_markers(run_dir: Path) -> None:
    """Remove FAILED markers and progress caches under a run directory."""
    sim_root = run_dir / "simulations"
    if not sim_root.exists():
        return
    removed = 0
    for path in sim_root.rglob("FAILED"):
        try:
            path.unlink()
            removed += 1
        except Exception:
            continue
    for path in sim_root.rglob("progress"):
        if not path.is_dir():
            continue
        for csv in path.glob("*.csv"):
            try:
                csv.unlink()
                removed += 1
            except Exception:
                continue
    if removed:
        logger.info(f"[cleanup] Removed {removed} failure/progress marker(s).")

    progress_root = run_dir / "artifacts" / "progress"
    if progress_root.exists():
        try:
            shutil.rmtree(progress_root)
            logger.info(f"[cleanup] Removed progress cache folder: {progress_root}")
        except Exception:
            logger.warning(f"[cleanup] Failed to remove progress cache folder: {progress_root}")


def _build_rbfe_network_plan(
    ligands: List[str],
    lig_map: Dict[str, str],
    rbfe_cfg,
    config_dir: Path,
) -> dict:
    from batter.rbfe import (
        RBFENetwork,
        resolve_mapping_fn,
        load_mapping_file,
        konnektor_pairs,
    )
    from batter.config.utils import sanitize_ligand_name

    available = [sanitize_ligand_name(x) for x in ligands if x]
    if len(available) < 2:
        raise RuntimeError("RBFE requires at least two ligands.")

    mapping_source: Dict[str, Any] = {}
    if rbfe_cfg.mapping_file:
        pairs = load_mapping_file(Path(rbfe_cfg.mapping_file))
        network = RBFENetwork.from_ligands(available, mapping_fn=lambda _: pairs)
        mapping_source["mapping_file"] = str(rbfe_cfg.mapping_file)
    else:
        mapping_name = rbfe_cfg.mapping or "default"
        if mapping_name == "konnektor":
            pairs = konnektor_pairs(
                available,
                {name: Path(lig_map[name]) for name in available},
                layout=rbfe_cfg.konnektor_layout,
            )
            network = RBFENetwork.from_ligands(available, mapping_fn=lambda _: pairs)
            mapping_source["mapping"] = "konnektor"
            if rbfe_cfg.konnektor_layout:
                mapping_source["konnektor_layout"] = rbfe_cfg.konnektor_layout
        elif mapping_name in {"default", "star", "first"}:
            try:
                pairs = konnektor_pairs(
                    available,
                    {name: Path(lig_map[name]) for name in available},
                    layout="star",
                )
                network = RBFENetwork.from_ligands(available, mapping_fn=lambda _: pairs)
                mapping_source["mapping"] = mapping_name
                mapping_source["konnektor_layout"] = "star"
            except Exception as exc:
                logger.warning(
                    f"RBFE default mapping requested StarNetworkGenerator but failed "
                    f"({exc}); falling back to internal default mapping."
                )
                mapping_fn = resolve_mapping_fn(mapping_name)
                network = RBFENetwork.from_ligands(available, mapping_fn=mapping_fn)
                mapping_source["mapping"] = mapping_name
        else:
            mapping_fn = resolve_mapping_fn(mapping_name)
            network = RBFENetwork.from_ligands(available, mapping_fn=mapping_fn)
            mapping_source["mapping"] = mapping_name

    payload = network.to_mapping()
    if bool(getattr(rbfe_cfg, "both_directions", False)):
        bidirectional_pairs: List[List[str]] = []
        seen: set[tuple[str, str]] = set()
        for ref, alt in payload.get("pairs", []):
            for pair in ((ref, alt), (alt, ref)):
                if pair in seen:
                    continue
                seen.add(pair)
                bidirectional_pairs.append([pair[0], pair[1]])
        payload["pairs"] = bidirectional_pairs
        mapping_source["both_directions"] = True
    payload.update(mapping_source)
    rbfe_network_path = config_dir / "rbfe_network.json"
    rbfe_network_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        f"RBFE network planned: {len(network.ligands)} ligands, {len(network.pairs)} pairs."
    )
    return payload


def _materialize_extra_conf_restraints(
    source: Path | str | None, run_dir: Path, yaml_dir: Path
) -> Path | None:
    """Copy extra_conformation_restraints into artifacts/config for reuse and return the stored path."""
    if not source:
        return None
    src = Path(source)
    if not src.is_absolute():
        src = (yaml_dir / src).resolve()

    dest_dir = run_dir / "artifacts" / "config"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if dest.exists():
        return dest
    if src.exists():
        try:
            shutil.copy2(src, dest)
            return dest
        except Exception as exc:
            logger.warning(f"Could not copy extra_conformation_restraints from {src}: {exc}")
            return None

    logger.warning(
        f"extra_conformation_restraints missing at {src} and no stored copy under {dest}"
    )
    return None


def run_from_yaml(
    path: Path | str,
    on_failure: Literal["prune", "raise", "retry"] = None,
    run_overrides: Dict[str, Any] | None = None,
) -> None:
    """Execute a BATTER workflow described by a YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the top-level run YAML file.
    on_failure : {"prune", "raise", "retry"}, optional
        Override for the failure policy applied to ligand pipelines.
    run_overrides : dict, optional
        Overrides applied to the ``run`` section (e.g., only FE preparation).
    """
    path = Path(path)
    logger.info(f"Starting BATTER run from {path}")

    # Configs
    rc = RunConfig.load(path)

    if run_overrides:
        logger.info(f"Applying run overrides: {run_overrides}")
        rc = rc.model_copy(update={"run": rc.run.model_copy(update=run_overrides)})
    if on_failure:
        rc.run.on_failure = on_failure

    logger.info(
    "Run configuration:\n{}",
    yaml.safe_dump(rc.model_dump(mode="json"), sort_keys=False)
    )
    yaml_dir = path.parent

    # ligand params output directory
    if rc.create.param_outdir is None:
        rc.create.param_outdir = str(rc.run.output_folder / "ligand_params")
    else:
        logger.info(
            f"Using user-specified ligand param_outdir: {rc.create.param_outdir}"
        )

    sim_cfg = rc.resolved_sim_config()
    logger.info(f"Loaded simulation config for system: {sim_cfg.system_name}")

    # Backend
    if rc.backend == "slurm":
        backend = SlurmBackend()
    else:
        backend = LocalBackend()
        register_local_handlers(backend)

    # Shared System Build (system-level assets live under sys.root)
    builder = _select_system_builder(rc.protocol, rc.run.system_type)

    requested_run_id = getattr(rc.run, "run_id", "auto")
    config_signature, config_payload = _compute_run_signature(path, run_overrides)

    while True:
        run_id, run_dir = select_run_id(
            rc.run.output_folder,
            rc.protocol,
            rc.create.system_name,
            requested_run_id,
        )
        stored_sig, sig_path = _stored_signature(run_dir)
        stored_payload = _stored_payload(run_dir)
        if _resolve_signature_conflict(
            stored_sig,
            config_signature,
            requested_run_id,
            rc.run.allow_run_id_mismatch,
            run_id=run_id,
            run_dir=run_dir,
            stored_payload=stored_payload,
            current_payload=config_payload,
        ):
            break
        logger.info(
            f"Existing execution {run_dir} uses different configuration hash ({stored_sig[:12]}); creating a fresh run.",
        )
        requested_run_id = "auto"
        run_id = generate_run_id(rc.protocol, rc.create.system_name)
        run_dir = Path(rc.run.output_folder) / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        continue

    logger.info(f"Using run_id='{run_id}' under {run_dir}")
    _, sig_path = _stored_signature(run_dir)

    _store_run_yaml_copy(run_dir, path)

    # Ligands
    lig_original_names: Dict[str, str] = {}
    staged_lig_map = discover_staged_ligands(run_dir)
    stored_names = _load_stored_ligand_names(run_dir)
    if staged_lig_map:
        lig_map = staged_lig_map
        lig_original_names = stored_names
        if lig_original_names:
            logger.debug(
                "Loaded %d original ligand names from %s",
                len(lig_original_names),
                _ligand_names_path(run_dir),
            )
        logger.info(
            f"Resuming with {len(lig_map)} staged ligands discovered under {run_dir}"
        )
    else:
        # Fall back to YAML resolution (requires original paths/files to exist)
        lig_map, lig_original_names = resolve_ligand_map(rc, yaml_dir)
        if lig_original_names:
            _store_ligand_names(run_dir, lig_original_names)
    rc.create.ligand_paths = {k: str(v) for k, v in lig_map.items()}

    # Build system-prep params exactly once (after run_dir is known)
    extra_conf_path = _materialize_extra_conf_restraints(
        rc.create.extra_conformation_restraints, run_dir, yaml_dir
    )
    sys_params = {
        "param_outdir": str(rc.create.param_outdir),
        "system_name": rc.create.system_name,
        "protein_input": str(rc.create.protein_input),
        "system_input": str(rc.create.system_input) if rc.create.system_input else None,
        "system_coordinate": (
            str(rc.create.system_coordinate) if rc.create.system_coordinate else None
        ),
        "ligand_paths": rc.create.ligand_paths,
        "anchor_atoms": list(rc.create.anchor_atoms or []),
        "protein_align": str(rc.create.protein_align),
        "lipid_mol": list(rc.create.lipid_mol or []),
        "other_mol": list(rc.create.other_mol or []),
        "ligand_ff": rc.create.ligand_ff,
        "retain_lig_prot": bool(rc.create.retain_lig_prot),
        "charge": rc.create.param_charge,
        "yaml_dir": str(yaml_dir),
        "extra_restraints": rc.create.extra_restraints,
        "extra_restraint_fc": rc.create.extra_restraint_fc,
        "extra_conformation_restraints": extra_conf_path
        or rc.create.extra_conformation_restraints,
    }

    base_meta = {}
    if rc.protocol == "rbfe":
        base_meta["mode"] = "RBFE"
    sys_exec = SimSystem(name=rc.create.system_name, root=run_dir, meta=base_meta)
    sys_exec = builder.build(sys_exec, rc.create)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(config_signature + "\n")
    _payload_path(run_dir).write_text(
        json.dumps(config_payload, sort_keys=True, indent=2)
    )

    # Per-execution run directory (auto-resume latest when 'auto')
    logger.add(run_dir / "batter.run.log", level="DEBUG")

    dry_run = rc.run.dry_run
    if dry_run:
        logger.warning("DRY RUN mode enabled: no SLURM jobs will be submitted.")

    # SLURM manager (registry per execution)
    slurm_flags = rc.run.slurm.to_sbatch_flags() if rc.run.slurm else None
    batch_mode = bool(getattr(rc.run, "batch_mode", False))
    batch_poll = 10.0 if batch_mode else 60 * 15
    registry_file = None if batch_mode else _slurm_registry_path(run_dir)
    job_mgr = SlurmJobManager(
        poll_s=batch_poll,
        max_retries=3,
        resubmit_backoff_s=30,
        registry_file=registry_file,
        dry_run=dry_run,
        sbatch_flags=slurm_flags,
        batch_mode=batch_mode,
        batch_gpus=getattr(rc.run, "batch_gpus", None),
        gpus_per_task=getattr(rc.run, "batch_gpus_per_task", 1),
        srun_extra=getattr(rc.run, "batch_srun_extra", None),
        max_active_jobs=rc.run.max_active_jobs,
        partition=rc.run.slurm.partition if rc.run.slurm else None,
    )

    # Build pipeline with explicit sys_params
    tpl = select_pipeline(
        rc.protocol,
        sim_cfg,
        rc.run.only_fe_preparation,
        sys_params=sys_params,
        partition=rc.run.slurm.partition if rc.run.slurm else None,
    )

    # Run parent-only steps at run_dir by using a run-scoped SimSystem
    run_sys = SimSystem(
        name=f"{sys_exec.name}:{run_id}",
        root=run_dir,
        protein=sys_exec.protein,
        topology=sys_exec.topology,
        coordinates=sys_exec.coordinates,
        ligands=tuple(),  # parent steps don't need per-ligand sdf
        lipid_mol=sys_exec.lipid_mol,
        other_mol=sys_exec.other_mol,
        anchors=sys_exec.anchors,
        meta=sys_exec.meta,
    )
    # Stage ligands under this execution
    lig_root = run_dir / "simulations"
    lig_root.mkdir(parents=True, exist_ok=True)
    for lig_name, lig_path in lig_map.items():
        sub = lig_root / lig_name / "inputs"
        if not (sub / f"ligand{Path(lig_path).suffix}").exists():
            builder.make_child_for_ligand(sys_exec, lig_name, lig_path)
    logger.debug(f"Staged {len(lig_map)} ligand subsystems under {lig_root}")

    parent_failure = False
    parent_only = Pipeline(
        [
            s
            for s in tpl.ordered_steps()
            if s.name in {"system_prep", "system_prep_asfe", "param_ligands"}
        ]
    )
    if parent_only.ordered_steps():
        names = [s.name for s in parent_only.ordered_steps()]
        logger.debug(f"Executing parent-only steps at {run_dir}: {names}")
        for step in parent_only.ordered_steps():
            if is_done(run_sys, step.name):
                logger.info(f"[skip] {step.name}: finished.")
                continue
            try:
                step_params = step.params
                if isinstance(step_params, dict):
                    step_params = dict(step_params)
                    step_params["on_failure"] = rc.run.on_failure
                elif hasattr(step_params, "copy_with"):
                    step_params = step_params.copy_with(on_failure=rc.run.on_failure)
                backend.run(step, run_sys, step_params)
            except Exception as exc:
                if step.name == "param_ligands" and (rc.run.on_failure or "").lower() in {"prune", "retry"}:
                    parent_failure = True
                    logger.error(
                        "[param_ligands] encountered error with on_failure=%s: %s — continuing with successful ligands only.",
                        rc.run.on_failure,
                        exc,
                    )
                    break
                raise

    # Locate sim_overrides from system_prep (under run_dir)
    config_dir = run_dir / "artifacts" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = config_dir / "sim_overrides.json"
    sim_cfg_updated = sim_cfg

    if rc.protocol == "rbfe":
        rbfe_network_path = config_dir / "rbfe_network.json"
        if not rbfe_network_path.exists():
            from batter.config.run import RBFENetworkArgs

            rbfe_cfg = rc.rbfe or RBFENetworkArgs()
            _build_rbfe_network_plan(
                list(lig_map.keys()), lig_map, rbfe_cfg, config_dir
            )
    if overrides_path.exists():
        upd = json.loads(overrides_path.read_text()) or {}
        sim_cfg_updated = sim_cfg.model_copy(
            update={k: v for k, v in upd.items() if v is not None}
        )

    from batter.config.io import write_yaml_config

    write_yaml_config(sim_cfg_updated, config_dir / "sim.resolved.yaml")

    run_meta_path = config_dir / "run_meta.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "protocol": rc.protocol,
                "backend": rc.backend,
                "system_name": rc.create.system_name,
                "run_id": run_id,
            },
            indent=2,
        )
    )

    # Now build a fresh pipeline for per-ligand steps using the UPDATED sim
    removed = {"system_prep", "system_prep_asfe", "param_ligands"}
    per_lig_steps: List[Step] = []
    for s in tpl.ordered_steps():
        if s.name in removed:
            continue
        payload = s.payload.copy_with() if s.payload is not None else None
        per_lig_steps.append(
            Step(
                name=s.name,
                requires=[r for r in s.requires if r not in removed],
                payload=payload,
            )
        )
    per_lig = Pipeline(per_lig_steps)

    # IMPORTANT: also update the `sim` param on each remaining step
    patched = []
    for s in per_lig.ordered_steps():
        payload = s.payload
        if payload is not None and payload.sim is not None:
            payload = payload.copy_with(sim=sim_cfg_updated)
        patched.append(Step(name=s.name, requires=s.requires, payload=payload))
    per_lig = Pipeline(patched)

    # --- define phases explicitly ---
    PH_PREPARE_EQUIL = {"prepare_equil"}
    PH_EQUIL = {"equil"}
    PH_EQUIL_ANALYSIS = {"equil_analysis"}
    PH_PRE_PREPARE_FE = {"pre_prepare_fe"}
    PH_PRE_FE_EQUIL = {"pre_fe_equil"}
    PH_PREPARE_FE = {"prepare_fe", "prepare_fe_windows"}
    PH_FE_EQUIL = {"fe_equil"}
    PH_FE = {"fe"}
    PH_ANALYZE = {"analyze"}

    def _phase(names: set[str]) -> Pipeline:
        selected = [s for s in per_lig.ordered_steps() if s.name in names]
        selected_names = {s.name for s in selected}
        pruned = [
            Step(
                name=s.name,
                requires=[r for r in s.requires if r in selected_names],
                payload=s.payload,
            )
            for s in selected
        ]
        return Pipeline(pruned)

    phase_prepare_equil = _phase(PH_PREPARE_EQUIL)
    phase_equil = _phase(PH_EQUIL)
    phase_equil_analysis = _phase(PH_EQUIL_ANALYSIS)
    phase_pre_prepare_fe = _phase(PH_PRE_PREPARE_FE)
    phase_pre_fe_equil = _phase(PH_PRE_FE_EQUIL)
    phase_prepare_fe = _phase(PH_PREPARE_FE)
    phase_fe_equil = _phase(PH_FE_EQUIL)
    phase_fe = _phase(PH_FE)
    phase_analyze = _phase(PH_ANALYZE)

    # --- build SimSystem children ---
    param_idx_path = run_dir / "artifacts" / "ligand_params" / "index.json"
    if not param_idx_path.exists():
        if parent_failure and (rc.run.on_failure or "").lower() in {"prune", "retry"}:
            logger.warning(
                "Parametrization failed and no ligand param index was written; continuing with 0 ligands due to on_failure=%s.",
                rc.run.on_failure,
            )
            param_index = {"ligands": []}
        else:
            raise FileNotFoundError(f"Missing ligand param index: {param_idx_path}")
    else:
        param_index = json.loads(param_idx_path.read_text())
    param_dir_dict = {e["residue_name"]: e["store_dir"] for e in param_index["ligands"]}

    lig_resname_map = {}
    for entry in param_index["ligands"]:
        lig = entry.get("ligand")
        resn = entry.get("residue_name")
        lig_resname_map[lig] = resn

    # keep all children for now
    children_all: List[SimSystem] = []
    for lig_name, resn in lig_resname_map.items():
        d = run_dir / "simulations" / lig_name
        child_meta = sys_exec.meta.merge(
            ligand=lig_name,
            residue_name=resn,
            param_dir_dict=param_dir_dict,
        )
        children_all.append(
            SimSystem(
                name=f"{sys_exec.name}:{lig_name}:{run_id}",
                root=d,
                protein=sys_exec.protein,
                topology=sys_exec.topology,
                coordinates=sys_exec.coordinates,
                ligands=tuple([d / "inputs" / "ligand.sdf"]),
                lipid_mol=sys_exec.lipid_mol,
                other_mol=sys_exec.other_mol,
                anchors=sys_exec.anchors,
                meta=child_meta,
            )
        )
    # start with all children
    children = children_all
    fe_children_all: List[SimSystem] = children_all
    if getattr(rc.run, "clean_failures", False):
        _clear_failure_markers(run_dir)

    # --------------------
    # PHASE 1: prepare_equil (parallel)
    # --------------------
    if phase_prepare_equil.ordered_steps():
        run_phase_skipping_done(
            phase_prepare_equil,
            children,
            "prepare_equil",
            backend,
            max_workers=rc.run.max_workers,
            on_failure=rc.run.on_failure,
        )
        children = handle_phase_failures(children, "prepare_equil", rc.run.on_failure)
    else:
        logger.info("[skip] prepare_equil: no steps in this protocol.")

    # --------------------
    # PHASE 2: equil (parallel) → must COMPLETE for all ligands
    # --------------------
    def _inject_mgr(
        p: Pipeline, stage_name: str, extra_payload: dict[str, Any] | None = None
    ) -> Pipeline:
        job_mgr.set_stage(stage_name)
        patched = []
        for s in p.ordered_steps():
            base_payload = s.payload or StepPayload()
            updates = {"job_mgr": job_mgr, "job_stage": stage_name}
            if rc.run.max_active_jobs is not None:
                updates["max_active_jobs"] = rc.run.max_active_jobs
            updates["batch_mode"] = batch_mode
            updates["batch_run_root"] = run_dir / "batch_run"
            updates["batch_gpus"] = getattr(rc.run, "batch_gpus", None)
            updates["batch_gpus_per_task"] = getattr(rc.run, "batch_gpus_per_task", 1)
            if extra_payload:
                updates.update(extra_payload)
            payload = base_payload.copy_with(**updates)
            patched.append(Step(name=s.name, requires=s.requires, payload=payload))
        return Pipeline(patched)

    def _inject_payload(p: Pipeline, **updates: Any) -> Pipeline:
        patched = []
        for s in p.ordered_steps():
            base_payload = s.payload or StepPayload()
            payload = base_payload.copy_with(**updates)
            patched.append(Step(name=s.name, requires=s.requires, payload=payload))
        return Pipeline(patched)

    phase_equil = _inject_mgr(phase_equil, "equil")
    if phase_equil.ordered_steps():
        finished = run_phase_skipping_done(
            phase_equil,
            children,
            "equil",
            backend,
            max_workers=1,
            on_failure=rc.run.on_failure,
        )
        if not finished:
            job_mgr.wait_all()
            if dry_run and job_mgr.triggered:
                logger.success(
                    "[DRY-RUN] Reached first SLURM submission point (equil). Exiting without submitting."
                )
                return
        children = handle_phase_failures(children, "equil", rc.run.on_failure)
    else:
        logger.info("[skip] equil: no steps in this protocol.")

    # --------------------
    # PHASE 2.5: equil_analysis (parallel) → prune UNBOUND if requested
    # --------------------
    # prune UNBOUND ligands before FE prep
    unbound_children: list[SimSystem] = []

    def _filter_bound(children_list):
        keep = []
        for c in children_list:
            if (c.root / "equil" / "UNBOUND").exists():
                lig = c.meta.get("ligand", c.name)
                logger.warning(f"Pruning UNBOUND ligand after equil: {lig}")
                unbound_children.append(c)
                continue
            keep.append(c)
        return keep

    if phase_equil_analysis.ordered_steps():
        run_phase_skipping_done(
            phase_equil_analysis,
            children,
            "equil_analysis",
            backend,
            max_workers=rc.run.max_workers,
            on_failure=rc.run.on_failure,
        )
        children = handle_phase_failures(children, "equil_analysis", rc.run.on_failure)
        children = _filter_bound(children)
    else:
        logger.info("[skip] equil_analysis: no steps in this protocol.")

    # --------------------
    # PHASE 2.6: pre_prepare_fe (RBFE ligand prep) → z-1 only
    # --------------------
    if phase_pre_prepare_fe.ordered_steps():
        phase_pre_prepare_fe = _inject_payload(
            phase_pre_prepare_fe,
            components=["z"],
            component_lambdas={"z": [0.0]},
            phase_name="pre_prepare_fe",
        )
        run_phase_skipping_done(
            phase_pre_prepare_fe,
            children,
            "pre_prepare_fe",
            backend,
            max_workers=rc.run.max_workers,
            on_failure=rc.run.on_failure,
        )
        children = handle_phase_failures(children, "pre_prepare_fe", rc.run.on_failure)
    else:
        logger.info("[skip] pre_prepare_fe: no steps in this protocol.")

    # --------------------
    # PHASE 2.7: pre_fe_equil → must COMPLETE for all ligands
    # --------------------
    phase_pre_fe_equil = _inject_mgr(
        phase_pre_fe_equil,
        "pre_fe_equil",
        extra_payload={"phase_name": "pre_fe_equil", "extra_env": {"SKIP_WINDOW_EQ": "1"}},
    )
    if phase_pre_fe_equil.ordered_steps():
        finished = run_phase_skipping_done(
            phase_pre_fe_equil,
            children,
            "pre_fe_equil",
            backend,
            max_workers=1,
            on_failure=rc.run.on_failure,
        )
        if not finished:
            job_mgr.wait_all()
            if dry_run and job_mgr.triggered:
                logger.success(
                    "[DRY-RUN] Reached first SLURM submission point (pre_fe_equil). Exiting without submitting."
                )
                return
        children = handle_phase_failures(children, "pre_fe_equil", rc.run.on_failure)
    else:
        logger.info("[skip] pre_fe_equil: no steps in this protocol.")

    # --------------------
    # RBFE: build transformation systems (pairs) after pre_fe_equil
    # --------------------
    if rc.protocol == "rbfe":
        from batter.rbfe import RBFENetwork
        from batter.config.utils import sanitize_ligand_name

        available = [c.meta.get("ligand") for c in children if c.meta.get("ligand")]
        if len(available) < 2:
            raise RuntimeError(
                "RBFE requires at least two ligands that completed equilibration."
            )
        available = [sanitize_ligand_name(x) for x in available]
        available_set = set(available)

        rbfe_network_path = config_dir / "rbfe_network.json"
        if rbfe_network_path.exists():
            payload = json.loads(rbfe_network_path.read_text())
        else:
            from batter.config.run import RBFENetworkArgs

            rbfe_cfg = rc.rbfe or RBFENetworkArgs()
            payload = _build_rbfe_network_plan(
                list(lig_map.keys()), lig_map, rbfe_cfg, config_dir
            )

        pairs = payload.get("pairs") or []
        if not pairs:
            raise RuntimeError("RBFE mapping produced no ligand pairs.")

        cleaned_pairs = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuntimeError(f"RBFE mapping entries must be 2-tuples; got {pair!r}.")
            cleaned_pairs.append(
                (
                    sanitize_ligand_name(str(pair[0])),
                    sanitize_ligand_name(str(pair[1])),
                )
            )

        if (rc.run.on_failure or "").lower() in {"prune", "retry"}:
            pruned = [
                p
                for p in cleaned_pairs
                if p[0] in available_set and p[1] in available_set
            ]
            if not pruned:
                raise RuntimeError(
                    "RBFE mapping does not include any available ligands after pruning."
                )
            if len(pruned) != len(cleaned_pairs):
                logger.warning(
                    "Pruned %d RBFE pair(s) due to on_failure=%s.",
                    len(cleaned_pairs) - len(pruned),
                    rc.run.on_failure,
                )
            cleaned_pairs = pruned

        network = RBFENetwork.from_ligands(available, mapping_fn=lambda _: cleaned_pairs)

        # Build transformation systems under simulations/transformations/
        trans_root = run_dir / "simulations" / "transformations"
        trans_root.mkdir(parents=True, exist_ok=True)
        rbfe_children: List[SimSystem] = []

        for ref, alt in network.pairs:
            pair_id = f"{ref}~{alt}"
            pair_dir = trans_root / pair_id
            inputs_dir = pair_dir / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)

            ref_src = Path(lig_map[ref])
            alt_src = Path(lig_map[alt])
            ref_dst = inputs_dir / f"{ref}{ref_src.suffix}"
            alt_dst = inputs_dir / f"{alt}{alt_src.suffix}"
            if not ref_dst.exists():
                shutil.copy2(ref_src, ref_dst)
            if not alt_dst.exists():
                shutil.copy2(alt_src, alt_dst)

            resn_ref = lig_resname_map.get(ref)
            resn_alt = lig_resname_map.get(alt)
            if not resn_ref or not resn_alt:
                raise RuntimeError(
                    f"Missing residue names for RBFE pair {pair_id}: {ref}={resn_ref}, {alt}={resn_alt}."
                )

            pair_meta = sys_exec.meta.merge(
                ligand=pair_id,
                residue_name=resn_ref,
                mode="RBFE",
                param_dir_dict=param_dir_dict,
                pair_id=pair_id,
                ligand_ref=ref,
                ligand_alt=alt,
                residue_ref=resn_ref,
                residue_alt=resn_alt,
                input_ref=str(ref_dst),
                input_alt=str(alt_dst),
            )

            rbfe_children.append(
                SimSystem(
                    name=f"{sys_exec.name}:{pair_id}:{run_id}",
                    root=pair_dir,
                    protein=sys_exec.protein,
                    topology=sys_exec.topology,
                    coordinates=sys_exec.coordinates,
                    ligands=tuple([ref_dst, alt_dst]),
                    lipid_mol=sys_exec.lipid_mol,
                    other_mol=sys_exec.other_mol,
                    anchors=sys_exec.anchors,
                    meta=pair_meta,
                )
            )

        # Switch to transformation systems for FE stages/results
        children = rbfe_children
        fe_children_all = rbfe_children
    
    # --------------------
    # PHASE 3: prepare_fe (parallel)
    # --------------------
    if phase_prepare_fe.ordered_steps():
        run_phase_skipping_done(
            phase_prepare_fe,
            children,
            "prepare_fe",
            backend,
            max_workers=rc.run.max_workers,
            on_failure=rc.run.on_failure,
        )
        children = handle_phase_failures(children, "prepare_fe", rc.run.on_failure)
    else:
        logger.info("[skip] prepare_fe: no steps in this protocol.")

    # --------------------
    # PHASE 4: fe_equil → must COMPLETE for all ligands
    # --------------------
    phase_fe_equil = _inject_mgr(phase_fe_equil, "fe_equil")
    if phase_fe_equil.ordered_steps():
        finished = run_phase_skipping_done(
            phase_fe_equil,
            children,
            "fe_equil",
            backend,
            max_workers=1,
            on_failure=rc.run.on_failure,
        )
        if not finished:
            job_mgr.wait_all()
            if dry_run and job_mgr.triggered:
                logger.success(
                    "[DRY-RUN] Reached first SLURM submission point (fe_equil). Exiting without submitting."
                )
                return
        children = handle_phase_failures(children, "fe_equil", rc.run.on_failure)
    else:
        logger.info("[skip] fe_equil: no steps in this protocol.")

    # --------------------
    # PHASE 5: fe → must COMPLETE for all ligands
    # --------------------
    phase_fe = _inject_mgr(phase_fe, "fe")
    has_fe_phase = bool(phase_fe.ordered_steps())
    if has_fe_phase:
        finished = run_phase_skipping_done(
            phase_fe,
            children,
            "fe",
            backend,
            max_workers=1,
            on_failure=rc.run.on_failure,
        )
        if not finished:
            job_mgr.wait_all()
            if dry_run and job_mgr.triggered:
                logger.success(
                    "[DRY-RUN] Reached first SLURM submission point (fe). Exiting without submitting."
                )
                return
        children = handle_phase_failures(children, "fe", rc.run.on_failure)
    else:
        logger.info("[skip] fe: no steps in this protocol.")

    # --------------------
    # PHASE 6: analyze (parallel)
    # --------------------
    def _inject_analysis_workers(p: Pipeline) -> Pipeline:
        patched = []
        for s in p.ordered_steps():
            payload = (s.payload or StepPayload()).copy_with(
                analysis_n_workers=rc.run.max_workers
            )
            patched.append(Step(name=s.name, requires=s.requires, payload=payload))
        return Pipeline(patched)

    phase_analyze = _inject_analysis_workers(phase_analyze)
    if phase_analyze.ordered_steps():
        run_phase_skipping_done(
            phase_analyze,
            children,
            "analyze",
            backend,
            max_workers=rc.run.max_workers,
            on_failure=rc.run.on_failure,
        )
        children = handle_phase_failures(children, "analyze", rc.run.on_failure)
    else:
        logger.info("[skip] analyze: no steps in this protocol.")

    # --------------------
    # FE record save
    # --------------------
    if not has_fe_phase:
        logger.info(
            "FE production skipped (--only-equil); ending run without FE record export."
        )
        return

    store = ArtifactStore(rc.run.output_folder)
    repo = FEResultsRepository(store)
    analysis_start_step = sim_cfg_updated.analysis_start_step
    if analysis_start_step is not None:
        analysis_start_step = int(analysis_start_step)
    failures: list[tuple[str, str, str]] = []
    if rc.protocol != "rbfe":
        for child in unbound_children:
            ligand = child.meta["ligand"]
            reason = "UNBOUND detected during equilibration"
            canonical_smiles, original_name, original_path = extract_ligand_metadata(
                child, lig_original_names
            )
            repo.record_failure(
                run_id=run_id,
                ligand=ligand,
                system_name=sim_cfg_updated.system_name,
                temperature=sim_cfg_updated.temperature,
                status="unbound",
                reason=reason,
                canonical_smiles=canonical_smiles,
                original_name=original_name,
                original_path=original_path,
                protocol=rc.protocol,
                analysis_start_step=analysis_start_step,
            )
            failures.append((ligand, "unbound", reason))
    failures.extend(
        save_fe_records(
            run_dir=run_dir,
            run_id=run_id,
            children_all=fe_children_all,
            sim_cfg_updated=sim_cfg_updated,
            repo=repo,
            protocol=rc.protocol,
            analysis_start_step=analysis_start_step,
        )
    )

    if failures:
        failed = ", ".join(
            [f"{n} ({status}: {reason})" for n, status, reason in failures]
        )
        logger.warning(f"{len(failures)} ligand(s) had post-run issues: {failed}")
    logger.success(
        f"All phases completed {run_dir}. FE records saved to repository {rc.run.output_folder}/results/."
    )

    _notify_run_completion(rc, run_id, run_dir, failures)


def _notify_run_completion(
    rc: RunConfig,
    run_id: str,
    run_dir: Path,
    failures: list[tuple[str, str, str]],
) -> None:
    recipient = rc.run.email_on_completion
    if not recipient:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    subject = f"BATTER run '{run_id}' of {rc.create.system_name} completed"
    results_path = Path(rc.run.output_folder) / "results"

    body_lines = [
        "Hi there!",
        "",
        f"Your BATTER run '{rc.create.system_name}' (run_id='{run_id}') completed at {timestamp} UTC.",
        f"Protocol: {rc.protocol}",
        f"Output folder: {run_dir}",
        f"FE records stored under: {results_path}",
        "",
    ]

    if failures:
        body_lines.append(
            "The following ligand(s) had post-run issues (see logs for additional context):"
        )
        for ligand, status, reason in failures:
            body_lines.append(f"- {ligand} ({status}): {reason}")
    else:
        body_lines.append("No ligand failures were detected.")

    body_lines.extend(
        [
            "",
            "Best wishes,",
            "BATTER",
        ]
    )

    message_body = "\n".join(body_lines)
    sender = rc.run.email_sender
    if not sender:
        logger.warning(
            "No sender email configured; cannot send completion notification. set `run.email_sender` in your YAML."
        )
        return
    message = (
        f"From: batter <{sender}>\n"
        f"To: {recipient}\n"
        f"Subject: {subject}\n\n"
        f"{message_body}"
    )

    try:
        with smtplib.SMTP("localhost") as smtp:
            smtp.sendmail(sender, [recipient], message)
        logger.info(f"Sent completion notification to {recipient}")
    except SMTPException as exc:
        logger.warning(f"Failed to send completion email to {recipient}: {exc}")
    except Exception as exc:  # pragma: no cover - best-effort notification
        logger.warning(f"Unexpected error while sending completion email: {exc}")
