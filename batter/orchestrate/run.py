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
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Type

import json

from loguru import logger

from batter.config.run import RunConfig
from batter.systems.core import SimSystem, SystemBuilder
from batter.systems.mabfe import MABFEBuilder
from batter.systems.masfe import MASFEBuilder
from batter.exec.local import LocalBackend
from batter.exec.slurm import SlurmBackend

from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step
from batter.pipeline.payloads import StepPayload

from batter.runtime.portable import ArtifactStore
from batter.runtime.fe_repo import FEResultsRepository, FERecord

from batter.exec.slurm_mgr import SlurmJobManager

from batter.orchestrate.backend import register_local_handlers
from batter.orchestrate.ligands import discover_staged_ligands, resolve_ligand_map
from batter.orchestrate.markers import (
    handle_phase_failures,
    run_phase_skipping_done,
    is_done,
)
from batter.orchestrate.pipeline_utils import select_pipeline
from batter.orchestrate.results_io import fallback_totals_from_json, parse_results_dat


def _normalize_for_hash(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        drop_keys = {"output_folder"}
        return {k: _normalize_for_hash(v) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def _compute_run_signature(
    yaml_path: Path,
    system_overrides: Dict[str, Any] | None,
    run_overrides: Dict[str, Any] | None,
) -> str:
    data = Path(yaml_path).read_bytes()
    payload = {
        "system_overrides": _normalize_for_hash(system_overrides or {}),
        "run_overrides": _normalize_for_hash(run_overrides or {}),
    }
    frozen = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256()
    h.update(data)
    h.update(b"\0")
    h.update(frozen)
    return h.hexdigest()


def _stored_signature(run_dir: Path) -> tuple[str | None, Path]:
    sig_path = run_dir / "artifacts" / "config" / "run_config.hash"
    if sig_path.exists():
        return sig_path.read_text().strip(), sig_path
    return None, sig_path


def _resolve_signature_conflict(
    stored_sig: str | None,
    config_signature: str,
    requested_run_id: str | None,
    allow_run_id_mismatch: bool,
    *,
    run_id: str,
    run_dir: Path,
) -> bool:
    """
    Decide whether to continue using ``run_dir`` given the stored signature.

    Returns
    -------
    bool
        ``True`` if the caller should continue with ``run_dir``; ``False`` if a new
        run directory must be created (only possible when ``requested_run_id`` is
        ``"auto"``). Raises ``RuntimeError`` when a mismatch is not permitted.
    """
    if stored_sig is None or stored_sig == config_signature:
        return True
    if requested_run_id == "auto":
        return False
    if allow_run_id_mismatch:
        logger.warning(
            f"Execution '{run_id}' already exists with configuration hash {stored_sig[:12]} (current {config_signature[:12]}); "
            "continuing because --allow-run-id-mismatch is enabled.",
        )
        return True
    raise RuntimeError(
        f"Execution '{run_id}' already exists with a different configuration. "
        "Choose a different --run-id, enable --allow-run-id-mismatch, or update the existing run."
    )


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


def _select_system_builder(protocol: str, system_type: str | None) -> SystemBuilder:
    builder_cls, expected_type = _builder_info_for_protocol(protocol)
    if system_type and system_type != expected_type:
        raise ValueError(
            f"system.type={system_type!r} is incompatible with protocol '{protocol}'. "
            f"Expected '{expected_type}'. Remove or update 'system.type'."
        )
    return builder_cls()


def select_run_id(
    sys_root: Path | str, protocol: str, system_name: str, requested: str | None
) -> Tuple[str, Path]:
    """Resolve the execution run identifier and backing directory.

    Parameters
    ----------
    sys_root : str or pathlib.Path
        Root directory that stores executions for the system.
    protocol : str
        Protocol name used to label the run identifier.
    system_name : str
        Logical system name; used when autogenerating identifiers.
    requested : str or None
        User-specified run id. ``"auto"`` or ``None`` triggers automatic selection.

    Returns
    -------
    tuple
        Two-element tuple ``(run_id, run_dir)`` where ``run_dir`` is guaranteed to
        exist on disk.
    """
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
    """Generate a timestamped identifier for a new execution.

    Parameters
    ----------
    protocol : str
        Name of the protocol (e.g., ``"abfe"``) that will run.
    system_name : str
        Logical system label used to tie runs to systems on disk.

    Returns
    -------
    str
        Identifier formatted as ``"{protocol}-{system_name}-{timestamp}"``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{protocol}-{system_name}-{ts}"


def run_from_yaml(
    path: Path | str,
    on_failure: Literal["prune", "raise", "retry"] = None,
    system_overrides: Dict[str, Any] = None,
    run_overrides: Dict[str, Any] | None = None,
) -> None:
    """Execute a BATTER workflow described by a YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the top-level run YAML file.
    on_failure : {"prune", "raise", "retry"}, optional
        Override for the failure policy applied to ligand pipelines.
    system_overrides : dict, optional
        Mapping of fields that should override ``system`` section values at load time.
    run_overrides : dict, optional
        Overrides applied to the ``run`` section (e.g., only FE preparation).
    """
    path = Path(path)
    logger.info(f"Starting BATTER run from {path}")

    # Configs
    rc = RunConfig.load(path)
    if system_overrides:
        logger.info(f"Applying system overrides: {system_overrides}")
        rc = rc.model_copy(
            update={"system": rc.system.model_copy(update=system_overrides)}
        )
    if run_overrides:
        logger.info(f"Applying run overrides: {run_overrides}")
        rc = rc.model_copy(update={"run": rc.run.model_copy(update=run_overrides)})
    if on_failure:
        logger.info(f"on_failure behavior: {on_failure}")
        rc.run.on_failure = on_failure

    yaml_dir = path.parent

    # ligand params output directory
    if rc.create.param_outdir is None:
        rc.create.param_outdir = str(Path(rc.system.output_folder) / "ligand_params")
    else:
        logger.info(
            f"Using user-specified ligand param_outdir: {rc.create.param_outdir}"
        )

    # Build system-prep params exactly once
    sys_params = {
        "param_outdir": str(rc.create.param_outdir),
        "system_name": rc.create.system_name,
        "protein_input": str(rc.create.protein_input),
        "system_input": str(rc.create.system_input),
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
        "extra_conformation_restraints": rc.create.extra_conformation_restraints,
    }

    sim_cfg = rc.resolved_sim_config()
    logger.info(f"Loaded simulation config for system: {sim_cfg.system_name}")

    # Backend
    if rc.backend == "slurm":
        backend = SlurmBackend()
    else:
        backend = LocalBackend()
        register_local_handlers(backend)

    # Shared System Build (system-level assets live under sys.root)
    builder = _select_system_builder(rc.protocol, rc.system.type)

    requested_run_id = getattr(rc.run, "run_id", "auto")
    config_signature = _compute_run_signature(path, system_overrides, run_overrides)

    while True:
        run_id, run_dir = select_run_id(
            rc.system.output_folder,
            rc.protocol,
            rc.create.system_name,
            requested_run_id,
        )
        stored_sig, sig_path = _stored_signature(run_dir)
        if _resolve_signature_conflict(
            stored_sig,
            config_signature,
            requested_run_id,
            rc.run.allow_run_id_mismatch,
            run_id=run_id,
            run_dir=run_dir,
        ):
            break
        logger.info(
            f"Existing execution {run_dir} uses different configuration hash ({stored_sig[:12]}); creating a fresh run.",
        )
        requested_run_id = "auto"
        run_id = generate_run_id(rc.protocol, rc.create.system_name)
        run_dir = Path(rc.system.output_folder) / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        continue

    logger.info(f"Using run_id='{run_id}' under {run_dir}")
    _, sig_path = _stored_signature(run_dir)

    # Ligands
    staged_lig_map = discover_staged_ligands(run_dir)
    if staged_lig_map:
        lig_map = staged_lig_map
        logger.info(
            f"Resuming with {len(lig_map)} staged ligands discovered under {run_dir}"
        )
    else:
        # Fall back to YAML resolution (requires original paths/files to exist)
        lig_map = resolve_ligand_map(rc, yaml_dir)
    rc.create.ligand_paths = {k: str(v) for k, v in lig_map.items()}
    sys_params.update({"ligand_paths": rc.create.ligand_paths})

    sys_exec = SimSystem(name=rc.create.system_name, root=run_dir)
    sys_exec = builder.build(sys_exec, rc.create)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(config_signature + "\n")

    # Per-execution run directory (auto-resume latest when 'auto')
    logger.add(run_dir / "batter.run.log", level="DEBUG")

    dry_run = rc.run.dry_run
    if dry_run:
        logger.warning("DRY RUN mode enabled: no SLURM jobs will be submitted.")

    # SLURM manager (registry per execution)
    slurm_flags = rc.run.slurm.to_sbatch_flags() if rc.run.slurm else None
    job_mgr = SlurmJobManager(
        poll_s=60 * 15,
        max_retries=3,
        resubmit_backoff_s=30,
        registry_file=(run_dir / ".slurm" / "queue.jsonl"),
        dry_run=dry_run,
        sbatch_flags=slurm_flags,
    )

    # Build pipeline with explicit sys_params
    tpl = select_pipeline(
        rc.protocol, sim_cfg, rc.run.only_fe_preparation, sys_params=sys_params
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
            backend.run(step, run_sys, step.params)

    # Locate sim_overrides from system_prep (under run_dir)
    overrides_path = run_dir / "artifacts" / "config" / "sim_overrides.json"
    sim_cfg_updated = sim_cfg
    if overrides_path.exists():
        upd = json.loads(overrides_path.read_text()) or {}
        sim_cfg_updated = sim_cfg.model_copy(
            update={k: v for k, v in upd.items() if v is not None}
        )

        from batter.config.io import write_yaml_config

        (run_dir / "artifacts" / "config").mkdir(parents=True, exist_ok=True)
        write_yaml_config(
            sim_cfg_updated, run_dir / "artifacts" / "config" / "sim.resolved.yaml"
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
    phase_prepare_fe = _phase(PH_PREPARE_FE)
    phase_fe_equil = _phase(PH_FE_EQUIL)
    phase_fe = _phase(PH_FE)
    phase_analyze = _phase(PH_ANALYZE)

    # --- build SimSystem children ---
    param_idx_path = run_dir / "artifacts" / "ligand_params" / "index.json"
    if not param_idx_path.exists():
        raise FileNotFoundError(f"Missing ligand param index: {param_idx_path}")
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
        )
        children = handle_phase_failures(children, "prepare_equil", rc.run.on_failure)
    else:
        logger.info(f"[skip] prepare_equil: no steps in this protocol.")

    # --------------------
    # PHASE 2: equil (parallel) → must COMPLETE for all ligands
    # --------------------
    def _inject_mgr(p: Pipeline) -> Pipeline:
        patched = []
        for s in p.ordered_steps():
            base_payload = s.payload or StepPayload()
            updates = {"job_mgr": job_mgr}
            if rc.run.max_active_jobs is not None:
                updates["max_active_jobs"] = rc.run.max_active_jobs
            payload = base_payload.copy_with(**updates)
            patched.append(Step(name=s.name, requires=s.requires, payload=payload))
        return Pipeline(patched)

    phase_equil = _inject_mgr(phase_equil)
    if phase_equil.ordered_steps():
        finished = run_phase_skipping_done(
            phase_equil, children, "equil", backend, max_workers=rc.run.max_workers
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
        logger.info(f"[skip] equil: no steps in this protocol.")

    # --------------------
    # PHASE 2.5: equil_analysis (parallel) → prune UNBOUND if requested
    # --------------------
    # prune UNBOUND ligands before FE prep
    def _filter_bound(children_list):
        keep = []
        for c in children_list:
            if (c.root / "equil" / "UNBOUND").exists():
                lig = c.meta.get("ligand", c.name)
                logger.warning(f"Pruning UNBOUND ligand after equil: {lig}")
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
        )
        children = handle_phase_failures(children, "equil_analysis", rc.run.on_failure)
        children = _filter_bound(children)
    else:
        logger.info("[skip] equil_analysis: no steps in this protocol.")

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
        )
        children = handle_phase_failures(children, "prepare_fe", rc.run.on_failure)
    else:
        logger.info("[skip] prepare_fe: no steps in this protocol.")

    # --------------------
    # PHASE 4: fe_equil → must COMPLETE for all ligands
    # --------------------
    phase_fe_equil = _inject_mgr(phase_fe_equil)
    if phase_fe_equil.ordered_steps():
        finished = run_phase_skipping_done(
            phase_fe_equil,
            children,
            "fe_equil",
            backend,
            max_workers=rc.run.max_workers,
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
    phase_fe = _inject_mgr(phase_fe)
    has_fe_phase = bool(phase_fe.ordered_steps())
    if has_fe_phase:
        finished = run_phase_skipping_done(
            phase_fe, children, "fe", backend, max_workers=rc.run.max_workers
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
            phase_analyze, children, "analyze", backend, max_workers=rc.run.max_workers
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

    # Store at the system store (shared across executions of this system)
    store = ArtifactStore(rc.system.output_folder)
    repo = FEResultsRepository(store)

    failures: list[tuple[str, str]] = []
    for child in children_all:
        lig_name = child.meta["ligand"]
        mol_name = child.meta["residue_name"]
        results_dir = child.root / "fe" / "Results"
        total_dG, total_se = None, None

        # Preferred: parse Results.dat
        dat = results_dir / "Results.dat"
        if dat.exists():
            try:
                tdg, tse, _ = parse_results_dat(dat)
                total_dG, total_se = tdg, tse
            except Exception as e:
                logger.warning(f"[{lig_name}] Failed to parse Results.dat: {e}")

        # Fallback: try JSON component results
        if total_dG is None or total_se is None:
            tdg, tse = fallback_totals_from_json(results_dir)
            total_dG = tdg if total_dG is None else total_dG
            total_se = tse if total_se is None else total_se

        if total_dG is None or total_se is None:
            failures.append((lig_name, "no_totals_found"))
            logger.warning(f"[{lig_name}] No totals found under {results_dir}")
            continue

        try:
            rec = FERecord(
                run_id=run_id,
                ligand=lig_name,
                mol_name=mol_name,
                system_name=sim_cfg_updated.system_name,
                fe_type=sim_cfg_updated.fe_type,
                temperature=sim_cfg_updated.temperature,
                method=sim_cfg_updated.dec_int,
                total_dG=total_dG,
                total_se=total_se,
                components=list(sim_cfg_updated.components),
                windows=[],  # optional: can be populated later
            )
            repo.save(rec, copy_from=results_dir)
            logger.info(
                f"Saved FE record for ligand {lig_name}"
                f"(ΔG={total_dG:.2f} ± {total_se:.2f} kcal/mol; run_id={run_id})"
            )
        except Exception as e:
            logger.warning(f"Could not save FE record for {lig_name}: {e}")
            failures.append((lig_name, f"save_failed: {e}"))

    if failures:
        failed = ", ".join([f"{n} ({m})" for n, m in failures])
        logger.warning(f"{len(failures)} ligand(s) had post-run issues: {failed}")
    logger.success(
        f"All phases completed {run_dir}. FE records saved to repository {rc.system.output_folder}/results/."
    )
