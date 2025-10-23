"""
batter.orchestrate.run
======================

Top-level orchestration entry for BATTER runs.

This module wires:
YAML (RunConfig) → shared system build → bulk ligand staging →
single param job ("param_ligands") → per-ligand pipelines → FE record save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
from datetime import datetime, timezone
import json
import shutil
import re
import os

from loguru import logger

from batter.config.run import RunConfig
from batter.systems.core import SimSystem
from batter.systems.mabfe import MABFEBuilder
from batter.exec.local import LocalBackend
from batter.exec.slurm import SlurmBackend

from batter.pipeline.factory import make_abfe_pipeline, make_asfe_pipeline
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step, ExecResult

from batter.runtime.portable import ArtifactStore
from batter.runtime.fe_repo import FEResultsRepository, FERecord

from batter.exec.slurm_mgr import SlurmJobManager

from batter.utils import components_under

# -----------------------------------------------------------------------------
# Pipeline utilities
# -----------------------------------------------------------------------------

def _select_pipeline(
    protocol: str,
    sim_cfg,
    only_fe_prep: bool,
    sys_params: dict | None = None,
) -> Pipeline:
    p = (protocol or "abfe").lower()
    if p == "abfe":
        return make_abfe_pipeline(
            sim_cfg,
            sys_params=sys_params or {},
            only_fe_preparation=only_fe_prep,
        )
    if p == "asfe":
        return make_asfe_pipeline(sim_cfg, only_fe_preparation=only_fe_prep)
    if p == "rbfe":
        raise NotImplementedError("RBFE protocol is not yet implemented.")
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def _pipeline_prune_requires(p: Pipeline, removed: set[str]) -> Pipeline:
    """Return a new Pipeline where any 'requires' that reference removed steps are dropped."""
    new_steps: List[Step] = []
    for s in p.ordered_steps():
        if s.name in removed:
            continue
        new_steps.append(
            Step(
                name=s.name,
                requires=[r for r in s.requires if r not in removed],
                params=dict(s.params),
            )
        )
    return Pipeline(new_steps)


def _gen_run_id(protocol: str, ligand: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{protocol}-{ligand}-{ts}"


def resolve_placeholders(text: str, sys: SimSystem) -> str:
    return text.replace("{WORK}", str(sys.root)).replace("{SYSTEM}", sys.name)


def partition_children_by_status(
    children: List[SimSystem],
    phase: str,
    finished_name: str = "FINISHED",
    failed_name: str = "FAILED",
) -> Tuple[List[SimSystem], List[SimSystem]]:
    """Return (ok_children, failed_children) based on sentinel files under <child.root>/<phase>/."""
    ok, bad = [], []
    for child in children:
        p = child.root / phase
        if (p / finished_name).exists():
            ok.append(child)
        elif (p / failed_name).exists():
            bad.append(child)
        else:
            # If neither exists, treat as failed (job manager should have resolved it already).
            bad.append(child)
    return ok, bad

# -----------------------------------------------------------------------------
# Ligand resolution
# -----------------------------------------------------------------------------

def _resolve_ligand_map(rc, yaml_dir: Path) -> dict[str, Path]:
    """
    Resolve ligands from either `create.ligand_paths` (list) or
    `create.ligand_input` (JSON file; can be dict or list).
    Returns {LIG_NAME: absolute Path}.

    Ligand names are sanitized to be uppercase alphanumeric/underscore only.
    """
    def _sanitize_name(name: str) -> str:
        """
        Make ligand name safe for filesystem usage:
            - Uppercase
            - Replace unsafe chars (non-alphanumeric/underscore) with '_'
            - Trim trailing underscores
        """
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
        return safe.strip("_").upper()

    lig_map: dict[str, Path] = {}

    # Case A: explicit list in YAML
    paths = getattr(rc.create, "ligand_paths", None) or []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else (yaml_dir / p)
        name = _sanitize_name(p.stem)
        lig_map[name] = p.resolve()

    # Case B: JSON file mapping
    lig_json = getattr(rc.create, "ligand_input", None)
    if lig_json:
        jpath = Path(lig_json)
        jpath = jpath if jpath.is_absolute() else (yaml_dir / jpath)
        data = json.loads(jpath.read_text())

        if isinstance(data, dict):
            for name, p in data.items():
                p = Path(p)
                p = p if p.is_absolute() else (jpath.parent / p)
                name = _sanitize_name(name)
                lig_map[name] = p.resolve()
        elif isinstance(data, list):
            for p in data:
                p = Path(p)
                p = p if p.is_absolute() else (jpath.parent / p)
                name = _sanitize_name(p.stem)
                lig_map[name] = p.resolve()
        else:
            raise TypeError(f"{jpath} must be a dict or list, got {type(data).__name__}")

    if not lig_map:
        raise ValueError(
            "No ligands provided. Specify `create.ligand_paths` or `create.ligand_input` in your YAML."
        )

    missing = [str(p) for p in lig_map.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Ligand file(s) not found: {missing}")

    return lig_map


# -----------------------------------------------------------------------------
# Local backend minimal handlers
# -----------------------------------------------------------------------------

def _register_local_handlers(backend: LocalBackend) -> None:
    from batter.systems.core import SimSystem
    from batter.exec.handlers.system_prep import system_prep as _system_prep
    from batter.exec.handlers.param_ligands import param_ligands as _param_ligands
    from batter.exec.handlers.prepare_equil import prepare_equil_handler as _prepare_equil
    from batter.exec.handlers.equil import equil_handler as _equil
    from batter.exec.handlers.equil_analysis import equil_analysis_handler as _equil_analysis
    from batter.exec.handlers.prepare_fe import prepare_fe_handler as _prepare_fe
    from batter.exec.handlers.prepare_fe import prepare_fe_windows_handler as _prepare_fe_windows
    from batter.exec.handlers.fe import fe_equil_handler as _fe_equil
    from batter.exec.handlers.fe import fe_handler as _fe
    from batter.exec.handlers.fe_analysis import analyze_handler as _analyze

    backend.register("system_prep", _system_prep)
    backend.register("param_ligands", _param_ligands)
    backend.register("prepare_equil", _prepare_equil)
    backend.register("equil", _equil)
    backend.register("equil_analysis", _equil_analysis)
    backend.register("prepare_fe", _prepare_fe)
    backend.register("prepare_fe_windows", _prepare_fe_windows)
    backend.register("fe_equil", _fe_equil)
    backend.register("fe", _fe)
    backend.register("analyze", _analyze)

    logger.debug(f"Registered LOCAL handlers: {list(backend._handlers.keys())}")

# --- phase skipping utilities -----------------------------------------------

REQUIRED_MARKERS = {
    "system_prep": [["artifacts/config/sim_overrides.json"]],
    "param_ligands": [["artifacts/ligand_params/index.json"]],
    "prepare_equil": [["equil/full.prmtop", "equil/artifacts/prepare_equil.ok"]],
    "equil":         [["equil/FINISHED"], ["equil/FAILED"]],
    "equil_analysis":[["equil/representative.pdb"], ["equil/UNBOUND"]],
    "prepare_fe":    [["fe/artifacts/prepare_fe.ok", "fe/artifacts/prepare_fe_windows.ok"]],
    "fe_equil":      [["fe/{comp}/{comp}-1/EQ_FINISHED"]],
    "fe":            [["fe/{comp}/{comp}{win:02d}/FINISHED"]],
    "analyze":       [["fe/artifacts/analyze.ok"]],
}


def _production_windows_under(root: Path, comp: str) -> list[int]:
    """
    Return sorted list of integer window indices for <ligand>/fe/<comp>/<compN>
    (exclude equil dir <comp>-1).
    """
    base = root / "fe" / comp
    if not base.exists():
        return []
    out: list[int] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name == f"{comp}-1":
            continue  # equil folder
        if not name.startswith(comp):
            continue
        tail = name[len(comp):]  # e.g., '0', '1', '-3', etc.
        try:
            idx = int(tail)
        except ValueError:
            continue
        # production windows are >= 0
        if idx >= 0:
            out.append(idx)
    return sorted(out)

def _dnf_satisfied(root: Path, marker_spec) -> bool:
    """
    marker_spec can be:
      - list[str]          → ANY of these (back-compat)
      - list[list[str]]    → DNF: ANY group satisfied; each group is ALL-of
    """ 
    if not marker_spec:
        return False

    # Backward-compatible: flat list means ANY-of
    if all(isinstance(m, str) for m in marker_spec):
        return any((root / m).exists() for m in marker_spec)

    # DNF groups: satisfied if ANY group has ALL files present
    for group in marker_spec:
        if all((root / p).exists() for p in group):
            return True
    
    return False

def _is_done(system: SimSystem, phase_name: str) -> bool:
    root = system.root
    spec = REQUIRED_MARKERS.get(phase_name, [])
    if not spec:
        return False

    # Simple phases use the generic DNF logic as-is
    if phase_name not in {"fe_equil", "fe"}:
        return _dnf_satisfied(root, spec)

    # Expand placeholders and require ALL components (and ALL windows for 'fe')
    comps = components_under(root)
    if not comps:
        return False

    if phase_name == "fe_equil":
        # Every component must satisfy: fe/{comp}/{comp}-1/EQ_FINISHED
        for comp in comps:
            expanded_groups = []
            for group in spec:
                expanded_groups.append([p.format(comp=comp, win="") for p in group])
            if not _dnf_satisfied(root, expanded_groups):
                return False
        
        return True

    if phase_name == "fe":
        # Every component AND every production window must satisfy: fe/{comp}/{comp}{win:02d}/FINISHED
        for comp in comps:
            wins = _production_windows_under(root, comp)
            if not wins:
                return False  # no windows → not done
            for win in wins:
                expanded_groups = []
                for group in spec:
                    expanded_groups.append([p.format(comp=comp, win=win) for p in group])
                if not _dnf_satisfied(root, expanded_groups):
                    return False
        return True

    return False

def _filter_needing_phase(children: list[SimSystem], phase_name: str) -> list[SimSystem]:
    if phase_name not in REQUIRED_MARKERS:
        return list(children)
    need = [c for c in children if not _is_done(c, phase_name)]
    done = [c for c in children if c not in need]
    if done:
        names = ", ".join(c.meta.get("ligand", c.name) for c in done)
        logger.debug(f"[skip] {phase_name}: {len(done)} ligand(s) already complete → {names}")
    return need

def _run_phase_skipping_done(phase: Pipeline, children: list[SimSystem],
                             phase_name: str, backend,
                             max_workers: int | None = None
                             ) -> list[SimSystem]:
    """Run a phase only for ligands that still need it. Returns True if all were already done."""
    todo = _filter_needing_phase(children, phase_name)
    if not todo:
        logger.info(f"[skip] {phase_name}: all ligands already complete.")
        return True
    logger.info(f"{phase_name}: {len(todo)} ligand(s) not finished → running phase..."
                f"(of {len(children)} total).")
    backend.run_parallel(phase, todo, description=phase_name, max_workers=max_workers)
    return False

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------



def run_from_yaml(path: Path | str, on_failure: Literal["prune", "raise"] = None,
                  system_overrides: Dict[str, Any] = None) -> None:
    """
    Run a full BATTER workflow from a YAML configuration.

    Process
    -------
    1. Load RunConfig and SimulationConfig
    2. Choose backend (local/slurm)
    3. Build shared system once
    4. Stage **all ligands at once** under <work>/ligands/<NAME>/
    5. Run **system_prep** and **param_ligands** ONCE at the parent root
    6. For each ligand, run the rest of the pipeline on its child system
    7. Save one FE record per ligand
    """
    path = Path(path)
    logger.info(f"Starting BATTER run from {path}")

    # Configs
    rc = RunConfig.load(path)
    if system_overrides:
        logger.info(f"Applying system overrides: {system_overrides}")
        rc = rc.model_copy(update={
            "system": rc.system.model_copy(update=system_overrides)
        })
    yaml_path = Path(path)
    yaml_dir = yaml_path.parent

    # lf ligand_input load json and set ligand_paths 
    lig_map = _resolve_ligand_map(rc, yaml_dir)
    rc.create.ligand_paths = {k: str(v) for k, v in lig_map.items()}

    # if rc.create.param_outdir is None, set to default under work/
    if rc.create.param_outdir is None:
        rc.create.param_outdir = str(Path(rc.system.output_folder) / "ligand_params")
    else:
        logger.info(f"Using user-specified ligand param_outdir: {rc.create.param_outdir}")

    # update on_failure
    if on_failure:
        rc.run.on_failure = on_failure

    # Build system-prep params exactly once
    sys_params = {
        "param_outdir": str(rc.create.param_outdir),
        "system_name": rc.create.system_name,
        "protein_input": str(rc.create.protein_input),
        "system_input": str(rc.create.system_input),
        "system_coordinate": (str(rc.create.system_coordinate) if rc.create.system_coordinate else None),
        "ligand_paths": rc.create.ligand_paths,
        "anchor_atoms": list(rc.create.anchor_atoms or []),
        "protein_align": str(rc.create.protein_align),
        "lipid_mol": list(rc.create.lipid_mol or []),
        "other_mol": list(rc.create.other_mol or []),
        "ligand_ff": rc.create.ligand_ff,
        "retain_lig_prot": bool(rc.create.retain_lig_prot),
        "yaml_dir": str(yaml_dir),
        "extra_restraints": rc.create.extra_restraints or "",
        "extra_restraint_fc": rc.create.extra_restraint_fc,
        "extra_conformation_restraints": rc.create.extra_conformation_restraints or "",
    }


    sim_cfg = rc.resolved_sim_config()
    logger.info(f"Loaded simulation config for system: {sim_cfg.system_name}")
    # add debug logging to the output
    logger.add(rc.system.output_folder / "batter.log", level="DEBUG")

    # Backend
    if rc.backend == "slurm":
        backend = SlurmBackend()
    else:
        backend = LocalBackend()
        _register_local_handlers(backend)

    # Shared System Build
    if rc.system.type != "MABFE":
        raise ValueError(f"Unsupported system.type={rc.system.type!r}. Only 'MABFE' is implemented.")


    def _inject_mgr(p: Pipeline) -> Pipeline:
        patched = []
        for s in p.ordered_steps():
            prm = dict(s.params)
            prm["job_mgr"] = job_mgr
            patched.append(Step(name=s.name, requires=s.requires, params=prm))
        return Pipeline(patched)

    builder = MABFEBuilder()
    sys = SimSystem(name=rc.create.system_name, root=rc.system.output_folder)
    sys = builder.build(sys, rc.create)

    dry_run = rc.run.dry_run
    if dry_run:
        logger.warning("DRY RUN mode enabled: no SLRUM jobs will be submitted.")

    slurm_flags = rc.run.slurm.to_sbatch_flags() if rc.run.slurm else None
    job_mgr = SlurmJobManager(poll_s=60*15, max_retries=3, resubmit_backoff_s=30,
        registry_file=(Path(sys.root) / ".slurm" / "queue.jsonl"), dry_run=dry_run,
        sbatch_flags=slurm_flags
    )
    
    lig_root = sys.root / "simulations"
    lig_root.mkdir(parents=True, exist_ok=True)

    for lig_name, lig_path in lig_map.items():
        builder.make_child_for_ligand(sys, lig_name, lig_path)

    logger.debug(f"Staged {len(lig_map)} ligand subsystems under {lig_root}")

    # Build pipeline with explicit sys_params
    tpl = _select_pipeline(
        rc.protocol,
        sim_cfg,
        rc.run.only_fe_preparation,
        sys_params=sys_params,
    )

    # Run parent-only steps (system_prep, param_ligands) at system root
    parent_only = Pipeline([s for s in tpl.ordered_steps() if s.name in {"system_prep", "param_ligands"}])
    if parent_only.ordered_steps():
        names = [s.name for s in parent_only.ordered_steps()]
        logger.debug(f"Executing parent-only steps at {sys.root}: {names}")
        for step in parent_only.ordered_steps():
            # ---- Check whether to skip ----
            if _is_done(sys, step.name):
                logger.info(f"[skip] {step.name}: finished.")
                continue
            backend.run(step, sys, step.params)
    
    # Locate sim_overrides from system_prep
    overrides_path = sys.root / "artifacts" / "config" / "sim_overrides.json"
    sim_cfg_updated = sim_cfg
    if overrides_path.exists():
        upd = json.loads(overrides_path.read_text()) or {}
        sim_cfg_updated = sim_cfg.model_copy(update={k: v for k, v in upd.items() if v is not None})

        from batter.config.io import write_yaml_config
        (sys.root / "artifacts" / "config").mkdir(parents=True, exist_ok=True)
        write_yaml_config(sim_cfg_updated, sys.root / "artifacts" / "config" / "sim.resolved.yaml")


    # Now build a fresh pipeline for per-ligand steps using the UPDATED sim
    removed = {"system_prep", "param_ligands"}

    per_lig_steps: List[Step] = []
    for s in tpl.ordered_steps():
        if s.name in removed:
            continue
        p = dict(s.params)
        per_lig_steps.append(
            Step(
                name=s.name,
                requires=[r for r in s.requires if r not in removed],
                params=p,
            )
        )
    per_lig = Pipeline(per_lig_steps)

    # IMPORTANT: also update the `sim` param on each remaining step
    patched = []
    for s in per_lig.ordered_steps():
        p = dict(s.params)
        if "sim" in p:
            p["sim"] = sim_cfg_updated.model_dump()
        patched.append(Step(name=s.name, requires=s.requires, params=p))
    per_lig = Pipeline(patched)

    # --- define phases explicitly ---
    PH_PREPARE_EQUIL = {"prepare_equil"}
    PH_EQUIL         = {"equil"}
    PH_EQUIL_ANALYSIS  = {"equil_analysis"}
    PH_PREPARE_FE    = {"prepare_fe", "prepare_fe_windows"}
    PH_FE_EQUIL      = {"fe_equil"}
    PH_FE            = {"fe"}
    PH_ANALYZE       = {"analyze"}

    def _phase(names: set[str]) -> Pipeline:
        """
        Build a sub-pipeline containing only steps in `names`,
        with `requires` pruned to dependencies that are also in `names`.
        """
        selected = [s for s in per_lig.ordered_steps() if s.name in names]
        selected_names = {s.name for s in selected}
        pruned = [
            Step(
                name=s.name,
                requires=[r for r in s.requires if r in selected_names],
                params=dict(s.params),
            )
            for s in selected
        ]
        return Pipeline(pruned)

    phase_prepare_equil = _phase(PH_PREPARE_EQUIL)
    phase_equil         = _phase(PH_EQUIL)
    phase_equil_analysis  = _phase(PH_EQUIL_ANALYSIS)
    phase_prepare_fe    = _phase(PH_PREPARE_FE)
    phase_fe_equil      = _phase(PH_FE_EQUIL)
    phase_fe            = _phase(PH_FE)
    phase_analyze       = _phase(PH_ANALYZE)

    # --- build SimSystem children ---
    param_idx_path = sys.root / "artifacts" / "ligand_params" / "index.json"
    if not param_idx_path.exists():
        raise FileNotFoundError(f"Missing ligand param index: {param_idx_path}")
    param_index = json.loads(param_idx_path.read_text())
    param_dir_dict = {e["residue_name"]: e["store_dir"] for e in param_index["ligands"]}

    # get mapping of ligand name → residue name
    lig_resname_map = {}
    for entry in param_index["ligands"]:
        lig = entry.get("ligand")
        resn = entry.get("residue_name")
        lig_resname_map[lig] = resn

    children: List[SimSystem] = []
    for lig_name, resn in lig_resname_map.items():
        d = sys.root / "simulations" / lig_name
        children.append(
            SimSystem(
                name=f"{sys.name}:{lig_name}",
                root=d,
                protein=sys.protein,
                topology=sys.topology,
                coordinates=sys.coordinates,
                ligands=tuple([d / "inputs" / "ligand.sdf"]),
                lipid_mol=sys.lipid_mol,
                other_mol=sys.other_mol,
                anchors=sys.anchors,
                meta={**(sys.meta or {}),
                "ligand": lig_name,
                "residue_name": resn,
                "param_dir_dict": param_dir_dict},
            )
        )

    # --------------------
    # PHASE 1: prepare_equil (parallel)
    # --------------------
    _run_phase_skipping_done(phase_prepare_equil, children, "prepare_equil", backend,  max_workers=rc.run.max_workers)

    # --------------------
    # PHASE 2: equil (parallel) → must COMPLETE for all ligands
    # --------------------
    phase_equil = _inject_mgr(phase_equil)
    finished = _run_phase_skipping_done(phase_equil, children, "equil", backend, max_workers=rc.run.max_workers)
    if not finished:
        job_mgr.wait_all()
        if dry_run and job_mgr.triggered:
            logger.success("[DRY-RUN] Reached first SLURM submission point (equil). Exiting without submitting.")
            raise SystemExit(0)

    # --------------------
    # PHASE 2.5: equil_analysis (parallel) → prune UNBOUND if requested
    # --------------------
    _run_phase_skipping_done(phase_equil_analysis, children, "equil_analysis", backend, max_workers=rc.run.max_workers)

    # prune UNBOUND ligands before FE prep
    def _filter_bound(children):
        keep = []
        for c in children:
            if (c.root / "equil" / "UNBOUND").exists():
                lig = (c.meta or {}).get("ligand", c.name)
                logger.warning(f"Pruning UNBOUND ligand after equil: {lig}")
                continue
            keep.append(c)
        return keep

    children = _filter_bound(children)

    # PHASE 3: prepare_fe (parallel)
    # --------------------
    _run_phase_skipping_done(phase_prepare_fe, children, "prepare_fe", backend, max_workers=rc.run.max_workers)

    # --------------------
    # PHASE 4: fe_equil → must COMPLETE for all ligands
    # --------------------
    phase_fe_equil = _inject_mgr(phase_fe_equil)
    finished = _run_phase_skipping_done(phase_fe_equil, children, "fe_equil", backend, max_workers=rc.run.max_workers)
    if not finished:
        job_mgr.wait_all()
        if dry_run and job_mgr.triggered:
            logger.success("[DRY-RUN] Reached first SLURM submission point (equil). Exiting without submitting.")
            raise SystemExit(0)

    
    # --------------------
    # PHASE 5: fe → must COMPLETE for all ligands
    # --------------------
    phase_fe = _inject_mgr(phase_fe)
    finished = _run_phase_skipping_done(phase_fe, children, "fe", backend, max_workers=rc.run.max_workers)
    if not finished:
        job_mgr.wait_all()
        if dry_run and job_mgr.triggered:
            logger.success("[DRY-RUN] Reached first SLURM submission point (equil). Exiting without submitting.")
            raise SystemExit(0)

    # --------------------
    # PHASE 6: analyze (parallel)
    # --------------------
    _run_phase_skipping_done(phase_analyze, children, "analyze", backend, max_workers=rc.run.max_workers)

    # --------------------
    # FE record save (no re-running the pipeline)
    # --------------------
    store = ArtifactStore(sys.root)
    repo = FEResultsRepository(store)

    failures: list[tuple[str, str]] = []
    for child in children:
        lig_name = child.meta["ligand"]
        # Try to read totals from windows.json (prefer artifacts/fe/windows.json)
        total_dG, total_se = -7.1, 0.3
        wjson = child.root / "artifacts" / "fe" / "windows.json"
        if not wjson.exists():
            wjson = child.root / "artifacts" / "windows.json"
        if wjson.exists():
            try:
                dct = json.loads(wjson.read_text())
                total_dG = float(dct.get("total_dG", total_dG))
                total_se = float(dct.get("total_se", total_se))
            except Exception:
                pass

        run_id = getattr(rc.run, "run_id", "auto")
        if run_id == "auto":
            run_id = _gen_run_id(rc.protocol, lig_name)

        try:
            rec = FERecord(
                run_id=run_id,
                system_name=sim_cfg_updated.system_name,
                fe_type=sim_cfg_updated.fe_type,
                temperature=sim_cfg_updated.temperature,
                method=sim_cfg_updated.dec_int,
                total_dG=total_dG,
                total_se=total_se,
                components=list(sim_cfg_updated.components),
                windows=[],
            )
            repo.save(rec)
            logger.info(f"Saved FE record for ligand {lig_name} under {sys.root}")
        except Exception as e:
            logger.warning(f"Could not save FE record for {lig_name}: {e}")
            failures.append((lig_name, f"save_failed: {e}"))

    if failures:
        failed = ", ".join([f"{n} ({m})" for n, m in failures])
        logger.warning(f"{len(failures)} ligand(s) had post-run issues: {failed}")
    logger.success(f"All phases completed. FE results written under {sys.root}")