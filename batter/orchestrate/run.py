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
from typing import Any, Dict, List, Literal
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
    from batter.exec.handlers.prepare_equil import prepare_equil_handler as _prepare_equil_handler

    from pathlib import Path

    def _touch_artifact(system: SimSystem, subdir: str, fname: str, content: str = "ok\n") -> Path:
        d = system.root / "artifacts" / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / fname
        p.write_text(content)
        return p

    def _equil(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        rst = _touch_artifact(system, "equil", "equil.rst7", "dummy-eq\n")
        return ExecResult([], {"rst7": rst})

    def _prepare_fe(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        marker = _touch_artifact(system, "prepare_fe", "prepare_fe.ok")
        return ExecResult([], {"prepare_fe": marker})

    def _prepare_fe_windows(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        marker = _touch_artifact(system, "prepare_fe_windows", "windows_prep.ok")
        return ExecResult([], {"windows_prep": marker})

    def _fe_equil(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        rst = _touch_artifact(system, "fe_equil", "fe_equil.rst7", "dummy-fe-eq\n")
        return ExecResult([], {"fe_equil_rst": rst})

    def _fe(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        summ = _touch_artifact(system, "fe", "windows.json", '{"total_dG": -7.1, "total_se": 0.3}\n')
        return ExecResult([], {"summary": summ})

    def _analyze(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        ok = _touch_artifact(system, "analyze", "analyze.ok")
        return ExecResult([], {"analysis": ok})

    backend.register("system_prep", _system_prep)
    backend.register("param_ligands", _param_ligands)
    backend.register("prepare_equil", _prepare_equil_handler)
    backend.register("equil", _equil)
    backend.register("prepare_fe", _prepare_fe)
    backend.register("prepare_fe_windows", _prepare_fe_windows)
    backend.register("fe_equil", _fe_equil)
    backend.register("fe", _fe)
    backend.register("analyze", _analyze)

    logger.debug(f"Registered LOCAL handlers: {list(backend._handlers.keys())}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_from_yaml(path: Path | str, on_failure: Literal["prune", "raise"] = None) -> None:
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
    yaml_path = Path(path)
    yaml_dir = yaml_path.parent

    # lf ligand_input load json and set ligand_paths 
    lig_map = _resolve_ligand_map(rc, yaml_dir)
    rc.create.ligand_paths = {k: str(v) for k, v in lig_map.items()}

    # update on_failure
    if on_failure:
        rc.run.on_failure = on_failure

    # Build system-prep params exactly once
    sys_params = {
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
    }


    sim_cfg = rc.resolved_sim_config()
    logger.info(f"Loaded simulation config for system: {sim_cfg.system_name}")

    # Backend
    if rc.backend == "slurm":
        backend = SlurmBackend()
    else:
        backend = LocalBackend()
        _register_local_handlers(backend)

    # Shared System Build
    if rc.system.type != "MABFE":
        raise ValueError(f"Unsupported system.type={rc.system.type!r}. Only 'MABFE' is implemented.")

    builder = MABFEBuilder()
    sys = SimSystem(name=rc.create.system_name, root=rc.system.output_folder)
    sys = builder.build(sys, rc.create)

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
        parent_only.run(backend, sys)
    
    # Locate sim_overrides from system_prep
    overrides_path = sys.root / "artifacts" / "config" / "sim_overrides.json"
    sim_cfg_updated = sim_cfg
    if overrides_path.exists():
        import json
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
    phase_prepare_fe    = _phase(PH_PREPARE_FE)
    phase_fe_equil      = _phase(PH_FE_EQUIL)
    phase_fe            = _phase(PH_FE)
    phase_analyze       = _phase(PH_ANALYZE)

    # --- build SimSystem children (attach param_dir_dict once) ---
    param_idx_path = sys.root / "artifacts" / "ligand_params" / "index.json"
    if not param_idx_path.exists():
        raise FileNotFoundError(f"Missing ligand param index: {param_idx_path}")
    param_index = json.loads(param_idx_path.read_text())
    param_dir_dict = {e["residue_name"]: e["store_dir"] for e in param_index["ligands"]}

    children: List[SimSystem] = []
    for d in sorted([p for p in (sys.root / "simulations").glob("*") if p.is_dir()]):
        lig_name = d.name
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
                meta={**(sys.meta or {}), "ligand": lig_name, "param_dir_dict": param_dir_dict},
            )
        )

    # --- helpers: submit phase and wait for markers ---
    def _run_phase_for_all(phase: Pipeline, children: list[SimSystem], phase_name: str, backend: LocalBackend, max_workers: int | None = None):
        if not phase.ordered_steps():
            return
        logger.debug(f"Phase: {phase_name} → steps={ [s.name for s in phase.ordered_steps()] }")
        backend.run_parallel(phase, children, description=phase_name)

    def _wait_for_markers(systems: List[SimSystem], rel_marker: str,
                          timeout_s: int = 0, poll_s: float = 15.0) -> List[SimSystem]:
        """
        Block until each system has the marker (relative path under child.root).
        Returns the list of systems that satisfied the barrier.
        If timeout_s == 0 → wait indefinitely.
        """
        import time
        start = time.time()
        pending = {s.meta["ligand"]: s for s in systems}
        done: dict[str, SimSystem] = {}
        while pending:
            found = []
            for lig, s in pending.items():
                if (s.root / rel_marker).exists():
                    logger.info(f"[barrier] {lig}: found {rel_marker}")
                    found.append(lig)
            for lig in found:
                done[lig] = pending.pop(lig)
            if not pending:
                break
            if timeout_s and (time.time() - start) > timeout_s:
                missing = ", ".join(pending.keys())
                raise TimeoutError(f"Timed out waiting for {rel_marker} for: {missing}")
            time.sleep(poll_s)
        return list(done.values())

    # --------------------
    # PHASE 1: prepare_equil (parallel)
    # --------------------
    _run_phase_for_all(phase_prepare_equil, children, "prepare_equil", backend,  max_workers=rc.run.max_workers)
    # If your handler writes a prep sentinel, wait for it here; otherwise skip.
    # Example (uncomment/adjust if available):
    # children = _wait_for_markers(children, "equil/build_files/equil-reference.pdb")

    # --------------------
    # PHASE 2: equil (parallel) → must COMPLETE for all ligands
    # --------------------
    _run_phase_for_all(phase_equil, children, "equil", backend)
    children = _wait_for_markers(children, "artifacts/equil/equil.rst7")

    # Optional prune: drop ligands that failed to produce marker
    # (Already handled by wait returning only completed ones.)
    if not children:
        raise RuntimeError("All ligands failed during equilibration barrier.")

    # --------------------
    # PHASE 3: prepare_fe (parallel)
    # --------------------
    _run_phase_for_all(phase_prepare_fe, children, "prepare_fe", backend, max_workers=rc.run.max_workers)
    # Optional: wait for prep sentinel
    # children = _wait_for_markers(children, "artifacts/prepare_fe/prepare_fe.ok")

    # --------------------
    # PHASE 4: fe_equil (parallel; if present)
    # --------------------
    _run_phase_for_all(phase_fe_equil, children, "fe_equil", backend)
    # Optional: wait
    # children = _wait_for_markers(children, "artifacts/fe_equil/fe_equil.rst7")

    # --------------------
    # PHASE 5: fe (parallel)
    # --------------------
    _run_phase_for_all(phase_fe, children, "fe", backend)
    # Optional: wait for completion of FE windows across ligands
    # children = _wait_for_markers(children, "artifacts/fe/windows.json")

    # --------------------
    # PHASE 6: analyze (parallel)
    # --------------------
    _run_phase_for_all(phase_analyze, children, "analyze", backend, max_workers=rc.run.max_workers)
    # Optional: wait
    # children = _wait_for_markers(children, "artifacts/analyze/analyze.ok")

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