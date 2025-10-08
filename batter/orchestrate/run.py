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
from typing import Any, Dict, List
from datetime import datetime, timezone
import json
import shutil
import os  # needed for relative symlinks

from loguru import logger

from batter.config.run import RunConfig
from batter.systems.core import SimSystem
from batter.systems.mabfe import MABFEBuilder
from batter.exec.local import LocalBackend
from batter.exec.slurm import SlurmBackend
from batter.pipeline.factory import make_abfe_pipeline, make_asfe_pipeline
from batter.pipeline.pipeline import Pipeline
from batter.pipeline.step import Step, ExecResult

from batter.param.ligand import batch_ligand_process
from batter.runtime.portable import ArtifactStore
from batter.runtime.fe_repo import FEResultsRepository, FERecord


# -----------------------------------------------------------------------------
# Pipeline utilities
# -----------------------------------------------------------------------------

def _select_pipeline(protocol: str, sim_cfg, only_fe_prep: bool) -> Pipeline:
    p = (protocol or "abfe").lower()
    if p == "abfe":
        return make_abfe_pipeline(sim_cfg, only_fe_prep)
    if p == "asfe":
        return make_asfe_pipeline(sim_cfg, only_fe_prep)
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
    return (
        text.replace("{WORK}", str(sys.root))
            .replace("{SYSTEM}", sys.name)
    )


# -----------------------------------------------------------------------------
# Ligand resolution
# -----------------------------------------------------------------------------

def _resolve_ligand_map(rc, yaml_dir: Path) -> dict[str, Path]:
    """
    Resolve ligands from either `create.ligand_paths` (list) or
    `create.ligand_input` (JSON file; can be dict or list).
    Returns {LIG_NAME: absolute Path}.
    """
    lig_map: dict[str, Path] = {}

    # Case A: explicit list in YAML
    paths = getattr(rc.create, "ligand_paths", None) or []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else (yaml_dir / p)
        lig_map[p.stem.upper()] = p.resolve()

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
                lig_map[name.upper()] = p.resolve()
        elif isinstance(data, list):
            for p in data:
                p = Path(p)
                p = p if p.is_absolute() else (jpath.parent / p)
                lig_map[p.stem.upper()] = p.resolve()
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
    from pathlib import Path
    import shutil, json

    # reuse helpers from content-addressed ligand store
    from batter.param.ligand import _rdkit_load, _canonical_payload, _hash_id, batch_ligand_process

    def _param_ligands(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        lig_root = system.root / "ligands"
        if not lig_root.exists():
            raise FileNotFoundError(f"[param_ligands] No 'ligands/' at {system.root}. Did staging run?")

        staged = [d for d in sorted(lig_root.glob("*")) if (d / "inputs" / "ligand.sdf").exists()]
        if not staged:
            raise FileNotFoundError(f"[param_ligands] No staged ligands found under {lig_root}.")

        outdir = Path(resolve_placeholders(params["outdir"], system)).expanduser().resolve()
        charge = params.get("charge", "am1bcc")  # fixed typo
        ligand_ff = params.get("ligand_ff", "openff-2.2.1")
        retain = bool(params.get("retain_lig_prot", True))

        lig_map: Dict[str, str] = {d.name: (d / "inputs" / "ligand.sdf").as_posix() for d in staged}

        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[LOCAL] parameterizing {len(lig_map)} ligands → {outdir} (charge={charge}, ff={ligand_ff}, retain={retain})"
        )

        hashes = batch_ligand_process(
            ligand_paths=lig_map,
            output_path=outdir,
            retain_lig_prot=retain,
            ligand_ff=ligand_ff,
            overwrite=False,
            run_with_slurm=False,
        )
        if not hashes:
            raise RuntimeError("[param_ligands] No ligands processed (empty hash list).")

        linked = []
        for d in staged:
            sdf = d / "inputs" / "ligand.sdf"
            mol = _rdkit_load(sdf, retain_h=retain)
            smi = _canonical_payload(mol)
            hid = _hash_id(smi, ligand_ff=ligand_ff, retain_h=retain)

            src_dir = outdir / hid
            meta = src_dir / "metadata.json"
            if not src_dir.exists() or not meta.exists():
                raise FileNotFoundError(
                    f"[param_ligands] Missing params for staged ligand {d.name}: expected {src_dir}"
                )

            child_params = d / "params"
            child_params.mkdir(parents=True, exist_ok=True)

            for src in sorted(src_dir.glob("*")):
                dst = child_params / src.name
                if not dst.exists():
                    try:
                        rel = os.path.relpath(src, start=child_params)
                        dst.symlink_to(rel)  # relative link for portability
                    except Exception:
                        shutil.copy2(src, dst)
            linked.append((d.name, hid))

        logger.info(f"[LOCAL] Linked params for staged ligands: {linked}")
        return ExecResult([], {"param_store": outdir, "hashes": hashes})

    def _touch_artifact(system: SimSystem, subdir: str, fname: str, content: str = "ok\n") -> Path:
        d = system.root / "artifacts" / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / fname
        p.write_text(content)
        return p

    def _prepare_equil(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        marker = _touch_artifact(system, "prepare_equil", "prepare_equil.ok")
        return ExecResult([], {"prepare_equil": marker})

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
        # write inside artifacts/fe/windows.json (matches reader below)
        summ = _touch_artifact(system, "fe", "windows.json", '{"total_dG": -7.1, "total_se": 0.3}\n')
        return ExecResult([], {"summary": summ})

    def _analyze(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
        ok = _touch_artifact(system, "analyze", "analyze.ok")
        return ExecResult([], {"analysis": ok})

    backend.register("param_ligands", _param_ligands)
    backend.register("prepare_equil", _prepare_equil)
    backend.register("equil", _equil)
    backend.register("prepare_fe", _prepare_fe)
    backend.register("prepare_fe_windows", _prepare_fe_windows)
    backend.register("fe_equil", _fe_equil)
    backend.register("fe", _fe)
    backend.register("analyze", _analyze)

    logger.info(f"Registered LOCAL handlers: {list(backend._handlers.keys())}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_from_yaml(path: Path | str, on_failure: Literal["prune", "raise"] = "raise") -> None:
    """
    Run a full BATTER workflow from a YAML configuration.

    Process
    -------
    1. Load RunConfig and SimulationConfig
    2. Choose backend (local/slurm)
    3. Build shared system once
    4. Stage **all ligands at once** under <work>/ligands/<NAME>/
    5. Run **param_ligands** ONCE at the parent root (single job)
    6. For each ligand, run the rest of the pipeline on its child system
    7. Save one FE record per ligand

    Parameters
    ----------
    path
        Path to the top-level YAML configuration.
    on_failure
        Failure policy for per-ligand pipelines:
        - "prune": skip a ligand if any step fails and continue with others.
        - "raise": immediately raise the error and abort the whole run.
    """
    path = Path(path)
    logger.info(f"Starting BATTER run from {path}")

    # Configs
    rc = RunConfig.load(path)
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

    # Stage ligands
    yaml_dir = path.parent
    lig_map = _resolve_ligand_map(rc, yaml_dir)  # raises on 0 ligands or missing files

    lig_root = sys.root / "ligands"
    lig_root.mkdir(parents=True, exist_ok=True)

    for lig_name, lig_path in lig_map.items():
        builder.make_child_for_ligand(sys, lig_name, lig_path)

    logger.info(f"Staged {len(lig_map)} ligand subsystems under {lig_root}")
    logger.info(f"System built successfully in {sys.root}")

    # Pipeline template
    tpl = _select_pipeline(rc.protocol, sim_cfg, rc.run.only_fe_preparation)

    # 5) Run param_ligands ONCE at parent
    parent_only = Pipeline([s for s in tpl.ordered_steps() if s.name == "param_ligands"])
    if parent_only.ordered_steps():
        logger.info(f"Executing single param_ligands job at {sys.root}")
        parent_only.run(backend, sys)

    # 6) Per-ligand: remove the step and its dependency before running
    per_lig = _pipeline_prune_requires(tpl, {"param_ligands"})

    # 7) Run remaining steps per ligand child system
    store = ArtifactStore(sys.root)
    repo = FEResultsRepository(store)

    child_dirs = sorted([d for d in (sys.root / "ligands").glob("*") if d.is_dir()])
    
    failures: list[tuple[str, str]] = []  # [(lig_name, str(error))]
    for d in child_dirs:
        lig_name = d.name
        child = SimSystem(
            name=f"{sys.name}:{lig_name}",
            root=d,
            protein=sys.protein,
            topology=sys.topology,
            coordinates=sys.coordinates,
            ligands=tuple([d / "inputs" / "ligand.sdf"]),
            lipid_mol=sys.lipid_mol,
            anchors=sys.anchors,
            meta={**(sys.meta or {}), "ligand": lig_name},
        )

        logger.info(f"=== Ligand {lig_name} === root: {child.root}")
        try:
            results = per_lig.run(backend, child)
            logger.success(f"Completed ligand {lig_name}. Steps: {list(results)}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if on_failure == "prune":
                logger.error(f"Ligand {lig_name} failed; pruning this ligand. Error: {msg}")
                # leave a marker so users can see which ligand was skipped
                (child.root / "artifacts" / "FAILED.txt").write_text(f"{msg}\n")
                failures.append((lig_name, msg))
                continue
            # on_failure == "raise"
            logger.error(f"Ligand {lig_name} failed; aborting entire run due to --on-failure=raise. Error: {msg}")
            raise

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
                system_name=sim_cfg.system_name,
                fe_type=sim_cfg.fe_type,
                temperature=sim_cfg.temperature,
                method=sim_cfg.dec_int,
                total_dG=total_dG,
                total_se=total_se,
                components=list(sim_cfg.components),
                windows=[],
            )
            repo.save(rec)
            logger.info(f"Saved FE record for ligand {lig_name} under {sys.root}")
        except Exception as e:
            logger.warning(f"Could not save FE record for {lig_name}: {e}")
    
    if failures:
        failed = ", ".join([f"{n} ({m})" for n, m in failures])
        logger.warning(f"{len(failures)} ligand(s) were pruned due to failures: {failed}")
    logger.success(f"All ligands completed. FE results written under {sys.root}")