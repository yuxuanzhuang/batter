"""Command-line interface for BATTER."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import math
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import click
import pandas as pd
import yaml
from loguru import logger

from batter.api import (
    __version__,
    clone_execution,
    list_fe_runs,
    load_fe_run,
    run_analysis_from_execution,
    run_from_yaml,
)
from batter.config.run import RunConfig
from batter.data import job_manager
from batter.utils.components import components_under
from batter.utils.slurm_templates import (
    render_slurm_with_header_body,
    seed_default_headers,
)
from batter.utils import natural_keys
from batter.cli.fek import fek_schedule
from batter.orchestrate.run_support import (
    compute_run_signature,
    generate_run_id,
    resolve_signature_conflict,
    select_run_id,
    stored_payload,
    stored_signature,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="batter")
def cli() -> None:
    """Root command group for BATTER."""
    seed_default_headers()


@cli.command("seed-headers")
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Destination directory for Slurm headers (defaults to ~/.batter).",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Overwrite existing headers if present.",
)
def seed_headers(dest: Path | None, force: bool) -> None:
    """Copy packaged Slurm headers into dest (default: ~/.batter)."""
    copied = seed_default_headers(dest, overwrite=force)
    dest_dir = dest or Path.home() / ".batter"
    if copied:
        click.echo(f"Seeded headers into {dest_dir}:")
        for path in copied:
            click.echo(f"  - {path}")
    else:
        click.echo(
            f"No headers copied; existing headers already present under {dest_dir}."
        )
        if not force:
            click.echo("Use --force to overwrite existing header files.")


@cli.command("diff-headers")
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Location of Slurm headers (defaults to ~/.batter).",
)
def diff_headers_cmd(dest: Path | None) -> None:
    """Show differences between user headers and packaged defaults."""
    from batter.utils.slurm_templates import diff_headers as _diff_headers

    diffs = _diff_headers(dest)
    if not diffs:
        click.echo("No headers found.")
        return
    for name, diff in diffs.items():
        click.echo(f"=== {name} ===")
        if not diff:
            click.echo("No differences.")
            continue
        click.echo(diff)


# -------------------------------- run ----------------------------------


def hash_run_input(yaml_path: Path, **options) -> str:
    """
    Return a stable hash for the YAML contents and CLI overrides.

    Parameters
    ----------
    yaml_path : Path
        Path to the run YAML file.
    **options
        CLI overrides that should affect the hash.

    Returns
    -------
    str
        First 12 characters of the SHA-256 digest.
    """
    p = Path(yaml_path)
    data = p.read_bytes()  # raw bytes to avoid newline normalization issues
    # freeze options dict deterministically (sorted keys)
    frozen = json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256()
    h.update(data)
    h.update(b"\0")  # separator byte to avoid accidental concatenation collisions
    h.update(frozen)
    return h.hexdigest()[:12]


def _which_batter() -> str:
    """
    Resolve the executable used to invoke ``batter``.

    Returns
    -------
    str
        Shell-escaped token (``batter`` path or ``python -m batter.cli``).
    """
    import shutil

    exe = shutil.which("batter")
    if exe:
        return shlex.quote(exe)
    # last resort: run module (works inside editable installs)
    return shlex.quote(sys.executable) + " -m batter.cli"


def _resolve_run_dir_for_submission(
    cfg: RunConfig, yaml_path: Path, run_overrides: Dict[str, Any]
) -> Tuple[str, Path]:
    """
    Mirror the run_id resolution used by ``run_from_yaml`` to keep SLURM job names stable.
    """
    config_signature, config_payload = compute_run_signature(
        yaml_path, run_overrides or None
    )
    requested_run_id = getattr(cfg.run, "run_id", "auto")

    while True:
        run_id, run_dir = select_run_id(
            cfg.run.output_folder,
            cfg.protocol,
            cfg.create.system_name,
            requested_run_id,
        )
        stored_sig, _ = stored_signature(run_dir)
        prev_payload = stored_payload(run_dir)
        if resolve_signature_conflict(
            stored_sig,
            config_signature,
            requested_run_id,
            cfg.run.allow_run_id_mismatch,
            stored_payload=prev_payload,
            current_payload=config_payload,
            run_id=run_id,
            run_dir=run_dir,
        ):
            return run_id, run_dir

        # mismatch with auto run_id → generate a fresh execution directory
        requested_run_id = "auto"
        run_id = generate_run_id(cfg.protocol, cfg.create.system_name)
        run_dir = Path(cfg.run.output_folder) / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)


def _upsert_sbatch_option(text: str, flag: str, value: str) -> str:
    """
    Replace or insert a ``#SBATCH --<flag>=...`` line with ``value``.
    """
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(flag)}=.*$", re.MULTILINE)
    repl = f"#SBATCH --{flag}={value}"
    if pattern.search(text):
        return pattern.sub(repl, text, count=1)

    lines = text.splitlines()
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    lines.insert(insert_idx, repl)
    return "\n".join(lines)


REMD_HEADER_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "_internal"
    / "templates"
    / "remd_run_files"
    / "SLURMM-BATCH-remd.header"
)


class RemdTask(NamedTuple):
    execution: Path
    ligand: str
    component: str
    comp_dir: Path
    n_windows: int


def _hash_path_list(paths: Sequence[Path]) -> str:
    joined = "\n".join(sorted(str(p.resolve()) for p in paths))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _resolve_ligand_dirs(exec_path: Path) -> List[Path]:
    """
    Return ligand directories under an execution path or the path itself if it is already a ligand root.
    """
    if (exec_path / "simulations").is_dir():
        lig_base = exec_path / "simulations"
    elif exec_path.name == "simulations" and exec_path.is_dir():
        lig_base = exec_path
    elif (exec_path / "fe").is_dir():
        return [exec_path]
    else:
        raise ValueError(
            f"{exec_path} is not an execution folder (missing simulations/ or fe/)."
        )

    return [p for p in lig_base.iterdir() if p.is_dir()]


def _load_windows_counts(fe_root: Path) -> dict[str, int]:
    meta_path = fe_root / "artifacts" / "windows.json"
    if not meta_path.is_file():
        return {}

    try:
        data = json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning(f"[remd-batch] Failed to read {meta_path}: {exc}")
        return {}

    counts: dict[str, int] = {}
    for comp, meta in data.items():
        if not isinstance(meta, dict):
            continue
        try:
            if meta.get("n_windows") is not None:
                counts[comp] = int(meta["n_windows"])
                continue
            if "lambdas" in meta:
                counts[comp] = len(meta["lambdas"])
        except Exception:
            continue
    return counts


def _extract_n_windows_from_run_script(run_script: Path) -> int | None:
    """
    Pull N_WINDOWS from run-local-remd.bash if present.
    """
    try:
        for line in run_script.read_text().splitlines():
            if line.strip().startswith("N_WINDOWS="):
                _, _, val = line.partition("=")
                val = val.strip()
                if val.isdigit():
                    return int(val)
    except Exception:
        return None
    return None


def _count_component_windows(comp_dir: Path, comp: str) -> int:
    if not comp_dir.is_dir():
        return 0

    count = 0
    for entry in comp_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith(comp):
            continue
        tail = name[len(comp) :]
        if tail.startswith("-"):
            continue
        if tail.isdigit():
            count += 1
    return count


def _collect_remd_tasks(exec_path: Path) -> List[RemdTask]:
    lig_dirs = _resolve_ligand_dirs(exec_path)
    tasks: List[RemdTask] = []

    for lig_dir in lig_dirs:
        comps = components_under(lig_dir)
        if not comps:
            logger.warning(f"[remd-batch] No components found under {lig_dir / 'fe'}")
            continue

        windows_counts = _load_windows_counts(lig_dir / "fe")
        for comp in comps:
            comp_dir = (lig_dir / "fe" / comp).resolve()
            finished_marker = comp_dir / "FINISHED"
            if finished_marker.exists():
                logger.info(
                    f"[remd-batch] {comp_dir} already finished; skipping."
                )
                continue
            run_script = comp_dir / "run-local-remd.bash"
            if not run_script.is_file():
                logger.warning(
                    f"[remd-batch] Missing run-local-remd.bash under {comp_dir}; skipping."
                )
                continue

            n_windows = windows_counts.get(comp)
            if n_windows is None:
                n_windows = _extract_n_windows_from_run_script(run_script)
            if n_windows is None:
                n_windows = _count_component_windows(comp_dir, comp)

            tasks.append(
                RemdTask(
                    execution=exec_path.resolve(),
                    ligand=lig_dir.name,
                    component=comp,
                    comp_dir=comp_dir,
                    n_windows=n_windows,
                )
            )

    return tasks


def _infer_gpus_per_node_from_text(text: str) -> int | None:
    """
    Attempt to extract per-node GPU count from SBATCH directives.
    """
    patterns = [
        r"--gres\s*=?\s*gpu:(\d+)",
        r"--gpus-per-node\s*=?\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _infer_header_gpus_per_node(header_root: Path | None) -> int | None:
    """
    Attempt to read per-node GPUs from the REMD header (gres or gpus-per-node).
    """
    root = header_root or (Path.home() / ".batter")
    header_path = root / "SLURMM-BATCH-remd.header"
    text = ""
    try:
        text = header_path.read_text()
    except Exception:
        try:
            text = REMD_HEADER_TEMPLATE.read_text()
        except Exception:
            return None

    gpn = _infer_gpus_per_node_from_text(text)
    if gpn is None:
        try:
            tmpl_text = REMD_HEADER_TEMPLATE.read_text()
            gpn = _infer_gpus_per_node_from_text(tmpl_text)
        except Exception:
            pass
    return gpn


def _render_remd_batch_script(
    body_text: str, replacements: dict[str, str], header_root: Path | None
) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(body_text)
        tmp_path = Path(tmp.name)

    try:
        return render_slurm_with_header_body(
            "SLURMM-BATCH-remd.header",
            REMD_HEADER_TEMPLATE,
            tmp_path,
            replacements,
            header_root=header_root,
        )
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


@cli.command("run")
@click.argument(
    "yaml_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--on-failure",
    type=click.Choice(["prune", "raise", "retry"], case_sensitive=False),
    default="raise",
    show_default=True,
)
@click.option(
    "--output-folder", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option(
    "--run-id",
    default=None,
    help="Override run_id (e.g., rep1). Use 'auto' to reuse latest.",
)
@click.option(
    "--allow-run-id-mismatch/--no-allow-run-id-mismatch",
    default=None,
    help="Allow reusing a provided run-id even if the stored configuration hash differs.",
)
@click.option("--dry-run/--no-dry-run", default=None, help="Override YAML run.dry_run.")
@click.option(
    "--only-equil/--full", default=None, help="Run only equil steps; override YAML."
)
@click.option(
    "--slurm-submit/--local-run",
    default=False,
    help="Submit this run via SLURM (sbatch) instead of running locally.",
)
@click.option(
    "--slurm-manager-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a SLURM header/template to prepend to the generated script.",
)
def cmd_run(
    yaml_path: Path,
    on_failure: str,
    output_folder: Optional[Path],
    run_id: Optional[str],
    allow_run_id_mismatch: Optional[bool],
    dry_run: Optional[bool],
    only_equil: Optional[bool],
    slurm_submit: bool,
    slurm_manager_path: Optional[Path],
) -> None:
    """
    Execute a BATTER workflow defined in ``YAML_PATH``.
    """
    run_over = {}
    if output_folder:
        run_over["output_folder"] = output_folder
    if run_id is not None:
        run_over["run_id"] = run_id
    if allow_run_id_mismatch is not None:
        run_over["allow_run_id_mismatch"] = allow_run_id_mismatch
    if dry_run is not None:
        run_over["dry_run"] = dry_run
    if only_equil is not None:
        run_over["only_fe_preparation"] = only_equil

    # first do a basic validation of the YAML (with any CLI overrides applied)
    try:
        base_cfg = RunConfig.load(yaml_path)
        cfg_for_validation = base_cfg
        if run_over:
            cfg_for_validation = cfg_for_validation.model_copy(
                update={"run": cfg_for_validation.run.model_copy(update=run_over)}
            )
        # Force resolution so missing/invalid fields are surfaced before submitting
        cfg_for_validation.resolved_sim_config()
    except Exception as e:
        raise click.ClickException(f"Invalid SimulationConfig YAML: {e}")

    if slurm_submit:
        _, run_dir = _resolve_run_dir_for_submission(
            cfg_for_validation, yaml_path, run_over
        )
        run_dir_abs = run_dir.resolve()
        manager_job_name = (
            "fep_" + (PurePosixPath(run_dir_abs) / "simulations" / "manager").as_posix()
        )
        log_base = f"manager-{run_dir_abs.name or 'run'}"

        batter_cmd = _which_batter()
        parts = [batter_cmd, "run", shlex.quote(str(Path(yaml_path).resolve()))]
        parts += ["--on-failure", shlex.quote(on_failure)]

        if output_folder:
            parts += [
                "--output-folder",
                shlex.quote(str(Path(output_folder).resolve())),
            ]
        if run_id is not None:
            parts += ["--run-id", shlex.quote(run_id)]
        if allow_run_id_mismatch is not None:
            parts += [
                (
                    "--allow-run-id-mismatch"
                    if allow_run_id_mismatch
                    else "--no-allow-run-id-mismatch"
                )
            ]
        if dry_run is not None:
            parts += ["--dry-run" if dry_run else "--no-dry-run"]
        if only_equil is not None:
            parts += ["--only-equil" if only_equil else "--full"]

        run_cmd = " ".join(parts)

        # create a hash based on contents of the yaml and options
        run_hash = hash_run_input(
            yaml_path,
            on_failure=on_failure.lower(),
            output_folder=str(Path(output_folder).resolve()) if output_folder else "",
            run_id=run_id or "",
            dry_run=("1" if dry_run else "0") if dry_run is not None else "",
            only_equil=("1" if only_equil else "0") if only_equil is not None else "",
        )
        base_path = (
            Path(slurm_manager_path) if slurm_manager_path else Path(job_manager)
        )
        tpl_header = base_path.with_suffix(".header")
        tpl_body = base_path.with_suffix(".body")
        manager_code = render_slurm_with_header_body(
            "job_manager.header",
            tpl_header,
            tpl_body,
            {
                "__JOB_NAME__": manager_job_name,
                "__JOB_LOG_BASE__": log_base,
            },
            header_root=cfg_for_validation.run.slurm_header_dir,
        )
        manager_code = _upsert_sbatch_option(manager_code, "job-name", manager_job_name)
        with open(f"{run_hash}_job_manager.sbatch", "w") as f:
            f.write(manager_code)
            f.write("\n")
            f.write(run_cmd)
            f.write("\n")
            f.write("echo 'Job completed.'\n")
            f.write("\n")

        # submit slurm job
        result = subprocess.run(
            ["sbatch", f"{run_hash}_job_manager.sbatch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        click.echo(f"Submitted jobscript: {run_hash}_job_manager.sbatch")
        click.echo(f"STDOUT: {result.stdout}")
        click.echo(f"STDERR: {result.stderr}")
        return

    run_from_yaml(
        yaml_path,
        on_failure=on_failure.lower(),
        run_overrides=(run_over or None),
    )


@cli.command("run-exec")
@click.argument(
    "execution_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--on-failure",
    type=click.Choice(["prune", "raise", "retry"], case_sensitive=False),
    default="raise",
    show_default=True,
)
def cmd_run_exec(execution_dir: Path, on_failure: str) -> None:
    """
    Resume/extend a run using only an existing execution directory.
    """
    exec_dir = execution_dir.resolve()
    yaml_copy = exec_dir / "artifacts" / "config" / "run_config.yaml"
    if not yaml_copy.exists():
        raise click.ClickException(
            f"Could not find stored run_config.yaml under {yaml_copy}. "
            "Run once with `batter run` to seed artifacts/config."
        )

    run_overrides = {
        "output_folder": exec_dir.parent.parent,
        "run_id": exec_dir.name,
        "allow_run_id_mismatch": True,
    }
    try:
        run_from_yaml(
            yaml_copy,
            on_failure=on_failure.lower(),
            run_overrides=run_overrides,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@cli.command("remd-batch")
@click.option(
    "--execution",
    "-e",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Execution directories to include (run root or a ligand folder under simulations/).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination for the rendered sbatch script (defaults to CWD).",
)
@click.option(
    "--header-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory containing SLURM headers (default: ~/.batter).",
)
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Optional partition override for the sbatch header.",
)
@click.option(
    "--time-limit",
    type=str,
    default=None,
    help="Optional time limit override for the sbatch header (e.g., 08:00:00).",
)
@click.option(
    "--gpus",
    type=int,
    default=None,
    help="Total GPUs to request; defaults to the total REMD window count found.",
)
@click.option(
    "--nodes",
    type=int,
    default=None,
    help="Optional node count override for the sbatch header.",
)
@click.option(
    "--gpus-per-node",
    type=int,
    default=8,
    show_default=True,
    help="GPUs available per node (used to size per-task node allocations).",
)
def remd_batch(
    execution: tuple[Path, ...],
    output: Path | None,
    header_root: Path | None,
    partition: str | None,
    time_limit: str | None,
    gpus: int | None,
    nodes: int | None,
    gpus_per_node: int | None,
) -> None:
    """
    Generate an sbatch script that runs ``run-local-remd.bash`` for provided executions.
    """
    exec_paths = [p.resolve() for p in execution]
    tasks: List[RemdTask] = []
    seen: set[Path] = set()

    for path in exec_paths:
        try:
            new_tasks = _collect_remd_tasks(path)
        except ValueError as exc:
            raise click.ClickException(str(exc))

        for t in new_tasks:
            if t.comp_dir in seen:
                continue
            seen.add(t.comp_dir)
            tasks.append(t)
            win_note = t.n_windows if t.n_windows else "unknown"
            logger.info(
                f"[remd-batch] Queued {t.comp_dir} (windows={win_note})."
            )

    if not tasks:
        raise click.ClickException(
            "No unfinished REMD component folders found under the provided paths."
        )

    tasks.sort(key=lambda t: (str(t.execution), t.ligand, t.component))
    gpus_per_node_resolved = gpus_per_node
    if gpus_per_node_resolved is None:
        gpus_per_node_resolved = _infer_header_gpus_per_node(header_root)
    if gpus_per_node_resolved is None:
        gpus_per_node_resolved = 8
    if gpus_per_node_resolved <= 0:
        raise click.ClickException("--gpus-per-node must be >= 1.")

    task_specs: list[tuple[RemdTask, int, int]] = []
    total_windows = 0
    max_windows = 0
    total_nodes_needed = 0
    for t in tasks:
        n_windows = t.n_windows
        if n_windows <= 0:
            logger.warning(
                f"[remd-batch] Could not determine windows for {t.comp_dir}; "
                "assuming 1 for resource sizing."
            )
            n_windows = 1
        nodes_needed = int(math.ceil(n_windows / float(gpus_per_node_resolved)))
        task_specs.append((t, n_windows, nodes_needed))
        total_windows += n_windows
        max_windows = max(max_windows, n_windows)
        total_nodes_needed += nodes_needed

    gpu_request = gpus or (total_windows if total_windows > 0 else None)
    node_request = nodes
    if node_request is None and total_nodes_needed > 0:
        node_request = total_nodes_needed

    job_hash = _hash_path_list(exec_paths)
    job_name = f"fep_remd_batch_{job_hash}"
    body_lines = [
        "scontrol show job ${SLURM_JOB_ID:-}",
        'echo "Job started at $(date)"',
        "status=0",
        "pids=()",
        'mpi_base=$(echo "${MPI_EXEC:-mpirun}" | awk \'{print $1}\')',
        'mpi_base=${mpi_base##*/}',
        "use_srun=0",
        'if [[ "${mpi_base}" == srun* ]]; then',
        "  use_srun=1",
        "fi",
        "run_remd_task() {",
        '  local label="$1"',
        '  local dir="$2"',
        '  local win="$3"',
        '  local nodes="$4"',
        '  echo "Running ${label}${win:+ (windows=${win})}"',
        '  local mpi_flags="${MPI_FLAGS:-}"',
        '  if [[ "$use_srun" -eq 1 && "$nodes" -gt 0 && "$win" -gt 0 ]]; then',
        '    local extra_flags="--nodes=${nodes} --ntasks=${win} --exclusive --gpus-per-task=1"',
        '    if [[ -n "$mpi_flags" ]]; then',
        '      mpi_flags="${mpi_flags} ${extra_flags}"',
        "    else",
        '      mpi_flags="${extra_flags}"',
        "    fi",
        "  fi",
        '  ( cd "$dir" && MPI_FLAGS="$mpi_flags" bash ./run-local-remd.bash ) || status=1',
        "}",
        "",
    ]

    for t, n_windows, nodes_needed in task_specs:
        label = f"{t.ligand}/{t.component}"
        if t.execution.name:
            label = f"{t.execution.name}/{label}"
        body_lines.append(
            f'run_remd_task "{label}" "{t.comp_dir}" "{n_windows}" "{nodes_needed}" &'
        )
        body_lines.append('pids+=($!)')

    body_lines.append("")
    body_lines.append('for pid in "${pids[@]}"; do')
    body_lines.append('  wait "$pid" || status=1')
    body_lines.append("done")
    body_lines.append('echo "Job completed at $(date)"')
    body_lines.append("exit $status")
    body_text = "\n".join(body_lines) + "\n"

    script_text = _render_remd_batch_script(
        body_text,
        {
            "SYSTEMNAME": job_name,
            "STAGE": "remd-batch",
            "POSE": job_hash,
        },
        header_root=header_root,
    )
    if gpus_per_node_resolved is None:
        gpus_per_node_resolved = _infer_gpus_per_node_from_text(script_text)
    if node_request is None and gpu_request:
        if gpus_per_node_resolved:
            node_request = int(math.ceil(gpu_request / float(gpus_per_node_resolved)))
        else:
            node_request = gpu_request  # assume 1 GPU per node if unknown
    script_text = _upsert_sbatch_option(script_text, "job-name", job_name)
    if partition:
        script_text = _upsert_sbatch_option(script_text, "partition", partition)
    if time_limit:
        script_text = _upsert_sbatch_option(script_text, "time", time_limit)
    if node_request:
        script_text = _upsert_sbatch_option(script_text, "nodes", str(node_request))
    if gpu_request:
        # Slurm gres is per-node; when nodes and per-node GPUs are known, use that value.
        # Otherwise fall back to total GPUs requested only for single-node allocations.
        gres_val = None
        if node_request and gpus_per_node_resolved:
            gres_val = gpus_per_node_resolved
        elif not node_request:
            gres_val = gpu_request
        if gres_val:
            script_text = _upsert_sbatch_option(script_text, "gres", f"gpu:{gres_val}")
        if node_request and not gpus_per_node_resolved:
            # assume 1 GPU per node when nothing better is known
            script_text = _upsert_sbatch_option(script_text, "gres", "gpu:1")
        script_text = _upsert_sbatch_option(script_text, "gpus-per-task", "1")
        script_text = _upsert_sbatch_option(script_text, "ntasks", str(gpu_request))

    # Align log filenames with job name for clarity
    script_text = _upsert_sbatch_option(script_text, "output", f"{job_name}-%j.out")
    script_text = _upsert_sbatch_option(script_text, "error", f"{job_name}-%j.err")

    output_path = output or Path.cwd() / f"run_remd_batch_{job_hash}.sbatch"
    output_path.write_text(script_text)
    try:
        output_path.chmod(0o755)
    except Exception:
        pass

    click.echo(f"Wrote sbatch script to {output_path}")
    click.echo(
        f"Components queued: {len(tasks)} | max windows: {max_windows or 'unknown'} | "
        f"total windows: {total_windows or 'unknown'} | GPUs: {gpu_request or 'unset'} | "
        f"nodes: {node_request or 'unset'} | gpus-per-node: {gpus_per_node_resolved} | "
        f"job-name: {job_name}"
    )


# ---------------------------- free energy results check ------------------------------


@cli.group("fe")
def fe() -> None:
    """Query and inspect free-energy results."""


@fe.command("list")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv", "tsv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
def fe_list(work_dir: Path, fmt: str) -> None:
    """
    List free-energy runs stored within ``WORK_DIR``.

    Parameters
    ----------
    work_dir : Path
        Portable work directory containing the ``results/`` tree.
    fmt : {"table", "json", "csv", "tsv"}
        Output formatting option (defaults to ``"table"``).
    """
    try:
        df = list_fe_runs(work_dir)
    except Exception as e:
        raise click.ClickException(str(e))

    if df.empty:
        click.secho("No FE runs found.", fg="yellow")
        return

    # ensure expected cols exist
    cols = [
        "system_name",
        "run_id",
        "ligand",
        "mol_name",
        "temperature",
        "total_dG",
        "total_se",
        "original_name",
        "status",
        "protocol",
        "created_at",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # parse datetime for stable sort, but keep original text if non-parseable
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = (
        df.assign(_created=created)
        .sort_values("_created", na_position="last")
        .drop(columns=["_created"])
    )
    df = df[cols]

    if fmt.lower() == "json":
        click.echo(df.to_json(orient="records", date_unit="s"))
        return
    if fmt.lower() == "csv":
        click.echo(df.to_csv(index=False))
        return
    if fmt.lower() == "tsv":
        click.echo(df.to_csv(index=False, sep="\t"))
        return

    # pretty table
    with pd.option_context("display.max_columns", None, "display.width", 120):
        # format floats if present
        def _fmt(v):
            try:
                if pd.isna(v):
                    return ""
                if isinstance(v, (float, int)) and str(v) != "":
                    return f"{float(v):.3f}"
            except Exception:
                pass
            return v

        show = df.copy()
        if "total_dG" in show.columns:
            show["total_dG"] = show["total_dG"].map(_fmt)
        if "total_se" in show.columns:
            show["total_se"] = show["total_se"].map(_fmt)
        click.echo(show.to_string(index=False))


@fe.command("show")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("run_id", type=str)
@click.option(
    "--ligand",
    "-l",
    type=str,
    default=None,
    help="Specify a ligand identifier when multiple records share the same run_id.",
)
def fe_show(work_dir: Path, run_id: str, ligand: str | None) -> None:
    """
    Display a single free-energy record from ``WORK_DIR``.

    Parameters
    ----------
    work_dir : Path
        Portable work directory.
    run_id : str
        Run identifier returned by :func:`fe_list`.
    ligand : str, optional
        Ligand identifier to disambiguate when multiple ligands are stored under the same run_id.
    """
    try:
        rec = load_fe_run(work_dir, run_id, ligand=ligand)
    except FileNotFoundError:
        raise click.ClickException(f"Run '{run_id}' not found under {work_dir}.")
    except Exception as e:
        raise click.ClickException(str(e))

    def f3(x):
        try:
            if x is None:
                return "NA"
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    click.secho("Summary", fg="cyan", bold=True)
    click.echo(
        f"- run_id     : {rec.run_id}\n"
        f"- system     : {rec.system_name}\n"
        f"- fe_type    : {rec.fe_type}\n"
        f"- method     : {rec.method}\n"
        f"- temperature: {rec.temperature}\n"
        f"- components : {', '.join(rec.components) if rec.components else ''}\n"
        f"- total_dG   : {f3(getattr(rec, 'total_dG', None))} kcal/mol\n"
        f"- total_se   : {f3(getattr(rec, 'total_se', None))} kcal/mol\n"
        f"- created_at : {getattr(rec, 'created_at', '')}"
    )

    if getattr(rec, "windows", None):
        click.secho("\nPer-window", fg="cyan", bold=True)
        df = pd.DataFrame([w.model_dump() for w in rec.windows])
        # stable, readable column order
        order = [
            c
            for c in ["component", "lam", "dG", "dG_se", "n_samples"]
            if c in df.columns
        ]
        df = df[order + [c for c in df.columns if c not in order]]
        # format numbers
        for col in ["dG", "dG_se"]:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda v: f"{float(v):.3f}" if pd.notna(v) else ""
                )
        with pd.option_context("display.max_columns", None, "display.width", 120):
            click.echo(df.to_string(index=False))
    else:
        click.secho("\n(no per-window data saved)", fg="yellow")


@fe.command("analyze")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("run_id", type=str)
@click.option(
    "--ligand",
    "-l",
    type=str,
    default=None,
    help="Select a single ligand when multiple records exist for the run.",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Number of local workers to pass to the FE analysis handler.",
)
@click.option(
    "--raise-on-error/--no-raise-on-error",
    default=True,
    help="Whether analysis failures should raise (default) or be logged and skipped.",
)
@click.option(
    "--sim-range",
    type=str,
    default=None,
    help="Subset of lambda windows to analyze, formatted as ``start,end``.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Logging level for analysis stage.",
)
def fe_analyze(
    work_dir: Path,
    run_id: str,
    ligand: str | None,
    workers: int | None,
    raise_on_error: bool,
    sim_range: str | None,
    log_level: str = "INFO",
) -> None:
    """
    Re-run the FE analysis stage for a stored execution.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    
    parsed_range: tuple[int, int] | None = None
    if sim_range:
        parts = sim_range.split(",")
        if len(parts) != 2:
            raise click.ClickException(
                "`--sim-range` expects two comma-separated integers."
            )
        try:
            parsed_range = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise click.ClickException(f"Invalid `--sim-range`: {exc}")

    try:
        run_analysis_from_execution(
            work_dir,
            run_id,
            ligand=ligand,
            n_workers=workers,
            sim_range=parsed_range,
            raise_on_error=raise_on_error,
        )
    except Exception as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"Analysis run finished for '{run_id}'"
        f"{' (ligand ' + ligand + ')' if ligand else ''}."
    )


# ------------------------ execution cloning ---------------------------
@cli.command("clone-exec")
@click.argument(
    "work_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("src_run_id", type=str)
@click.argument("dst_run_id", required=False)
@click.option(
    "--dst-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination work directory (defaults to WORK_DIR).",
)
@click.option(
    "--only-equil/--full",
    default=True,
    show_default=True,
    help="Clone only equilibration artifacts or the full FE layout.",
)
@click.option(
    "--mode",
    type=click.Choice(["copy", "hardlink", "symlink"], case_sensitive=False),
    default="symlink",
    show_default=True,
    help="Copy strategy for cloning files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite destination execution folder if it exists.",
)
def cmd_clone_exec(
    work_dir: Path,
    src_run_id: str,
    dst_run_id: str | None,
    dst_root: Path | None,
    only_equil: bool,
    force: bool,
    mode: str,
) -> None:
    """
    Clone an existing execution directory.
    """
    dst_root = dst_root or work_dir
    if dst_run_id is None:
        dst_run_id = f"{src_run_id}-clone"

    # Basic sanity checks to give nice CLI errors before calling the underlying function
    src_exec = work_dir / "executions" / src_run_id
    if not src_exec.is_dir():
        raise click.ClickException(f"Source execution not found: {src_exec}")

    dst_exec = dst_root / "executions" / dst_run_id
    if dst_exec.exists() and not force:
        raise click.ClickException(
            f"Destination already exists: {dst_exec} (use --force to overwrite)"
        )

    # Delegate to your existing implementation
    try:
        clone_execution(
            work_dir=work_dir,
            src_run_id=src_run_id,
            dst_root=dst_root,
            dst_run_id=dst_run_id,
            mode=mode,
            only_equil=only_equil,
            reset_states=True,
            overwrite=force,
        )
    except Exception as e:
        raise click.ClickException(str(e)) from e

    click.secho(
        f"Cloned execution '{src_run_id}' → '{dst_run_id}' under {dst_root}",
        fg="green",
    )


# ----------------------------- check status -------------------------------


def _parse_jobname(jobname: str) -> dict[str, Optional[object]] | None:
    """
    Parse BATTER job names emitted by SLURM handlers.
    """
    if not jobname.startswith("fep_"):
        return None

    body = jobname[4:]
    if "/simulations/" not in body:
        return None

    system_root, tail = body.split("/simulations/", 1)
    if "/" in tail:
        tail = tail.split("/", 1)[0]

    if not tail:
        return None

    stage = "unknown"
    comp = None
    win: int | None = None

    ligand = tail
    if tail == "manager":
        ligand = "MANAGER_JOB"
        stage = "manager"
    elif tail.endswith("_eq"):
        ligand = tail[: -len("_eq")]
        stage = "eq"
    elif tail.endswith("_fe_equil"):
        core = tail[: -len("_fe_equil")]
        parts = core.rsplit("_", 1)
        if len(parts) == 2:
            ligand, comp = parts
        stage = "fe_equil"
    elif tail.endswith("_fe"):
        core = tail[: -len("_fe")]
        m = re.match(r"(?P<lig>.+)_(?P<comp>[A-Za-z]+)_(?P<win>[A-Za-z]+\d+)$", core)
        if m:
            ligand = m.group("lig")
            comp = m.group("comp")
            win_match = re.search(r"(\d+)$", m.group("win"))
            if win_match:
                try:
                    win = int(win_match.group(1))
                except ValueError:
                    win = None
        stage = "fe"
    elif tail.endswith("_remd"):
        core = tail[: -len("_remd")]
        m = re.match(r"(?P<lig>.+)_(?P<comp>[A-Za-z]+)$", core)
        if m:
            ligand = m.group("lig")
            comp = m.group("comp")
        stage = "remd"

    run_id = None
    mrun = re.search(r"/executions/([^/]+)$", system_root)
    if mrun:
        run_id = mrun.group(1)

    return {
        "stage": stage,
        "run_id": run_id,
        "system_root": system_root,
        "ligand": ligand,
        "comp": comp,
        "win": win,
    }


def _natural_keys(val: str | None):
    """Return a tuple suitable for natural sorting of strings containing digits."""
    s = "" if val is None else str(val)
    parts = natural_keys(s)
    return tuple(p.lower() if isinstance(p, str) else p for p in parts)


def _natkey_series(s: pd.Series) -> pd.Series:
    """Vectorised version of :func:`_natural_keys` for pandas Series."""
    assert pd is not None  # for type-checkers
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).map(_natural_keys)


@cli.command("report-jobs")
@click.option("--partition", "-p", default=None, help="SLURM partition filter.")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed job lines.")
def report_jobs(partition=None, detailed=False):
    """Report SLURM job status for BATTER jobs prefixed with ``fep_``."""
    try:
        cmd = ["squeue", "--user", os.getenv("USER"), "--format=%i %j %T"]
        if partition:
            cmd = [
                "squeue",
                "--partition",
                partition,
                "--user",
                os.getenv("USER"),
                "--format=%i %j %T",
            ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Failed to get SLURM job list: {e.stderr}")
        return

    lines = res.stdout.strip().splitlines()
    if not lines or len(lines) == 1:
        click.echo("No SLURM jobs found.")
        return
    _, *jobs = lines  # drop header

    rows = []
    for line in jobs:
        # robust split: job name may contain spaces → use maxsplit=2
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        jobid, jobname, status = parts
        if not jobname.startswith("fep_"):
            continue
        meta = _parse_jobname(jobname)
        if meta is None:
            continue
        rows.append(
            {
                "jobid": jobid,
                "status": status,
                "stage": meta["stage"],
                "run_id": meta["run_id"],
                "identifier": (
                    meta["system_root"] + "/" + meta["run_id"]
                    if meta["run_id"]
                    else meta["system_root"]
                ),
                "system_root": meta["system_root"],
                "ligand": meta["ligand"],
                "comp": meta["comp"],
                "win": meta["win"],
                "jobname": jobname,
            }
        )

    if not rows:
        click.echo("No BATTER jobs (fep_*) found.")
        return

    df = pd.DataFrame(rows)

    # topline
    total = len(df)
    running = (df["status"] == "RUNNING").sum()
    pending = (df["status"] == "PENDING").sum()
    click.echo(
        click.style(
            f"Total jobs: {total}, Running: {running}, Pending: {pending}", bold=True
        )
    )

    grp_key = df["identifier"].where(df["identifier"].notna())
    df = df.assign(_group=grp_key)
    for gid, sub in df.groupby("_group"):
        sys_root = sub["system_root"].dropna().unique()
        label = gid if gid is not None else "(unknown)"
        if sys_root.size > 0:
            label = sys_root[0]
        click.echo(click.style(f"\nRun: {label}", bold=True))
        stages = ", ".join(sorted(sub["stage"].dropna().unique()))
        click.echo(f"Stages present: {stages or '(unknown)'}")
        click.echo("-" * 70)

        label_col = "ligand"

        summary = (
            sub.assign(label=sub[label_col])
            .groupby(["label"])["status"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )

        for need_col in ("RUNNING", "PENDING"):
            if need_col not in summary.columns:
                summary[need_col] = 0

        # natural sort labels
        summary = summary.sort_values(by="label", key=lambda s: s.map(_natural_keys))

        # print compact two columns
        line_buf = []
        for _, r in summary.iterrows():
            label = r["label"]
            p = int(r.get("PENDING", 0))
            r_ = int(r.get("RUNNING", 0))
            colored = click.style(
                f"{label}(P={p},R={r_})",
                fg=("green" if r_ > 0 else "yellow" if p > 0 else "red"),
                bold=(r_ > 0),
            )
            line_buf.append(colored)

        for i in range(0, len(line_buf), 4):
            click.echo("   ".join(line_buf[i : i + 4]))

        if detailed:
            click.echo(click.style("\nDetailed:", bold=True))
            det = (
                sub[["jobid", "status", "stage", "ligand", "comp", "win"]]
                .assign(win=sub["win"])
                .sort_values(["stage", "ligand", "comp", "win"], key=_natkey_series)
            )
            with pd.option_context("display.width", 140, "display.max_columns", None):
                click.echo(det.to_string(index=False))
        click.echo("-" * 70)

    click.echo("To cancel, run: batter cancel-jobs --contains '<folder_listed_above>'")


@cli.command("cancel-jobs")
@click.option(
    "--contains",
    "-c",
    required=True,
    help="Cancel all jobs whose SLURM job name contains this substring (match against full 'fep_...').",
)
def cancel_jobs(contains: str):
    """Cancel all SLURM jobs whose names contain ``contains``."""
    try:
        res = subprocess.run(
            ["squeue", "--user", os.getenv("USER"), "--format=%i %j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Error querying SLURM: {e.stderr}")
        return

    ids = []
    for line in res.stdout.strip().splitlines()[1:]:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        jobid, jobname = parts
        if contains in jobname:
            ids.append(jobid)

    if not ids:
        click.echo(f"No jobs found containing '{contains}'.")
        return

    click.echo(f"Cancelling {len(ids)} job(s)")
    for i in range(0, len(ids), 30):
        batch = ids[i : i + 30]
        try:
            subprocess.run(["scancel"] + batch, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"Failed to cancel {batch}: {e.stderr}")


cli.add_command(fek_schedule)
