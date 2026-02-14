"""Batch and REMD batch command implementations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Sequence

import click
from loguru import logger

from batter.cli.root import cli
from batter.cli.shared import _upsert_sbatch_option, _which_batter
from batter.utils.components import components_under
from batter.utils.slurm_templates import render_slurm_with_header_body


REMD_HEADER_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "_internal"
    / "templates"
    / "remd_run_files"
    / "SLURMM-BATCH-remd.header"
)
BATCH_RUN_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "_internal"
    / "templates"
    / "remd_run_files"
    / "run-local-batch.bash"
)
BATCH_CHECK_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "_internal"
    / "templates"
    / "run_files_orig"
    / "check_run.bash"
)


class RemdTask(NamedTuple):
    execution: Path
    ligand: str
    component: str
    comp_dir: Path
    n_windows: int


class BatchTask(NamedTuple):
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
    Return FE leaf directories under an execution path.

    Supports both ABFE layout:
      executions/<run>/simulations/<ligand>/fe/...
    and RBFE layout:
      executions/<run>/simulations/transformations/<pair>/fe/...
    """
    def _leaf_dirs_under_simulations(sim_root: Path) -> List[Path]:
        out: List[Path] = []
        for entry in sim_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name == "transformations":
                for pair_dir in entry.iterdir():
                    if pair_dir.is_dir() and (pair_dir / "fe").is_dir():
                        out.append(pair_dir)
                continue
            if (entry / "fe").is_dir():
                out.append(entry)
        return out

    if (exec_path / "simulations").is_dir():
        return _leaf_dirs_under_simulations(exec_path / "simulations")

    if exec_path.name == "simulations" and exec_path.is_dir():
        return _leaf_dirs_under_simulations(exec_path)

    if exec_path.name == "transformations" and exec_path.is_dir():
        return [p for p in exec_path.iterdir() if p.is_dir() and (p / "fe").is_dir()]

    if (exec_path / "fe").is_dir():
        return [exec_path]

    raise ValueError(
        f"{exec_path} is not an execution folder (missing simulations/ or fe/)."
    )


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


def _component_window_dirs(comp_dir: Path, comp: str) -> List[Path]:
    if not comp_dir.is_dir():
        return []

    out: List[Path] = []
    for entry in comp_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name == f"{comp}-1":
            continue
        if not name.startswith(comp):
            continue
        tail = name[len(comp) :]
        if tail and tail.lstrip("-").isdigit():
            out.append(entry)
    return sorted(out)


def _component_finished(comp_dir: Path, comp: str, windows: Sequence[Path]) -> bool:
    if (comp_dir / "FINISHED").exists():
        return True
    if windows and all((w / "FINISHED").exists() for w in windows):
        return True
    return False


def _write_batch_run_script(comp_dir: Path, comp: str, n_windows: int) -> Path:
    from batter._internal.ops.remd import patch_batch_component_inputs
    text = BATCH_RUN_TEMPLATE.read_text()
    text = text.replace("COMPONENT", comp).replace("NWINDOWS", str(n_windows))
    run_script = comp_dir / "run-local-batch.bash"
    run_script.write_text(text)
    try:
        run_script.chmod(0o755)
    except Exception:
        pass

    check_dst = comp_dir / "check_run.bash"
    if not check_dst.exists() and BATCH_CHECK_TEMPLATE.exists():
        check_dst.write_text(BATCH_CHECK_TEMPLATE.read_text())
        try:
            check_dst.chmod(0o755)
        except Exception:
            pass

    patch_batch_component_inputs(comp_dir, comp)

    return run_script


def _collect_batch_tasks(exec_path: Path) -> List[BatchTask]:
    lig_dirs = _resolve_ligand_dirs(exec_path)
    tasks: List[BatchTask] = []

    for lig_dir in lig_dirs:
        comps = components_under(lig_dir)
        if not comps:
            logger.warning(f"[batch] No components found under {lig_dir / 'fe'}")
            continue

        windows_counts = _load_windows_counts(lig_dir / "fe")
        for comp in comps:
            comp_dir = (lig_dir / "fe" / comp).resolve()
            if not comp_dir.is_dir():
                continue

            window_dirs = _component_window_dirs(comp_dir, comp)
            if not window_dirs:
                logger.warning(
                    f"[batch] No window directories found under {comp_dir}; skipping."
                )
                continue

            n_windows = windows_counts.get(comp)
            if n_windows is None or n_windows <= 0:
                n_windows = len(window_dirs)
            elif len(window_dirs) and n_windows != len(window_dirs):
                logger.warning(
                    f"[batch] Window count mismatch under {comp_dir}: "
                    f"metadata={n_windows} dirs={len(window_dirs)}; using dirs count."
                )
                n_windows = len(window_dirs)

            if _component_finished(comp_dir, comp, window_dirs):
                logger.debug(f"[batch] {comp_dir} already finished; skipping.")
                continue

            _write_batch_run_script(comp_dir, comp, n_windows)

            tasks.append(
                BatchTask(
                    execution=exec_path.resolve(),
                    ligand=lig_dir.name,
                    component=comp,
                    comp_dir=comp_dir,
                    n_windows=n_windows,
                )
            )
            logger.info(f"[batch] {comp_dir} Queued (windows={n_windows}).")

    return tasks


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
            run_script = comp_dir / "run-local-remd.bash"
            if not run_script.is_file():
                logger.warning(
                    f"[remd-batch] Missing run-local-remd.bash under {comp_dir}; skipping."
                )
                continue

            finish_time = _remd_finished_time(comp_dir, comp)
            total_ps = _remd_total_ps(comp_dir, comp) if finish_time else None
            is_finished = finished_marker.exists()
            if not is_finished and finish_time and total_ps is not None:
                try:
                    remaining_ps = total_ps - float(finish_time)
                except Exception:
                    remaining_ps = None
                if (
                    remaining_ps is not None
                    and total_ps >= 100.0
                    and remaining_ps <= 100.0
                ):
                    is_finished = True
            status_note = "finished" if is_finished else "pending"
            if finish_time:
                logger.debug(
                    f"[remd-batch] {comp_dir} window0 time(ps)={finish_time} ({status_note})."
                )
            else:
                logger.debug(
                    f"[remd-batch] {comp_dir} window0 time unavailable ({status_note})."
                )

            if is_finished:
                logger.debug(f"[remd-batch] {comp_dir} already finished; skipping.")
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
            logger.info(f"[remd-batch] {comp_dir} Queued (windows={n_windows}).")

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


def _remd_time_from_rst(rst_path: Path) -> str | None:
    if not rst_path.is_file():
        return None
    ncdump = shutil.which("ncdump")
    if not ncdump:
        return None
    try:
        result = subprocess.run(
            [ncdump, "-v", "time", str(rst_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    match = re.search(
        r"^\s*time\s*=\s*([-+0-9.eE]+)\s*;",
        result.stdout,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1)
    return None


def _remd_finished_time(comp_dir: Path, comp: str) -> str | None:
    win0 = comp_dir / f"{comp}00"
    return _remd_time_from_rst(win0 / "md-current.rst7") or _remd_time_from_rst(
        win0 / "md-previous.rst7"
    )


def _remd_total_ps(comp_dir: Path, comp: str) -> float | None:
    tmpl = comp_dir / f"{comp}00" / "mdin-remd-template"
    if not tmpl.is_file():
        return None
    try:
        text = tmpl.read_text()
    except Exception:
        return None
    match = re.search(
        r"^[!#]\s*total_steps\s*=\s*([0-9]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    total_steps = int(match.group(1))
    dt_match = re.search(
        r"^\s*dt\s*=\s*([-+0-9.eEdD]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    dt = 0.001
    if dt_match:
        try:
            dt = float(dt_match.group(1).replace("d", "e").replace("D", "e"))
        except Exception:
            dt = 0.001
    return total_steps * dt


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


def _run_remd_batch(
    execution: tuple[Path, ...],
    output: Path | None,
    header_root: Path | None,
    partition: str | None,
    time_limit: str | None,
    gpus: int | None,
    nodes: int | None,
    gpus_per_node: int | None,
    auto_resubmit: bool,
    signal_mins: float,
    max_resubmit_count: int,
    current_submission_time: int,
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
            logger.debug(f"[remd-batch] Queued {t.comp_dir} (windows={win_note}).")

    if not tasks:
        raise click.ClickException(
            "No unfinished REMD component folders found under the provided paths."
        )

    tasks.sort(key=lambda t: (str(t.execution), t.ligand, t.component))
    if auto_resubmit and signal_mins <= 0:
        raise click.ClickException(
            "--signal-mins must be > 0 when auto-resubmit is enabled."
        )
    if auto_resubmit and max_resubmit_count <= 0:
        raise click.ClickException(
            "--max-resubmit-count must be > 0 when auto-resubmit is enabled."
        )
    if auto_resubmit and current_submission_time < 0:
        raise click.ClickException(
            "--current-submission-time must be >= 0 when auto-resubmit is enabled."
        )
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
    output_path = output or Path.cwd() / f"run_remd_batch_{job_hash}.sbatch"
    output_path_abs = output_path.resolve()
    resubmit_cmd = None
    if auto_resubmit:
        batter_cmd = _which_batter()
        resubmit_args = ["batch", "--remd"]
        for p in exec_paths:
            resubmit_args.extend(["-e", str(p)])
        resubmit_args.extend(["--output", str(output_path_abs)])
        if header_root:
            resubmit_args.extend(["--header-root", str(header_root.resolve())])
        if partition:
            resubmit_args.extend(["--partition", partition])
        if time_limit:
            resubmit_args.extend(["--time-limit", time_limit])
        if gpus is not None:
            resubmit_args.extend(["--gpus", str(gpus)])
        if nodes is not None:
            resubmit_args.extend(["--nodes", str(nodes)])
        if gpus_per_node is not None:
            resubmit_args.extend(["--gpus-per-node", str(gpus_per_node)])
        resubmit_args.extend(
            [
                "--auto-resubmit",
                "--signal-mins",
                str(signal_mins),
                "--max-resubmit-count",
                str(max_resubmit_count),
                "--current-submission-time",
                str(current_submission_time + 1),
            ]
        )
        resubmit_cmd = f"{batter_cmd} " + " ".join(
            shlex.quote(arg) for arg in resubmit_args
        )
    body_lines = [
        "scontrol show job ${SLURM_JOB_ID:-}",
        'echo "Job started at $(date)"',
        "status=0",
        "pids=()",
        "mpi_base=$(echo \"${MPI_EXEC:-mpirun}\" | awk '{print $1}')",
        "mpi_base=${mpi_base##*/}",
        "use_srun=0",
        'if [[ "${mpi_base}" == srun* ]]; then',
        "  use_srun=1",
        "fi",
    ]
    if auto_resubmit:
        body_lines += [
            "resubmit_done=0",
            f'RESUBMIT_CMD="{resubmit_cmd}"',
            f'RESUBMIT_OUTPUT="{output_path_abs}"',
            f"MAX_RESUBMIT_COUNT={max_resubmit_count}",
            f"CURRENT_SUBMISSION_TIME={current_submission_time}",
            "resubmit_allowed() {",
            "  if (( CURRENT_SUBMISSION_TIME + 1 >= MAX_RESUBMIT_COUNT )); then",
            '    echo "[INFO] Auto-resubmit: max submission count reached (next=${CURRENT_SUBMISSION_TIME}+1 >= ${MAX_RESUBMIT_COUNT})."',
            "    return 1",
            "  fi",
            "  return 0",
            "}",
            "regen_and_submit() {",
            '  if [[ "$resubmit_done" -eq 1 ]]; then return; fi',
            "  resubmit_done=1",
            '  echo "[INFO] Auto-resubmit triggered at $(date)"',
            "  resubmit_allowed || return",
            '  eval "$RESUBMIT_CMD" || { echo "[ERROR] Auto-resubmit command failed."; return; }',
            '  if [[ -n "${SLURM_JOB_ID:-}" ]]; then',
            '    sbatch --dependency=afterany:${SLURM_JOB_ID} "$RESUBMIT_OUTPUT"',
            "  else",
            '    sbatch "$RESUBMIT_OUTPUT"',
            "  fi",
            "}",
            'trap "regen_and_submit" USR1',
        ]
    body_lines += [
        "run_remd_task() {",
        '  local label="$1"',
        '  local dir="$2"',
        '  local win="$3"',
        '  local nodes="$4"',
        '  echo "Running ${label}${win:+ (windows=${win})} dir=${dir}"',
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
        body_lines.append("pids+=($!)")

    body_lines.append("")
    body_lines.append('for pid in "${pids[@]}"; do')
    body_lines.append('  wait "$pid" || status=1')
    body_lines.append("done")
    if auto_resubmit:
        body_lines.append("pending=0")
        body_lines.append("for d in \\")
        for t, _, _ in task_specs:
            body_lines.append(f'  "{t.comp_dir}" \\')
        body_lines.append("  ; do")
        body_lines.append('  if [[ ! -f "$d/FINISHED" ]]; then')
        body_lines.append("    pending=1")
        body_lines.append("    break")
        body_lines.append("  fi")
        body_lines.append("done")
        body_lines.append('if [[ "$pending" -eq 1 ]]; then')
        body_lines.append('  echo "[INFO] Auto-resubmit: pending components remain."')
        body_lines.append("  regen_and_submit")
        body_lines.append("else")
        body_lines.append('  echo "[INFO] Auto-resubmit: all components finished."')
        body_lines.append("fi")
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
    if auto_resubmit:
        signal_seconds = int(math.ceil(signal_mins * 60.0))
        script_text = _upsert_sbatch_option(
            script_text, "signal", f"B:USR1@{signal_seconds}"
        )
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
        f"auto-resubmit: {'yes' if auto_resubmit else 'no'} | "
        f"signal-mins: {signal_mins if auto_resubmit else 'n/a'} | "
        f"max-resubmit-count: {max_resubmit_count if auto_resubmit else 'n/a'} | "
        f"current-submission-time: {current_submission_time if auto_resubmit else 'n/a'} | "
        f"job-name: {job_name}"
    )


@cli.command("batch")
@click.option(
    "--execution",
    "-e",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Execution paths to include (run root, simulations/, transformations/, or a leaf folder containing fe/).",
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
    help="Total GPUs to request; defaults to the total window count found.",
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
@click.option(
    "--auto-resubmit/--no-auto-resubmit",
    default=True,
    show_default=True,
    help="Regenerate and resubmit the batch script until all components finish.",
)
@click.option(
    "--signal-mins",
    type=float,
    default=90.0,
    show_default=True,
    help="Minutes before time limit to trigger auto-resubmit (requires --auto-resubmit).",
)
@click.option(
    "--max-resubmit-count",
    type=int,
    default=4,
    show_default=True,
    help="Maximum total submissions (including the first run) when auto-resubmitting.",
)
@click.option(
    "--current-submission-time",
    type=int,
    default=0,
    show_default=True,
    help="Internal counter for auto-resubmit; increments on each resubmission.",
)
@click.option(
    "--remd/--no-remd",
    default=False,
    show_default=True,
    help="Run in REMD mode (uses run-local-remd.bash).",
)
def batch(
    execution: tuple[Path, ...],
    output: Path | None,
    header_root: Path | None,
    partition: str | None,
    time_limit: str | None,
    gpus: int | None,
    nodes: int | None,
    gpus_per_node: int | None,
    auto_resubmit: bool,
    signal_mins: float,
    max_resubmit_count: int,
    current_submission_time: int,
    remd: bool,
) -> None:
    """
    Generate an sbatch script that runs batch workflows for provided executions.
    """
    if remd:
        # Reuse the REMD implementation so behavior stays consistent.
        _run_remd_batch(
            execution=execution,
            output=output,
            header_root=header_root,
            partition=partition,
            time_limit=time_limit,
            gpus=gpus,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            auto_resubmit=auto_resubmit,
            signal_mins=signal_mins,
            max_resubmit_count=max_resubmit_count,
            current_submission_time=current_submission_time,
        )
        return

    exec_paths = [p.resolve() for p in execution]
    tasks: List[BatchTask] = []
    seen: set[Path] = set()

    for path in exec_paths:
        try:
            new_tasks = _collect_batch_tasks(path)
        except ValueError as exc:
            raise click.ClickException(str(exc))

        for t in new_tasks:
            if t.comp_dir in seen:
                continue
            seen.add(t.comp_dir)
            tasks.append(t)
            win_note = t.n_windows if t.n_windows else "unknown"
            logger.debug(f"[batch] Queued {t.comp_dir} (windows={win_note}).")

    if not tasks:
        raise click.ClickException(
            "No unfinished component folders found under the provided paths."
        )

    tasks.sort(key=lambda t: (str(t.execution), t.ligand, t.component))
    if auto_resubmit and signal_mins <= 0:
        raise click.ClickException(
            "--signal-mins must be > 0 when auto-resubmit is enabled."
        )
    if auto_resubmit and max_resubmit_count <= 0:
        raise click.ClickException(
            "--max-resubmit-count must be > 0 when auto-resubmit is enabled."
        )
    if auto_resubmit and current_submission_time < 0:
        raise click.ClickException(
            "--current-submission-time must be >= 0 when auto-resubmit is enabled."
        )

    gpus_per_node_resolved = gpus_per_node
    if gpus_per_node_resolved is None:
        gpus_per_node_resolved = _infer_header_gpus_per_node(header_root)
    if gpus_per_node_resolved is None:
        gpus_per_node_resolved = 8
    if gpus_per_node_resolved <= 0:
        raise click.ClickException("--gpus-per-node must be >= 1.")

    task_specs: list[tuple[BatchTask, int, int]] = []
    total_windows = 0
    max_windows = 0
    total_nodes_needed = 0
    for t in tasks:
        n_windows = t.n_windows
        if n_windows <= 0:
            logger.warning(
                f"[batch] Could not determine windows for {t.comp_dir}; "
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
    job_name = f"fep_batch_{job_hash}"
    output_path = output or Path.cwd() / f"run_batch_{job_hash}.sbatch"
    output_path_abs = output_path.resolve()
    resubmit_cmd = None
    if auto_resubmit:
        batter_cmd = _which_batter()
        resubmit_args = ["batch"]
        for p in exec_paths:
            resubmit_args.extend(["-e", str(p)])
        resubmit_args.extend(["--output", str(output_path_abs)])
        if header_root:
            resubmit_args.extend(["--header-root", str(header_root.resolve())])
        if partition:
            resubmit_args.extend(["--partition", partition])
        if time_limit:
            resubmit_args.extend(["--time-limit", time_limit])
        if gpus is not None:
            resubmit_args.extend(["--gpus", str(gpus)])
        if nodes is not None:
            resubmit_args.extend(["--nodes", str(nodes)])
        if gpus_per_node is not None:
            resubmit_args.extend(["--gpus-per-node", str(gpus_per_node)])
        resubmit_args.extend(
            [
                "--auto-resubmit",
                "--signal-mins",
                str(signal_mins),
                "--max-resubmit-count",
                str(max_resubmit_count),
                "--current-submission-time",
                str(current_submission_time + 1),
            ]
        )
        resubmit_cmd = f"{batter_cmd} " + " ".join(
            shlex.quote(arg) for arg in resubmit_args
        )
    body_lines = [
        "scontrol show job ${SLURM_JOB_ID:-}",
        'echo "Job started at $(date)"',
        "status=0",
        "pids=()",
        "mpi_base=$(echo \"${MPI_EXEC:-mpirun}\" | awk '{print $1}')",
        "mpi_base=${mpi_base##*/}",
        "use_srun=0",
        'if [[ "${mpi_base}" == srun* ]]; then',
        "  use_srun=1",
        "fi",
    ]
    if auto_resubmit:
        body_lines += [
            "resubmit_done=0",
            f'RESUBMIT_CMD="{resubmit_cmd}"',
            f'RESUBMIT_OUTPUT="{output_path_abs}"',
            f"MAX_RESUBMIT_COUNT={max_resubmit_count}",
            f"CURRENT_SUBMISSION_TIME={current_submission_time}",
            "resubmit_allowed() {",
            "  if (( CURRENT_SUBMISSION_TIME + 1 >= MAX_RESUBMIT_COUNT )); then",
            '    echo "[INFO] Auto-resubmit: max submission count reached (next=${CURRENT_SUBMISSION_TIME}+1 >= ${MAX_RESUBMIT_COUNT})."',
            "    return 1",
            "  fi",
            "  return 0",
            "}",
            "regen_and_submit() {",
            '  if [[ "$resubmit_done" -eq 1 ]]; then return; fi',
            "  resubmit_done=1",
            '  echo "[INFO] Auto-resubmit triggered at $(date)"',
            "  resubmit_allowed || return",
            '  eval "$RESUBMIT_CMD" || { echo "[ERROR] Auto-resubmit command failed."; return; }',
            '  if [[ -n "${SLURM_JOB_ID:-}" ]]; then',
            '    sbatch --dependency=afterany:${SLURM_JOB_ID} "$RESUBMIT_OUTPUT"',
            "  else",
            '    sbatch "$RESUBMIT_OUTPUT"',
            "  fi",
            "}",
            'trap "regen_and_submit" USR1',
        ]
    body_lines += [
        "run_batch_task() {",
        '  local label="$1"',
        '  local dir="$2"',
        '  local win="$3"',
        '  local nodes="$4"',
        '  echo "Running ${label}${win:+ (windows=${win})} dir=${dir}"',
        '  local mpi_flags="${MPI_FLAGS:-}"',
        '  if [[ "$use_srun" -eq 1 && "$nodes" -gt 0 && "$win" -gt 0 ]]; then',
        '    local extra_flags="--nodes=${nodes} --ntasks=${win} --exclusive --gpus-per-task=1"',
        '    if [[ -n "$mpi_flags" ]]; then',
        '      mpi_flags="${mpi_flags} ${extra_flags}"',
        "    else",
        '      mpi_flags="${extra_flags}"',
        "    fi",
        "  fi",
        '  ( cd "$dir" && MPI_FLAGS="$mpi_flags" bash ./run-local-batch.bash ) || status=1',
        "}",
        "",
    ]

    for t, n_windows, nodes_needed in task_specs:
        label = f"{t.ligand}/{t.component}"
        if t.execution.name:
            label = f"{t.execution.name}/{label}"
        body_lines.append(
            f'run_batch_task "{label}" "{t.comp_dir}" "{n_windows}" "{nodes_needed}" &'
        )
        body_lines.append("pids+=($!)")

    body_lines.append("")
    body_lines.append('for pid in "${pids[@]}"; do')
    body_lines.append('  wait "$pid" || status=1')
    body_lines.append("done")
    if auto_resubmit:
        body_lines.append("pending=0")
        body_lines.append("for d in \\")
        for t, _, _ in task_specs:
            body_lines.append(f'  "{t.comp_dir}" \\')
        body_lines.append("  ; do")
        body_lines.append('  if [[ ! -f "$d/FINISHED" ]]; then')
        body_lines.append("    pending=1")
        body_lines.append("    break")
        body_lines.append("  fi")
        body_lines.append("done")
        body_lines.append('if [[ "$pending" -eq 1 ]]; then')
        body_lines.append('  echo "[INFO] Auto-resubmit: pending components remain."')
        body_lines.append("  regen_and_submit")
        body_lines.append("else")
        body_lines.append('  echo "[INFO] Auto-resubmit: all components finished."')
        body_lines.append("fi")
    body_lines.append('echo "Job completed at $(date)"')
    body_lines.append("exit $status")
    body_text = "\n".join(body_lines) + "\n"

    script_text = _render_remd_batch_script(
        body_text,
        {
            "SYSTEMNAME": job_name,
            "STAGE": "batch",
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
    if auto_resubmit:
        signal_seconds = int(math.ceil(signal_mins * 60.0))
        script_text = _upsert_sbatch_option(
            script_text, "signal", f"B:USR1@{signal_seconds}"
        )
    if gpu_request:
        gres_val = None
        if node_request and gpus_per_node_resolved:
            gres_val = gpus_per_node_resolved
        elif not node_request:
            gres_val = gpu_request
        if gres_val:
            script_text = _upsert_sbatch_option(script_text, "gres", f"gpu:{gres_val}")
        if node_request and not gpus_per_node_resolved:
            script_text = _upsert_sbatch_option(script_text, "gres", "gpu:1")
        script_text = _upsert_sbatch_option(script_text, "gpus-per-task", "1")
        script_text = _upsert_sbatch_option(script_text, "ntasks", str(gpu_request))

    script_text = _upsert_sbatch_option(script_text, "output", f"{job_name}-%j.out")
    script_text = _upsert_sbatch_option(script_text, "error", f"{job_name}-%j.err")

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
        f"auto-resubmit: {'yes' if auto_resubmit else 'no'} | "
        f"signal-mins: {signal_mins if auto_resubmit else 'n/a'} | "
        f"max-resubmit-count: {max_resubmit_count if auto_resubmit else 'n/a'} | "
        f"current-submission-time: {current_submission_time if auto_resubmit else 'n/a'} | "
        f"job-name: {job_name}"
    )
