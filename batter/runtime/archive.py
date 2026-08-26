from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Callable, Iterable, Sequence


REPRODUCIBLE_ROOT_DIRS = (
    Path("artifacts/config"),
    Path("artifacts/ligand_params"),
    Path("artifacts/ligands"),
    Path("inputs"),
    Path("all-ligands"),
)
REPRODUCIBLE_SIM_DIR_NAMES = frozenset({"inputs", "params"})
REPRODUCIBLE_ROOT_SUFFIXES = frozenset(
    {".yaml", ".yml", ".json", ".toml", ".txt", ".md", ".csv", ".tsv", ".log"}
)
RESULTS_DIR_PATTERNS = (
    "results",
    "simulations/*/results",
    "simulations/*/equil/results",
    "simulations/*/fe/results",
    "simulations/*/fe/*/*/results",
    "simulations/*/*/results",
    "simulations/*/*/equil/results",
    "simulations/*/*/fe/results",
    "simulations/*/*/fe/*/*/results",
)


@dataclass(frozen=True)
class ExecutionArchiveSummary:
    execution: Path
    archive_prefix: str
    results_dirs: tuple[Path, ...]
    associated_results: tuple[Path, ...]
    reproducible_inputs: tuple[Path, ...]


@dataclass(frozen=True)
class ExecutionArchiveResult:
    archive_path: Path
    members: tuple[str, ...]
    executions: tuple[ExecutionArchiveSummary, ...]


@dataclass(frozen=True)
class ArchiveMember:
    source: Path
    arcname: str


@dataclass(frozen=True)
class ExecutionArchivePlan:
    output_path: Path
    root_name: str
    mode: str
    manifest: bytes
    summaries: tuple[ExecutionArchiveSummary, ...]
    members: tuple[ArchiveMember, ...]


def discover_execution_paths(paths: Iterable[Path | str]) -> list[Path]:
    """Resolve execution directories from explicit executions or execution parents."""
    discovered: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            return
        if not resolved.is_dir():
            raise FileNotFoundError(f"Execution path is not a directory: {candidate}")
        seen.add(resolved)
        discovered.append(resolved)

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Execution path is not a directory: {raw_path}")
        if path.name == "executions":
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    add(child)
            continue
        if (path / "executions").is_dir() and not (
            (path / "simulations").is_dir()
            or (path / "artifacts").is_dir()
            or (path / "results").is_dir()
        ):
            for child in sorted((path / "executions").iterdir()):
                if child.is_dir():
                    add(child)
            continue
        add(path)
    return discovered


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_results_dirs(execution: Path) -> list[Path]:
    """Find BATTER results directories without walking raw simulation trees."""
    results_dirs: list[Path] = []
    for pattern in RESULTS_DIR_PATTERNS:
        for path in sorted(execution.glob(pattern)):
            if path.is_dir():
                results_dirs.append(path)
    return _dedupe_paths(results_dirs)


def discover_associated_results(execution: Path) -> list[Path]:
    """Find sibling output/results entries associated with one execution run_id."""
    if execution.parent.name != "executions":
        return []
    output_root = execution.parent.parent
    results_root = output_root / "results"
    if not results_root.is_dir():
        return []
    candidates: list[Path] = []
    for name in ("index.csv", "README.txt", "README.md"):
        path = results_root / name
        if path.exists():
            candidates.append(path)
    run_results = results_root / execution.name
    if run_results.exists():
        candidates.append(run_results)
    return _dedupe_paths(candidates)


def discover_reproducible_inputs(execution: Path) -> list[Path]:
    """Collect small run inputs needed to understand or reproduce archived results."""
    candidates: list[Path] = []
    for rel_dir in REPRODUCIBLE_ROOT_DIRS:
        path = execution / rel_dir
        if path.exists():
            candidates.append(path)

    for child in sorted((execution / "simulations").glob("*")):
        if not child.is_dir():
            continue
        for name in sorted(REPRODUCIBLE_SIM_DIR_NAMES):
            path = child / name
            if path.exists():
                candidates.append(path)
        phase_state = child / "artifacts" / "phase_state.json"
        if phase_state.exists():
            candidates.append(phase_state)

    for child in sorted(execution.iterdir()):
        if child.is_file() and child.suffix.lower() in REPRODUCIBLE_ROOT_SUFFIXES:
            candidates.append(child)

    return _dedupe_paths(candidates)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _tar_mode(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        return "w:gz"
    if name.endswith((".tar.bz2", ".tbz", ".tbz2")):
        return "w:bz2"
    if name.endswith((".tar.xz", ".txz")):
        return "w:xz"
    return "w"


def _execution_prefixes(executions: Sequence[Path]) -> dict[Path, str]:
    if not executions:
        return {}
    common = Path(os.path.commonpath([str(path) for path in executions]))
    prefixes: dict[Path, str] = {}
    used: set[str] = set()
    for execution in executions:
        try:
            rel = execution.relative_to(common)
        except ValueError:
            rel = Path(execution.name)
        prefix = rel.as_posix()
        if prefix in {"", "."}:
            prefix = execution.name
        if prefix in used:
            prefix = execution.as_posix().lstrip("/").replace("/", "__")
        used.add(prefix)
        prefixes[execution] = prefix
    return prefixes


def _extra_prefixes(extra_paths: Sequence[Path]) -> dict[Path, str]:
    prefixes: dict[Path, str] = {}
    used: set[str] = set()
    for path in extra_paths:
        name = path.name or path.resolve().as_posix().lstrip("/").replace("/", "__")
        prefix = name
        if prefix in used:
            prefix = path.resolve().as_posix().lstrip("/").replace("/", "__")
        used.add(prefix)
        prefixes[path] = prefix
    return prefixes


def _iter_archive_members(source: Path, arcname: str) -> Iterable[ArchiveMember]:
    if source.is_dir() and not source.is_symlink():
        yield ArchiveMember(source=source, arcname=arcname)
        for root, dirnames, filenames in os.walk(source, followlinks=False):
            dirnames.sort()
            filenames.sort()
            root_path = Path(root)
            for dirname in dirnames:
                child = root_path / dirname
                child_rel = child.relative_to(source).as_posix()
                yield ArchiveMember(source=child, arcname=f"{arcname}/{child_rel}")
            for filename in filenames:
                child = root_path / filename
                child_rel = child.relative_to(source).as_posix()
                yield ArchiveMember(source=child, arcname=f"{arcname}/{child_rel}")
        return
    yield ArchiveMember(source=source, arcname=arcname)


def _manifest_for(
    *,
    summaries: Sequence[ExecutionArchiveSummary],
    extra_paths: Sequence[Path],
    root_name: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive_root": root_name,
        "policy": {
            "results": "all directories named results below each execution",
            "reproducible_inputs": [
                rel.as_posix() for rel in REPRODUCIBLE_ROOT_DIRS
            ]
            + [
                "simulations/<ligand>/inputs",
                "simulations/<ligand>/params",
                "simulations/<ligand>/artifacts/phase_state.json",
                "root-level *.yaml/*.yml/*.json/*.toml/*.txt/*.md/*.csv/*.tsv/*.log",
            ],
        },
        "executions": [
            {
                "source": str(summary.execution),
                "archive_prefix": summary.archive_prefix,
                "results_dirs": [
                    path.relative_to(summary.execution).as_posix()
                    for path in summary.results_dirs
                ],
                "associated_results": [
                    path.relative_to(summary.execution.parent.parent).as_posix()
                    for path in summary.associated_results
                    if (
                        summary.execution.parent.name == "executions"
                        and _is_relative_to(
                            path.resolve(), summary.execution.parent.parent
                        )
                    )
                ],
                "reproducible_inputs": [
                    path.relative_to(summary.execution).as_posix()
                    for path in summary.reproducible_inputs
                    if _is_relative_to(path.resolve(), summary.execution)
                ],
            }
            for summary in summaries
        ],
        "extra_inputs": [str(path) for path in extra_paths],
    }


def build_execution_archive_plan(
    executions: Iterable[Path | str],
    output: Path | str,
    *,
    include: Iterable[Path | str] = (),
    root_name: str = "batter_archive",
    overwrite: bool = False,
) -> ExecutionArchivePlan:
    """Plan a compact tar archive for BATTER execution results and inputs."""
    execution_paths = discover_execution_paths(executions)
    if not execution_paths:
        raise ValueError("No execution directories were provided.")

    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra_paths = _dedupe_paths([Path(path).expanduser().resolve() for path in include])
    for path in extra_paths:
        if not path.exists():
            raise FileNotFoundError(f"Included input path does not exist: {path}")

    execution_prefixes = _execution_prefixes(execution_paths)
    summaries: list[ExecutionArchiveSummary] = []
    for execution in execution_paths:
        summaries.append(
            ExecutionArchiveSummary(
                execution=execution,
                archive_prefix=execution_prefixes[execution],
                results_dirs=tuple(discover_results_dirs(execution)),
                associated_results=tuple(discover_associated_results(execution)),
                reproducible_inputs=tuple(discover_reproducible_inputs(execution)),
            )
        )

    manifest = json.dumps(
        _manifest_for(
            summaries=summaries,
            extra_paths=extra_paths,
            root_name=root_name,
        ),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    members: list[ArchiveMember] = []
    added_sources: set[Path] = set()

    def add_path(source: Path, arcname: str) -> None:
        for member in _iter_archive_members(source, arcname):
            resolved = member.source.resolve()
            if resolved == output_path or resolved in added_sources:
                continue
            added_sources.add(resolved)
            members.append(member)

    for summary in summaries:
        base = f"{root_name}/{summary.archive_prefix}"
        for path in summary.results_dirs:
            rel = path.relative_to(summary.execution).as_posix()
            add_path(path, f"{base}/{rel}")
        for path in summary.associated_results:
            rel = path.relative_to(summary.execution.parent.parent).as_posix()
            add_path(path, f"{root_name}/{rel}")
        for path in summary.reproducible_inputs:
            rel = path.relative_to(summary.execution).as_posix()
            add_path(path, f"{base}/{rel}")

    for path, prefix in _extra_prefixes(extra_paths).items():
        add_path(path, f"{root_name}/extra_inputs/{prefix}")

    return ExecutionArchivePlan(
        output_path=output_path,
        root_name=root_name,
        mode=_tar_mode(output_path),
        manifest=manifest,
        summaries=tuple(summaries),
        members=tuple(members),
    )


def write_execution_archive_plan(
    plan: ExecutionArchivePlan,
    *,
    progress: Callable[[str], None] | None = None,
) -> ExecutionArchiveResult:
    """Write a prepared execution archive plan."""
    members: list[str] = []
    with tarfile.open(plan.output_path, plan.mode) as tar:
        info = tarfile.TarInfo(f"{plan.root_name}/archive_manifest.json")
        info.size = len(plan.manifest)
        info.mtime = datetime.now(timezone.utc).timestamp()
        tar.addfile(info, fileobj=io.BytesIO(plan.manifest))
        members.append(info.name)
        if progress is not None:
            progress(info.name)

        for member in plan.members:
            tar.add(member.source, arcname=member.arcname, recursive=False)
            members.append(member.arcname)
            if progress is not None:
                progress(member.arcname)

    return ExecutionArchiveResult(
        archive_path=plan.output_path,
        members=tuple(members),
        executions=plan.summaries,
    )


def create_execution_archive(
    executions: Iterable[Path | str],
    output: Path | str,
    *,
    include: Iterable[Path | str] = (),
    root_name: str = "batter_archive",
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExecutionArchiveResult:
    """Create a compact tar archive for BATTER execution results and inputs."""
    plan = build_execution_archive_plan(
        executions,
        output,
        include=include,
        root_name=root_name,
        overwrite=overwrite,
    )
    return write_execution_archive_plan(plan, progress=progress)
