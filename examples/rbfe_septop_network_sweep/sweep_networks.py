#!/usr/bin/env python3
"""Sweep rbfe_septop network-building parameters on the CHK1 example."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "batter").is_dir():
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SOURCE = Path(
    "/home/users/yuzhuang/yuzhuang_scratch/public_binding_free_energy_benchmark_new/"
    "fep_benchmark_inputs/structure_inputs/batter_run/CHK1_scaffold_hopping/"
    "CHK1_3u9n_corehop_rbfe"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs"

DEFAULT_KARTOGRAF_OPTIONS: dict[str, Any] = {
    "atom_max_distance": 0.95,
    "map_exact_ring_matches_only": True,
    "allow_partial_fused_rings": True,
    "allow_bond_breaks": False,
    "filter_element_changes": True,
    "filter_mismatched_attached_h_count": False,
}

DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "star_shape_oneway",
        "layout": "star",
        "network_scorer": "auto",
        "both_directions": False,
    },
    {
        "name": "mst_shape_oneway",
        "layout": "minimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
    },
    {
        "name": "rmst_shape_default_oneway",
        "layout": "redundantminimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
    },
    {
        "name": "rmst_shape_default_bidir",
        "layout": "redundantminimalspanningtree",
        "network_scorer": "auto",
        "both_directions": True,
    },
    {
        "name": "rmst_shape_n1_oneway",
        "layout": "redundantminimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
        "generator_kwargs": {"n_redundancy": 1},
    },
    {
        "name": "rmst_shape_n2_oneway",
        "layout": "redundantminimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
        "generator_kwargs": {"n_redundancy": 2},
    },
    {
        "name": "rmst_shape_n3_oneway",
        "layout": "redundantminimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
        "generator_kwargs": {"n_redundancy": 3},
    },
    {
        "name": "mst_lomap_scorer_oneway",
        "layout": "minimalspanningtree",
        "network_scorer": "lomap",
        "both_directions": False,
    },
    {
        "name": "mst_shape_loose_atomdist_oneway",
        "layout": "minimalspanningtree",
        "network_scorer": "auto",
        "both_directions": False,
        "kartograf": {"atom_max_distance": 1.25},
    },
]


def _load_batter_network_helpers():
    try:
        from batter.config.utils import sanitize_ligand_name
        from batter.rbfe import (
            _build_konnektor_atom_mapper,
            _load_rdkit_mol,
            _network_scorer_callable,
            _pairs_from_konnektor_network,
            _resolve_konnektor_generator,
            resolve_network_scorer_name,
        )
        from gufe import SmallMoleculeComponent
    except Exception as exc:
        raise SystemExit(
            "Could not import BATTER/Konnektor network dependencies. Run with the "
            "production BATTER environment, for example:\n"
            "/home/users/yuzhuang/yuzhuang_scratch/miniforge3_0424/envs/batter/bin/python "
            f"{Path(__file__).relative_to(REPO_ROOT)}\n"
            f"Import error: {exc}"
        ) from exc

    return {
        "sanitize_ligand_name": sanitize_ligand_name,
        "build_mapper": _build_konnektor_atom_mapper,
        "load_rdkit_mol": _load_rdkit_mol,
        "network_scorer_callable": _network_scorer_callable,
        "pairs_from_network": _pairs_from_konnektor_network,
        "resolve_generator": _resolve_konnektor_generator,
        "resolve_network_scorer_name": resolve_network_scorer_name,
        "SmallMoleculeComponent": SmallMoleculeComponent,
    }


def _candidate_ligand_dicts(source: Path) -> list[Path]:
    return [
        source / "ligand_dict.json",
        source.parent / "ligand_dict.json",
        source.parent.parent / "ligand_dict.json",
        source / "executions" / "rep1" / "artifacts" / "config" / "ligand_dict.json",
    ]


def _resolve_ligands(
    source: Path,
    ligand_dict_path: Path | None,
    sanitize_ligand_name,
) -> tuple[list[str], dict[str, Path], Path]:
    if ligand_dict_path is None:
        for candidate in _candidate_ligand_dicts(source):
            if candidate.is_file():
                ligand_dict_path = candidate
                break
    if ligand_dict_path is None:
        searched = "\n".join(str(path) for path in _candidate_ligand_dicts(source))
        raise SystemExit(f"Could not find ligand_dict.json. Searched:\n{searched}")

    ligand_dict_path = ligand_dict_path.resolve()
    raw = json.loads(ligand_dict_path.read_text())
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{ligand_dict_path} must contain a JSON object.")

    ligands: list[str] = []
    ligand_files: dict[str, Path] = {}
    for raw_name, raw_path in raw.items():
        name = sanitize_ligand_name(str(raw_name))
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = ligand_dict_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Missing ligand file for {name}: {path}")
        ligands.append(name)
        ligand_files[name] = path

    return ligands, ligand_files, ligand_dict_path


def _case_kartograf_options(case: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(DEFAULT_KARTOGRAF_OPTIONS)
    options.update(dict(case.get("kartograf") or {}))
    return options


def _build_pairs(
    *,
    ligands: Sequence[str],
    ligand_files: Mapping[str, Path],
    case: Mapping[str, Any],
    helpers: Mapping[str, Any],
) -> list[tuple[str, str]]:
    layout = str(case.get("layout") or "star")
    network_scorer = str(case.get("network_scorer") or "auto")
    atom_mapper = str(case.get("atom_mapper") or "kartograf")
    generator_kwargs = dict(case.get("generator_kwargs") or {})

    generator_cls = helpers["resolve_generator"](layout)
    mapper = helpers["build_mapper"](
        atom_mapper,
        kartograf_options=_case_kartograf_options(case),
        lomap_options=dict(case.get("lomap") or {}),
    )
    scorer = helpers["network_scorer_callable"](
        network_scorer,
        protocol="rbfe_septop",
    )
    generator = generator_cls(
        mappers=mapper,
        scorer=scorer,
        **generator_kwargs,
    )

    components = []
    for ligand in ligands:
        mol = helpers["load_rdkit_mol"](ligand_files[ligand])
        components.append(helpers["SmallMoleculeComponent"](mol, name=ligand))

    if hasattr(generator, "generate_ligand_network"):
        network = generator.generate_ligand_network(components)
    elif hasattr(generator, "generate_network"):
        network = generator.generate_network(components)
    else:
        network = generator(components)

    return list(helpers["pairs_from_network"](network))


def _add_reverse_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref, alt in pairs:
        for pair in ((ref, alt), (alt, ref)):
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
    return out


def _undirected_edges(pairs: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
    return {tuple(sorted((str(ref), str(alt)))) for ref, alt in pairs if ref != alt}


def _degree_by_ligand(
    ligands: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> Counter[str]:
    degree: Counter[str] = Counter({ligand: 0 for ligand in ligands})
    for ref, alt in _undirected_edges(pairs):
        degree[ref] += 1
        degree[alt] += 1
    return degree


def _write_case_outputs(
    *,
    case_dir: Path,
    ligands: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    raw_pairs: Sequence[tuple[str, str]],
    case: Mapping[str, Any],
    resolved_scorer: str,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    undirected = sorted(_undirected_edges(pairs))
    degree = _degree_by_ligand(ligands, pairs)
    max_degree = max(degree.values(), default=0)

    network_payload = {
        "ligands": list(ligands),
        "pairs": [[ref, alt] for ref, alt in pairs],
        "raw_generator_pairs": [[ref, alt] for ref, alt in raw_pairs],
        "layout": case.get("layout"),
        "network_scorer": resolved_scorer,
        "network_scorer_requested": case.get("network_scorer", "auto"),
        "atom_mapper": case.get("atom_mapper", "kartograf"),
        "both_directions": bool(case.get("both_directions", False)),
        "generator_kwargs": dict(case.get("generator_kwargs") or {}),
        "kartograf": _case_kartograf_options(case),
        "n_ligands": len(ligands),
        "n_directed_pairs": len(pairs),
        "n_undirected_edges": len(undirected),
        "max_undirected_degree": max_degree,
        "degrees": dict(degree),
    }
    (case_dir / "network.json").write_text(json.dumps(network_payload, indent=2))

    with (case_dir / "edges.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["ref", "alt", "undirected_key"])
        for ref, alt in pairs:
            writer.writerow([ref, alt, "~".join(sorted((ref, alt)))])

    with (case_dir / "degree.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["ligand", "undirected_degree"])
        for ligand in ligands:
            writer.writerow([ligand, degree[ligand]])

    return network_payload


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(case) for case in DEFAULT_CASES]
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, list):
        raise SystemExit(f"{path} must contain a JSON list of case objects.")
    return [dict(case) for case in loaded]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep Konnektor/BATTER rbfe_septop network parameters."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="CHK1 input folder or output folder. Defaults to the existing CHK1 run.",
    )
    parser.add_argument(
        "--ligand-dict",
        type=Path,
        default=None,
        help="Optional ligand_dict.json path. Relative ligand paths resolve from this file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for sweep summaries.",
    )
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="JSON list of cases. See cases.example.json.",
    )
    args = parser.parse_args(argv)

    helpers = _load_batter_network_helpers()
    ligands, ligand_files, ligand_dict_path = _resolve_ligands(
        args.source.resolve(),
        args.ligand_dict.resolve() if args.ligand_dict else None,
        helpers["sanitize_ligand_name"],
    )
    cases = _load_cases(args.cases_json)

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        name = str(case.get("name") or case.get("layout") or "network_case")
        case_dir = args.out / name
        raw_pairs = _build_pairs(
            ligands=ligands,
            ligand_files=ligand_files,
            case=case,
            helpers=helpers,
        )
        pairs = (
            _add_reverse_pairs(raw_pairs)
            if bool(case.get("both_directions", False))
            else list(raw_pairs)
        )
        resolved_scorer = helpers["resolve_network_scorer_name"](
            case.get("network_scorer", "auto"),
            protocol="rbfe_septop",
        )
        payload = _write_case_outputs(
            case_dir=case_dir,
            ligands=ligands,
            pairs=pairs,
            raw_pairs=raw_pairs,
            case=case,
            resolved_scorer=resolved_scorer,
        )
        summary_rows.append(
            {
                "case": name,
                "layout": case.get("layout"),
                "network_scorer": resolved_scorer,
                "atom_mapper": case.get("atom_mapper", "kartograf"),
                "both_directions": bool(case.get("both_directions", False)),
                "generator_kwargs": json.dumps(
                    dict(case.get("generator_kwargs") or {}),
                    sort_keys=True,
                ),
                "kartograf_overrides": json.dumps(
                    dict(case.get("kartograf") or {}),
                    sort_keys=True,
                ),
                "n_ligands": payload["n_ligands"],
                "n_generator_pairs": len(raw_pairs),
                "n_directed_pairs": payload["n_directed_pairs"],
                "n_undirected_edges": payload["n_undirected_edges"],
                "max_undirected_degree": payload["max_undirected_degree"],
                "degrees": json.dumps(payload["degrees"], sort_keys=True),
                "pairs": ";".join(f"{ref}~{alt}" for ref, alt in pairs),
            }
        )

    summary_path = args.out / "summary.tsv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(summary_rows[0].keys()) if summary_rows else ["case"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "source": str(args.source.resolve()),
        "ligand_dict": str(ligand_dict_path),
        "out": str(args.out.resolve()),
        "n_ligands": len(ligands),
        "ligands": list(ligands),
        "cases": [row["case"] for row in summary_rows],
    }
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {summary_path}")
    for row in summary_rows:
        print(
            "{case}: {n_undirected_edges} undirected, {n_directed_pairs} directed, "
            "max degree {max_undirected_degree}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

