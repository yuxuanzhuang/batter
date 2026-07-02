"""RBFE network helpers."""

from __future__ import annotations

import base64
import math
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List, Any, Mapping
from loguru import logger

from batter.config.utils import sanitize_ligand_name
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign, AllChem


def _normalize_atom_mapper(atom_mapper: str | None) -> str:
    mapper = str(atom_mapper or "kartograf").strip().lower()
    if mapper not in {"kartograf", "lomap"}:
        raise ValueError(
            f"Unknown atom mapper '{atom_mapper}'. Available: kartograf, lomap"
        )
    return mapper


def _normalize_protocol(protocol: str | None) -> str:
    return str(protocol or "").strip().lower().replace("-", "_")


def resolve_network_scorer_name(
    network_scorer: str | None = None,
    *,
    protocol: str | None = None,
) -> str:
    scorer = str(network_scorer or "auto").strip().lower().replace("-", "_")
    if scorer in {"", "auto", "default"}:
        return (
            "pocket_shape"
            if _normalize_protocol(protocol) == "rbfe_septop"
            else "lomap"
        )
    if scorer in {"lomap", "default_lomap"}:
        return "lomap"
    if scorer in {
        "shape",
        "shape_difference",
        "shape_mismatch",
        "kartograf_shape",
        "kartograf_shape_difference",
    }:
        return "shape_difference"
    if scorer in {
        "pocket",
        "pocket_shape",
        "grid_shape",
        "pocket_grid",
        "receptor_grid",
        "receptor_shape",
        "receptor_frame_shape",
    }:
        return "pocket_shape"
    raise ValueError(
        f"Unknown RBFE network scorer '{network_scorer}'. "
        "Available: auto, lomap, shape_difference, pocket_shape"
    )


def _shape_difference_network_score(mapping) -> float:
    """High-is-good score that minimizes Kartograf shape mismatch distance."""
    mapped_count = _mapping_mapped_atom_count(mapping)
    if mapped_count is not None and mapped_count < 2:
        return 0.0
    try:
        from kartograf.mapping_metrics.metric_shape_difference import (
            MappingShapeMismatchScorer,
        )

        scorer = MappingShapeMismatchScorer(ignore_hs=True)
        mol_shape_dist = scorer.get_rdmol_shape_distance(
            mapping.componentA.to_rdkit(),
            mapping.componentB.to_rdkit(),
        )
        mapped_shape_dist = scorer.get_mapped_structure_shape_distance(mapping)
        if not math.isfinite(float(mapped_shape_dist)):
            return 0.0
        distance = (float(mapped_shape_dist) + 2.0 * float(mol_shape_dist)) / 3.0
        if not math.isfinite(distance):
            return 0.0
        return max(0.0, min(1.0, 1.0 - distance))
    except Exception as exc:
        logger.debug(f"[rbfe] shape-difference network score failed: {exc}")
        return 0.0


def _rdkit_mol_from_component(component: Any) -> Chem.Mol | None:
    if component is None:
        return None
    if hasattr(component, "to_rdkit"):
        try:
            mol = component.to_rdkit()
            if mol is not None:
                return mol
        except Exception:
            pass
    mol = getattr(component, "_rdkit", None)
    if mol is not None:
        return mol
    mol = getattr(component, "mol", None)
    if mol is not None:
        return mol
    return None


def _mapping_component(mapping: Any, name: str) -> Any:
    component = getattr(mapping, name, None)
    if component is None:
        component = getattr(mapping, f"_{name}", None)
    return component


def _pocket_grid_occupancy(
    mol: Chem.Mol,
    *,
    spacing: float = 0.5,
    radius_buffer: float = 0.25,
) -> set[tuple[int, int, int]] | None:
    """Voxelized receptor-frame ligand heavy-atom occupancy.

    The input ligand poses are assumed to already be in the same receptor frame.
    A voxel is occupied if its center falls within a buffered vdW radius of any
    ligand heavy atom.
    """
    if mol is None or mol.GetNumConformers() < 1:
        return None
    if spacing <= 0:
        return None

    try:
        conf = mol.GetConformer()
    except Exception:
        return None

    periodic_table = Chem.GetPeriodicTable()
    voxels: set[tuple[int, int, int]] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        try:
            pos = conf.GetAtomPosition(idx)
        except Exception:
            continue
        try:
            radius = float(periodic_table.GetRvdw(atom.GetAtomicNum()))
        except Exception:
            radius = 1.7
        if not math.isfinite(radius) or radius <= 0:
            radius = 1.7
        radius = radius + radius_buffer

        ix0 = math.floor((float(pos.x) - radius) / spacing)
        ix1 = math.floor((float(pos.x) + radius) / spacing)
        iy0 = math.floor((float(pos.y) - radius) / spacing)
        iy1 = math.floor((float(pos.y) + radius) / spacing)
        iz0 = math.floor((float(pos.z) - radius) / spacing)
        iz1 = math.floor((float(pos.z) + radius) / spacing)

        for ix in range(ix0, ix1 + 1):
            cx = (ix + 0.5) * spacing
            dx2 = (cx - float(pos.x)) ** 2
            if dx2 > radius * radius:
                continue
            for iy in range(iy0, iy1 + 1):
                cy = (iy + 0.5) * spacing
                dxy2 = dx2 + (cy - float(pos.y)) ** 2
                if dxy2 > radius * radius:
                    continue
                for iz in range(iz0, iz1 + 1):
                    cz = (iz + 0.5) * spacing
                    if dxy2 + (cz - float(pos.z)) ** 2 <= radius * radius:
                        voxels.add((ix, iy, iz))

    return voxels or None


def _pocket_grid_overlap_score(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    *,
    spacing: float = 0.5,
    radius_buffer: float = 0.25,
) -> float | None:
    """High-is-good receptor-frame occupancy score for two ligand poses."""
    metrics = _pocket_grid_overlap_metrics(
        mol_a,
        mol_b,
        spacing=spacing,
        radius_buffer=radius_buffer,
    )
    if metrics is None:
        return None
    return metrics["pocket_grid_score"]


def _pocket_grid_overlap_metrics(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    *,
    spacing: float = 0.5,
    radius_buffer: float = 0.25,
) -> dict[str, float] | None:
    """Return receptor-frame occupancy metrics for two ligand poses."""
    vox_a = _pocket_grid_occupancy(
        mol_a,
        spacing=spacing,
        radius_buffer=radius_buffer,
    )
    vox_b = _pocket_grid_occupancy(
        mol_b,
        spacing=spacing,
        radius_buffer=radius_buffer,
    )
    if not vox_a or not vox_b:
        return None

    overlap = len(vox_a & vox_b)
    if overlap <= 0:
        return {
            "pocket_grid_score": 0.0,
            "pocket_grid_containment": 0.0,
            "pocket_grid_jaccard": 0.0,
            "pocket_grid_overlap_voxels": 0.0,
            "pocket_grid_ref_voxels": float(len(vox_a)),
            "pocket_grid_alt_voxels": float(len(vox_b)),
        }
    min_volume = min(len(vox_a), len(vox_b))
    union = len(vox_a | vox_b)
    if min_volume <= 0 or union <= 0:
        return None

    containment = overlap / min_volume
    jaccard = overlap / union
    score = 0.65 * containment + 0.35 * jaccard
    return {
        "pocket_grid_score": max(0.0, min(1.0, float(score))),
        "pocket_grid_containment": max(0.0, min(1.0, float(containment))),
        "pocket_grid_jaccard": max(0.0, min(1.0, float(jaccard))),
        "pocket_grid_overlap_voxels": float(overlap),
        "pocket_grid_ref_voxels": float(len(vox_a)),
        "pocket_grid_alt_voxels": float(len(vox_b)),
    }


def _pocket_similarity_metric_scores(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    mapping: Any | None = None,
) -> dict[str, float]:
    """Metrics stored in RBFE mapping artifacts for HTML edge visualization."""
    metrics = _pocket_grid_overlap_metrics(mol_a, mol_b)
    if metrics is None:
        return {}

    shape_score = (
        _shape_difference_network_score(mapping) if mapping is not None else 0.0
    )
    pocket_shape_score = metrics["pocket_grid_score"]
    if shape_score > 0:
        pocket_shape_score = 0.85 * metrics["pocket_grid_score"] + 0.15 * shape_score

    out = {
        "pocket_shape_score": max(0.0, min(1.0, float(pocket_shape_score))),
        **metrics,
    }
    if shape_score > 0:
        out["pocket_shape_kartograf_score"] = max(0.0, min(1.0, float(shape_score)))
    return out


def _voxel_center(
    voxel: tuple[int, int, int],
    spacing: float,
) -> tuple[float, float, float]:
    return (
        (float(voxel[0]) + 0.5) * spacing,
        (float(voxel[1]) + 0.5) * spacing,
        (float(voxel[2]) + 0.5) * spacing,
    )


def _sample_voxels(
    voxels: set[tuple[int, int, int]],
    *,
    max_points: int = 4500,
) -> list[tuple[int, int, int]]:
    if len(voxels) <= max_points:
        return sorted(voxels)
    ordered = sorted(voxels)
    stride = max(1, math.ceil(len(ordered) / max_points))
    return ordered[::stride][:max_points]


def _write_pocket_shape_overlap_png(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    out_path: Path,
    *,
    pair_id: str,
    spacing: float = 0.5,
    radius_buffer: float = 0.25,
) -> bool:
    """Write a receptor-frame voxel-overlap plot for a planned RBFE edge."""
    vox_a = _pocket_grid_occupancy(
        mol_a,
        spacing=spacing,
        radius_buffer=radius_buffer,
    )
    vox_b = _pocket_grid_occupancy(
        mol_b,
        spacing=spacing,
        radius_buffer=radius_buffer,
    )
    if not vox_a or not vox_b:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        logger.debug(
            f"Could not import matplotlib for pocket-shape plot {pair_id}: {exc}"
        )
        return False

    ref_only = vox_a - vox_b
    alt_only = vox_b - vox_a
    overlap = vox_a & vox_b
    all_voxels = vox_a | vox_b
    if not all_voxels:
        return False

    metrics = _pocket_grid_overlap_metrics(
        mol_a,
        mol_b,
        spacing=spacing,
        radius_buffer=radius_buffer,
    ) or {}

    categories = [
        ("reference only", ref_only, "#2563eb", 0.30, 2.0),
        ("target only", alt_only, "#f97316", 0.30, 2.0),
        ("overlap", overlap, "#16a34a", 0.78, 3.0),
    ]
    sampled: dict[str, list[tuple[float, float, float]]] = {}
    for label, voxels, _color, _alpha, _size in categories:
        sampled[label] = [
            _voxel_center(voxel, spacing)
            for voxel in _sample_voxels(voxels)
        ]

    all_points = [
        _voxel_center(voxel, spacing)
        for voxel in _sample_voxels(all_voxels, max_points=12000)
    ]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    zs = [point[2] for point in all_points]

    def _limits(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, 1.0)
        low = min(values)
        high = max(values)
        pad = max(1.0, 0.08 * max(high - low, 1.0))
        return low - pad, high + pad

    xlim = _limits(xs)
    ylim = _limits(ys)
    zlim = _limits(zs)

    try:
        fig = plt.figure(figsize=(9.8, 7.6), dpi=150)
        axes = [
            fig.add_subplot(2, 2, 1, projection="3d"),
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4),
        ]

        ax3d = axes[0]
        for label, _voxels, color, alpha, size in categories:
            points = sampled[label]
            if not points:
                continue
            ax3d.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                [point[2] for point in points],
                s=size,
                c=color,
                alpha=alpha,
                linewidths=0,
                depthshade=False,
            )
        ax3d.set_title("3D voxel occupancy", fontsize=10)
        ax3d.set_xlabel("x (A)", fontsize=8)
        ax3d.set_ylabel("y (A)", fontsize=8)
        ax3d.set_zlabel("z (A)", fontsize=8)
        ax3d.set_xlim(*xlim)
        ax3d.set_ylim(*ylim)
        ax3d.set_zlim(*zlim)
        ax3d.view_init(elev=25, azim=-55)
        try:
            ax3d.set_box_aspect(
                (
                    max(xlim[1] - xlim[0], 1.0),
                    max(ylim[1] - ylim[0], 1.0),
                    max(zlim[1] - zlim[0], 1.0),
                )
            )
        except Exception:
            pass

        projection_specs = [
            (axes[1], 0, 1, "XY projection", "x (A)", "y (A)", xlim, ylim),
            (axes[2], 0, 2, "XZ projection", "x (A)", "z (A)", xlim, zlim),
            (axes[3], 1, 2, "YZ projection", "y (A)", "z (A)", ylim, zlim),
        ]
        for ax, dim_x, dim_y, title, xlabel, ylabel, limit_x, limit_y in projection_specs:
            for label, _voxels, color, alpha, size in categories:
                points = sampled[label]
                if not points:
                    continue
                ax.scatter(
                    [point[dim_x] for point in points],
                    [point[dim_y] for point in points],
                    s=size,
                    c=color,
                    alpha=alpha,
                    linewidths=0,
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlim(*limit_x)
            ax.set_ylim(*limit_y)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#e5e7eb", linewidth=0.5)

        score = metrics.get("pocket_grid_score")
        containment = metrics.get("pocket_grid_containment")
        jaccard = metrics.get("pocket_grid_jaccard")
        metric_text = ""
        if score is not None and containment is not None and jaccard is not None:
            metric_text = (
                f"grid={score:.3f}  containment={containment:.3f}  "
                f"jaccard={jaccard:.3f}"
            )
        fig.suptitle(f"{pair_id} pocket occupancy overlap\n{metric_text}", fontsize=12)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=7,
                label=label,
            )
            for label, _voxels, color, _alpha, _size in categories
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.012),
        )
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.925))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    except Exception as exc:
        logger.debug(f"Could not write pocket-shape plot for {pair_id}: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass
        return False

    return out_path.is_file()


def _pocket_shape_network_score(mapping) -> float:
    """Score receptor-frame pocket occupancy, with Kartograf shape as fallback.

    This is intended for ``rbfe_septop`` where the whole ligand is in softcore
    and atom mapping is not a common-core requirement. The dominant term rewards
    overlapping ligand occupancy in the input receptor frame. A containment term
    lets a smaller ligand that fills one subpocket connect to a larger ligand
    spanning several subpockets, while the Jaccard term still ranks full
    x/y-to-x/y overlap above x-to-x containment.
    """
    shape_score = _shape_difference_network_score(mapping)
    mol_a = _rdkit_mol_from_component(_mapping_component(mapping, "componentA"))
    mol_b = _rdkit_mol_from_component(_mapping_component(mapping, "componentB"))
    if mol_a is None or mol_b is None:
        return shape_score

    grid_metrics = _pocket_grid_overlap_metrics(mol_a, mol_b)
    if grid_metrics is None:
        return shape_score

    grid_score = grid_metrics["pocket_grid_score"]
    if shape_score <= 0:
        return grid_score
    return max(0.0, min(1.0, 0.85 * grid_score + 0.15 * shape_score))


def _network_scorer_callable(
    network_scorer: str | None = None,
    *,
    protocol: str | None = None,
):
    scorer_name = resolve_network_scorer_name(network_scorer, protocol=protocol)
    if scorer_name == "pocket_shape":
        return _pocket_shape_network_score
    if scorer_name == "shape_difference":
        return _shape_difference_network_score

    from lomap.gufe_bindings.scorers import default_lomap_score

    return default_lomap_score


def _mapper_options_dict(options: Any | None) -> dict[str, Any]:
    if options is None:
        return {}
    if hasattr(options, "model_dump"):
        return dict(options.model_dump(exclude_none=True, exclude_unset=True))
    if isinstance(options, Mapping):
        return {str(key): value for key, value in options.items() if value is not None}
    return dict(options)


def _lomap_mapper_kwargs(options: Any | None = None) -> dict[str, Any]:
    kwargs = {
        "time": 20,
        "threed": True,
        "max3d": 1.5,
        "element_change": False,
        "shift": True,
    }
    kwargs.update(_mapper_options_dict(options))
    return kwargs


def _kartograf_mapper_kwargs(
    options: Any | None = None,
    *,
    atom_map_hydrogens_default: bool,
) -> dict[str, Any]:
    mapper_options = _mapper_options_dict(options)
    use_element_filter = mapper_options.pop("filter_element_changes", True)
    use_attached_h_filter = mapper_options.pop("filter_mismatched_attached_h_count", False)
    mapper_options.pop("atom_map_hydrogens", None)
    mapper_options.pop("map_hydrogens_on_hydrogens_only", None)

    kwargs = {
        "atom_max_distance": 0.95,
        "map_hydrogens_on_hydrogens_only": True,
        "atom_map_hydrogens": atom_map_hydrogens_default,
        "map_exact_ring_matches_only": True,
        "allow_partial_fused_rings": True,
        "allow_bond_breaks": False,
    }
    kwargs.update(mapper_options)

    additional_mapping_filter_functions = []
    if use_element_filter:
        additional_mapping_filter_functions.append(filter_element_changes)
    if use_attached_h_filter:
        additional_mapping_filter_functions.append(filter_mismatched_attached_h_count)
    kwargs["additional_mapping_filter_functions"] = additional_mapping_filter_functions
    return kwargs


def _build_konnektor_atom_mapper(
    atom_mapper: str,
    *,
    hmr: bool = True,
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapping_overrides: Any | None = None,
):
    mapper_name = _normalize_atom_mapper(atom_mapper)
    if mapper_name == "lomap":
        from lomap import LomapAtomMapper

        mapper = LomapAtomMapper(**_lomap_mapper_kwargs(lomap_options))
    else:
        mapper = _build_current_kartograf_atom_mapper_for_network(
            kartograf_options=kartograf_options
        )

    overrides = _coerce_atom_mapping_overrides(atom_mapping_overrides)
    if overrides:
        return _wrap_atom_mapper_with_overrides(mapper, overrides)
    return mapper


def _build_current_kartograf_atom_mapper_for_network(
    kartograf_options: Any | None = None,
):
    """Return the Kartograf mapper currently used for RBFE network generation."""
    from kartograf.atom_mapper import KartografAtomMapper

    return KartografAtomMapper(
        **_kartograf_mapper_kwargs(
            kartograf_options,
            atom_map_hydrogens_default=False,
        )
    )


def _build_current_kartograf_atom_mapper_for_simprep_x(
    kartograf_options: Any | None = None,
):
    """Return the Kartograf mapper used by RBFE x-component simprep."""
    from kartograf.atom_mapper import KartografAtomMapper

    return KartografAtomMapper(
        **_kartograf_mapper_kwargs(
            kartograf_options,
            atom_map_hydrogens_default=True,
        )
    )


def _wrap_atom_mapper_with_overrides(
    delegate: Any,
    overrides: ManualAtomMappingOverrides,
):
    """Return an AtomMapper that yields manual mappings before falling back."""
    try:
        from gufe import AtomMapper as GufeAtomMapper
    except Exception:
        GufeAtomMapper = None

    base_mapper_cls = (
        GufeAtomMapper
        if GufeAtomMapper is not None and isinstance(delegate, GufeAtomMapper)
        else object
    )

    class ManualOverrideAtomMapper(base_mapper_cls):
        def __init__(self, wrapped_mapper: Any, manual_overrides: ManualAtomMappingOverrides):
            self.wrapped_mapper = wrapped_mapper
            self.manual_overrides = manual_overrides

        def suggest_mappings(self, componentA: Any, componentB: Any):
            ref = _component_name(componentA)
            alt = _component_name(componentB)
            manual_mapping = self.manual_overrides.get_b_to_a(ref, alt)
            if manual_mapping is not None:
                yield _make_ligand_atom_mapping(componentA, componentB, manual_mapping)
                return
            yield from self.wrapped_mapper.suggest_mappings(componentA, componentB)

        @classmethod
        def _defaults(cls):
            try:
                return super()._defaults()
            except Exception:
                return {}

        @classmethod
        def _from_dict(cls, d):
            return cls(
                d.get("wrapped_mapper"),
                _manual_atom_mapping_overrides_from_dict(d.get("manual_overrides")),
            )

        def _to_dict(self):
            return {
                "wrapped_mapper": self.wrapped_mapper,
                "manual_overrides": _manual_atom_mapping_overrides_to_dict(
                    self.manual_overrides
                ),
            }

    return ManualOverrideAtomMapper(delegate, overrides)


def filter_element_changes(
    molA: Chem.Mol, molB: Chem.Mol, mapping: dict[int, int]
) -> dict[int, int]:
    """Forces a mapping to exclude any alchemical element changes in the core"""
    filtered_mapping = {}

    for i, j in mapping.items():
        if (
            molA.GetAtomWithIdx(i).GetAtomicNum()
            != molB.GetAtomWithIdx(j).GetAtomicNum()
        ):
            continue
        filtered_mapping[i] = j

    return filtered_mapping


def filter_mismatched_attached_h_count(
    molA: Chem.Mol, molB: Chem.Mol, mapping: dict[int, int]
) -> dict[int, int]:
    """
    Exclude mapped heavy-atom pairs where the number of directly attached H differs.
    This helps avoid HMR mass mismatches for 'common/core' atoms.
    """
    filtered = {}
    for i, j in mapping.items():
        a = molA.GetAtomWithIdx(i)
        b = molB.GetAtomWithIdx(j)

        hA = a.GetTotalNumHs(includeNeighbors=True)
        hB = b.GetTotalNumHs(includeNeighbors=True)

        if hA != hB:
            continue

        filtered[i] = j
    return filtered

RBFEPair = Tuple[str, str]
RBFEMapFn = Callable[[Sequence[str]], Iterable[RBFEPair]]
AtomIndexMapping = dict[int, int]


def _invert_atom_mapping(mapping: Mapping[int, int]) -> AtomIndexMapping:
    return {int(value): int(key) for key, value in mapping.items()}


def _component_name(component: Any) -> str:
    name = getattr(component, "name", None)
    if name is None and hasattr(component, "_name"):
        name = getattr(component, "_name", None)
    return sanitize_ligand_name(str(name if name is not None else component))


def _component_num_atoms(component: Any) -> int | None:
    mol = None
    if hasattr(component, "to_rdkit"):
        try:
            mol = component.to_rdkit()
        except Exception:
            mol = None
    if mol is None:
        mol = getattr(component, "_rdkit", None) or getattr(component, "mol", None)
    if mol is not None and hasattr(mol, "GetNumAtoms"):
        try:
            return int(mol.GetNumAtoms())
        except Exception:
            return None
    return None


def _mapping_mapped_atom_count(mapping: Any) -> int | None:
    for attr_name in ("componentB_to_componentA", "componentA_to_componentB"):
        try:
            mapped = getattr(mapping, attr_name)
        except Exception:
            continue
        if mapped is not None:
            try:
                return len(mapped)
            except Exception:
                pass
    return None


def _normalize_minimal_mapping_atom(value: int | None) -> int:
    if value is None:
        return 3
    try:
        minimum = int(value)
    except Exception as exc:
        raise ValueError("rbfe.minimal_mapping_atom must be an integer >= 1.") from exc
    if minimum < 1:
        raise ValueError("rbfe.minimal_mapping_atom must be an integer >= 1.")
    return minimum


def _validate_minimal_mapping_atom(
    pair_id: str,
    n_mapped: int | None,
    minimal_mapping_atom: int | None,
) -> None:
    if n_mapped is None:
        return
    minimum = _normalize_minimal_mapping_atom(minimal_mapping_atom)
    if n_mapped >= minimum:
        return
    atom_label = "atom" if n_mapped == 1 else "atoms"
    raise ValueError(
        f"RBFE atom mapping for planned pair {pair_id} maps only "
        f"{n_mapped} {atom_label}, below rbfe.minimal_mapping_atom={minimum}. "
        "You can lower rbfe.minimal_mapping_atom in the config if this "
        "transformation is intentional, but a mapping this small is often "
        "wrong; check the ligand pairing, ligand chemistry, and atom mapper."
    )


def _mol_num_atoms(mol: Any) -> int | None:
    if mol is None or not hasattr(mol, "GetNumAtoms"):
        return None
    try:
        return int(mol.GetNumAtoms())
    except Exception:
        return None


def _mol_heavy_atom_indices(mol: Any) -> set[int] | None:
    n_atoms = _mol_num_atoms(mol)
    if n_atoms is None or not hasattr(mol, "GetAtomWithIdx"):
        return None
    heavy: set[int] = set()
    try:
        for idx in range(n_atoms):
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1:
                heavy.add(idx)
    except Exception:
        return None
    return heavy


def _mapping_coverage_status(
    mol_ref: Any,
    mol_alt: Any,
    map_b_to_a: Mapping[Any, Any],
) -> dict[str, Any]:
    """Return atom-count metadata for a componentB-to-componentA mapping."""
    try:
        alt_to_ref = {int(key): int(value) for key, value in map_b_to_a.items()}
    except Exception:
        return {}

    n_ref_atoms = _mol_num_atoms(mol_ref)
    n_alt_atoms = _mol_num_atoms(mol_alt)
    if n_ref_atoms is None or n_alt_atoms is None:
        return {}

    alt_indices = set(alt_to_ref)
    ref_indices = set(alt_to_ref.values())
    status: dict[str, Any] = {
        "n_ref_atoms": n_ref_atoms,
        "n_alt_atoms": n_alt_atoms,
        "full_atom_mapping": (
            len(alt_to_ref) == n_alt_atoms == n_ref_atoms
            and alt_indices == set(range(n_alt_atoms))
            and ref_indices == set(range(n_ref_atoms))
        ),
    }

    ref_heavy = _mol_heavy_atom_indices(mol_ref)
    alt_heavy = _mol_heavy_atom_indices(mol_alt)
    if ref_heavy is not None and alt_heavy is not None:
        mapped_alt_heavy = {idx for idx in alt_indices if idx in alt_heavy}
        mapped_ref_heavy = {idx for idx in ref_indices if idx in ref_heavy}
        status.update(
            {
                "n_ref_heavy_atoms": len(ref_heavy),
                "n_alt_heavy_atoms": len(alt_heavy),
                "full_heavy_atom_mapping": (
                    len(mapped_alt_heavy) == len(alt_heavy) == len(ref_heavy)
                    and mapped_alt_heavy == alt_heavy
                    and mapped_ref_heavy == ref_heavy
                ),
            }
        )

    return status


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mapping_metric_scores(mapping: Any) -> dict[str, float]:
    """Compute optional Kartograf mapping metrics for network visualization.

    These scores are stored in each pair's ``mapping_status.json`` and surfaced
    by ``rbfe_network.html`` as selectable edge-color metrics. Missing optional
    Kartograf scorer modules are ignored so mapping artifact generation can
    continue in lean environments.
    """
    if mapping is None:
        return {}

    metric_calls: list[tuple[str, Any, str]] = []
    mapped_count = _mapping_mapped_atom_count(mapping)
    try:
        from kartograf.mapping_metrics.metric_mapping_rmsd import MappingRMSDScorer

        scorer = MappingRMSDScorer()
        metric_calls.append(("mapping_rmsd", scorer, "get_rmsd"))
        metric_calls.append(("mapping_score_rmsd", scorer, "get_score"))
    except Exception:
        pass
    try:
        from kartograf.mapping_metrics.metric_volume_ratio import (
            MappingRatioMappedAtomsScorer,
            MappingVolumeRatioScorer,
        )

        metric_calls.append(
            (
                "mapping_score_ratio_mapped_atoms",
                MappingRatioMappedAtomsScorer(),
                "get_score",
            )
        )
        if mapped_count is None or mapped_count >= 4:
            metric_calls.append(
                ("mapping_score_volume_ratio", MappingVolumeRatioScorer(), "get_score")
            )
    except Exception:
        pass
    if mapped_count is None or mapped_count >= 2:
        try:
            from kartograf.mapping_metrics.metric_shape_difference import (
                MappingShapeMismatchScorer,
                MappingShapeOverlapScorer,
            )

            metric_calls.append(
                (
                    "mapping_score_shape_mismatch",
                    MappingShapeMismatchScorer(),
                    "get_score",
                )
            )
            metric_calls.append(
                (
                    "mapping_score_shape_overlap",
                    MappingShapeOverlapScorer(),
                    "get_score",
                )
            )
        except Exception:
            pass

    scores: dict[str, float] = {}
    for key, scorer, method_name in metric_calls:
        try:
            method = getattr(scorer, method_name)
            value = _finite_float(method(mapping))
        except Exception as exc:
            logger.debug(f"Could not compute RBFE mapping metric {key}: {exc}")
            continue
        if value is not None:
            scores[key] = value
    return scores


def ligand_identity_key(path: Path | str) -> str:
    """Return a canonical molecule identity key for RBFE duplicate filtering."""
    mol = _load_rdkit_mol(Path(path))
    try:
        identity_mol = Chem.AddHs(Chem.RemoveHs(Chem.Mol(mol)), addCoords=False)
    except Exception:
        identity_mol = mol
    return Chem.MolToSmiles(identity_mol, isomericSmiles=True)


def deduplicate_identical_ligands(
    ligands: Sequence[str],
    ligand_files: Mapping[str, Path | str],
) -> tuple[list[str], dict[str, str], list[dict[str, str]]]:
    """
    Keep the first ligand for each exact molecular identity.

    Returns kept ligand names, a skipped->kept replacement map, and metadata for
    skipped duplicate ligands.
    """
    kept: list[str] = []
    replacements: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    first_by_identity: dict[str, str] = {}

    for ligand in ligands:
        path = ligand_files.get(ligand)
        if path is None:
            kept.append(ligand)
            continue
        try:
            identity = ligand_identity_key(path)
        except Exception as exc:
            logger.warning(
                f"Could not determine RBFE ligand identity for {ligand} ({path}): {exc}; keeping it."
            )
            kept.append(ligand)
            continue

        representative = first_by_identity.get(identity)
        if representative is None:
            first_by_identity[identity] = ligand
            kept.append(ligand)
            continue

        replacements[ligand] = representative
        skipped.append(
            {
                "ligand": ligand,
                "kept": representative,
                "identity": identity,
            }
        )

    return kept, replacements, skipped


class _SimpleLigandAtomMapping:
    """Fallback mapping object for environments without gufe.LigandAtomMapping."""

    def __init__(
        self,
        component_a: Any,
        component_b: Any,
        *,
        component_b_to_component_a: Mapping[int, int],
        annotations: Mapping[str, Any] | None = None,
    ) -> None:
        self._componentA = component_a
        self._componentB = component_b
        self._componentB_to_componentA = {
            int(key): int(value) for key, value in component_b_to_component_a.items()
        }
        self._componentA_to_componentB = _invert_atom_mapping(
            self._componentB_to_componentA
        )
        self.annotations = dict(annotations or {})

    @property
    def componentA(self) -> Any:
        return self._componentA

    @property
    def componentB(self) -> Any:
        return self._componentB

    @property
    def componentA_to_componentB(self) -> AtomIndexMapping:
        return dict(self._componentA_to_componentB)

    @property
    def componentB_to_componentA(self) -> AtomIndexMapping:
        return dict(self._componentB_to_componentA)

    @property
    def componentA_unique(self) -> tuple[int, ...]:
        n_atoms = _component_num_atoms(self._componentA)
        if n_atoms is None:
            return ()
        mapped = set(self._componentA_to_componentB)
        return tuple(index for index in range(n_atoms) if index not in mapped)

    @property
    def componentB_unique(self) -> tuple[int, ...]:
        n_atoms = _component_num_atoms(self._componentB)
        if n_atoms is None:
            return ()
        mapped = set(self._componentB_to_componentA)
        return tuple(index for index in range(n_atoms) if index not in mapped)

    def with_annotations(self, annotations: Mapping[str, Any]):
        merged = dict(self.annotations)
        merged.update(dict(annotations))
        return _SimpleLigandAtomMapping(
            self._componentA,
            self._componentB,
            component_b_to_component_a=self._componentB_to_componentA,
            annotations=merged,
        )


def _make_ligand_atom_mapping(
    component_a: Any,
    component_b: Any,
    map_b_to_a: Mapping[int, int],
) -> Any:
    mapping_b_to_a = {int(key): int(value) for key, value in map_b_to_a.items()}
    mapping_a_to_b = _invert_atom_mapping(mapping_b_to_a)

    if not (hasattr(component_a, "to_rdkit") and hasattr(component_b, "to_rdkit")):
        return _SimpleLigandAtomMapping(
            component_a,
            component_b,
            component_b_to_component_a=mapping_b_to_a,
        )

    try:
        from gufe import LigandAtomMapping
    except Exception:
        return _SimpleLigandAtomMapping(
            component_a,
            component_b,
            component_b_to_component_a=mapping_b_to_a,
        )

    constructors = (
        lambda: LigandAtomMapping(
            component_a,
            component_b,
            componentB_to_componentA=mapping_b_to_a,
        ),
        lambda: LigandAtomMapping(
            component_a,
            component_b,
            componentA_to_componentB=mapping_a_to_b,
        ),
        lambda: LigandAtomMapping(component_a, component_b, mapping_a_to_b),
    )
    for constructor in constructors:
        try:
            return constructor()
        except TypeError:
            continue

    return _SimpleLigandAtomMapping(
        component_a,
        component_b,
        component_b_to_component_a=mapping_b_to_a,
    )


def _normalize_atom_index_mapping(raw: Any, *, context: str) -> AtomIndexMapping:
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, Mapping):
                if {"from", "to"}.issubset(entry):
                    items.append((entry["from"], entry["to"]))
                elif {"source", "target"}.issubset(entry):
                    items.append((entry["source"], entry["target"]))
                elif {"b", "a"}.issubset(entry):
                    items.append((entry["b"], entry["a"]))
                else:
                    raise ValueError(
                        "Atom mapping entry for "
                        f"{context} must include from/to, source/target, or b/a keys."
                    )
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                items.append((entry[0], entry[1]))
            else:
                raise ValueError(
                    f"Atom mapping entry for {context} must be a 2-tuple; got {entry!r}."
                )
    else:
        raise ValueError(f"Atom mapping for {context} must be a dict or list of pairs.")

    mapping: AtomIndexMapping = {}
    used_values: set[int] = set()
    for key, value in items:
        try:
            src = int(key)
            dst = int(value)
        except Exception as exc:
            raise ValueError(
                f"Atom mapping indices for {context} must be integers: {key!r}->{value!r}."
            ) from exc
        if src < 0 or dst < 0:
            raise ValueError(f"Atom mapping indices for {context} must be >= 0.")
        if src in mapping:
            raise ValueError(f"Duplicate source atom index {src} in {context}.")
        if dst in used_values:
            raise ValueError(f"Duplicate target atom index {dst} in {context}.")
        mapping[src] = dst
        used_values.add(dst)

    if not mapping:
        raise ValueError(f"Atom mapping for {context} is empty.")
    return mapping


_ATOM_MAPPING_B_TO_A_KEYS = (
    "componentB_to_componentA",
    "component_b_to_component_a",
    "target_to_reference",
    "alt_to_ref",
    "b_to_a",
    "map_b_to_a",
    "mapping",
    "map",
)
_ATOM_MAPPING_A_TO_B_KEYS = (
    "componentA_to_componentB",
    "component_a_to_component_b",
    "reference_to_target",
    "ref_to_alt",
    "a_to_b",
    "map_a_to_b",
)
_ATOM_MAPPING_PAIR_KEYS = ("pair", "edge")
_ATOM_MAPPING_REF_KEYS = ("ref", "reference", "ligand_a", "ligandA", "componentA", "A")
_ATOM_MAPPING_ALT_KEYS = ("alt", "target", "ligand_b", "ligandB", "componentB", "B")


def _first_present(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _looks_like_atom_mapping_entry(data: Mapping[str, Any]) -> bool:
    has_pair = any(key in data for key in _ATOM_MAPPING_PAIR_KEYS) or (
        _first_present(data, _ATOM_MAPPING_REF_KEYS) is not None
        and _first_present(data, _ATOM_MAPPING_ALT_KEYS) is not None
    )
    has_mapping = any(key in data for key in _ATOM_MAPPING_B_TO_A_KEYS) or any(
        key in data for key in _ATOM_MAPPING_A_TO_B_KEYS
    )
    return bool(has_pair and has_mapping)


def _atom_mapping_payload_to_b_to_a(
    payload: Any,
    *,
    context: str,
) -> AtomIndexMapping:
    if isinstance(payload, Mapping):
        for key in _ATOM_MAPPING_B_TO_A_KEYS:
            if key in payload:
                return _normalize_atom_index_mapping(payload[key], context=context)
        for key in _ATOM_MAPPING_A_TO_B_KEYS:
            if key in payload:
                return _invert_atom_mapping(
                    _normalize_atom_index_mapping(payload[key], context=context)
                )
    return _normalize_atom_index_mapping(payload, context=context)


def _atom_mapping_entry_from_data(entry: Any) -> tuple[RBFEPair, AtomIndexMapping]:
    if isinstance(entry, Mapping):
        pair_value = _first_present(entry, _ATOM_MAPPING_PAIR_KEYS)
        if pair_value is not None:
            ref, alt = _normalize_pair(pair_value)
        else:
            ref_value = _first_present(entry, _ATOM_MAPPING_REF_KEYS)
            alt_value = _first_present(entry, _ATOM_MAPPING_ALT_KEYS)
            if ref_value is None or alt_value is None:
                raise ValueError(
                    "Atom mapping entry must include pair/edge or "
                    f"reference/target ligands: {entry!r}."
                )
            ref, alt = _normalize_pair((ref_value, alt_value))
        return (ref, alt), _atom_mapping_payload_to_b_to_a(
            entry,
            context=f"{ref}~{alt}",
        )

    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        ref, alt = _normalize_pair((entry[0], entry[1]))
        return (ref, alt), _normalize_atom_index_mapping(
            entry[2],
            context=f"{ref}~{alt}",
        )

    raise ValueError(f"Unsupported atom mapping entry: {entry!r}.")


def _atom_mapping_entries_from_data(data: Any) -> list[tuple[RBFEPair, AtomIndexMapping]]:
    if isinstance(data, Mapping):
        if _looks_like_atom_mapping_entry(data):
            return [_atom_mapping_entry_from_data(data)]

        for key in ("pairs", "edges", "mappings", "atom_mappings"):
            if key in data:
                return _atom_mapping_entries_from_data(data[key])

        entries: list[tuple[RBFEPair, AtomIndexMapping]] = []
        for pair_value, payload in data.items():
            ref, alt = _normalize_pair(pair_value)
            entries.append(
                (
                    (ref, alt),
                    _atom_mapping_payload_to_b_to_a(
                        payload,
                        context=f"{ref}~{alt}",
                    ),
                )
            )
        return entries

    if isinstance(data, list):
        return [_atom_mapping_entry_from_data(entry) for entry in data]

    raise ValueError(
        f"Unsupported RBFE atom mapping data type: {type(data).__name__}"
    )


@dataclass(frozen=True)
class ManualAtomMappingOverrides:
    """
    User-provided atom mappings keyed by directed ligand pair.

    Mappings are stored in BATTER's prepared artifact orientation:
    ``componentB_to_componentA`` (target/alternate atom index -> reference atom index).
    If a requested pair is present only in the reverse direction, the mapping is
    inverted automatically.
    """

    mappings: Mapping[RBFEPair, AtomIndexMapping]
    source: Path | None = None

    def __post_init__(self) -> None:
        normalized: dict[RBFEPair, AtomIndexMapping] = {}
        for pair, mapping in self.mappings.items():
            ref, alt = _normalize_pair(pair)
            normalized[(ref, alt)] = _normalize_atom_index_mapping(
                mapping,
                context=f"{ref}~{alt}",
            )
        object.__setattr__(self, "mappings", normalized)

    def __bool__(self) -> bool:
        return bool(self.mappings)

    def __len__(self) -> int:
        return len(self.mappings)

    def get_b_to_a(self, ref: str, alt: str) -> AtomIndexMapping | None:
        ref_name = sanitize_ligand_name(str(ref))
        alt_name = sanitize_ligand_name(str(alt))
        exact = self.mappings.get((ref_name, alt_name))
        if exact is not None:
            return dict(exact)
        reverse = self.mappings.get((alt_name, ref_name))
        if reverse is not None:
            return _invert_atom_mapping(reverse)
        return None

    def source_label(self, ref: str, alt: str) -> str:
        ref_name = sanitize_ligand_name(str(ref))
        alt_name = sanitize_ligand_name(str(alt))
        if (ref_name, alt_name) in self.mappings:
            direction = f"{ref_name}~{alt_name}"
        elif (alt_name, ref_name) in self.mappings:
            direction = f"{alt_name}~{ref_name} (inverted)"
        else:
            direction = "unknown"
        if self.source is None:
            return direction
        return f"{self.source}:{direction}"


def _manual_atom_mapping_overrides_to_dict(
    overrides: ManualAtomMappingOverrides,
) -> dict[str, Any]:
    """Return a stable, JSON-compatible representation for GUFE tokenization."""
    return {
        "source": str(overrides.source) if overrides.source is not None else None,
        "mappings": [
            {
                "ref": ref,
                "alt": alt,
                "componentB_to_componentA": {
                    str(key): int(value) for key, value in sorted(mapping.items())
                },
            }
            for (ref, alt), mapping in sorted(overrides.mappings.items())
        ],
    }


def _manual_atom_mapping_overrides_from_dict(
    data: Any | None,
) -> ManualAtomMappingOverrides:
    if isinstance(data, ManualAtomMappingOverrides):
        return data
    if data is None:
        return ManualAtomMappingOverrides({})
    source = None
    if isinstance(data, Mapping):
        raw_source = data.get("source")
        if raw_source not in (None, ""):
            source = Path(str(raw_source))
    entries = _atom_mapping_entries_from_data(data)
    return ManualAtomMappingOverrides(dict(entries), source=source)


def _coerce_atom_mapping_overrides(
    overrides: Any | None,
) -> ManualAtomMappingOverrides | None:
    if overrides is None:
        return None
    if isinstance(overrides, ManualAtomMappingOverrides):
        return overrides
    if isinstance(overrides, (str, Path)):
        return load_atom_mapping_file(Path(overrides))
    entries = _atom_mapping_entries_from_data(overrides)
    return ManualAtomMappingOverrides(dict(entries))


def load_atom_mapping_file(path: Path) -> ManualAtomMappingOverrides:
    """
    Load user-provided RBFE atom mappings from JSON/YAML.

    The simplest format is a mapping of pair labels to atom maps:
    ``{"LIGA~LIGB": {"0": 0, "1": 1}}``.  Pair maps are interpreted as
    ``componentB_to_componentA`` (target/alternate atom index -> reference atom
    index), matching BATTER's prepared ``mapping.json`` artifact.  Structured
    entries may also use ``componentA_to_componentB``/``reference_to_target``;
    those are inverted during loading.
    """
    if not path.exists():
        raise FileNotFoundError(f"RBFE atom mapping file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text())
    else:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "RBFE atom mapping files must be JSON or YAML; "
                f"could not parse {path} as JSON."
            ) from exc

    entries = _atom_mapping_entries_from_data(data)
    if not entries:
        raise ValueError(f"RBFE atom mapping file produced no mappings: {path}")

    mapping_by_pair: dict[RBFEPair, AtomIndexMapping] = {}
    for pair, mapping in entries:
        if pair in mapping_by_pair:
            raise ValueError(f"Duplicate RBFE atom mapping for pair {pair[0]}~{pair[1]}.")
        mapping_by_pair[pair] = mapping
    return ManualAtomMappingOverrides(mapping_by_pair, source=path)


def _dedupe_pairs(pairs: Iterable[RBFEPair]) -> List[RBFEPair]:
    seen: set[RBFEPair] = set()
    out: List[RBFEPair] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _normalize_pair(pair: Any) -> RBFEPair:
    if isinstance(pair, str):
        if "~" in pair:
            left, right = (p.strip() for p in pair.split("~", 1))
        elif "," in pair:
            left, right = (p.strip() for p in pair.split(",", 1))
        else:
            parts = [p for p in pair.split() if p]
            if len(parts) != 2:
                raise ValueError(f"RBFE mapping line must contain 2 tokens: {pair!r}")
            left, right = parts
    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
        left, right = pair
    else:
        raise ValueError(f"RBFE mapping entries must be 2-tuples; got {pair!r}.")

    return (sanitize_ligand_name(str(left)), sanitize_ligand_name(str(right)))


def validate_rbfe_network_ligand_coverage(
    ligands: Sequence[str],
    pairs: Sequence[Sequence[str] | tuple[str, str]],
    *,
    context: str = "RBFE network",
) -> None:
    lig_list = [sanitize_ligand_name(str(lig)) for lig in ligands if str(lig)]
    if not lig_list:
        return

    connected: set[str] = set()
    for pair in pairs:
        ref, alt = _normalize_pair(pair)
        connected.add(ref)
        connected.add(alt)

    missing = [lig for lig in lig_list if lig not in connected]
    if missing:
        raise ValueError(
            f"{context} does not include any mapping edge for ligand(s): "
            + ", ".join(missing)
        )


def _pairs_from_data(data: Any) -> List[RBFEPair]:
    if isinstance(data, dict):
        if "pairs" in data:
            raw = data["pairs"]
        elif "edges" in data:
            raw = data["edges"]
        else:
            # adjacency mapping: {LIG1: [LIG2, LIG3], ...}
            raw = []
            for src, targets in data.items():
                if not isinstance(targets, (list, tuple)):
                    raise ValueError(
                        "RBFE mapping dict must map ligands to list of targets."
                    )
                for tgt in targets:
                    raw.append([src, tgt])
        return [_normalize_pair(p) for p in raw]

    if isinstance(data, list):
        return [_normalize_pair(p) for p in data]

    raise ValueError(f"Unsupported RBFE mapping data type: {type(data).__name__}")


def load_mapping_file(path: Path) -> List[RBFEPair]:
    """
    Load RBFE mapping pairs from a file.

    Supported formats:
      - JSON/YAML: list of pairs, or dict with 'pairs'/'edges', or adjacency mapping.
      - Text: one pair per line, separated by '~', ',' or whitespace.
    """
    if not path.exists():
        raise FileNotFoundError(f"RBFE mapping file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        if suffix == ".json":
            data = json.loads(path.read_text())
        else:
            import yaml

            data = yaml.safe_load(path.read_text())
        pairs = _pairs_from_data(data)
    else:
        pairs = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            pairs.append(_normalize_pair(line))

    if not pairs:
        raise ValueError(f"RBFE mapping file produced no pairs: {path}")
    return pairs


def resolve_mapping_fn(name: str | None) -> RBFEMapFn:
    """
    Resolve a mapping function by name.
    """
    if not name:
        return RBFENetwork.default_mapping
    key = str(name).strip().lower()
    if key in {"default", "star", "first"}:
        return RBFENetwork.default_mapping
    if key in {"konnektor"}:
        raise ValueError(
            "RBFE mapping 'konnektor' requires ligand inputs; it must be resolved "
            "in the orchestrator when building the network."
        )
    raise ValueError(f"Unknown RBFE mapping '{name}'. Available: default, konnektor")


def _load_rdkit_mol(path: Path):
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".sd"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = supplier[0] if supplier and len(supplier) > 0 else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    elif suffix == ".pdb":
        from MDAnalysis import Universe

        u = Universe(str(path))
        mol = u.atoms.convert_to("RDKIT")
    else:
        mol = Chem.MolFromMolFile(str(path), removeHs=False)

    if mol is None:
        raise ValueError(f"Failed to load ligand from {path} with RDKit.")
    return mol


def _small_molecule_component(mol: Chem.Mol, name: str):
    from gufe import SmallMoleculeComponent

    if hasattr(SmallMoleculeComponent, "from_rdkit"):
        try:
            return SmallMoleculeComponent.from_rdkit(mol)
        except TypeError:
            return SmallMoleculeComponent.from_rdkit(mol, name=name)
    return SmallMoleculeComponent(mol, name=name)


def _mapping_png_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _edge_asset_from_mapping_dir(
    pair_id: str,
    pair_dir: Path,
    *,
    prefer_pocket_shape: bool = False,
) -> dict[str, Any]:
    asset: dict[str, Any] = {
        "mapping_path": (pair_dir / "mapping.json").as_posix(),
        "mapping_dir": pair_dir.as_posix(),
    }
    shape_png = pair_dir / "pocket_shape_overlap.png"
    if shape_png.is_file():
        asset["shape_overlap_path"] = shape_png.as_posix()
    mapping_png = pair_dir / "mapping.png"
    if mapping_png.is_file():
        mapping_uri = _mapping_png_data_uri(mapping_png)
        asset["atom_mapping_image_data_uri"] = mapping_uri
        asset["atom_mapping_image_alt"] = f"Atom mapping for {pair_id}"
    if prefer_pocket_shape and shape_png.is_file():
        asset["image_data_uri"] = _mapping_png_data_uri(shape_png)
        asset["image_alt"] = f"Pocket shape overlap for {pair_id}"
        asset["image_kind"] = "pocket_shape_overlap"
    elif mapping_png.is_file():
        asset["image_data_uri"] = asset.get("atom_mapping_image_data_uri")
        asset["image_alt"] = f"Atom mapping for {pair_id}"
        asset["image_kind"] = "atom_mapping"
    status = pair_dir / "mapping_status.json"
    if status.is_file():
        try:
            status_payload = json.loads(status.read_text())
            for key in (
                "n_mapped",
                "mapper",
                "n_ref_atoms",
                "n_alt_atoms",
                "n_ref_heavy_atoms",
                "n_alt_heavy_atoms",
                "full_atom_mapping",
                "full_heavy_atom_mapping",
                "mapping_rmsd",
                "mapping_score_rmsd",
                "mapping_score_ratio_mapped_atoms",
                "mapping_score_volume_ratio",
                "mapping_score_shape_mismatch",
                "mapping_score_shape_overlap",
                "pocket_shape_score",
                "pocket_grid_score",
                "pocket_grid_containment",
                "pocket_grid_jaccard",
                "pocket_grid_overlap_voxels",
                "pocket_grid_ref_voxels",
                "pocket_grid_alt_voxels",
                "pocket_shape_kartograf_score",
            ):
                if key in status_payload:
                    asset[key] = status_payload[key]
        except Exception:
            pass
    return asset


def _cached_pair_mapping_atom_count(
    pair_dir: Path,
    asset: Mapping[str, Any] | None = None,
) -> int | None:
    if asset is not None and asset.get("n_mapped") is not None:
        try:
            return int(asset["n_mapped"])
        except Exception:
            pass

    mapping_json = pair_dir / "mapping.json"
    if not mapping_json.is_file():
        return None
    try:
        data = json.loads(mapping_json.read_text())
        return len(_atom_mapping_payload_to_b_to_a(data, context=mapping_json.name))
    except Exception:
        logger.debug(f"Could not read cached RBFE mapping atom count: {mapping_json}")
        return None


def _serialize_atom_mapping(mapping: Mapping[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in mapping.items()}


def _mapping_status_was_manual(pair_dir: Path) -> bool:
    status = pair_dir / "mapping_status.json"
    if not status.is_file():
        return False
    try:
        payload = json.loads(status.read_text())
    except Exception:
        return False
    return bool(payload.get("mapping_override"))


def _remove_optional_mapping_artifacts(pair_dir: Path) -> None:
    for name in ("mapping.pkl", "mapping.png", "pocket_shape_overlap.png"):
        path = pair_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
        except OSError:
            pass


def _write_manual_pair_mapping_artifacts(
    *,
    ref: str,
    alt: str,
    ligand_files: Mapping[str, Path | str],
    pair_dir: Path,
    map_b_to_a: Mapping[int, int],
    source_label: str,
    include_pocket_shape: bool = False,
    minimal_mapping_atom: int | None = 3,
) -> dict[str, Any]:
    pair_id = f"{ref}~{alt}"
    serialized = _serialize_atom_mapping(map_b_to_a)
    if not serialized:
        raise ValueError(f"Manual RBFE atom mapping for {pair_id} is empty.")
    _validate_minimal_mapping_atom(
        pair_id,
        len(serialized),
        minimal_mapping_atom,
    )

    pair_dir.mkdir(parents=True, exist_ok=True)
    _remove_optional_mapping_artifacts(pair_dir)
    (pair_dir / "mapping.json").write_text(
        json.dumps(serialized, indent=2, sort_keys=True)
    )
    coverage_status: dict[str, Any] = {}
    metric_scores: dict[str, float] = {}
    atom_mapping_obj = None
    try:
        ref_path = Path(ligand_files[ref])
        alt_path = Path(ligand_files[alt])
        rdmol_ref = _load_rdkit_mol(ref_path)
        rdmol_alt = _load_rdkit_mol(alt_path)
        coverage_status = _mapping_coverage_status(
            rdmol_ref,
            rdmol_alt,
            serialized,
        )
        component_ref = _small_molecule_component(rdmol_ref, ref)
        component_alt = _small_molecule_component(rdmol_alt, alt)
        atom_mapping_obj = _make_ligand_atom_mapping(
            component_ref,
            component_alt,
            serialized,
        )
        metric_scores = _mapping_metric_scores(atom_mapping_obj)
        if include_pocket_shape:
            metric_scores.update(
                _pocket_similarity_metric_scores(
                    rdmol_ref,
                    rdmol_alt,
                    atom_mapping_obj,
                )
            )
            _write_pocket_shape_overlap_png(
                rdmol_ref,
                rdmol_alt,
                pair_dir / "pocket_shape_overlap.png",
                pair_id=pair_id,
            )
    except Exception as exc:
        logger.debug(
            f"Could not compute manual RBFE mapping coverage for {pair_id}: {exc}"
        )
    status_payload = {
        "pair_id": pair_id,
        "reference": ref,
        "target": alt,
        "mapper": "manual",
        "mapping_override": True,
        "mapping_source": source_label,
        "mapping_direction": "componentB_to_componentA",
        "n_mapped": len(serialized),
    }
    status_payload.update(coverage_status)
    status_payload.update(metric_scores)
    (pair_dir / "mapping_status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True)
    )

    if atom_mapping_obj is not None:
        try:
            with (pair_dir / "mapping.pkl").open("wb") as fh:
                pickle.dump(atom_mapping_obj, fh)
        except Exception as exc:
            logger.debug(
                f"Could not write manual RBFE atom-mapping pickle for {pair_id}: {exc}"
            )
        try:
            atom_mapping_obj.draw_to_file(fname=pair_dir / "mapping.png")
        except Exception as exc:
            logger.debug(
                f"Could not draw manual RBFE atom-mapping image for {pair_id}: {exc}"
            )

    return _edge_asset_from_mapping_dir(
        pair_id,
        pair_dir,
        prefer_pocket_shape=include_pocket_shape,
    )


def write_pair_mapping_artifacts(
    *,
    ref: str,
    alt: str,
    ligand_files: Mapping[str, Path | str],
    out_dir: Path,
    atom_mapper: str = "kartograf",
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapper_options: Any | None = None,
    atom_mapping_overrides: Any | None = None,
    overwrite: bool = False,
    include_pocket_shape: bool = False,
    minimal_mapping_atom: int | None = 3,
) -> dict[str, Any]:
    """Generate reusable atom-mapping artifacts for one planned RBFE pair."""
    pair_id = f"{ref}~{alt}"
    pair_dir = Path(out_dir) / pair_id
    mapping_json = pair_dir / "mapping.json"

    overrides = _coerce_atom_mapping_overrides(atom_mapping_overrides)
    manual_map = overrides.get_b_to_a(ref, alt) if overrides else None
    if manual_map is not None:
        return _write_manual_pair_mapping_artifacts(
            ref=ref,
            alt=alt,
            ligand_files=ligand_files,
            pair_dir=pair_dir,
            map_b_to_a=manual_map,
            source_label=overrides.source_label(ref, alt) if overrides else "manual",
            include_pocket_shape=include_pocket_shape,
            minimal_mapping_atom=minimal_mapping_atom,
        )

    if mapping_json.is_file() and not overwrite:
        if _mapping_status_was_manual(pair_dir):
            logger.debug(
                f"Prepared RBFE mapping for {pair_id} was manual but no current "
                "override covers it; regenerating."
            )
            _remove_optional_mapping_artifacts(pair_dir)
        else:
            cached_asset = _edge_asset_from_mapping_dir(
                pair_id,
                pair_dir,
                prefer_pocket_shape=include_pocket_shape,
            )
            _validate_minimal_mapping_atom(
                pair_id,
                _cached_pair_mapping_atom_count(pair_dir, cached_asset),
                minimal_mapping_atom,
            )
            if not include_pocket_shape:
                return cached_asset
            if "pocket_shape_score" in cached_asset and (
                cached_asset.get("image_kind") == "pocket_shape_overlap"
            ):
                return cached_asset
            logger.debug(
                f"Prepared RBFE mapping for {pair_id} lacks pocket-shape "
                "visualization metrics; refreshing mapping status."
            )

    mapper_name = _normalize_atom_mapper(atom_mapper)
    if overwrite:
        _remove_optional_mapping_artifacts(pair_dir)
    ref_path = Path(ligand_files[ref])
    alt_path = Path(ligand_files[alt])
    rdmol_ref = _load_rdkit_mol(ref_path)
    rdmol_alt = _load_rdkit_mol(alt_path)
    component_ref = _small_molecule_component(rdmol_ref, ref)
    component_alt = _small_molecule_component(rdmol_alt, alt)

    atom_mapping_obj = None
    if mapper_name == "lomap":
        from lomap import LomapAtomMapper

        mapper = LomapAtomMapper(
            **_lomap_mapper_kwargs(atom_mapper_options or lomap_options)
        )
        atom_mapping_obj = next(
            mapper.suggest_mappings(component_ref, component_alt), None
        )
    else:
        from kartograf.atom_aligner import align_mol_shape

        component_alt = align_mol_shape(component_alt, ref_mol=component_ref)
        mapper = _build_current_kartograf_atom_mapper_for_simprep_x(
            atom_mapper_options or kartograf_options
        )
        atom_mapping_obj = next(
            mapper.suggest_mappings(component_ref, component_alt), None
        )

    map_b_to_a = getattr(atom_mapping_obj, "componentB_to_componentA", {}) or {}
    map_b_to_a = _serialize_atom_mapping(map_b_to_a)
    if not map_b_to_a:
        raise ValueError(f"No atom mapping found for planned RBFE pair {pair_id}.")
    _validate_minimal_mapping_atom(
        pair_id,
        len(map_b_to_a),
        minimal_mapping_atom,
    )

    pair_dir.mkdir(parents=True, exist_ok=True)
    mapping_json.write_text(json.dumps(map_b_to_a, indent=2, sort_keys=True))
    status_payload = {
        "pair_id": pair_id,
        "reference": ref,
        "target": alt,
        "mapper": mapper_name,
        "n_mapped": len(map_b_to_a),
    }
    status_payload.update(_mapping_coverage_status(rdmol_ref, rdmol_alt, map_b_to_a))
    status_payload.update(_mapping_metric_scores(atom_mapping_obj))
    if include_pocket_shape:
        status_payload.update(
            _pocket_similarity_metric_scores(
                rdmol_ref,
                rdmol_alt,
                atom_mapping_obj,
            )
        )
        _write_pocket_shape_overlap_png(
            rdmol_ref,
            rdmol_alt,
            pair_dir / "pocket_shape_overlap.png",
            pair_id=pair_id,
        )
    (pair_dir / "mapping_status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True)
    )
    try:
        with (pair_dir / "mapping.pkl").open("wb") as fh:
            pickle.dump(atom_mapping_obj, fh)
    except Exception as exc:
        logger.debug(f"Could not write RBFE atom-mapping pickle for {pair_id}: {exc}")

    try:
        atom_mapping_obj.draw_to_file(fname=pair_dir / "mapping.png")
    except Exception as exc:
        logger.debug(f"Could not draw RBFE atom-mapping image for {pair_id}: {exc}")

    return _edge_asset_from_mapping_dir(
        pair_id,
        pair_dir,
        prefer_pocket_shape=include_pocket_shape,
    )


def write_planned_mapping_artifacts(
    *,
    pairs: Sequence[Sequence[str] | tuple[str, str]],
    ligand_files: Mapping[str, Path | str],
    out_dir: Path,
    atom_mapper: str = "kartograf",
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapper_options: Any | None = None,
    atom_mapping_overrides: Any | None = None,
    overwrite: bool = False,
    protocol: str | None = None,
    minimal_mapping_atom: int | None = 3,
) -> dict[str, dict[str, Any]]:
    """
    Generate reusable atom-mapping artifacts for a planned RBFE network.

    Each edge gets ``mapping.json``, optional ``mapping.pkl``/``mapping.png``,
    and ``mapping_status.json`` under ``out_dir``. For ``rbfe_septop`` only,
    pocket-shape overlap metrics and images are also generated. The returned
    metadata is fed directly into the interactive network HTML so users can
    inspect mapping coverage, mapper identity, and metric scores before
    production.
    """
    assets: dict[str, dict[str, Any]] = {}
    overrides = _coerce_atom_mapping_overrides(atom_mapping_overrides)
    include_pocket_shape = _normalize_protocol(protocol) == "rbfe_septop"
    for ref_raw, alt_raw in pairs:
        ref = sanitize_ligand_name(str(ref_raw))
        alt = sanitize_ligand_name(str(alt_raw))
        missing = [name for name in (ref, alt) if name not in ligand_files]
        if missing:
            raise FileNotFoundError(
                f"Missing ligand file(s) for RBFE mapping {ref}~{alt}: {missing}"
            )
        assets[f"{ref}~{alt}"] = write_pair_mapping_artifacts(
            ref=ref,
            alt=alt,
            ligand_files=ligand_files,
            out_dir=out_dir,
            atom_mapper=atom_mapper,
            kartograf_options=kartograf_options,
            lomap_options=lomap_options,
            atom_mapper_options=atom_mapper_options,
            atom_mapping_overrides=overrides,
            overwrite=overwrite,
            include_pocket_shape=include_pocket_shape,
            minimal_mapping_atom=minimal_mapping_atom,
        )
    return assets


def _resolve_konnektor_generator(layout: str | None):
    try:
        from konnektor import network_planners as gen
    except ImportError as exc:
        raise RuntimeError(
            "Konnektor mapping requires the 'konnektor' package to be installed."
        ) from exc

    layout_key = (layout or "star").strip().lower()
    candidates: dict[str, type] = {}
    for name in dir(gen):
        if not name.endswith("NetworkGenerator"):
            continue
        cls = getattr(gen, name)
        short = name[: -len("NetworkGenerator")].lower()
        candidates[short] = cls
        candidates[name.lower()] = cls
    logger.debug(f'Available Konnektor network generators: {list(candidates.keys())}')
    if layout_key not in candidates:
        raise ValueError(
            f"Unknown Konnektor layout '{layout_key}'. Available: {', '.join(candidates.keys())}"
        )
    return candidates[layout_key]


def _pairs_from_konnektor_network(network) -> List[RBFEPair]:
    edges = getattr(network, "edges", None)
    if edges is None and hasattr(network, "to_edges"):
        edges = network.to_edges()
    if edges is None:
        raise RuntimeError("Konnektor network did not expose edges.")

    pairs: List[RBFEPair] = []
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = edge
        elif hasattr(edge, "componentA") and hasattr(edge, "componentB"):
            a, b = edge.componentA, edge.componentB
        elif hasattr(edge, "component1") and hasattr(edge, "component2"):
            a, b = edge.component1, edge.component2
        elif hasattr(edge, "components"):
            comps = list(edge.components)
            if len(comps) != 2:
                raise RuntimeError("Konnektor edge did not include two components.")
            a, b = comps
        else:
            raise RuntimeError("Unsupported Konnektor edge object format.")

        name_a = sanitize_ligand_name(getattr(a, "name", str(a)))
        name_b = sanitize_ligand_name(getattr(b, "name", str(b)))
        pairs.append((name_a, name_b))
    return pairs


def konnektor_pairs(
    ligands: Sequence[str],
    ligand_files: Mapping[str, Path],
    layout: str | None = None,
    plot_path: Path | None = None,
    hmr: bool = True,
    atom_mapper: str = "kartograf",
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapping_overrides: Any | None = None,
    network_scorer: str | None = None,
    protocol: str | None = None,
) -> List[RBFEPair]:
    """
    Build RBFE pairs using Konnektor network planners.

    When ``plot_path`` is supplied, BATTER also asks Konnektor for a static
    network PNG and writes ``network.graphml`` next to it. The richer
    BATTER-authored HTML network is generated later from the resolved pair list
    and prepared atom-mapping artifacts.
    """
    try:
        from gufe import SmallMoleculeComponent

    except ImportError as exc:
        raise RuntimeError(
            "konnektor mapping requires 'gufe' to be installed."
        ) from exc


    generator_cls = _resolve_konnektor_generator(layout)
    if generator_cls.__name__.lower().startswith("explicit"):
        raise ValueError(
            "Konnektor 'explicit' layout requires explicit edges; use rbfe.mapping_file."
        )
    
    mapper = _build_konnektor_atom_mapper(
        atom_mapper,
        hmr=hmr,
        kartograf_options=kartograf_options,
        lomap_options=lomap_options,
        atom_mapping_overrides=atom_mapping_overrides,
    )
    scorer = _network_scorer_callable(network_scorer, protocol=protocol)

    generator = generator_cls(mappers=mapper, scorer=scorer)

    components: List[SmallMoleculeComponent] = []
    for lig in ligands:
        path = Path(ligand_files[lig])
        mol = _load_rdkit_mol(path)
        components.append(SmallMoleculeComponent(mol, name=lig))

    if hasattr(generator, "generate_ligand_network"):
        network = generator.generate_ligand_network(components)
    elif hasattr(generator, "generate_network"):
        network = generator.generate_network(components)
    elif callable(generator):
        network = generator(components)
    else:
        raise RuntimeError("Unsupported Konnektor generator API.")

    if plot_path is not None:
        try:
            from konnektor.visualization import draw_ligand_network

            fig = draw_ligand_network(network=network, title=getattr(network, "name", None))
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=200)
            with open(f"{plot_path.parent}/network.graphml", "w") as writer:
                writer.write(network.to_graphml())
        except Exception:
            pass

    pairs = _pairs_from_konnektor_network(network)
    if not pairs:
        raise ValueError("Konnektor mapping produced no ligand pairs.")
    return pairs


def draw_explicit_konnektor_network(
    pairs: Sequence[Sequence[str] | tuple[str, str]],
    ligand_files: Mapping[str, Path],
    plot_path: Path,
    hmr: bool = True,
    atom_mapper: str = "kartograf",
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapping_overrides: Any | None = None,
    network_scorer: str | None = None,
    protocol: str | None = None,
) -> None:
    """Build an explicit Konnektor network from pairs and draw it."""
    mapper_name = _normalize_atom_mapper(atom_mapper)
    try:
        from konnektor.network_planners import ExplicitNetworkGenerator
        from konnektor.visualization import draw_ligand_network
        from gufe import SmallMoleculeComponent
        align_mol_shape = None
        if mapper_name == "kartograf":
            from kartograf.atom_aligner import align_mol_shape as _align_mol_shape

            align_mol_shape = _align_mol_shape
    except Exception:
        return

    try:
        mapper = _build_konnektor_atom_mapper(
            mapper_name,
            hmr=hmr,
            kartograf_options=kartograf_options,
            lomap_options=lomap_options,
            atom_mapping_overrides=atom_mapping_overrides,
        )
    except Exception:
        return

    comp_by_name: dict[str, SmallMoleculeComponent] = {}
    edges = []
    nodes_by_name: dict[str, SmallMoleculeComponent] = {}
    for ref, alt in pairs:
        name_a = str(ref)
        name_b = str(alt)
        if name_a not in ligand_files or name_b not in ligand_files:
            continue
        if name_a not in comp_by_name:
            mol_a = _load_rdkit_mol(Path(ligand_files[name_a]))
            comp_by_name[name_a] = SmallMoleculeComponent(mol_a, name=name_a)
        if name_b not in comp_by_name:
            mol_b = _load_rdkit_mol(Path(ligand_files[name_b]))
            comp_by_name[name_b] = SmallMoleculeComponent(mol_b, name=name_b)

        comp_a = comp_by_name[name_a]
        comp_b = comp_by_name[name_b]
        if align_mol_shape is not None:
            try:
                comp_b = align_mol_shape(comp_b, ref_mol=comp_a)
            except Exception:
                pass
        edges.append((comp_a, comp_b))
        nodes_by_name.setdefault(name_a, comp_a)
        nodes_by_name.setdefault(name_b, comp_b)

    if not edges:
        return

    nodes = list(nodes_by_name.values())
    try:
        scorer = _network_scorer_callable(network_scorer, protocol=protocol)
    except Exception:
        return
    generator = ExplicitNetworkGenerator(mappers=mapper, scorer=scorer)

    try:
        network = generator.generate_ligand_network(edges=edges, nodes=nodes)
        fig = draw_ligand_network(network=network, title=getattr(network, "name", None))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200)
        with open(f"{plot_path.parent}/network.graphml", "w") as writer:
            writer.write(network.to_graphml())
    except Exception:
        return


@dataclass(frozen=True)
class RBFENetwork:
    """
    Record the RBFE simulation mapping as ligand pairs.

    Parameters
    ----------
    ligands : Sequence[str]
        Ordered ligand identifiers participating in the network.
    pairs : Sequence[tuple[str, str]]
        Directed pairs describing simulations to run (reference, target).
    """

    ligands: Tuple[str, ...]
    pairs: Tuple[RBFEPair, ...]

    @staticmethod
    def default_mapping(ligands: Sequence[str]) -> List[RBFEPair]:
        """
        Default RBFE mapping: first ligand paired to each subsequent ligand.
        """
        if len(ligands) < 2:
            return []
        root = ligands[0]
        return [(root, lig) for lig in ligands[1:]]

    @classmethod
    def from_ligands(
        cls,
        ligands: Sequence[str],
        mapping_fn: RBFEMapFn | None = None,
    ) -> "RBFENetwork":
        """
        Build an RBFE network from ligand identifiers and a mapping function.

        Parameters
        ----------
        ligands : Sequence[str]
            Ordered ligand identifiers.
        mapping_fn : callable, optional
            Function that returns iterable of (ref, target) pairs. When omitted,
            defaults to mapping the first ligand to all others.
        """
        if not ligands:
            raise ValueError("RBFE network requires at least two ligands.")

        lig_list = [sanitize_ligand_name(str(lig)) for lig in ligands]
        if len(lig_list) < 2:
            raise ValueError("RBFE network requires at least two ligands.")

        if len(set(lig_list)) != len(lig_list):
            raise ValueError("RBFE network ligand identifiers must be unique.")

        builder = mapping_fn or cls.default_mapping
        raw_pairs = list(builder(lig_list))

        if not raw_pairs:
            raise ValueError("RBFE mapping function returned no ligand pairs.")

        lig_set = set(lig_list)
        cleaned: List[RBFEPair] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"RBFE mapping entries must be 2-tuples; got {pair!r}.")
            ref, tgt = str(pair[0]), str(pair[1])
            if ref not in lig_set or tgt not in lig_set:
                raise ValueError(
                    f"RBFE mapping contains unknown ligand(s): {(ref, tgt)!r}."
                )
            if ref == tgt:
                raise ValueError("RBFE mapping cannot include self-pairs.")
            cleaned.append((ref, tgt))

        deduped = _dedupe_pairs(cleaned)
        return cls(ligands=tuple(lig_list), pairs=tuple(deduped))

    def to_mapping(self) -> dict:
        """
        Return a JSON-serializable mapping payload.
        """
        return {
            "ligands": list(self.ligands),
            "pairs": [list(p) for p in self.pairs],
        }
