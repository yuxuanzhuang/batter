"""RBFE network helpers."""

from __future__ import annotations

import base64
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
):
    mapper_name = _normalize_atom_mapper(atom_mapper)
    if mapper_name == "lomap":
        from lomap import LomapAtomMapper

        return LomapAtomMapper(**_lomap_mapper_kwargs(lomap_options))

    return _build_current_kartograf_atom_mapper_for_network(
        kartograf_options=kartograf_options
    )


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
    from kartograf import SmallMoleculeComponent

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


def _edge_asset_from_mapping_dir(pair_id: str, pair_dir: Path) -> dict[str, Any]:
    asset: dict[str, Any] = {
        "mapping_path": (pair_dir / "mapping.json").as_posix(),
        "mapping_dir": pair_dir.as_posix(),
    }
    png = pair_dir / "mapping.png"
    if png.is_file():
        asset["image_data_uri"] = _mapping_png_data_uri(png)
        asset["image_alt"] = f"Atom mapping for {pair_id}"
    status = pair_dir / "mapping_status.json"
    if status.is_file():
        try:
            status_payload = json.loads(status.read_text())
            if "n_mapped" in status_payload:
                asset["n_mapped"] = status_payload["n_mapped"]
            if "mapper" in status_payload:
                asset["mapper"] = status_payload["mapper"]
        except Exception:
            pass
    return asset


def _serialize_atom_mapping(mapping: Mapping[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in mapping.items()}


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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate reusable atom-mapping artifacts for one planned RBFE pair."""
    pair_id = f"{ref}~{alt}"
    pair_dir = Path(out_dir) / pair_id
    mapping_json = pair_dir / "mapping.json"
    if mapping_json.is_file() and not overwrite:
        return _edge_asset_from_mapping_dir(pair_id, pair_dir)

    mapper_name = _normalize_atom_mapper(atom_mapper)
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

    pair_dir.mkdir(parents=True, exist_ok=True)
    mapping_json.write_text(json.dumps(map_b_to_a, indent=2, sort_keys=True))
    status_payload = {
        "pair_id": pair_id,
        "reference": ref,
        "target": alt,
        "mapper": mapper_name,
        "n_mapped": len(map_b_to_a),
    }
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

    return _edge_asset_from_mapping_dir(pair_id, pair_dir)


def write_planned_mapping_artifacts(
    *,
    pairs: Sequence[Sequence[str] | tuple[str, str]],
    ligand_files: Mapping[str, Path | str],
    out_dir: Path,
    atom_mapper: str = "kartograf",
    kartograf_options: Any | None = None,
    lomap_options: Any | None = None,
    atom_mapper_options: Any | None = None,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate reusable atom-mapping artifacts for a planned RBFE network."""
    assets: dict[str, dict[str, Any]] = {}
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
            overwrite=overwrite,
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
) -> List[RBFEPair]:
    """
    Build RBFE pairs using Konnektor network planners.
    """
    try:
        from gufe import SmallMoleculeComponent
        from lomap.gufe_bindings.scorers import default_lomap_score

    except ImportError as exc:
        raise RuntimeError(
            "Konnektor mapping requires 'gufe' and 'lomap' dependencies."
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
    )

    generator = generator_cls(mappers=mapper, scorer=default_lomap_score)

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
) -> None:
    """Build an explicit Konnektor network from pairs and draw it."""
    mapper_name = _normalize_atom_mapper(atom_mapper)
    try:
        from konnektor.network_planners import ExplicitNetworkGenerator
        from konnektor.visualization import draw_ligand_network
        from gufe import SmallMoleculeComponent
        from lomap.gufe_bindings.scorers import default_lomap_score
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
    generator = ExplicitNetworkGenerator(mappers=mapper, scorer=default_lomap_score)

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
