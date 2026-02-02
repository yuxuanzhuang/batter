"""RBFE network helpers."""

from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List, Any, Mapping
from loguru import logger

from batter.config.utils import sanitize_ligand_name

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
) -> List[RBFEPair]:
    """
    Build RBFE pairs using Konnektor network planners.
    """
    try:
        from gufe import SmallMoleculeComponent
        from kartograf.atom_mapper import KartografAtomMapper
    except ImportError as exc:
        raise RuntimeError(
            "Konnektor mapping requires 'gufe' and 'kartograf' dependencies."
        ) from exc

    generator_cls = _resolve_konnektor_generator(layout)
    if generator_cls.__name__.lower().startswith("explicit"):
        raise ValueError(
            "Konnektor 'explicit' layout requires explicit edges; use rbfe.mapping_file."
        )
    mapper = KartografAtomMapper()

    def _null_scorer(_mapping):
        return 0.0

    try:
        generator = generator_cls(mappers=mapper, scorer=_null_scorer)
    except TypeError:
        generator = generator_cls(mappers=mapper)

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

    pairs = _pairs_from_konnektor_network(network)
    if not pairs:
        raise ValueError("Konnektor mapping produced no ligand pairs.")
    return pairs


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
