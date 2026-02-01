"""RBFE network helpers."""

from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List, Any

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


def load_edges_file(path: Path) -> List[RBFEPair]:
    """
    Load RBFE mapping pairs from a JSON file containing a dict of edges.

    Supported dict forms:
      - {"edges": [[A,B], ...]}
      - {"pairs": [[A,B], ...]}
      - adjacency mapping: {A: [B, C], ...}
    """
    if not path.exists():
        raise FileNotFoundError(f"RBFE edges file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("RBFE edges file must contain a JSON object (dict).")
    pairs = _pairs_from_data(data)
    if not pairs:
        raise ValueError(f"RBFE edges file produced no pairs: {path}")
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
    raise ValueError(f"Unknown RBFE mapping '{name}'. Available: default")


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
