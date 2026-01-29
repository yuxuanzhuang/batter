"""RBFE network helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple, List

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

        lig_list = [str(lig) for lig in ligands]
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
