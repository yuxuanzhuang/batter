from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Protocol


__all__ = [
    "SimSystem",
    "CreateSystemLike",
    "SystemBuilder",
]


@dataclass(frozen=True, slots=True)
class SimSystem:
    """
    Immutable descriptor of a simulation system and its on-disk artifacts.

    Parameters
    ----------
    name : str
        Logical system name (e.g., ``"AT1R_AAI"``).
    root : pathlib.Path
        Working directory where artifacts live. This directory is considered
        **relocatable**; other modules should store relative paths when possible.
    topology : pathlib.Path, optional
        Path to an explicit topology (e.g., AMBER PRMTOP). May be ``None`` if the
        builder generates it later.
    coordinates : pathlib.Path, optional
        Coordinates or restart file (e.g., RST7/INPCRD).
    protein : pathlib.Path, optional
        Input protein structure file (PDB/mmCIF).
    ligands : tuple[pathlib.Path, ...]
        One or more ligand structure files.
    lipid_mol : tuple[str, ...]
        Lipid names present in the system (e.g., ``("POPC",)``).
    other_mol : tuple[str, ...]
        Other cofactor present in the system``).
    anchors : tuple[str, ...]
        Anchor atoms in the form ``"RESID@ATOM"`` (e.g., ``"85@CA"``).
    meta : dict
        Free-form metadata for provenance (e.g., software versions).
    """
    name: str
    root: Path
    topology: Optional[Path] = None
    coordinates: Optional[Path] = None
    protein: Optional[Path] = None
    ligands: Tuple[Path, ...] = ()
    lipid_mol: Tuple[str, ...] = ()
    other_mol: Tuple[str, ...] = ()
    anchors: Tuple[str, ...] = ()
    meta: Dict[str, str] = None

    def with_artifacts(self, **kw) -> "SimSystem":
        """
        Return a new :class:`SimSystem` with updated artifact attributes.

        Examples
        --------
        >>> sys = SimSystem(name="X", root=Path("work/X"))
        >>> sys2 = sys.with_artifacts(topology=Path("work/X/top.prmtop"))
        """
        return replace(self, **kw)


class CreateSystemLike(Protocol):
    """
    Structural typing interface for inputs to a system builder.

    Notes
    -----
    This Protocol is intentionally minimal to avoid import cycles with
    Pydantic models. Any object with these attributes (e.g., a Pydantic
    model instance) satisfies the protocol.
    """
    system_name: str
    protein_input: Optional[Path]
    system_topology: Optional[Path]
    system_coordinate: Optional[Path]
    ligand_paths: Sequence[Path]
    ligand_ff: str
    overwrite: bool
    retain_lig_prot: bool
    lipid_mol: Sequence[str]
    other_mol: Sequence[str]
    anchor_atoms: Sequence[str]


class SystemBuilder(Protocol):
    """
    Interface for creating or updating on-disk artifacts for a system.

    Methods
    -------
    build(system, args)
        Materialize artifacts for ``system`` using ``args``, returning an
        updated :class:`SimSystem`. Implementations must be **idempotent**:
        calling ``build`` twice with the same inputs must produce the same
        state without corrupting outputs.
    """

    def build(self, system: SimSystem, args: CreateSystemLike) -> SimSystem: ...