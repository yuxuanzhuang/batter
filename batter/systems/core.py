from __future__ import annotations

from dataclasses import dataclass, replace, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Sequence, Protocol


__all__ = [
    "SimSystem",
    "CreateSystemLike",
    "SystemBuilder",
    "SystemMeta",
]


@dataclass(frozen=True, slots=True)
class SystemMeta:
    """
    Structured metadata attached to a :class:`SimSystem`.

    Parameters
    ----------
    ligand : str, optional
        Ligand identifier associated with the system (if applicable).
    residue_name : str, optional
        Residue name used for the ligand.
    mode : str, optional
        High-level mode indicator (e.g., ``"MABFE"`` vs ``"MASFE"``).
    param_dir_dict : dict[str, str]
        Mapping from residue names to parameter storage directories.
    extras : dict[str, Any]
        Additional context stored alongside the known fields.
    """

    ligand: Optional[str] = None
    residue_name: Optional[str] = None
    mode: Optional[str] = None
    param_dir_dict: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "SystemMeta":
        """
        Construct a :class:`SystemMeta` from a mapping-like object.

        Parameters
        ----------
        data : mapping or None
            Source metadata. If already a :class:`SystemMeta`, it is returned.

        Returns
        -------
        SystemMeta
            Normalised metadata object.
        """
        if data is None:
            return cls()
        if isinstance(data, SystemMeta):
            return data
        mapping = dict(data)
        known = {
            "ligand": mapping.pop("ligand", None),
            "residue_name": mapping.pop("residue_name", None),
            "mode": mapping.pop("mode", None),
            "param_dir_dict": mapping.pop("param_dir_dict", {}) or {},
        }
        return cls(
            ligand=known["ligand"],
            residue_name=known["residue_name"],
            mode=known["mode"],
            param_dir_dict=dict(known["param_dir_dict"]),
            extras=dict(mapping),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metadata to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            All known fields plus extra entries.
        """
        data: Dict[str, Any] = {}
        if self.ligand is not None:
            data["ligand"] = self.ligand
        if self.residue_name is not None:
            data["residue_name"] = self.residue_name
        if self.mode is not None:
            data["mode"] = self.mode
        if self.param_dir_dict:
            data["param_dir_dict"] = dict(self.param_dir_dict)
        data.update(self.extras)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by key with an optional default.

        Parameters
        ----------
        key : str
            Metadata key.
        default : Any, optional
            Value returned when the key is missing.

        Returns
        -------
        Any
            Stored value or the default.
        """
        if key == "ligand":
            return self.ligand if self.ligand is not None else default
        if key == "residue_name":
            return self.residue_name if self.residue_name is not None else default
        if key == "mode":
            return self.mode if self.mode is not None else default
        if key == "param_dir_dict":
            return self.param_dir_dict if self.param_dir_dict else default
        return self.extras.get(key, default)

    def __getitem__(self, item: str) -> Any:
        """
        Access metadata using dictionary-style syntax.

        Parameters
        ----------
        item : str
            Requested key.

        Returns
        -------
        Any
            Stored value.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        value = self.get(item, None)
        if value is None and item not in {"ligand", "residue_name", "mode", "param_dir_dict"} and item not in self.extras:
            raise KeyError(item)
        return value

    def merge(self, **updates: Any) -> "SystemMeta":
        """
        Create a new :class:`SystemMeta` with updated values.

        Parameters
        ----------
        **updates
            Keyword overrides applied to the existing metadata.

        Returns
        -------
        SystemMeta
            New instance containing the merged metadata.
        """
        data = self.to_dict()
        data.update(updates)
        return SystemMeta.from_mapping(data)


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
    meta : SystemMeta
        Free-form metadata bundle for provenance (e.g., software versions).
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
    meta: SystemMeta = field(default_factory=SystemMeta)

    def __post_init__(self) -> None:
        if not isinstance(self.meta, SystemMeta):
            object.__setattr__(self, "meta", SystemMeta.from_mapping(self.meta))

    def with_artifacts(self, **kw) -> "SimSystem":
        """
        Return a new :class:`SimSystem` with updated artifact attributes.

        Examples
        --------
        >>> sys = SimSystem(name="X", root=Path("work/X"))
        >>> sys2 = sys.with_artifacts(topology=Path("work/X/top.prmtop"))
        """
        if "meta" in kw and not isinstance(kw["meta"], SystemMeta):
            kw["meta"] = SystemMeta.from_mapping(kw["meta"])
        return replace(self, **kw)

    def path(self, *parts: str | Path) -> Path:
        """
        Join ``root`` with the provided path segments.

        Parameters
        ----------
        *parts : str or Path
            Relative path components appended in order.

        Returns
        -------
        pathlib.Path
            Absolute path pointing inside ``root``.
        """
        p = self.root
        for part in parts:
            p = p / Path(part)
        return p

    def with_meta(self, **updates: Any) -> "SimSystem":
        """
        Return a copy of the system with merged metadata.

        Parameters
        ----------
        **updates
            Keyword arguments forwarded to :meth:`SystemMeta.merge`.

        Returns
        -------
        SimSystem
            Copy of the system containing the updated metadata bundle.
        """
        merged = self.meta.merge(**updates)
        return replace(self, meta=merged)


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
