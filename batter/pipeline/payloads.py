from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from batter.config.simulation import SimulationConfig


class SystemParams(BaseModel):
    """
    System-level inputs shared by multiple pipeline steps.

    This wrapper normalises common fields (paths, anchor atoms, etc.) while still
    allowing arbitrary extra keys. Paths are converted to :class:`pathlib.Path`
    instances, making downstream usage safer.

    Parameters
    ----------
    param_outdir : Path, optional
        Directory where ligand parameter outputs should be written.
    system_name : str, optional
        Logical system name propagated to child steps.
    protein_input, system_input, system_coordinate : Path, optional
        Paths to the protein topology/coordinate inputs if supplied.
    ligand_paths : dict[str, Path]
        Mapping of ligand identifiers to staged files.
    yaml_dir : Path, optional
        Directory containing the originating YAML (useful for resolving relatives).
    anchor_atoms : tuple[str, ...]
        Anchor atom labels used for restraint placement.
    extra_restraints : str, optional
        Optional positional restraint selection string.
    extra_restraint_fc : float, optional
        Force constant (kcal/mol/Å^2) applied to ``extra_restraints``.
    extra_conformation_restraints : Path, optional
        Path to a conformational restraint JSON file.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    param_outdir: Optional[Path] = None
    system_name: Optional[str] = None
    protein_input: Optional[Path] = None
    system_input: Optional[Path] = None
    system_coordinate: Optional[Path] = None
    ligand_paths: Dict[str, Path] = Field(default_factory=dict)
    yaml_dir: Optional[Path] = None
    anchor_atoms: tuple[str, ...] = ()
    extra_restraints: Optional[str] = None
    extra_restraint_fc: Optional[float] = None
    extra_conformation_restraints: Optional[Path] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, SystemParams):
            return value.to_mapping()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(f"Cannot construct SystemParams from {type(value)!r}")

    def __getitem__(self, item: str) -> Any:
        """
        Return a field or extra value by key.

        Parameters
        ----------
        item : str
            Key to fetch.

        Returns
        -------
        Any
            Stored value for the key.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        if item in type(self).model_fields:
            return getattr(self, item)
        if self.model_extra is not None and item in self.model_extra:
            return self.model_extra[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """
        Safe lookup for a field or extra value with a default.

        Parameters
        ----------
        item : str
            Key to fetch.
        default : Any, optional
            Value returned when the key is missing or None.

        Returns
        -------
        Any
            Requested value or the default.
        """
        if item in type(self).model_fields:
            value = getattr(self, item)
            return default if value is None else value
        if self.model_extra is not None:
            return self.model_extra.get(item, default)
        return default

    def to_mapping(self) -> Dict[str, Any]:
        """
        Convert the model (including extras) to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Merged view of standard fields and extras.
        """
        data = self.model_dump()
        if self.model_extra:
            data.update(self.model_extra)
        return data

    def copy_with(self, **updates: Any) -> "SystemParams":
        """
        Create a new :class:`SystemParams` with additional updates.

        Parameters
        ----------
        **updates
            Keyword overrides applied atop the existing data.

        Returns
        -------
        SystemParams
            A new instance incorporating the updates.
        """
        data = self.to_mapping()
        data.update(updates)
        return SystemParams(**data)


class StepPayload(BaseModel):
    """
    Typed payload passed to pipeline step handlers.

    The payload binds the :class:`~batter.config.simulation.SimulationConfig` and
    :class:`SystemParams` objects used by most handlers while permitting arbitrary
    extra values for backwards compatibility or specialised needs.

    Parameters
    ----------
    sim : SimulationConfig, optional
        Resolved simulation configuration for the step.
    sys_params : SystemParams, optional
        Shared system-level parameters.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    sim: Optional[SimulationConfig] = None
    sys_params: Optional[SystemParams] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, StepPayload):
            return value.to_mapping()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(f"Cannot construct StepPayload from {type(value)!r}")

    @model_validator(mode="after")
    def _coerce_nested(self) -> "StepPayload":
        if self.sys_params is not None and not isinstance(self.sys_params, SystemParams):
            object.__setattr__(self, "sys_params", SystemParams(self.sys_params))
        if self.sim is not None and not isinstance(self.sim, SimulationConfig):
            object.__setattr__(self, "sim", SimulationConfig.model_validate(self.sim))
        return self

    def __getitem__(self, item: str) -> Any:
        """
        Return a stored value by key, searching typed fields first.

        Parameters
        ----------
        item : str
            Key to fetch.

        Returns
        -------
        Any
            Stored value.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        if item in type(self).model_fields:
            return getattr(self, item)
        if self.model_extra is not None and item in self.model_extra:
            return self.model_extra[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """
        Safe lookup for a payload value with a default.

        Parameters
        ----------
        item : str
            Key to fetch.
        default : Any, optional
            Value returned when the key is missing or None.

        Returns
        -------
        Any
            Requested value or the default.
        """
        if item in type(self).model_fields:
            value = getattr(self, item)
            return default if value is None else value
        if self.model_extra is not None:
            return self.model_extra.get(item, default)
        return default

    def to_mapping(self) -> Dict[str, Any]:
        """
        Convert the payload (including extras) to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Merged representation of fields and extras.
        """
        data = self.model_dump()
        if self.model_extra:
            data.update(self.model_extra)
        return data

    def copy_with(self, **updates: Any) -> "StepPayload":
        """
        Create a new :class:`StepPayload` with additional updates.

        Parameters
        ----------
        **updates
            Keyword overrides applied to the current payload.

        Returns
        -------
        StepPayload
            New payload containing the merged data.
        """
        data = self.to_mapping()
        data.update(updates)
        return StepPayload(**data)
