from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from batter.config.simulation import SimulationConfig


class SystemParams(BaseModel):
    """Typed wrapper around system-level parameters shared across pipeline steps."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    param_outdir: Optional[Path] = None
    system_name: Optional[str] = None
    protein_input: Optional[Path] = None
    system_input: Optional[Path] = None
    system_coordinate: Optional[Path] = None
    ligand_paths: Dict[str, Path] = Field(default_factory=dict)
    yaml_dir: Optional[Path] = None
    anchor_atoms: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, SystemParams):
            return value.to_mapping()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(f"Cannot construct SystemParams from {type(value)!r}")

    def __getitem__(self, item: str) -> Any:
        if item in self.model_fields:
            return getattr(self, item)
        if self.model_extra is not None and item in self.model_extra:
            return self.model_extra[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        if item in self.model_fields:
            value = getattr(self, item)
            return default if value is None else value
        if self.model_extra is not None:
            return self.model_extra.get(item, default)
        return default

    def to_mapping(self) -> Dict[str, Any]:
        data = self.model_dump()
        if self.model_extra:
            data.update(self.model_extra)
        return data

    def copy_with(self, **updates: Any) -> "SystemParams":
        data = self.to_mapping()
        data.update(updates)
        return SystemParams(**data)


class StepPayload(BaseModel):
    """Typed payload passed to pipeline step handlers."""

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
        if item in self.model_fields:
            return getattr(self, item)
        if self.model_extra is not None and item in self.model_extra:
            return self.model_extra[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        if item in self.model_fields:
            value = getattr(self, item)
            return default if value is None else value
        if self.model_extra is not None:
            return self.model_extra.get(item, default)
        return default

    def to_mapping(self) -> Dict[str, Any]:
        data = self.model_dump()
        if self.model_extra:
            data.update(self.model_extra)
        return data

    def copy_with(self, **updates: Any) -> "StepPayload":
        data = self.to_mapping()
        data.update(updates)
        return StepPayload(**data)
