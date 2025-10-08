from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, PrivateAttr

from batter.config.simulation import SimulationConfig
from batter.config.io import read_yaml_config




# ----------------------------- Sections ---------------------------------

class SystemSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["MABFE"] = "MABFE"
    output_folder: Path

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return None if v is None else Path(v)


class CreateArgs(BaseModel):
    """
    Inputs for system creation/staging.

    You must provide either:
      - `ligand_paths`: list of files, OR
      - `ligand_input`: a JSON file (dict {name: path} or list [path, ...])
    """
    model_config = ConfigDict(extra="ignore")

    system_name: str
    protein_input: Path
    system_input: Optional[Path] = None
    system_coordinate: Optional[Path] = None

    # IMPORTANT: use built-in generics (list[...]) not typing.List
    ligand_paths: list[Path] = Field(default_factory=list, description="List of ligand files")
    ligand_input: Optional[Path] = Field(default=None, description="JSON mapping/list of ligand files")

    ligand_ff: str = "gaff2"
    retain_lig_prot: bool = True
    param_method: Literal["amber", "openff"] = "amber"
    param_charge: Literal["bcc", "gas", "am1bcc"] = "am1bcc"
    param_outdir: Optional[Path] = None

    anchor_atoms: list[str] = Field(default_factory=list)
    lipid_mol: list[str] = Field(default_factory=lambda: ["POPC"])
    extra_restraints: Optional[str] = None
    overwrite: bool = True

    # ---- coercion & validation ----

    @field_validator("protein_input", "system_input", "system_coordinate", "param_outdir", mode="before")
    @classmethod
    def _coerce_opt_paths(cls, v):
        return None if v in (None, "") else Path(v)

    @field_validator("ligand_paths", mode="before")
    @classmethod
    def _coerce_paths_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return [Path(p) for p in parts]
        return [Path(p) for p in v]

    @field_validator("ligand_input", mode="before")
    @classmethod
    def _coerce_ligand_input(cls, v):
        if v in (None, ""):
            return None
        return Path(v)

    @model_validator(mode="after")
    def _require_ligands(self):
        if not self.ligand_paths and not self.ligand_input:
            raise ValueError("You must provide either `ligand_paths` or `ligand_input`.")
        return self


class RunSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    only_fe_preparation: bool = False
    dry_run: bool = False
    run_id: str = "auto"


class RunConfig(BaseModel):
    """
    Top-level YAML config.
    """
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    protocol: Literal["abfe", "asfe"] = "abfe"
    backend: Literal["local", "slurm"] = "local"

    system: SystemSection
    create: CreateArgs
    sim_config_path: Path

    _yaml_dir: Path = PrivateAttr(default_factory=lambda: Path.cwd())

    run: RunSection = Field(default_factory=RunSection)

    @field_validator("sim_config_path", mode="before")
    @classmethod
    def _coerce_sim_cfg(cls, v):
        return Path(v)

    # helper to load from path
    @classmethod
    def load(cls, path: Path | str) -> "RunConfig":
        import yaml
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        return cls(**data)

    # convenience if you already have a YAML string
    @classmethod
    def model_validate_yaml(cls, yaml_text: str) -> "RunConfig":
        import yaml
        return cls(**(yaml.safe_load(yaml_text) or {}))

    def resolved_sim_config(self) -> SimulationConfig:
        cfg_path = self.sim_config_path
        if not cfg_path.is_absolute():
            cfg_path = self._yaml_dir / cfg_path
        return read_yaml_config(cfg_path)