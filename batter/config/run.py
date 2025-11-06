from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, Literal, List, Mapping, Iterable
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from batter.config.simulation import SimulationConfig
from batter.config.utils import (
    coerce_yes_no,
    expand_env_vars,
    normalize_optional_path,
    sanitize_ligand_name,
)

# ----------------------------- SLURM ---------------------------------

class SlurmConfig(BaseModel):
    """SLURM-specific configuration."""
    model_config = ConfigDict(extra="ignore")

    partition: Optional[str] = Field(None, description="SLURM partition / queue")
    time: Optional[str] = Field(None, description="Walltime, e.g. '04:00:00'")
    nodes: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    mem_per_cpu: Optional[str] = None
    gres: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    extra_sbatch: List[str] = Field(default_factory=list)

    def to_sbatch_flags(self) -> List[str]:
        flags: List[str] = []
        if self.partition:       flags += ["-p", self.partition]
        if self.time:            flags += ["-t", self.time]
        if self.nodes:           flags += ["-N", str(self.nodes)]
        if self.ntasks_per_node: flags += ["--ntasks-per-node", str(self.ntasks_per_node)]
        if self.mem_per_cpu:     flags += ["--mem-per-cpu", self.mem_per_cpu]
        if self.gres:            flags += ["--gres", self.gres]
        if self.account:         flags += ["--account", self.account]
        if self.qos:             flags += ["--qos", self.qos]
        if self.constraint:      flags += ["--constraint", self.constraint]
        flags += list(self.extra_sbatch or [])
        return flags

# ----------------------------- Sections ---------------------------------

class SystemSection(BaseModel):
    type: Literal["MABFE", "MASFE"] = "MABFE"
    output_folder: Path

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("`system.output_folder` is required.")
        return Path(v)

class CreateArgs(BaseModel):
    """
    Inputs for system creation/staging (pre-simulation).
    """
    model_config = ConfigDict(extra="forbid")

    system_name: Optional[str] = 'unnamed_system'
    protein_input: Optional[Path] = None
    system_input: Optional[Path] = None
    system_coordinate: Optional[Path] = None
    protein_align: Optional[str] = "name CA"

    # Ligand staging
    ligand_paths: dict[str, Path] = Field(default_factory=dict)
    ligand_input: Optional[Path] = None

    # Param settings
    ligand_ff: str = "gaff2"
    retain_lig_prot: bool = True
    param_method: Literal["amber", "openff"] = "amber"
    param_charge: Literal["bcc", "gas", "am1bcc"] = "am1bcc"
    param_outdir: Optional[Path] = None

    # Environment / anchors
    anchor_atoms: list[str] = Field(default_factory=list)
    lipid_mol: list[str] = Field(default_factory=list)
    other_mol: list[str] = Field(default_factory=list)
    overwrite: bool = True

    # Extra restraints
    # position restraints on selected string
    extra_restraints: Optional[str] = None
    extra_restraint_fc: float = 10.0

    # additional conformational restraints file (NFE)
    extra_conformation_restraints: Optional[Path] = None

    # Box / chemistry basics that are used before FE
    receptor_ff: str = "protein.ff14SB"
    lipid_ff: str = "lipid21"
    solv_shell: float = 15.0
    cation: str = "Na+"
    anion: str = "Cl-"
    ion_conc: float = 0.15
    neutralize_only: Literal["yes", "no"] = "no"
    water_model: str = "TIP3P"

    l1_range: float = 6.0
    min_adis: float = 3.0
    max_adis: float = 7.0

    @field_validator("protein_input", "system_input", "system_coordinate", "param_outdir", mode="before")
    @classmethod
    def _coerce_opt_paths(cls, v):
        return normalize_optional_path(v)

    @field_validator("ligand_paths", mode="before")
    @classmethod
    def _normalize_ligand_paths(cls, v):
        """
        Accept:
          - dict[str, path-like]
          - list/tuple of path-like
          - CSV string of paths
        Normalize → dict[str, Path] with sanitized names from stems if needed.
        """
        if v is None:
            return {}
        # CSV string
        if isinstance(v, str):
            items = [p.strip() for p in v.split(",") if p.strip()]
            d: dict[str, Path] = {}
            for s in items:
                p = normalize_optional_path(s)
                if p is None:
                    continue
                d[sanitize_ligand_name(p.stem)] = p
            return d
        # mapping
        if isinstance(v, Mapping):
            out: dict[str, Path] = {}
            for k, p in v.items():
                path_obj = normalize_optional_path(p)
                if path_obj is None:
                    continue
                out[sanitize_ligand_name(str(k))] = path_obj
            return out
        # iterable of paths
        if isinstance(v, Iterable):
            d: dict[str, Path] = {}
            for p in v:
                path_obj = normalize_optional_path(p)
                if path_obj is None:
                    continue
                d[sanitize_ligand_name(path_obj.stem)] = path_obj
            return d
        raise ValueError(f"Unsupported ligand_paths type: {type(v).__name__}")

    @field_validator("neutralize_only", mode="before")
    @classmethod
    def _coerce_create_yes_no(cls, v):
        return coerce_yes_no(v)

    @field_validator("ligand_input", mode="before")
    @classmethod
    def _coerce_ligand_input(cls, v):
        return normalize_optional_path(v)

    @model_validator(mode="after")
    def _require_ligands(self):
        if not self.ligand_paths and not self.ligand_input:
            raise ValueError("You must provide either `ligand_paths` or `ligand_input`.")
        return self

    @model_validator(mode="after")
    def _check_extra_restraints(self):
        if self.extra_conformation_restraints and self.extra_restraints:
            raise ValueError("Cannot specify both `extra_conformation_restraints` and `extra_restraints`.")

        if self.extra_conformation_restraints:
            p = Path(self.extra_conformation_restraints)
            if not p.exists():
                raise ValueError(f"extra_conformation_restraints file does not exist: {p}")
            # (optional) schema check if you expect JSON:
            try:
                data = json.loads(p.read_text())
            except Exception as e:
                raise ValueError(f"Could not parse {p}: {e}")
            if not isinstance(data, (list, tuple)) or not all(isinstance(r, (list, tuple)) for r in data):
                raise ValueError("JSON must be a list of rows [dir, res1, res2, cutoff, k].")
        return self

class FESimArgs(BaseModel):
    """
    Free-energy simulation knobs (lives under `fe_sim:`).
    """
    model_config = ConfigDict(extra="forbid")

    fe_type: str = "uno_rest"
    dec_int: str = "mbar"
    remd: Literal["yes", "no"] = "no"
    rocklin_correction: Literal["yes", "no"] = "no"

    lambdas: List[float] = Field(default_factory=list)
    sdr_dist: float = 0.0
    blocks: int = 0
    lig_buffer: float = 0.0

    # Restraint forces
    lig_distance_force: float = 0.0
    lig_angle_force: float = 0.0
    lig_dihcf_force: float = 0.0
    rec_com_force: float = 0.0
    lig_com_force: float = 0.0

    # Box padding (used by some builders)
    buffer_x: float = 0.0
    buffer_y: float = 0.0
    buffer_z: float = 0.0

    # Step counts / reporting
    release_eq: list[float] = Field(default_factory=list)
    eq_steps1: int = 500_000
    eq_steps2: int = 1_000_000
    z_steps1: int = 50_000
    z_steps2: int = 300_000
    y_steps1: int = 50_000
    y_steps2: int = 300_000
    ntpr: int = 1000
    ntwr: int = 10000
    ntwe: int = 0
    ntwx: int = 2500
    cut: float = 9.0
    gamma_ln: float = 1.0
    dt: float = 0.004
    hmr: Literal["yes", "no"] = "no"
    temperature: float = 310.0
    barostat: int = 2

    @field_validator("remd", "rocklin_correction", "hmr", mode="before")
    @classmethod
    def _coerce_fe_yes_no(cls, v):
        return coerce_yes_no(v)

class RunSection(BaseModel):
    """Run-related settings."""
    model_config = ConfigDict(extra="forbid")
    only_fe_preparation: bool = False
    on_failure: str = Field("raise", description="Behavior on ligand failure: 'raise' or 'prune'")
    max_workers: int | None = Field(
        None,
        description="Parallel workers for local backend (None = auto, 0 = serial)"
    )
    dry_run: bool = False
    run_id: str = "auto"

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

class RunConfig(BaseModel):
    """Top-level YAML config."""
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    protocol: Literal["abfe", "asfe"] = "abfe"
    backend: Literal["local", "slurm"] = "local"

    system: SystemSection
    create: CreateArgs
    fe_sim: FESimArgs = Field(default_factory=FESimArgs)
    run: RunSection = Field(default_factory=RunSection)

    @field_validator("protocol", mode="before")
    @classmethod
    def _lower_protocol(cls, v):
        return str(v).lower() if v else v

    @field_validator("backend", mode="before")
    @classmethod
    def _lower_backend(cls, v):
        return str(v).lower() if v else v

    @classmethod
    def load(cls, path: Path | str) -> "RunConfig":
        import yaml
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        data = expand_env_vars(data, base_dir=p.parent)
        return cls.model_validate(data)

    @classmethod
    def model_validate_yaml(cls, yaml_text: str) -> "RunConfig":
        import yaml
        raw = yaml.safe_load(yaml_text) or {}
        return cls.model_validate(expand_env_vars(raw))

    def resolved_sim_config(self) -> SimulationConfig:
        """Merge create/fe_sim into a SimulationConfig and surface SLURM bits."""
        return SimulationConfig.from_sections(
            self.create,
            self.fe_sim,
            partition=self.run.slurm.partition,
        )
