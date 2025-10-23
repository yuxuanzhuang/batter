from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from batter.config.simulation import SimulationConfig

# ----------------------------- utils ---------------------------------

def _coerce_yes_no(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return "yes" if v else "no"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"yes", "no"}: return s
        if s in {"true", "t", "1"}: return "yes"
        if s in {"false", "f", "0"}: return "no"
    raise ValueError(f"Expected yes/no (or boolean), got {v!r}")

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
    type: Literal["MABFE"] = "MABFE"
    output_folder: Path

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return None if v is None else Path(v)

class CreateArgs(BaseModel):
    """
    Inputs for system creation/staging (pre-simulation).
    """
    #model_config = ConfigDict(extra="ignore")

    system_name: str
    protein_input: Path
    system_input: Optional[Path] = None
    system_coordinate: Optional[Path] = None
    protein_align: Optional[str] = None

    # Ligand staging
    ligand_paths: list[Path] = Field(default_factory=list)
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
    extra_restraints: Optional[str] = None
    overwrite: bool = True

    # Box / chemistry basics that are used before FE
    receptor_ff: str = "ff14SB"
    lipid_ff: str = "lipid21"
    solv_shell: float = 15.0
    cation: str = "Na+"
    anion: str = "Cl-"
    ion_conc: float = 0.15
    neutralize_only: Literal["yes", "no"] = "no"
    water_model: str = "TIP3P"
    hmr: Literal["yes", "no"] = "no"
    dt: float = 0.004
    temperature: float = 310.0

    release_eq: list[float] = Field(default_factory=list)
    l1_range: float = 6.0
    min_adis: float = 3.0
    max_adis: float = 7.0

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
            return [Path(p.strip()) for p in v.split(",") if p.strip()]
        return [Path(p) for p in v]

    @field_validator("neutralize_only", "hmr", mode="before")
    @classmethod
    def _coerce_create_yes_no(cls, v):
        return _coerce_yes_no(v)

    @field_validator("ligand_input", mode="before")
    @classmethod
    def _coerce_ligand_input(cls, v):
        return None if v in (None, "") else Path(v)

    @model_validator(mode="after")
    def _require_ligands(self):
        if not self.ligand_paths and not self.ligand_input:
            raise ValueError("You must provide either `ligand_paths` or `ligand_input`.")
        return self

class FESimArgs(BaseModel):
    """
    Free-energy simulation knobs (lives under `fe_sim:`).
    """
    #model_config = ConfigDict(extra="ignore")

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
    eq_steps1: int = 500_000
    eq_steps2: int = 1_000_000
    z_steps1: int = 50_000
    z_steps2: int = 300_000
    ntpr: int = 1000
    ntwr: int = 10000
    ntwe: int = 0
    ntwx: int = 2500
    cut: float = 9.0
    gamma_ln: float = 1.0
    barostat: int = 2

    @field_validator("remd", "rocklin_correction", mode="before")
    @classmethod
    def _coerce_fe_yes_no(cls, v):
        return _coerce_yes_no(v)

class RunSection(BaseModel):
    """Run-related settings."""
    model_config = ConfigDict(extra="ignore")
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
    model_config = ConfigDict(extra="ignore")

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
        data = yaml.safe_load(p.read_text())
        return cls(**data)

    @classmethod
    def model_validate_yaml(cls, yaml_text: str) -> "RunConfig":
        import yaml
        return cls(**(yaml.safe_load(yaml_text) or {}))

    def resolved_sim_config(self) -> SimulationConfig:
        """Merge create/fe_sim into a SimulationConfig and surface SLURM bits."""
        base: dict[str, Any] = {}

        c = self.create
        overlay_create = {
            "system_name": c.system_name,
            "receptor_ff": getattr(c, "receptor_ff", "ff14SB"),
            "ligand_ff": c.ligand_ff,
            "lipid_ff": getattr(c, "lipid_ff", "lipid21"),
            "lipid_mol": list(c.lipid_mol or []),
            "other_mol": list(c.other_mol or []),
            "water_model": getattr(c, "water_model", "TIP3P"),
            "temperature": float(getattr(c, "temperature", 310.0)),
            "dt": float(getattr(c, "dt", 0.004)),
            "neutralize_only": _coerce_yes_no(getattr(c, "neutralize_only", "no")),
            "hmr": _coerce_yes_no(getattr(c, "hmr", "no")),
            "ion_conc": float(getattr(c, "ion_conc", 0.15)),
            "cation": getattr(c, "cation", "Na+"),
            "anion": getattr(c, "anion", "Cl-"),
            "solv_shell": float(getattr(c, "solv_shell", 15.0)),
            "protein_align": getattr(c, "protein_align", "name CA"),
            "release_eq": list(getattr(c, "release_eq", [])),
            "l1_range": float(getattr(c, "l1_range", 6.0)),
            "min_adis": float(getattr(c, "min_adis", 3.0)),
            "max_adis": float(getattr(c, "max_adis", 7.0)),
        }

        f = self.fe_sim
        overlay_fe = {
            "fe_type": f.fe_type,
            "dec_int": f.dec_int,
            "remd": _coerce_yes_no(getattr(f, "remd", "no")),
            "rocklin_correction": _coerce_yes_no(getattr(f, "rocklin_correction", "no")),
            "attach_rest": list(getattr(f, "attach_rest", [])),
            "lambdas": list(getattr(f, "lambdas", [])),
            "sdr_dist": float(getattr(f, "sdr_dist", 0.0)),
            "blocks": int(getattr(f, "blocks", 0)),
            "lig_buffer": float(getattr(f, "lig_buffer", 0.0)),
            "rec_dihcf_force": float(getattr(f, "rec_dihcf_force", 0.0)),
            "rec_discf_force": float(getattr(f, "rec_discf_force", 0.0)),
            "lig_distance_force": float(getattr(f, "lig_distance_force", 0.0)),
            "lig_angle_force": float(getattr(f, "lig_angle_force", 0.0)),
            "lig_dihcf_force": float(getattr(f, "lig_dihcf_force", 0.0)),
            "rec_com_force": float(getattr(f, "rec_com_force", 0.0)),
            "lig_com_force": float(getattr(f, "lig_com_force", 0.0)),
            "buffer_x": float(getattr(f, "buffer_x", 0.0)),
            "buffer_y": float(getattr(f, "buffer_y", 0.0)),
            "buffer_z": float(getattr(f, "buffer_z", 0.0)),
            "eq_steps1": int(getattr(f, "eq_steps1", 500_000)),
            "eq_steps2": int(getattr(f, "eq_steps2", 1_000_000)),
            "n_steps_dict": {
                "z_steps1": int(getattr(f, "z_steps1", 50_000)),
                "z_steps2": int(getattr(f, "z_steps2", 300_000)),
            },
            "ntpr": int(getattr(f, "ntpr", 1000)),
            "ntwr": int(getattr(f, "ntwr", 10000)),
            "ntwe": int(getattr(f, "ntwe", 0)),
            "ntwx": int(getattr(f, "ntwx", 2500)),
            "cut": float(getattr(f, "cut", 9.0)),
            "gamma_ln": float(getattr(f, "gamma_ln", 1.0)),
            "barostat": int(getattr(f, "barostat", 2)),
        }

        merged: dict[str, Any] = {**base, **overlay_create, **overlay_fe}

        # Surface SLURM selections to sim config for handlers to consume
        slurm= self.run.slurm
        merged["partition"] = slurm.partition

        return SimulationConfig(**merged)