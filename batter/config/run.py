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
        """
        Produce a flat list of ``sbatch`` command-line flags.

        Returns
        -------
        list of str
            Sequence suitable for passing to :func:`subprocess.run`.
        """
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

    def resolve_paths(self, base: Path) -> "SystemSection":
        """
        Return a copy where ``output_folder`` is absolute relative to ``base``.
        """
        folder = self.output_folder
        if not folder.is_absolute():
            folder = (base / folder).resolve()
        return self.model_copy(update={"output_folder": folder})

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("`system.output_folder` is required.")
        return Path(v)

class CreateArgs(BaseModel):
    """
    Inputs for system creation and staging.

    Notes
    -----
    This section mirrors the ``create`` block in the run YAML file.
    """
    model_config = ConfigDict(extra="forbid")

    system_name: Optional[str] = Field(
        "unnamed_system",
        description="Logical system name; used to label outputs when not provided.",
    )
    protein_input: Optional[Path] = Field(
        None,
        description="Path to the receptor structure (PDB/mmCIF).",
    )
    system_input: Optional[Path] = Field(
        None,
        description="Optional pre-built system topology (e.g., PRMTOP).",
    )
    system_coordinate: Optional[Path] = Field(
        None,
        description="Optional starting coordinates (e.g., INPCRD/RST7).",
    )
    protein_align: Optional[str] = Field(
        "name CA",
        description="Selection string used to align the protein prior to staging.",
    )

    # Ligand staging
    ligand_paths: dict[str, Path] = Field(
        default_factory=dict,
        description="Mapping of ligand identifiers to structure files.",
    )
    ligand_input: Optional[Path] = Field(
        None,
        description="Alternative JSON file describing ligands (dict or list).",
    )

    # Param settings
    ligand_ff: str = Field(
        "gaff2",
        description="Ligand force field identifier passed to parameterization tools.",
    )
    retain_lig_prot: bool = Field(
        True,
        description="Whether to retain ligand protomers generated during staging.",
    )
    param_method: Literal["amber", "openff"] = Field(
        "amber",
        description="Parameterization backend to use for ligands.",
    )
    param_charge: Literal["bcc", "gas", "am1bcc"] = Field(
        "am1bcc",
        description="Charge derivation method for ligands.",
    )
    param_outdir: Optional[Path] = Field(
        None,
        description="Optional override for the ligand parameter output directory.",
    )

    # Environment / anchors
    anchor_atoms: list[str] = Field(
        default_factory=list,
        description="List of anchor atom selections used for restraint placement.",
    )
    lipid_mol: list[str] = Field(
        default_factory=list,
        description="Names of lipid molecules present in the system.",
    )
    other_mol: list[str] = Field(
        default_factory=list,
        description="Names of non-lipid cofactors or co-binders.",
    )
    overwrite: bool = Field(
        True,
        description="If true, overwrite existing artifacts in the staging directory.",
    )

    # Extra restraints
    # position restraints on selected string
    extra_restraints: Optional[str] = Field(
        None,
        description="Optional positional restraint specification string.",
    )
    extra_restraint_fc: float = Field(
        10.0,
        description="Force constant (kcal/mol/Å^2) applied to ``extra_restraints``.",
    )

    # additional conformational restraints file (NFE)
    extra_conformation_restraints: Optional[Path] = Field(
        None,
        description="Path to conformational restraint JSON (used for NFE workflows).",
    )

    # Box / chemistry basics that are used before FE
    receptor_ff: str = Field(
        "protein.ff14SB",
        description="Protein force-field identifier.",
    )
    lipid_ff: str = Field(
        "lipid21",
        description="Lipid force-field identifier.",
    )
    solv_shell: float = Field(
        15.0,
        description="Initial solvent shell radius (Å).",
    )
    cation: str = Field(
        "Na+",
        description="Cation species for ion placement.",
    )
    anion: str = Field(
        "Cl-",
        description="Anion species for ion placement.",
    )
    ion_conc: float = Field(
        0.15,
        description="Target salt concentration (M).",
    )
    neutralize_only: Literal["yes", "no"] = Field(
        "no",
        description="If ``\"yes\"``, neutralize the system without adding bulk salt.",
    )
    water_model: str = Field(
        "TIP3P",
        description="Water model used for solvation.",
    )

    l1_range: float = Field(
        6.0,
        description="Radius (Å) for L1 search when identifying pocket positions.",
    )
    min_adis: float = Field(
        3.0,
        description="Minimum anchor-atom distance used during pose selection (Å).",
    )
    max_adis: float = Field(
        7.0,
        description="Maximum anchor-atom distance used during pose selection (Å).",
    )

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

    def resolve_paths(self, base: Path) -> "CreateArgs":
        """
        Return a copy where path fields are absolute relative to ``base``.
        """
        updates: dict[str, object] = {}
        path_fields = [
            "protein_input",
            "system_input",
            "system_coordinate",
            "param_outdir",
            "ligand_input",
            "extra_conformation_restraints",
        ]
        for name in path_fields:
            path_val = getattr(self, name)
            if isinstance(path_val, Path) and not path_val.is_absolute():
                updates[name] = (base / path_val).resolve()

        if self.ligand_paths:
            resolved = {}
            for key, path_val in self.ligand_paths.items():
                resolved[key] = (base / path_val).resolve() if not path_val.is_absolute() else path_val
            updates["ligand_paths"] = resolved

        return self.model_copy(update=updates)

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
        """Load and validate a run configuration from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Location of the YAML file to parse.

        Returns
        -------
        RunConfig
            Fully validated configuration object.
        """
        import yaml
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        data = expand_env_vars(data, base_dir=p.parent)
        cfg = cls.model_validate(data)
        return cfg.with_base_dir(p.parent)

    @classmethod
    def model_validate_yaml(cls, yaml_text: str) -> "RunConfig":
        """Validate a run configuration from an in-memory YAML string.

        Parameters
        ----------
        yaml_text : str
            Raw YAML content describing the run configuration.

        Returns
        -------
        RunConfig
            Validated configuration model.
        """
        import yaml
        raw = yaml.safe_load(yaml_text) or {}
        cfg = cls.model_validate(expand_env_vars(raw))
        return cfg.with_base_dir(Path.cwd())

    def resolved_sim_config(self) -> SimulationConfig:
        """Build the effective simulation configuration for this run.

        Returns
        -------
        SimulationConfig
            Simulation parameters derived from ``create`` and ``fe_sim`` sections.
        """
        return SimulationConfig.from_sections(
            self.create,
            self.fe_sim,
            partition=self.run.slurm.partition,
        )

    def with_base_dir(self, base_dir: Path) -> "RunConfig":
        """
        Return a copy with relative paths resolved against ``base_dir``.
        """
        resolved_system = self.system.resolve_paths(base_dir)
        resolved_create = self.create.resolve_paths(base_dir)
        return self.model_copy(update={"system": resolved_system, "create": resolved_create})
