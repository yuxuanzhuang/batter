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
    """
    SLURM-specific configuration.

    Parameters
    ----------
    partition : str, optional
        SLURM partition/queue name.
    time : str, optional
        Walltime in the ``HH:MM:SS`` format.
    nodes : int, optional
        Number of nodes to request.
    ntasks_per_node : int, optional
        Number of tasks per node.
    mem_per_cpu : str, optional
        Memory per CPU (e.g., ``16G``).
    gres : str, optional
        Generic resource string (e.g., GPU spec).
    account : str, optional
        Account to charge for jobs.
    qos : str, optional
        QoS string if required by the cluster.
    constraint : str, optional
        Constraint string passed to ``sbatch``.
    extra_sbatch : list[str]
        Additional arguments appended to the ``sbatch`` submission command.
    """

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
        if self.partition:
            flags += ["-p", self.partition]
        if self.time:
            flags += ["-t", self.time]
        if self.nodes:
            flags += ["-N", str(self.nodes)]
        if self.ntasks_per_node:
            flags += ["--ntasks-per-node", str(self.ntasks_per_node)]
        if self.mem_per_cpu:
            flags += ["--mem-per-cpu", self.mem_per_cpu]
        if self.gres:
            flags += ["--gres", self.gres]
        if self.account:
            flags += ["--account", self.account]
        if self.qos:
            flags += ["--qos", self.qos]
        if self.constraint:
            flags += ["--constraint", self.constraint]
        flags += list(self.extra_sbatch or [])
        return flags


# ----------------------------- Sections ---------------------------------


class SystemSection(BaseModel):
    type: Literal["MABFE", "MASFE"] = Field(
        "MABFE",
        description="System builder type. ``MABFE`` supports membrane ABFE workflows; ``MASFE`` enables ASFE.",
    )
    output_folder: Path = Field(
        ...,
        description="Directory where system artifacts and executions will be written.",
    )

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
    ligand_paths: dict[str, Path | str] = Field(
        default_factory=dict,
        description="Mapping of ligand identifiers to structure files (relative paths are resolved at runtime).",
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
        description='If ``"yes"``, neutralize the system without adding bulk salt.',
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

    @field_validator(
        "protein_input",
        "system_input",
        "system_coordinate",
        "param_outdir",
        mode="before",
    )
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
                resolved[key] = (
                    (base / path_val).resolve()
                    if not path_val.is_absolute()
                    else path_val
                )
            updates["ligand_paths"] = resolved

        return self.model_copy(update=updates)

    @model_validator(mode="after")
    def _require_ligands(self):
        if not self.ligand_paths and not self.ligand_input:
            raise ValueError(
                "You must provide either `ligand_paths` or `ligand_input`."
            )
        return self

    @model_validator(mode="after")
    def _check_extra_restraints(self):
        if self.extra_conformation_restraints and self.extra_restraints:
            raise ValueError(
                "Cannot specify both `extra_conformation_restraints` and `extra_restraints`."
            )

        if self.extra_conformation_restraints:
            p = Path(self.extra_conformation_restraints)
            if not p.exists():
                raise ValueError(
                    f"extra_conformation_restraints file does not exist: {p}"
                )
            # (optional) schema check if you expect JSON:
            try:
                data = json.loads(p.read_text())
            except Exception as e:
                raise ValueError(f"Could not parse {p}: {e}")
            if not isinstance(data, (list, tuple)) or not all(
                isinstance(r, (list, tuple)) for r in data
            ):
                raise ValueError(
                    "JSON must be a list of rows [dir, res1, res2, cutoff, k]."
                )
        return self


class FESimArgs(BaseModel):
    """
    Free-energy simulation knobs loaded from the ``fe_sim`` section.

    The fields map directly onto :class:`batter.config.simulation.SimulationConfig`
    overrides. Most values are optional and fall back to the defaults assembled in
    :class:`SimulationConfig`.
    """

    model_config = ConfigDict(extra="forbid")

    fe_type: str = Field(
        "uno_rest",
        description="Free-energy protocol type passed to the simulation configuration.",
    )
    dec_int: str = Field(
        "mbar",
        description="Free-energy integration scheme (``mbar`` or ``ti``).",
    )
    remd: Literal["yes", "no"] = Field(
        "no",
        description='Enable replica-exchange MD (currently unsupported; must remain ``"no"``).',
    )
    rocklin_correction: Literal["yes", "no"] = Field(
        "no",
        description="Apply Rocklin correction during analysis.",
    )

    lambdas: List[float] = Field(
        default_factory=list,
        description="Default lambda schedule when component-specific overrides are not provided.",
    )
    sdr_dist: float = Field(
        0.0,
        description="SDR placement distance for SDR protocols (Å).",
    )
    blocks: int = Field(
        0,
        description="Number of MBAR blocks to use during analysis.",
    )
    lig_buffer: float = Field(
        15.0,
        description="Ligand-specific box buffer (Å) for solvation boxes.",
    )

    # Restraint forces
    lig_distance_force: float = Field(
        5.0,
        description="Ligand COM distance restraint spring constant (kcal/mol/Å^2).",
    )
    lig_angle_force: float = Field(
        250.0,
        description="Ligand angle restraint spring constant (kcal/mol/rad^2).",
    )
    lig_dihcf_force: float = Field(
        0.0,
        description="Ligand dihedral restraint spring constant (kcal/mol/rad^2).",
    )
    rec_com_force: float = Field(
        10.0,
        description="Protein COM restraint spring constant (kcal/mol/Å^2).",
    )
    lig_com_force: float = Field(
        10.0,
        description="Ligand COM restraint spring constant (kcal/mol/Å^2).",
    )

    # Box padding (used by some builders)
    buffer_x: float = Field(10.0, description="Box padding along X (Å).")
    buffer_y: float = Field(10.0, description="Box padding along Y (Å).")
    buffer_z: float = Field(10.0, description="Box padding along Z (Å).")

    # Step counts / reporting
    release_eq: list[float] = Field(
        default_factory=list,
        description="Release-equilibration schedule weights.",
    )
    eq_steps1: int = Field(500_000, description="Equilibration stage 1 steps.")
    eq_steps2: int = Field(1_000_000, description="Equilibration stage 2 steps.")
    z_steps1: int = Field(50_000, description="Stage 1 steps for the 'z' component.")
    z_steps2: int = Field(300_000, description="Stage 2 steps for the 'z' component.")
    y_steps1: int = Field(50_000, description="Stage 1 steps for the 'y' component.")
    y_steps2: int = Field(300_000, description="Stage 2 steps for the 'y' component.")
    ntpr: int = Field(1000, description="Energy print frequency.")
    ntwr: int = Field(10_000, description="Restart write frequency.")
    ntwe: int = Field(0, description="Energy write frequency (0 disables).")
    ntwx: int = Field(2500, description="Trajectory write frequency.")
    cut: float = Field(9.0, description="Nonbonded cutoff (Å).")
    gamma_ln: float = Field(1.0, description="Langevin gamma value (ps^-1).")
    dt: float = Field(0.004, description="MD timestep (ps).")
    hmr: Literal["yes", "no"] = Field(
        "no", description="Hydrogen mass repartitioning toggle."
    )
    temperature: float = Field(310.0, description="Simulation temperature (K).")
    barostat: int = Field(2, description="Barostat selection (1=Berendsen, 2=MC).")

    @field_validator("remd", "rocklin_correction", "hmr", mode="before")
    @classmethod
    def _coerce_fe_yes_no(cls, v):
        return coerce_yes_no(v)

    @field_validator("lambdas")
    @classmethod
    def _validate_lambdas(cls, v: List[float]) -> List[float]:
        if not v:
            return v
        if any(left > right for left, right in zip(v, v[1:])):
            raise ValueError("Lambda values must be in ascending order.")
        return v

    @field_validator(
        "lig_distance_force",
        "lig_angle_force",
        "rec_com_force",
        "lig_com_force",
    )
    @classmethod
    def _validate_force_const(cls, value: float) -> float:
        if value is None:
            return value
        if value <= 0.0:
            raise ValueError("Force constants must be non-zero and positive.")
        return value


class RunSection(BaseModel):
    """Run-related settings."""

    model_config = ConfigDict(extra="forbid")
    only_fe_preparation: bool = Field(
        False,
        description="When true, stop the workflow after FE preparation.",
    )
    on_failure: Literal["raise", "prune", "retry"] = Field(
        "raise",
        description="Behavior on ligand failure: 'raise', 'prune', or 'retry' (clear FAILED sentinels and rerun once).",
    )
    max_workers: int | None = Field(
        None,
        description="Parallel workers for local backend (None = auto, 0 = serial).",
    )
    dry_run: bool = Field(
        False, description="Force dry-run mode regardless of YAML setting."
    )
    run_id: str = Field(
        "auto", description="Run identifier to use (``auto`` picks latest)."
    )

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)


class RunConfig(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(1, description="Schema version of the run configuration.")
    protocol: Literal["abfe", "asfe"] = Field(
        "abfe", description="High-level protocol to execute."
    )
    backend: Literal["local", "slurm"] = Field(
        "local", description="Execution backend."
    )

    system: SystemSection = Field(..., description="System-level configuration block.")
    create: CreateArgs = Field(..., description="Settings for system creation/staging.")
    fe_sim: FESimArgs = Field(
        default_factory=FESimArgs, description="Simulation parameter overrides."
    )
    run: RunSection = Field(
        default_factory=RunSection, description="Execution controls."
    )

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
        return self.model_copy(
            update={"system": resolved_system, "create": resolved_create}
        )
