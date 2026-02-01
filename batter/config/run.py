from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Dict, Optional, Literal, List, Mapping, Iterable, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from batter.config.simulation import PROTOCOL_TO_FE_TYPE, SimulationConfig
from batter.config.remd import RemdArgs
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
    param_charge: str = Field(
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

    The fields feed directly into :class:`batter.config.simulation.SimulationConfig`
    overrides. ``fe_type`` is resolved internally from ``protocol`` rather than
    being set by users.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_knobs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        if "num_equil_extends" in data:
            raise ValueError(
                "fe_sim.num_equil_extends is no longer supported; set fe_sim.eq_steps to the total equilibration steps."
            )
        if "num_fe_extends" in data:
            raise ValueError(
                "fe_sim.num_fe_extends is no longer supported; set fe_sim.n_steps (or <comp>_n_steps) to the total production steps."
            )
        if "analysis_range" in data:
            raise ValueError(
                "fe_sim.analysis_range is no longer supported; set fe_sim.analysis_start_step to the first step to include in analysis."
            )
        return data

    dec_int: str = Field(
        "mbar",
        description="Free-energy integration scheme (``mbar`` or ``ti``).",
    )
    remd: RemdArgs = Field(
        default_factory=RemdArgs,
        description="Replica-exchange MD controls (nstlim).",
    )
    rocklin_correction: Literal["yes", "no"] = Field(
        "no",
        description="Apply Rocklin correction during analysis.",
    )

    lambdas: List[float] = Field(
        default_factory=list,
        description="Default lambda schedule when component-specific overrides are not provided.",
    )
    component_lambdas: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Per-component lambda overrides (key = letter).",
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
    buffer_x: float = Field(20.0, description="Box padding along X (Å).")
    buffer_y: float = Field(20.0, description="Box padding along Y (Å).")
    buffer_z: float = Field(20.0, description="Box padding along Z (Å).")

    # Equilibration schedule
    num_equil_extends: int = Field(
        0,
        ge=0,
        description="Deprecated: equilibration extensions are ignored; keep 0.",
    )
    eq_steps: int = Field(
        1_000_000,
        gt=0,
        description="Total equilibration steps (entire equilibration run).",
    )
    n_steps: Dict[str, int] = Field(
        default_factory=lambda: {"x": 300_000, "y": 300_000},
        description="Total production steps per component (key = letter).",
    )
    ntpr: int = Field(100, description="Energy print frequency.")
    ntwr: int = Field(2_500, description="Restart write frequency.")
    ntwe: int = Field(0, description="Energy write frequency (0 disables).")
    ntwx: int = Field(25_000, description="Trajectory write frequency.")
    cut: float = Field(9.0, description="Nonbonded cutoff (Å).")
    gamma_ln: float = Field(1.0, description="Langevin gamma value (ps^-1).")
    dt: float = Field(0.004, description="MD timestep (ps).")
    hmr: Literal["yes", "no"] = Field(
        "no", description="Hydrogen mass repartitioning toggle."
    )
    enable_mcwat: Literal["yes", "no"] = Field(
        "yes",
        description="Enable MC water exchange moves during equilibration (1 = on).",
    )
    temperature: float = Field(298.15, description="Simulation temperature (K).")
    barostat: int = Field(2, description="Barostat selection (1=Berendsen, 2=MC).")
    unbound_threshold: float = Field(
        8.0,
        ge=0.0,
        description="Distance threshold (Å) used to flag ligands as unbound during equilibration analysis.",
    )
    analysis_start_step: int = Field(
        0,
        ge=0,
        description="Only analyze FE production steps after this step (per window).",
    )

    @field_validator("rocklin_correction", "hmr", "enable_mcwat", mode="before")
    @classmethod
    def _coerce_fe_yes_no(cls, v):
        return coerce_yes_no(v)

    @field_validator("remd", mode="before")
    @classmethod
    def _coerce_remd(cls, v):
        if isinstance(v, RemdArgs):
            return v
        if isinstance(v, Mapping):
            return RemdArgs(**v)
        if v is None:
            return RemdArgs()
        raise ValueError(
            "fe_sim.remd only accepts REMD timing settings (nstlim); "
            "use run.remd to enable or disable REMD submissions."
        )

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

    @model_validator(mode="before")
    @classmethod
    def _ingest_component_lambda_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        payload = dict(data)
        comp_map = dict(payload.get("component_lambdas") or {})

        def _parse_lambda_value(val: Any) -> List[float]:
            if val is None:
                return []
            if isinstance(val, str):
                parts = [p for p in re.split(r"[,\s]+", val.strip()) if p]
                return [float(p) for p in parts]
            if isinstance(val, (list, tuple)):
                return [float(v) for v in val]
            return [float(val)]

        for key in list(payload.keys()):
            m = re.match(r"^([a-z])_lambdas$", key)
            if not m:
                continue
            comp = m.group(1)
            comp_map.setdefault(comp, _parse_lambda_value(payload.pop(key)))

        payload["component_lambdas"] = comp_map
        return payload

    @model_validator(mode="before")
    @classmethod
    def _ingest_legacy_step_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        payload = dict(data)
        n_steps = dict(payload.get("n_steps") or {})
        # Allow legacy 'steps2' while migrating; raise on steps1
        legacy_steps2 = dict(payload.pop("steps2", {}) or {})
        for k, v in legacy_steps2.items():
            n_steps.setdefault(k, v)

        for key in list(payload.keys()):
            m_n = re.match(r"^([a-z])_n_steps$", key)
            if m_n:
                comp = m_n.group(1)
                val = payload.pop(key)
                try:
                    val = int(val)
                except Exception:
                    pass
                n_steps.setdefault(comp, val)
                continue
            m = re.match(r"^([a-z])_steps([12])$", key)
            if not m:
                continue
            comp, stage = m.groups()
            val = payload.pop(key)
            try:
                val = int(val)
            except Exception:
                pass
            if stage == "1":
                raise ValueError(
                    f"{comp}_steps1 is no longer supported; set {comp}_n_steps to the total production steps."
                )
            n_steps.setdefault(comp, val)

        payload["n_steps"] = n_steps
        return payload


class MDSimArgs(BaseModel):
    """
    Simulation overrides used when ``protocol == "md"``.

    These runs reuse the equilibration steps from ABFE but never schedule FE windows,
    so only generic MD knobs are required (no lambdas, SDR restraints, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_knobs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        if "num_equil_extends" in data:
            raise ValueError(
                "fe_sim.num_equil_extends is no longer supported; set fe_sim.eq_steps to the total equilibration steps."
            )
        if "num_fe_extends" in data:
            raise ValueError(
                "fe_sim.num_fe_extends is no longer supported; set fe_sim.n_steps (or <comp>_n_steps) to the total production steps."
            )
        if "analysis_range" in data:
            raise ValueError(
                "fe_sim.analysis_range is no longer supported; set fe_sim.analysis_start_step to the first step to include in analysis."
            )
        return data

    dt: float = Field(0.004, description="MD timestep (ps).")
    temperature: float = Field(298.15, description="Simulation temperature (K).")
    num_equil_extends: int = Field(
        0,
        ge=0,
        description="Deprecated: equilibration extensions are ignored; keep 0.",
    )
    eq_steps: int = Field(
        100_000,
        gt=0,
        description="Total equilibration steps (entire equilibration run).",
    )
    ntpr: int = Field(100, description="Energy print frequency.")
    ntwr: int = Field(10_000, description="Restart write frequency.")
    ntwe: int = Field(0, description="Energy write frequency (0 disables).")
    ntwx: int = Field(25_000, description="Trajectory write frequency.")
    cut: float = Field(9.0, description="Nonbonded cutoff (Å).")
    gamma_ln: float = Field(1.0, description="Langevin gamma value (ps^-1).")
    barostat: int = Field(2, description="Barostat selection (1=Berendsen, 2=MC).")
    hmr: Literal["yes", "no"] = Field(
        "yes", description="Hydrogen mass repartitioning toggle."
    )
    enable_mcwat: Literal["yes", "no"] = Field(
        "yes",
        description="Enable MC water exchange moves during equilibration (1 = on).",
    )

    @field_validator("hmr", "enable_mcwat", mode="before")
    @classmethod
    def _coerce_hmr(cls, v):
        return coerce_yes_no(v) or "no"


class RBFENetworkArgs(BaseModel):
    """
    RBFE network mapping controls.

    Users can specify a mapping strategy by name (``mapping``) or provide
    an explicit mapping file (``mapping_file``).
    """

    model_config = ConfigDict(extra="forbid")

    mapping: Optional[str] = Field(
        "default",
        description="Mapping strategy name (e.g., 'default').",
    )
    mapping_file: Optional[Path] = Field(
        None,
        description="Optional path to a mapping file (JSON/YAML/text).",
    )
    edges_file: Optional[Path] = Field(
        None,
        description="Optional path to a JSON file containing a dict of edges.",
    )

    def resolve_paths(self, base: Path) -> "RBFENetworkArgs":
        mf = self.mapping_file
        if mf is not None and not mf.is_absolute():
            mf = (base / mf).resolve()
        ef = self.edges_file
        if ef is not None and not ef.is_absolute():
            ef = (base / ef).resolve()
        return self.model_copy(update={"mapping_file": mf, "edges_file": ef})

    @field_validator("mapping", mode="before")
    @classmethod
    def _lower_mapping(cls, v):
        if v is None:
            return None
        text = str(v).strip()
        return text.lower() if text else None

    @model_validator(mode="after")
    def _validate_mapping(self) -> "RBFENetworkArgs":
        if self.mapping_file is None and self.edges_file is None and not self.mapping:
            raise ValueError(
                "rbfe.mapping, rbfe.mapping_file, or rbfe.edges_file must be provided."
            )
        return self


class RunSection(BaseModel):
    """Run-related settings, including where outputs land."""

    model_config = ConfigDict(extra="forbid")
    output_folder: Path = Field(
        ...,
        description="Directory where system artifacts and executions are stored.",
    )
    system_type: Literal["MABFE", "MASFE"] | None = Field(
        None,
        description=(
            "Optional override for the system builder. When omitted, the orchestrator "
            "chooses the builder based on the protocol."
        ),
    )
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
    max_active_jobs: int | None = Field(
        1000,
        ge=0,
        description="Max concurrent SLURM jobs for FE submissions (0 disables throttling).",
    )
    batch_mode: bool = Field(
        False,
        description="When true, run SLURM jobs inline via srun inside the manager allocation instead of submitting with sbatch.",
    )
    batch_gpus: int | None = Field(
        None,
        ge=0,
        description="GPUs available to the manager process for batch_mode; auto-detected from SLURM env when omitted.",
    )
    batch_gpus_per_task: int = Field(
        1,
        ge=1,
        description="GPUs to assign per task when batch_mode is enabled.",
    )
    batch_srun_extra: List[str] = Field(
        default_factory=list,
        description="Extra srun flags appended when launching tasks in batch_mode.",
    )
    dry_run: bool = Field(
        False, description="Force dry-run mode regardless of YAML setting."
    )
    clean_failures: bool = Field(
        False,
        description="Clear FAILED markers and progress caches before rerunning.",
    )
    remd: Literal["yes", "no"] = Field(
        "no",
        description="Enable REMD execution.",
    )
    run_id: str = Field(
        "auto", description="Run identifier to use (``auto`` picks latest)."
    )
    allow_run_id_mismatch: bool = Field(
        False,
        description=(
            "When ``True``, allow reusing an explicit ``run_id`` even if the "
            "configuration hash differs from the existing execution."
        ),
    )
    slurm_header_dir: Path | None = Field(
        None,
        description="Optional directory containing user Slurm headers (defaults to ~/.batter).",
    )

    email_sender: str = Field(
        "nobody@stanford.edu",
        description=("Sender address used for completion email notifications."),
    )
    email_on_completion: str | None = Field(
        None,
        description=(
            "Email address that should receive a notification once the run "
            "finishes (successfully or with warnings)."
        ),
    )
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

    def resolve_paths(self, base: Path) -> "RunSection":
        """
        Return a copy where ``output_folder`` is absolute relative to ``base``.
        """
        folder = self.output_folder
        if not folder.is_absolute():
            folder = (base / folder).resolve()
        hdr = self.slurm_header_dir
        if hdr is not None and not hdr.is_absolute():
            hdr = (base / hdr).resolve()
        return self.model_copy(
            update={
                "output_folder": folder,
                "slurm_header_dir": hdr,
                "remd": coerce_yes_no(self.remd),
            }
        )

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_output_folder(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("`run.output_folder` is required.")
        return Path(v)

    @field_validator("remd", mode="before")
    @classmethod
    def _coerce_remd(cls, v):
        return coerce_yes_no(v)

    @field_validator("system_type", mode="before")
    @classmethod
    def _normalize_system_type(cls, value):
        if value is None:
            return None
        text = str(value).strip().upper()
        if text not in {"MABFE", "MASFE"}:
            raise ValueError("run.system_type must be 'MABFE', 'MASFE', or omitted.")
        return text


class RunConfig(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(1, description="Schema version of the run configuration.")
    protocol: Literal["abfe", "rbfe", "asfe", "md"] = Field(
        "abfe", description="High-level protocol to execute."
    )
    backend: Literal["local", "slurm"] = Field(
        "local", description="Execution backend."
    )

    create: CreateArgs = Field(..., description="Settings for system creation/staging.")
    fe_sim: Dict[str, Any] | FESimArgs | MDSimArgs = Field(
        default_factory=dict, description="Simulation parameter overrides."
    )
    run: RunSection = Field(
        ..., description="Execution controls and artifact destination."
    )
    rbfe: RBFENetworkArgs | None = Field(
        default=None, description="RBFE network mapping configuration."
    )

    @model_validator(mode="after")
    def _coerce_fe_sim_model(self) -> "RunConfig":
        proto = getattr(self, "protocol", "abfe")
        current = self.fe_sim
        if proto == "md":
            target = MDSimArgs
        else:
            target = FESimArgs
        if isinstance(current, target):
            return self
        if isinstance(current, BaseModel):
            payload = current.model_dump()
        else:
            payload = dict(current or {})
        self.fe_sim = target.model_validate(payload)
        return self

    @model_validator(mode="after")
    def _validate_rbfe_section(self) -> "RunConfig":
        if self.rbfe is not None and self.protocol != "rbfe":
            raise ValueError("The 'rbfe' section is only valid when protocol='rbfe'.")
        return self

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
        fe_args = self.fe_sim
        if self.protocol == "md":
            if isinstance(fe_args, dict):
                fe_args = MDSimArgs(**fe_args)
            elif isinstance(fe_args, MDSimArgs):
                pass
            else:
                fe_args = MDSimArgs.model_validate(fe_args)
        else:
            if isinstance(fe_args, dict):
                fe_args = FESimArgs(**fe_args)
            elif isinstance(fe_args, FESimArgs):
                pass
            else:
                fe_args = FESimArgs.model_validate(fe_args)

        desired_fe_type = PROTOCOL_TO_FE_TYPE.get(self.protocol)
        return SimulationConfig.from_sections(
            self.create,
            fe_args,
            protocol=self.protocol,
            fe_type=desired_fe_type,
            slurm_header_dir=self.run.slurm_header_dir,
            run_remd=self.run.remd,
        )

    def with_base_dir(self, base_dir: Path) -> "RunConfig":
        """
        Return a copy with relative paths resolved against ``base_dir``.
        """
        resolved_create = self.create.resolve_paths(base_dir)
        resolved_run = self.run.resolve_paths(base_dir)
        resolved_rbfe = self.rbfe.resolve_paths(base_dir) if self.rbfe else None
        return self.model_copy(
            update={
                "create": resolved_create,
                "run": resolved_run,
                "rbfe": resolved_rbfe,
            }
        )


RunConfig.model_rebuild()
