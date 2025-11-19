from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Literal, TYPE_CHECKING, Tuple
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator, model_validator
import re
from loguru import logger
from batter.utils import COMPONENTS_LAMBDA_DICT
from batter.config.utils import coerce_yes_no

if TYPE_CHECKING:
    from batter.config.run import CreateArgs, FESimArgs

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())
_ANCHOR_RE = re.compile(r"^:?\d+@[\w\d]+$")  # e.g., ":85@CA" or "85@CA"

MEMBRANE_EXEMPT_COMPONENTS = {"y", "m"}

PROTOCOL_TO_FE_TYPE = {
    "abfe": "uno_rest",
    "asfe": "asfe",
    "md": "md",
}

class SimulationConfig(BaseModel):
    """
    Simulation configuration for ABFE/ASFE workflows.
    Values are fed by RunConfig.resolved_sim_config(), which merges `create:` and `fe_sim:`.
    """

    @classmethod
    def from_sections(
        cls,
        create: "CreateArgs",
        fe: "FESimArgs",
        *,
        partition: str | None = None,
        protocol: str | None = None,
        fe_type: str | None = None,
    ) -> "SimulationConfig":
        """Construct a :class:`SimulationConfig` from run sections.

        Parameters
        ----------
        create : CreateArgs
            System creation inputs taken from the ``create`` YAML section.
        fe : FESimArgs
            Free-energy simulation overrides from the ``fe_sim`` section.
        partition : str, optional
            Cluster partition specified in the run section.

        Returns
        -------
        SimulationConfig
            Fully merged simulation configuration ready for downstream use.
        """
        l1_range = create.l1_range if create.l1_range is not None else 6.0
        min_adis = create.min_adis if create.min_adis is not None else 3.0
        max_adis = create.max_adis if create.max_adis is not None else 7.0

        create_data: dict[str, Any] = {
            "system_name": create.system_name or "unnamed_system",
            "receptor_ff": create.receptor_ff,
            "ligand_ff": create.ligand_ff,
            "lipid_ff": create.lipid_ff,
            "lipid_mol": list(create.lipid_mol or []),
            "other_mol": list(create.other_mol or []),
            "water_model": create.water_model,
            "neutralize_only": coerce_yes_no(create.neutralize_only),
            "ion_conc": float(create.ion_conc),
            "cation": create.cation,
            "anion": create.anion,
            "solv_shell": float(create.solv_shell),
            "protein_align": create.protein_align,
            "l1_range": float(l1_range),
            "min_adis": float(min_adis),
            "max_adis": float(max_adis),
        }

        def _fe_attr(name: str, default):
            if hasattr(fe, name):
                return getattr(fe, name)
            if isinstance(fe, Mapping) and name in fe:
                return fe[name]
            return default() if callable(default) else default

        resolved_fe_type = fe_type
        if resolved_fe_type is None and protocol:
            resolved_fe_type = PROTOCOL_TO_FE_TYPE.get(protocol.lower())
        if resolved_fe_type is None:
            resolved_fe_type = _fe_attr("fe_type", lambda: None)
        if resolved_fe_type is None:
            resolved_fe_type = "md"

        proto_key = (protocol or "").lower()
        _component_steps_requirements = {
            "abfe": ["z_steps1", "z_steps2"],
            "asfe": ["y_steps1", "y_steps2", "m_steps1", "m_steps2"],
        }
        for field in _component_steps_requirements.get(proto_key, []):
            value = _fe_attr(field, lambda: 0)
            if value is None or value <= 0:
                raise ValueError(
                    f"{proto_key.upper()} protocol requires `{field}` to be positive, got {value!r}."
                )

        num_equil_extends = max(0, int(_fe_attr("num_equil_extends", lambda: 0)))
        eq_steps_value = int(_fe_attr("eq_steps", lambda: 1_000_000))
        release_count = max(1, num_equil_extends + 1)
        fe_release_eq = [0.0] * release_count

        extra_conf_rest = create.extra_conformation_restraints
        extra_restraints = create.extra_restraints

        num_fe_extends_value = int(_fe_attr("num_fe_extends", lambda: 10))

        def _analysis_range_default():
            if num_fe_extends_value < 4:
                logger.warning(
                    "num_fe_extends={} is < 4; default analysis_fe_range will start at 0.",
                    num_fe_extends_value,
                )
                return (0, -1)
            return (2, -1)

        analysis_fe_range_value = getattr(fe, "analysis_fe_range", None) if hasattr(fe, "analysis_fe_range") else None
        if analysis_fe_range_value is None:
            analysis_fe_range_value = _analysis_range_default()

        fe_data: dict[str, Any] = {
            "fe_type": resolved_fe_type,
            "dec_int": _fe_attr("dec_int", lambda: "mbar"),
            "remd": coerce_yes_no(_fe_attr("remd", lambda: "no")),
            "rocklin_correction": coerce_yes_no(_fe_attr("rocklin_correction", lambda: "no")),
            "enable_mcwat": coerce_yes_no(_fe_attr("enable_mcwat", lambda: "yes")),
            "lambdas": list(_fe_attr("lambdas", list) or []),
            "blocks": int(_fe_attr("blocks", lambda: 0)),
            "lig_buffer": float(_fe_attr("lig_buffer", lambda: 15.0)),
            "lig_distance_force": float(_fe_attr("lig_distance_force", lambda: 5.0)),
            "lig_angle_force": float(_fe_attr("lig_angle_force", lambda: 250.0)),
            "lig_dihcf_force": float(_fe_attr("lig_dihcf_force", lambda: 0.0)),
            "rec_com_force": float(_fe_attr("rec_com_force", lambda: 10.0)),
            "lig_com_force": float(_fe_attr("lig_com_force", lambda: 10.0)),
            "buffer_x": float(_fe_attr("buffer_x", lambda: 15.0)),
            "buffer_y": float(_fe_attr("buffer_y", lambda: 15.0)),
            "buffer_z": float(_fe_attr("buffer_z", lambda: 15.0)),
            "temperature": float(_fe_attr("temperature", lambda: 310.0)),
            "dt": float(_fe_attr("dt", lambda: 0.004)),
            "hmr": coerce_yes_no(_fe_attr("hmr", lambda: "yes")),
            "release_eq": fe_release_eq,
            "num_equil_extends": num_equil_extends,
            "eq_steps": eq_steps_value,
            "eq_steps1": eq_steps_value,
            "eq_steps2": eq_steps_value,
            "ntpr": int(_fe_attr("ntpr", lambda: 1000)),
            "ntwr": int(_fe_attr("ntwr", lambda: 10_000)),
            "ntwe": int(_fe_attr("ntwe", lambda: 0)),
            "ntwx": int(_fe_attr("ntwx", lambda: 50_000)),
            "cut": float(_fe_attr("cut", lambda: 9.0)),
            "gamma_ln": float(_fe_attr("gamma_ln", lambda: 1.0)),
            "barostat": int(_fe_attr("barostat", lambda: 2)),
            "unbound_threshold": float(_fe_attr("unbound_threshold", lambda: 8.0)),
            "analysis_fe_range": analysis_fe_range_value,
            "num_fe_extends": num_fe_extends_value,
        }

        infe_flag = bool(extra_conf_rest)
        if extra_conf_rest:
            fe_data["barostat"] = 2
        elif extra_restraints is not None:
            fe_data["barostat"] = 1

        n_steps_dict = {
            "z_steps1": int(_fe_attr("z_steps1", lambda: 50_000)),
            "z_steps2": int(_fe_attr("z_steps2", lambda: 300_000)),
            "y_steps1": int(_fe_attr("y_steps1", lambda: 50_000)),
            "y_steps2": int(_fe_attr("y_steps2", lambda: 300_000)),
        }

        merged: dict[str, Any] = {
            **create_data,
            **fe_data,
            "n_steps_dict": n_steps_dict,
            "infe": infe_flag,
        }
        if partition:
            merged["partition"] = partition

        return cls(**merged)

    #model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_default=True)

    # --- Required / core ---
    system_name: str = Field(..., description="System name (required)")
    fe_type: Literal[
        "custom","rest","sdr","dd","sdr-rest","express","relative",
        "uno","uno_com","uno_rest","self","uno_dd","dd-rest","asfe","md"
    ] = Field(..., description="Free-energy protocol type")

    # --- Global switches ---
    dec_int: Literal["mbar", "ti"] = Field("mbar", description="Integration method (mbar/ti)")
    remd: Literal["yes", "no"] = Field("no", description="H-REMD toggle")
    partition: str = Field("owners", description="Cluster partition/queue")
    infe: bool = Field(False, description="Enable NFE (infinite) equilibration when true.")

    # --- Anchors / molecular definitions ---
    p1: str = Field("", description='Anchor P1 "RESID@ATOM" (e.g., "85@CA")')
    p2: str = Field("", description='Anchor P2 "RESID@ATOM"')
    p3: str = Field("", description='Anchor P3 "RESID@ATOM"')

    other_mol: List[str] = Field(default_factory=list, description="Other co-binders")
    lipid_mol: List[str] = Field(default_factory=list, description="Lipid molecules")
    solv_shell: Optional[float] = Field(15.0, description="Initial solvent shell radius (Å)")

    rocklin_correction: Literal["yes","no"] = Field("no", description="Rocklin correction")

    # --- FE controls / analysis ---
    release_eq: List[float] = Field(default_factory=list, description="Equilibration release weights (derived)")
    num_equil_extends: int = Field(0, description="Number of equilibration extends (derived)")
    ti_points: Optional[int] = Field(0, description="(#) TI points (not implemented)")
    lambdas: List[float] = Field(default_factory=list, description="default lambda values")
    component_windows: Dict[str, List[float]] = Field(default_factory=dict, description="Per-component lambda values for overrides")
    sdr_dist: Optional[float] = Field(0.0, description="SDR placement distance (Å)")
    dec_method: Optional[str] = Field(None, description="Decoupling method (set for fe_type='custom')")
    blocks: int = Field(0, description="MBAR blocks")
    unbound_threshold: float = Field(
        8.0,
        ge=0.0,
        description="Distance (Å) between ligand COMs that classifies equilibration as unbound.",
    )
    analysis_fe_range: Optional[Tuple[int, int]] = Field(
        (2, -1),
        description="Optional tuple (start, end) limiting FE simulations analyzed per window.",
    )

    # --- Force constants ---
    lig_distance_force: float = Field(0.0, description="Ligand COM distance spring (kcal/mol/Å^2)")
    lig_angle_force: float = Field(0.0, description="Ligand angle/dihedral spring (kcal/mol/rad^2)")
    lig_dihcf_force: float = Field(0.0, description="Ligand dihedral spring (kcal/mol/rad^2)")
    rec_com_force: float = Field(0.0, description="Protein COM spring")
    lig_com_force: float = Field(0.0, description="Ligand COM spring")

    # --- Solvent / box ---
    water_model: Literal["SPCE", "TIP4PEW", "TIP3P", "TIP3PF", "OPC"] = Field("TIP3P", description="Water model")
    buffer_x: float = Field(10.0, description="Box buffer X (Å)")
    buffer_y: float = Field(10.0, description="Box buffer Y (Å)")
    buffer_z: float = Field(10.0, description="Box buffer Z (Å)")
    lig_buffer: float = Field(10.0, description="Ligand box buffer (Å)")

    # --- Ions ---
    neutralize_only: Literal["yes","no"] = Field("no", description="Neutralize only")
    cation: str = Field("Na+", description="Cation species")
    anion: str = Field("Cl-", description="Anion species")
    ion_conc: float = Field(0.15, description="Target salt concentration (M)")

    # --- Simulation params ---
    hmr: Literal["yes","no"] = Field("no", description="Hydrogen mass repartitioning")
    enable_mcwat: Literal["yes","no"] = Field(
        "yes",
        description="Enable MC water exchange moves during equilibration templates.",
    )
    temperature: float = Field(310.0, description="Temperature (K)")
    eq_steps: int = Field(1_000_000, description="Steps per equilibration segment (derived)")
    eq_steps1: int = Field(500_000, description="Equilibration stage 1 steps (legacy mirror of eq_steps)")
    eq_steps2: int = Field(1_000_000, description="Equilibration stage 2 steps (legacy mirror of eq_steps)")
    n_steps_dict: Dict[str, int] = Field(
        default_factory=lambda: {
            f"{comp}_steps{ind}": 50_000 if ind == "1" else 1_000_000
            for comp in FEP_COMPONENTS for ind in ("1", "2")
        },
        description="Per-component steps (keys: '{comp}_steps1|2')"
    )

    # --- L1 search (optional) ---
    l1_x: Optional[float] = Field(None, description="L1 center offset X (Å)")
    l1_y: Optional[float] = Field(None, description="L1 center offset Y (Å)")
    l1_z: Optional[float] = Field(None, description="L1 center offset Z (Å)")
    l1_range: Optional[float] = Field(None, description="L1 search radius (Å)")
    min_adis: Optional[float] = Field(None, description="Min anchor distance (Å)")
    max_adis: Optional[float] = Field(None, description="Max anchor distance (Å)")

    # --- Amber i/o ---
    ntpr: int = Field(1000, description="Print energy every ntpr steps")
    ntwr: int = Field(10_000, description="Write restart every ntwr steps")
    ntwe: int = Field(0, description="Write energy every ntwe steps")
    ntwx: int = Field(2500, description="Write trajectory every ntwx steps")
    cut: float = Field(9.0, description="Nonbonded cutoff (Å)")
    gamma_ln: float = Field(1.0, description="Langevin γ (ps^-1)")
    barostat: Literal[1, 2] = Field(2, description="1=Berendsen, 2=MC barostat")
    dt: float = Field(0.004, description="Time step (ps)")
    num_fe_extends: int = Field(10, description="# restarts per λ")
    all_atoms: Literal["yes","no"] = Field("no", description="save all atoms for FE")

    # --- Force fields ---
    receptor_ff: str = Field("protein.ff14SB", description="Receptor FF")
    ligand_ff: str = Field("gaff2", description="Ligand FF")
    lipid_ff: str = Field("lipid21", description="Lipid FF")

    # --- Derived/public state (not user-set) ---
    ligand_dict: Dict[str, Any] = Field(default_factory=dict, description="Ligand dictionary")
    rng: int = Field(0, description="Range of release_eq")
    ion_def: List[Any] = Field(default_factory=list, description="Ion tuple [cation, anion, conc]")
    dic_steps1: Dict[str, int] = Field(default_factory=dict, description="Stage1 steps per component")
    dic_steps2: Dict[str, int] = Field(default_factory=dict, description="Stage2 steps per component")
    rest: List[float] = Field(default_factory=list, description="Packed restraint constants")
    neut: str = Field("", description="Alias of neutralize_only")
    protein_align: str = Field("name CA", description="Alignment selection")
    receptor_segment: Optional[str] = Field(None, description="Segment to embed in membrane")

    # --- Private/internal runtime ---
    components: List[str] = Field(default_factory=list, description="List of components (v, o, z, etc.)")
    component_lambdas: Dict[str, List[float]] = Field(default_factory=dict, description="Lambda schedule for each component")
    membrane_simulation: bool = Field(default=True, description="Whether system includes a membrane")

    # ---------------- validators / coercers ----------------
    @field_validator("fe_type", "dec_int", mode="before")
    @classmethod
    def _lower_enums(cls, v: Any) -> Any:
        return v if v is None else str(v).lower()

    @field_validator("remd", "neutralize_only", "hmr", "rocklin_correction", "enable_mcwat", mode="before")
    @classmethod
    def _coerce_yes_no(cls, v: Any) -> str | None:
        if v is None: return None
        if isinstance(v, bool): return "yes" if v else "no"
        if isinstance(v, (int, float)): return "yes" if v else "no"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"yes","no"}: return s
            if s in {"true","t","1"}: return "yes"
            if s in {"false","f","0"}: return "no"
        raise ValueError(f"Invalid yes/no: {v!r}")

    @field_validator("lambdas", mode="before")
    @classmethod
    def _parse_lambdas(cls, v: Any) -> Any:
        """
        Accept a YAML list, or a single space/comma-separated string.
        """
        if v is None:
            return []
        if isinstance(v, str):
            # split on commas or whitespace
            parts = [p for p in re.split(r"[,\s]+", v.strip()) if p]
            return [float(x) for x in parts]
        return v

    @field_validator("barostat", mode="before")
    @classmethod
    def _coerce_barostat(cls, v: Any) -> Any:
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return v
            try:
                return int(text)
            except ValueError:
                return v
        if isinstance(v, float):
            return int(v)
        return v

    @field_validator("p1", "p2", "p3")
    @classmethod
    def _validate_anchor(cls, v: str) -> str:
        if not v:
            return v
        if not _ANCHOR_RE.match(v):
            raise ValueError(f"Anchor must look like ':85@CA' (got {v!r})")
        return v

    @model_validator(mode="after")
    def _finalize(self) -> "SimulationConfig":
        # REMD not implemented
        if self.remd == "yes":
            raise NotImplementedError("REMD not implemented; set remd to 'no'.")
        # TI not implemented
        if self.dec_int == "ti":
            raise NotImplementedError("TI integration not implemented; use 'mbar'.")

        # derived fields
        self.rng = len(self.release_eq) - 1
        self.ion_def = [self.cation, self.anion, self.ion_conc]
        self.neut = self.neutralize_only

        # stage dicts (copy from n_steps_dict only for ACTIVE components)
        self.dic_steps1.clear()
        self.dic_steps2.clear()
        for comp in FEP_COMPONENTS:
            k1, k2 = f"{comp}_steps1", f"{comp}_steps2"
            self.dic_steps1[comp] = int(self.n_steps_dict.get(k1, 0))
            self.dic_steps2[comp] = int(self.n_steps_dict.get(k2, 0))

        # pack restraints (order-sensitive, matches legacy)
        self.rest = [
            0, 0,
            self.lig_distance_force, self.lig_angle_force, self.lig_dihcf_force,
            self.rec_com_force, self.lig_com_force,
        ]

        # friendly notices
        if self.buffer_z == 0:
            logger.debug("buffer_z=0; automatic Z buffer will be applied for membranes.")

        # Set components/dec_method by fe_type
        match self.fe_type:
            case "custom":
                if self.dec_method is None:
                    raise ValueError("For fe_type='custom', set dec_method to one of: dd, sdr, exchange.")
                self.components = []
            case "rest":
                self.components, self.dec_method = ['c', 'a', 'l', 't', 'r'], "dd"
            case "sdr":
                self.components, self.dec_method = ['e', 'v'], "sdr"
            case "dd":
                self.components, self.dec_method = ['e', 'v', 'f', 'w'], "dd"
            case "sdr-rest":
                self.components, self.dec_method = ['c', 'a', 'l', 't', 'r', 'e', 'v'], "sdr"
            case "express":
                self.components, self.dec_method = ['m', 'n', 'e', 'v'], "sdr"
            case "dd-rest":
                self.components, self.dec_method = ['c', 'a', 'l', 't', 'r', 'e', 'v', 'f', 'w'], "dd"
            case "relative":
                self.components, self.dec_method = ['x', 'e', 'n', 'm'], "exchange"
            case "uno":
                self.components, self.dec_method = ['m', 'n', 'o'], "sdr"
            case "uno_rest":
                self.components, self.dec_method = ['z'], "sdr"
            case "uno_com":
                self.components, self.dec_method = ['o'], "sdr"
            case "self":
                self.components, self.dec_method = ['s'], "sdr"
            case "uno_dd":
                self.components, self.dec_method = ['z', 'y'], "dd"
            case "asfe":
                self.components, self.dec_method = ['y', 'm'], "sdr"
            case "md":
                self.components, self.dec_method = [], None

        # sanity checks for active components only
        for comp in self.components:
            s1, s2 = self.dic_steps1.get(comp, 0), self.dic_steps2.get(comp, 0)
            if s1 <= 0:
                raise ValueError(f"{comp}: stage 1 steps must be > 0 (key '{comp}_steps1').")
            if s2 <= 0:
                raise ValueError(f"{comp}: stage 2 steps must be > 0 (key '{comp}_steps2').")

        # update per-component lambdas
        self.component_lambdas.clear()
        for comp in self.components:
            lambdas = self.component_windows.get(comp) or []
            if not lambdas:
                lambdas = self.lambdas
                if not lambdas:
                    raise ValueError(f"No lambdas defined for component '{comp}'.")
                logger.debug(f"No per-component lambdas for '{comp}'; using default lambdas.")
            self.component_lambdas[comp] = lambdas

        # membrane simulation if lipids defined
        self.membrane_simulation = len(self.lipid_mol) > 0
        if self.membrane_simulation:
            self._check_membrane_compatibility()
        else:
            self._check_water_compatibility()

        return self

    def _check_membrane_compatibility(self) -> None:
        pass

    def _check_water_compatibility(self) -> None:
        # make sure buffer_x/y/z is > 5.0 Å
        for dim, buf in zip(("X","Y","Z"), (self.buffer_x, self.buffer_y, self.buffer_z)):
            if buf < 15.0:
                raise ValueError(f"For water simulations, buffer_{dim.lower()} must be >= 15.0 Å (got {buf}).")

    # convenience
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
