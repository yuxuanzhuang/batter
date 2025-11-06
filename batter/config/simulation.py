from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator, model_validator
import re
from loguru import logger
from batter.utils import COMPONENTS_LAMBDA_DICT
from batter.config.utils import coerce_yes_no

if TYPE_CHECKING:
    from batter.config.run import CreateArgs, FESimArgs

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())
_ANCHOR_RE = re.compile(r"^:?\d+@[\w\d]+$")  # e.g., ":85@CA" or "85@CA"

MEMBRANE_EXEMPT_COMPONENTS = {"y"}

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
    ) -> "SimulationConfig":
        create_data: dict[str, Any] = {
            "system_name": getattr(create, "system_name", "unnamed_system") or "unnamed_system",
            "receptor_ff": getattr(create, "receptor_ff", "protein.ff14SB"),
            "ligand_ff": getattr(create, "ligand_ff", "gaff2"),
            "lipid_ff": getattr(create, "lipid_ff", "lipid21"),
            "lipid_mol": list(getattr(create, "lipid_mol", []) or []),
            "other_mol": list(getattr(create, "other_mol", []) or []),
            "water_model": getattr(create, "water_model", "TIP3P"),
            "neutralize_only": coerce_yes_no(getattr(create, "neutralize_only", "no")),
            "ion_conc": float(getattr(create, "ion_conc", 0.15)),
            "cation": getattr(create, "cation", "Na+"),
            "anion": getattr(create, "anion", "Cl-"),
            "solv_shell": float(getattr(create, "solv_shell", 15.0)),
            "protein_align": getattr(create, "protein_align", "name CA"),
            "l1_range": float(getattr(create, "l1_range", 6.0)),
            "min_adis": float(getattr(create, "min_adis", 3.0)),
            "max_adis": float(getattr(create, "max_adis", 7.0)),
        }

        release_eq_value = getattr(fe, "release_eq", None)
        fe_release_eq = release_eq_value if release_eq_value is not None else [0]

        fe_data: dict[str, Any] = {
            "fe_type": getattr(fe, "fe_type", "uno_rest"),
            "dec_int": getattr(fe, "dec_int", "mbar"),
            "remd": coerce_yes_no(getattr(fe, "remd", "no")),
            "rocklin_correction": coerce_yes_no(getattr(fe, "rocklin_correction", "no")),
            "lambdas": list(getattr(fe, "lambdas", []) or []),
            "sdr_dist": float(getattr(fe, "sdr_dist", 0.0)),
            "blocks": int(getattr(fe, "blocks", 0)),
            "lig_buffer": float(getattr(fe, "lig_buffer", 0.0)),
            "lig_distance_force": float(getattr(fe, "lig_distance_force", 5.0)),
            "lig_angle_force": float(getattr(fe, "lig_angle_force", 250.0)),
            "lig_dihcf_force": float(getattr(fe, "lig_dihcf_force", 0.0)),
            "rec_com_force": float(getattr(fe, "rec_com_force", 10.0)),
            "lig_com_force": float(getattr(fe, "lig_com_force", 10.0)),
            "buffer_x": float(getattr(fe, "buffer_x", 0.0)),
            "buffer_y": float(getattr(fe, "buffer_y", 0.0)),
            "buffer_z": float(getattr(fe, "buffer_z", 0.0)),
            "temperature": float(getattr(fe, "temperature", 310.0)),
            "dt": float(getattr(fe, "dt", 0.004)),
            "hmr": coerce_yes_no(getattr(fe, "hmr", "no")),
            "release_eq": list(fe_release_eq),
            "eq_steps1": int(getattr(fe, "eq_steps1", 500_000)),
            "eq_steps2": int(getattr(fe, "eq_steps2", 1_000_000)),
            "ntpr": int(getattr(fe, "ntpr", 1000)),
            "ntwr": int(getattr(fe, "ntwr", 10_000)),
            "ntwe": int(getattr(fe, "ntwe", 0)),
            "ntwx": int(getattr(fe, "ntwx", 2_500)),
            "cut": float(getattr(fe, "cut", 9.0)),
            "gamma_ln": float(getattr(fe, "gamma_ln", 1.0)),
            "barostat": int(getattr(fe, "barostat", 2)),
        }

        n_steps_dict = {
            "z_steps1": int(getattr(fe, "z_steps1", 50_000)),
            "z_steps2": int(getattr(fe, "z_steps2", 300_000)),
            "y_steps1": int(getattr(fe, "y_steps1", 50_000)),
            "y_steps2": int(getattr(fe, "y_steps2", 300_000)),
        }

        merged: dict[str, Any] = {**create_data, **fe_data, "n_steps_dict": n_steps_dict}
        if partition:
            merged["partition"] = partition

        return cls(**merged)

    #model_config = ConfigDict(extra="ignore", populate_by_name=True, validate_default=True)

    # --- Required / core ---
    system_name: str = Field(..., description="System name (required)")
    fe_type: Literal[
        "custom","rest","sdr","dd","sdr-rest","express","relative",
        "uno","uno_com","uno_rest","self","uno_dd","dd-rest","asfe"
    ] = Field(..., description="Free-energy protocol type")

    # --- Global switches ---
    dec_int: Literal["mbar", "ti"] = Field("mbar", description="Integration method (mbar/ti)")
    remd: Literal["yes", "no"] = Field("no", description="H-REMD toggle")
    partition: str = Field("owners", description="Cluster partition/queue")

    # --- Anchors / molecular definitions ---
    p1: str = Field("", description='Anchor P1 "RESID@ATOM" (e.g., "85@CA")')
    p2: str = Field("", description='Anchor P2 "RESID@ATOM"')
    p3: str = Field("", description='Anchor P3 "RESID@ATOM"')

    other_mol: List[str] = Field(default_factory=list, description="Other co-binders")
    lipid_mol: List[str] = Field(default_factory=list, description="Lipid molecules")
    solv_shell: Optional[float] = Field(15.0, description="Initial solvent shell radius (Å)")

    rocklin_correction: Literal["yes","no"] = Field("no", description="Rocklin correction")

    # --- FE controls / analysis ---
    release_eq: List[float] = Field(default_factory=list, description="Equilibration release weights")
    ti_points: Optional[int] = Field(0, description="(#) TI points (not implemented)")
    lambdas: List[float] = Field(default_factory=list, description="default lambda values")
    component_windows: Dict[str, List[float]] = Field(default_factory=dict, description="Per-component lambda values for overrides")
    sdr_dist: Optional[float] = Field(0.0, description="SDR placement distance (Å)")
    dec_method: Optional[str] = Field(None, description="Decoupling method (set for fe_type='custom')")
    blocks: int = Field(0, description="MBAR blocks")

    # --- Force constants ---
    lig_distance_force: float = Field(0.0, description="Ligand COM distance spring (kcal/mol/Å^2)")
    lig_angle_force: float = Field(0.0, description="Ligand angle/dihedral spring (kcal/mol/rad^2)")
    lig_dihcf_force: float = Field(0.0, description="Ligand dihedral spring (kcal/mol/rad^2)")
    rec_com_force: float = Field(0.0, description="Protein COM spring")
    lig_com_force: float = Field(0.0, description="Ligand COM spring")

    # --- Solvent / box ---
    water_model: Literal["SPCE", "TIP4PEW", "TIP3P", "TIP3PF", "OPC"] = Field("TIP3P", description="Water model")
    num_waters: int = Field(0, description="[DEPRECATED] Must remain 0 (automatic sizing)")
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
    temperature: float = Field(310.0, description="Temperature (K)")
    eq_steps1: int = Field(500_000, description="Equilibration stage 1 steps")
    eq_steps2: int = Field(1_000_000, description="Equilibration stage 2 steps")
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
    num_fe_range: int = Field(10, description="# restarts per λ")
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

    @field_validator("remd", "neutralize_only", "hmr", "rocklin_correction", mode="before")
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
        if self.num_waters != 0:
            raise ValueError("'num_waters' is deprecated and must remain 0.")

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
                self.components, self.dec_method = ['y'], "sdr"

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
            if buf < 5.0:
                raise ValueError(f"For water simulations, buffer_{dim.lower()} must be >= 5.0 Å (got {buf}).")

    # convenience
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
