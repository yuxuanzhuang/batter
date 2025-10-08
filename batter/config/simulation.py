from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator, model_validator
import re
from loguru import logger
from batter.utils import COMPONENTS_LAMBDA_DICT

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())
_ANCHOR_RE = re.compile(r"^\d+@[A-Za-z0-9]+$")


class SimulationConfig(BaseModel):
    """
    Simulation configuration for ABFE/ASFE workflows.

    Parameters
    ----------
    system_name : str
        Human-readable system identifier (required).
    fe_type : {"custom","rest","sdr","dd","sdr-rest","express","relative",
               "uno","uno_com","uno_rest","self","uno_dd","dd-rest","asfe"}
        Free-energy protocol flavor.
    dec_int : {"mbar","ti"}
        Integration method. ``"ti"`` is not implemented yet and will raise.
    remd : {"yes","no"}
        H-REMD toggle.
    partition : str
        Cluster partition/queue name.

    Notes
    -----
    - Boolean switches accept ``"yes"/"no"`` or YAML booleans (coerced).
    - Internal `components` is computed from `fe_type` and exposed read-only.
    """

    model_config = ConfigDict(
            extra="ignore",
            populate_by_name=True, validate_default=True)

    # --- Required / core ---
    system_name: str = Field(..., description="System name (required)")
    fe_type: Literal[
        "custom","rest","sdr","dd","sdr-rest","express","relative",
        "uno","uno_com","uno_rest","self","uno_dd","dd-rest","asfe"
    ] = Field(..., description="Free energy protocol type")

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

    rocklin_correction: str = Field("no", description="Rocklin correction (yes/no)")

    # --- FE controls / analysis ---
    release_eq: List[float] = Field(default_factory=list, description="Attach/release weights")
    attach_rest: List[float] = Field(default_factory=list, description="Attach/restraint weights")
    ti_points: Optional[int] = Field(0, description="# of TI points if dec_int='ti' (not implemented)")
    lambdas: List[float] = Field(default_factory=list, description="Lambda values (reserved)")
    sdr_dist: Optional[float] = Field(0.0, description="SDR placement distance (Å)")
    dec_method: Optional[str] = Field(
        None, description="Decoupling method; only user-settable when fe_type='custom'"
    )
    blocks: int = Field(0, description="MBAR blocks")

    # --- Force constants (kcal/mol, Å, rad units as noted) ---
    rec_dihcf_force: float = Field(0.0, description="Protein dihedral spring (kcal/mol/rad^2)")
    rec_discf_force: float = Field(0.0, description="Protein distance spring (kcal/mol/Å^2)")
    lig_distance_force: float = Field(0.0, description="Ligand COM distance spring (kcal/mol/Å^2)")
    lig_angle_force: float = Field(0.0, description="Ligand angle/dihedral spring (kcal/mol/rad^2)")
    lig_dihcf_force: float = Field(0.0, description="Ligand dihedral spring (kcal/mol/rad^2)")
    rec_com_force: float = Field(0.0, description="Protein COM spring")
    lig_com_force: float = Field(0.0, description="Ligand COM spring")

    # --- Solvent / box ---
    water_model: Literal["SPCE", "TIP4PEW", "TIP3P", "TIP3PF", "OPC"] = Field("TIP3P", description="Water model")
    num_waters: int = Field(0, description="[DEPRECATED] Must remain 0 (automatic sizing)")
    buffer_x: float = Field(0.0, description="Box buffer X (Å)")
    buffer_y: float = Field(0.0, description="Box buffer Y (Å)")
    buffer_z: float = Field(0.0, description="Box buffer Z (Å)")
    lig_buffer: float = Field(0.0, description="Ligand box buffer (Å)")

    # --- Ions ---
    neutralize_only: str = Field("no", description="Neutralize only (yes/no)")
    cation: str = Field("Na+", description="Cation species")
    anion: str = Field("Cl-", description="Anion species")
    ion_conc: float = Field(0.15, description="Target salt concentration (M)")

    # --- Simulation params ---
    hmr: str = Field("no", description="Hydrogen mass repartitioning (yes/no)")
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
    dlambda: float = Field(0.001, description="Δλ for splitting initial λ into two close windows")

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

    # --- Force fields ---
    receptor_ff: str = Field("ff14SB", description="Receptor FF")
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
    _components: List[str] = PrivateAttr(default_factory=list)
    _membrane_simulation: bool = PrivateAttr(True)

    # ---------------- properties ----------------

    @property
    def components(self) -> tuple[str, ...]:
        """Read-only list of active components derived from `fe_type`."""
        return tuple(self._components)

    @property
    def H1(self) -> str: return self.p1
    @property
    def H2(self) -> str: return self.p2
    @property
    def H3(self) -> str: return self.p3

    # ---------------- validators ----------------

    @field_validator("neutralize_only", "hmr", "rocklin_correction", mode="before")
    @classmethod
    def _coerce_yes_no(cls, v: Any) -> str | None:
        if v is None: return None
        if isinstance(v, bool): return "yes" if v else "no"
        if isinstance(v, (int, float)): return "yes" if v else "no"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"yes", "no"}: return s
            if s in {"true","t","1"}: return "yes"
            if s in {"false","f","0"}: return "no"
        raise ValueError(f"Invalid yes/no: {v!r}")

    @field_validator("p1", "p2", "p3")
    @classmethod
    def _validate_anchor(cls, v: str) -> str:
        if not v: return v  # allow empty
        if not _ANCHOR_RE.match(v):
            raise ValueError(f"Anchor must look like '85@CA' (got {v!r})")
        return v

    @model_validator(mode="after")
    def _finalize(self) -> "SimulationConfig":
        # TI not implemented
        if self.dec_int == "ti":
            raise NotImplementedError("TI integration not implemented; use 'mbar'.")

        # derived fields
        self.rng = len(self.release_eq) - 1
        self.ion_def = [self.cation, self.anion, self.ion_conc]
        self.neut = self.neutralize_only

        # stage dicts
        for comp in FEP_COMPONENTS:
            self.dic_steps1[comp] = self.n_steps_dict.get(f"{comp}_steps1", 0)
            self.dic_steps2[comp] = self.n_steps_dict.get(f"{comp}_steps2", 0)

        # pack restraints
        self.rest = [
            self.rec_dihcf_force, self.rec_discf_force,
            self.lig_distance_force, self.lig_angle_force, self.lig_dihcf_force,
            self.rec_com_force, self.lig_com_force,
        ]

        # friendly notices
        if self.buffer_z == 0:
            logger.info("buffer_z=0; automatic Z buffer will be applied for membranes.")
        if self.num_waters != 0:
            raise ValueError("'num_waters' is deprecated and must remain 0.")

        match self.fe_type:
            case "custom":
                if self.dec_method is None:
                    raise ValueError("For fe_type='custom', set dec_method to one of: dd, sdr, exchange.")
                # leave components empty; user/tools must set internally
            case "rest":
                self._components, self.dec_method = ['c', 'a', 'l', 't', 'r'], "dd"
            case "sdr":
                self._components, self.dec_method = ['e', 'v'], "sdr"
            case "dd":
                self._components, self.dec_method = ['e', 'v', 'f', 'w'], "dd"
            case "sdr-rest":
                self._components, self.dec_method = ['c', 'a', 'l', 't', 'r', 'e', 'v'], "sdr"
            case "express":
                self._components, self.dec_method = ['m', 'n', 'e', 'v'], "sdr"
            case "dd-rest":
                self._components, self.dec_method = ['c', 'a', 'l', 't', 'r', 'e', 'v', 'f', 'w'], "dd"
            case "relative":
                self._components, self.dec_method = ['x', 'e', 'n', 'm'], "exchange"
            case "uno":
                self._components, self.dec_method = ['m', 'n', 'o'], "sdr"
            case "uno_rest":
                self._components, self.dec_method = ['z'], "sdr"
            case "uno_com":
                self._components, self.dec_method = ['o'], "sdr"
            case "self":
                self._components, self.dec_method = ['s'], "sdr"
            case "uno_dd":
                self._components, self.dec_method = ['z', 'y'], "dd"
            case "asfe":
                self._components, self.dec_method = ['y'], "sdr"

        # sanity checks for active components
        for comp in self._components:
            s1, s2 = self.dic_steps1.get(comp, 0), self.dic_steps2.get(comp, 0)
            if s1 <= 0:
                raise ValueError(f"{comp}: stage 1 steps must be > 0 (key '{comp}_steps1').")
            if s2 <= 0:
                raise ValueError(f"{comp}: stage 2 steps must be > 0 (key '{comp}_steps2').")

        return self

    # convenience
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()