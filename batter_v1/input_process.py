from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
    PrivateAttr,
)
from typing import List, Optional, Dict, Union, Any, Literal, Sequence
import re
import numpy as np
from loguru import logger
from batter_v1.utils import COMPONENTS_LAMBDA_DICT

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())

_ANCHOR_RE = re.compile(r"^\d+@[A-Za-z0-9]+$")


class SimulationConfig(BaseModel):
    """
    Configuration for a molecular simulation.

    Notes
    -----
    - Internal fields like :pyattr:`components` are computed and read-only.
    - Boolean flags accept ``'yes'/'no'`` *or* YAML booleans (``true``/``false``).
    - ``num_waters`` is deprecated and must remain 0.
    """
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_default=True,
    )

    # --- Required ---
    system_name: str = Field(..., description="System name (required)")
    fe_type: Literal[
        "custom", "rest", "sdr", "dd", "sdr-rest", "express", "relative",
        "uno", "uno_com", "uno_rest", "self", "uno_dd", "dd-rest", "asfe"
    ] = Field(..., description="Free energy type")

    # --- Global switches ---
    dec_int: Literal["mbar", "ti"] = Field("mbar", description="Free-energy integration method (mbar/ti)")
    remd: Literal["yes", "no"] = Field("no", description="H-REMD (yes/no)")
    partition: str = Field("owners", description="Cluster partition/queue to submit jobs")

    # --- Molecular definitions ---
    p1: str = Field("", description='Protein anchor P1 (format "RESID@ATOM", e.g., "85@CA")')
    p2: str = Field("", description='Protein anchor P2 (format "RESID@ATOM")')
    p3: str = Field("", description='Protein anchor P3 (format "RESID@ATOM")')

    other_mol: List[str] = Field(default_factory=list, description="Other co-binding molecules")
    lipid_mol: List[str] = Field(default_factory=list, description="Lipid molecules")
    solv_shell: Optional[float] = Field(
        15.0,
        description="Keep solvent within this Å of the protein in the initial structure (None disables)"
    )
    rocklin_correction: str = Field("no", description="Apply Rocklin correction for charged ligands (yes/no)")

    # --- FE controls / analysis ---
    release_eq: List[float] = Field(default_factory=list, description="Attach/release weights (short)")
    attach_rest: List[float] = Field(default_factory=list, description="Attach/restraint weights (short)")
    ti_points: Optional[int] = Field(0, description="# of TI quadrature points (if dec_int='ti')")
    lambdas: List[float] = Field(default_factory=list, description="Lambda values (filled when TI is implemented)")
    sdr_dist: Optional[float] = Field(0.0, description="SDR placement distance (Å)")
    # user may specify dec_method ONLY for fe_type='custom'
    dec_method: Optional[Literal["dd", "sdr", "exchange"]] = Field(
        None, description="Decoupling method; only user-settable when fe_type='custom'"
    )
    blocks: int = Field(0, description="Number of blocks for MBAR")

    # --- Force constants ---
    rec_dihcf_force: float = Field(0.0, description="Protein dihedral spring (kcal/mol/rad^2)")
    rec_discf_force: float = Field(0.0, description="Protein distance spring (kcal/mol/Å^2)")
    lig_distance_force: float = Field(0.0, description="Ligand COM distance spring (kcal/mol/Å^2)")
    lig_angle_force: float = Field(0.0, description="Ligand angle/dihedral spring (kcal/mol/rad^2)")
    lig_dihcf_force: float = Field(0.0, description="Ligand dihedral spring (kcal/mol/rad^2)")
    rec_com_force: float = Field(0.0, description="Protein COM spring")
    lig_com_force: float = Field(0.0, description="Ligand COM spring (for simultaneous decoupling)")

    # --- Solvent / box ---
    water_model: Literal["SPCE", "TIP4PEW", "TIP3P", "TIP3PF", "OPC"] = Field(
        "TIP3P", description="Water model"
    )
    num_waters: int = Field(0, description="[DEPRECATED] Must remain 0 (automatic sizing used)")
    buffer_x: float = Field(0.0, description="Box buffer (Å) along X (ignored for membranes)")
    buffer_y: float = Field(0.0, description="Box buffer (Å) along Y (ignored for membranes)")
    buffer_z: float = Field(0.0, description="Box buffer (Å) along Z")
    lig_buffer: float = Field(0.0, description="Buffer around ligand box (Å)")

    # --- Ions ---
    neutralize_only: str = Field("no", description="Neutralize only, or also ionize to target concentration (yes/no)")
    cation: str = Field("Na+", description="Cation species")
    anion: str = Field("Cl-", description="Anion species")
    ion_conc: float = Field(0.15, description="Target salt concentration (M)")

    # --- Simulation params ---
    hmr: str = Field("no", description="Hydrogen mass repartitioning (yes/no)")
    temperature: float = Field(298.15, description="Temperature (K)")
    eq_steps1: int = Field(500_000, description="Equilibration stage 1 steps")
    eq_steps2: int = Field(1_000_000, description="Equilibration stage 2 steps")
    n_steps_dict: Dict[str, int] = Field(
        default_factory=lambda: {
            f"{comp}_steps{ind}": 50_000 if ind == "1" else 1_000_000
            for comp in FEP_COMPONENTS for ind in ("1", "2")
        },
        description="Per-component steps (AMBER/eq), keys: '{comp}_steps1|2'"
    )

    # --- Ligand anchor search ---
    l1_x: Optional[float] = Field(None, description="L1 search center offset X (Å)")
    l1_y: Optional[float] = Field(None, description="L1 search center offset Y (Å)")
    l1_z: Optional[float] = Field(None, description="L1 search center offset Z (Å)")
    l1_range: Optional[float] = Field(None, description="L1 search radius (Å)")
    min_adis: Optional[float] = Field(None, description="Minimum anchor distance (Å)")
    max_adis: Optional[float] = Field(None, description="Maximum anchor distance (Å)")
    dlambda: float = Field(0.001, description="Δλ to split initial λ into two close windows")

    # --- Amber i/o ---
    ntpr: int = Field(1000, description="Print energy every ntpr steps")
    ntwr: int = Field(10_000, description="Write restart every ntwr steps")
    ntwe: int = Field(0, description="Write energy every ntwe steps")
    ntwx: int = Field(2500, description="Write trajectory every ntwx steps")
    cut: float = Field(9.0, description="Nonbonded cutoff (Å)")
    gamma_ln: float = Field(1.0, description="Langevin γ (ps^-1)")
    barostat: int = Field(2, description="1=Berendsen, 2=MC barostat")
    dt: float = Field(0.004, description="Time step (ps)")
    num_fe_range: int = Field(
        10,
        description="# restarts per λ; total steps = num_fe_range × n_steps"
    )

    # --- Force fields ---
    receptor_ff: str = Field("ff14SB", description="Receptor force field")
    ligand_ff: str = Field("gaff2", description="Ligand force field")
    lipid_ff: str = Field("lipid21", description="Lipid force field")

    # --- Internal / derived (public state) ---
    ligand_dict: Dict[str, Any] = Field(default_factory=dict, description="Ligand dictionary")
    rng: int = Field(0, description="Range of release_eq")
    ion_def: List[Any] = Field(default_factory=list, description="Ion tuple [cation, anion, conc]")
    dic_steps1: Dict[str, int] = Field(default_factory=dict, description="Steps for stage 1 per component")
    dic_steps2: Dict[str, int] = Field(default_factory=dict, description="Steps for stage 2 per component")
    rest: List[float] = Field(default_factory=list, description="Restraint constants packing list")
    neut: str = Field("", description="Alias of neutralize_only (for legacy paths)")
    protein_align: str = Field("name CA", description="Selection used for alignment")
    receptor_segment: Optional[str] = Field(None, description="Segment to embed in membrane")
    poses_list: List[str] = Field(default_factory=list, description="Segment to embed in membrane")

    # --- Internal-only runtime/private ---
    _components: List[str] = PrivateAttr(default_factory=list)
    _membrane_simulation: bool = PrivateAttr(True)

    # ------------------- Read-only facade -------------------

    @property
    def components(self) -> tuple[str, ...]:
        """Computed component list (read-only)."""
        return tuple(self._components)

    @property
    def H1(self) -> str: return self.p1
    @property
    def H2(self) -> str: return self.p2
    @property
    def H3(self) -> str: return self.p3

    # ------------------- Validators -------------------

    @field_validator("neutralize_only", "hmr", "rocklin_correction", mode="before")
    @classmethod
    def _coerce_yes_no(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return "yes" if value else "no"
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"yes", "no"}: return v
            if v in {"true", "t", "1"}: return "yes"
            if v in {"false", "f", "0"}: return "no"
        raise ValueError(f"Invalid value: {value}. Must be 'yes' or 'no'.")

    @field_validator("dec_method", mode="before")
    @classmethod
    def _gate_dec_method(cls, v: Optional[str], info) -> Optional[str]:
        # Allow user to pass dec_method only if fe_type is 'custom' (checked after init).
        # We can't see fe_type here reliably; the after-model validator enforces it.
        return v

    # ------------------- Post-init normalization -------------------

    @model_validator(mode="after")
    def _finalize(self) -> "SimulationConfig":
        # 1) TI handling
        if self.dec_int == "ti":
            raise NotImplementedError("TI integration scheme is not implemented yet; use 'mbar'.")

        # 2) Derived small fields
        self.rng = len(self.release_eq) - 1
        self.ion_def = [self.cation, self.anion, self.ion_conc]
        self.neut = self.neutralize_only

        # 3) Copy n_steps_dict into stage dicts (ensure keys exist)
        for comp in FEP_COMPONENTS:
            s1 = self.n_steps_dict.get(f"{comp}_steps1", 0)
            s2 = self.n_steps_dict.get(f"{comp}_steps2", 0)
            self.dic_steps1[comp] = s1
            self.dic_steps2[comp] = s2

        # 4) Pack restraint constants
        self.rest = [
            self.rec_dihcf_force, self.rec_discf_force,
            self.lig_distance_force, self.lig_angle_force, self.lig_dihcf_force,
            self.rec_com_force, self.lig_com_force,
        ]

        # 5) Hints and deprecations
        if self.buffer_z == 0:
            logger.info("buffer_z is 0; automatic buffer will be applied (membrane setups handle Z).")
        if self.num_waters != 0:
            raise ValueError("'num_waters' is deprecated and must remain 0 (automatic solvent sizing).")

        # 6) Resolve components + dec_method by fe_type
        if self.fe_type != "custom" and self.dec_method is not None:
            raise ValueError("`dec_method` may only be set by users when fe_type='custom'.")

        match self.fe_type:
            case "custom":
                if self.dec_method is None:
                    raise ValueError("For fe_type='custom' please set dec_method to one of: dd, sdr, exchange.")
                # For 'custom' you may optionally pre-populate _components elsewhere.
                # If you still want to force a set, do it here.
                if not self._components:
                    logger.warning("fe_type='custom' with empty components; set internally before running.")
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
                self._components, self.dec_method = ['y'], "sdr"  # dec_method not used, default to sdr

        # 7) Sanity checks for active components
        for comp in self._components:
            s1 = self.dic_steps1.get(comp, 0)
            s2 = self.dic_steps2.get(comp, 0)
            logger.debug(f"Using component {comp}: stage1={s1}, stage2={s2}")
            if s1 <= 0:
                raise ValueError(f"{comp}: steps for stage 1 must be > 0 (key '{comp}_steps1').")
            if s2 <= 0:
                raise ValueError(f"{comp}: steps for stage 2 must be > 0 (key '{comp}_steps2').")

        return self

    # ------------------- Utilities -------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict (private attrs excluded)."""
        return self.model_dump()


# ------------------- Legacy parser (deprecated) -------------------

def parse_input_file(input_file: str) -> dict:
    """
    DEPRECATED: Use YAML with a proper loader. This parser supports a simple
    'key = value' format and folds '*_steps{1,2}' into 'n_steps_dict'.
    """
    parameters: Dict[str, Any] = {}

    with open(input_file) as f_in:
        lines = (line.strip(' \t\n\r') for line in f_in)
        lines = [line for line in lines if line and not line.startswith('#')]

        for line in lines:
            if '=' not in line:
                raise ValueError(f"Invalid line: {line}")
            key, value = line.split('#')[0].split('=', 1)
            key = key.strip().lower()
            value = value.strip()

            if '[' in value and ']' in value:
                value = value.strip('\'\"-,.:;#()][')
                split_sep = ',' if key in {
                    'poses_list', 'ligand_list', 'other_mol', 'celpp_receptor', 'ligand_name', 'bb_start', 'bb_end'
                } else None
                try:
                    value = [v.strip() for v in (value.split(split_sep) if split_sep else value.split())]
                except Exception:
                    pass
            parameters[key] = value

    # merge FEP_COMPONENTS into dict
    n_steps_dict: Dict[str, int] = {}
    for comp in FEP_COMPONENTS:
        for ind in ('1', '2'):
            k = f'{comp}_steps{ind}'
            n_steps_dict[k] = int(parameters[k]) if k in parameters else 0
    parameters['n_steps_dict'] = n_steps_dict

    return parameters


def get_configure_from_file(file_path: str) -> SimulationConfig:
    """
    Parse a legacy text file and return a validated configuration.
    Prefer the YAML loader in new workflows.
    """
    raw_params = parse_input_file(file_path)
    return SimulationConfig(**raw_params)
