from pydantic import BaseModel, Field, model_validator, field_validator

import sys
from typing import List, Optional, Dict, Union, Any
import numpy as np
from loguru import logger
import pandas as pd
from batter.data import charmmlipid2amber
from batter.utils import COMPONENTS_LAMBDA_DICT

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())

class SimulationConfig(BaseModel):
    software: str = Field("amber", description="Software to use (amber, openmm)")
    
    # all deprecated
    # Calculation definitions
    calc_type: Optional[str] = None
    celpp_receptor: Optional[str] = None
    poses_list: Optional[List[str]] = None

    # Molecular definitions
    # Protein anchor
    p1: str = Field("", description="Protein anchor P1")
    p2: str = Field("", description="Protein anchor P2")
    p3: str = Field("", description="Protein anchor P3")

    ligand_list: List[str] = Field(default_factory=list, description="List of ligands")
    other_mol: List[str] = Field(default_factory=list, description="Other co-binding molecules")
    solv_shell: Optional[float] = Field(
        None, description="Water molecules around the protein that will be kept in the initial structure (in angstroms)"
    )

    # Variables for setting up equilibrium and free energy calculations, also used on analysis
    fe_type: str = Field(..., description="Free energy type (rest, dd, sdr, etc.)")
    remd: Optional[str] = Field('no', description="H-REMD (yes or no)")
    components: List[str] = Field(
        default_factory=list,
        description="Used with custom option for fe_type. Do not include b component here."
    )
    release_eq: List[float] = Field(default_factory=list, description="Short attach/release weights")
    attach_rest: List[float] = Field(default_factory=list, description="Short attach/release weights")
    ti_points: Optional[int] = Field(0, description="# of TI points for Gaussian quadrature")
    lambdas: List[float] = Field(default_factory=list, description="Lambda values for TI")
    sdr_dist: Optional[float] = Field(0, description="SDR distance to place the ligand")
    dec_method: Optional[str] = Field(None, description="Decoupling method, can be `dd` or `sdr`")

    # Additional variables for analysis
    dec_int: Optional[str] = Field("mbar", description="Decoupling integration method (mbar/ti)")
    blocks: Optional[int] = Field(0, description="Number of blocks for MBAR")

    # Force constants
    rec_dihcf_force: float = Field(
        0.0, description="Protein conformational dihedral spring constant - kcal/mol/rad**2"
    )
    rec_discf_force: float = Field(
        0.0, description="Protein conformational distance spring constant - kcal/mol/Angstrom**2"
    )
    lig_distance_force: float = Field(
        0.0, description="Guest pulling distance spring constant kcal/mol/Angstrom**2"
    )
    lig_angle_force: float = Field(0.0, description="Guest angle/dihedral spring constant - kcal/mol/rad**2")
    lig_dihcf_force: float = Field(
        0.0, description="Guest conformational dihedral spring constant - kcal/mol/rad**2"
    )
    rec_com_force: float = Field(0.0, description="Protein COM spring constant")
    lig_com_force: float = Field(0.0, description="Guest COM spring constant for simultaneous decoupling")

    # Water model, number and box size in the x and y direction
    water_model: str = Field("TIP3P", description="Water model (SPCE, TIP4PEW, TIP3P, TIP3PF or OPC)")
    num_waters: Optional[int] = Field(0, description="Number of water molecules in the system")
    
    buffer_x: Optional[float] = Field(
        0, description="Buffer size along X-axis; this will be omitted in membrane simulations"
    )
    buffer_y: Optional[float] = Field(
        0, description="Buffer size along Y-axis; this will be omitted in membrane simulations"
    )
    buffer_z: Optional[float] = Field(0, description="Buffer size along Z-axis")
    lig_buffer: Optional[float] = Field(0, description="Buffer size around the ligand box")

    # Counterions
    neutralize_only: str = Field("no", description="Neutralize only or also ionize (yes or no)")
    cation: str = Field("Na+", description="Cation")
    anion: str = Field("Cl-", description="Anion")
    ion_conc: Optional[float] = Field(0.15, description="Ionic concentration")

    # Simulation parameters
    hmr: str = Field("no", description="Apply hydorgen mass repartitioning (yes/no)")
    temperature: float = Field(..., description="Simulation temperature")
    # n_steps
    eq_steps1: int = Field(0, description="Number of steps for equilibration stage 1")
    eq_steps2: int = Field(0, description="Number of steps for equilibration stage 2")
    n_steps_dict: Dict[str, int] = Field(
        default_factory=lambda: {
            f"{comp}_steps{ind}": 0 for comp in FEP_COMPONENTS for ind in ["1", "2"]
        },
        description="Number of steps for each stage in AMBER and eq"
    )
    n_iter_dict: Dict[str, int] = Field(
        default_factory=lambda: {
            f"{comp}_itera{ind}": 0 for comp in FEP_COMPONENTS for ind in ["1", "2"]
        },
        description="Number of steps for each stage in OpenMM"
    )

    # Conformational restraints on the protein backbone
    rec_bb: str = Field("no", description="Use protein backbone dihedrals conformational restraints")
    bb_start: Optional[Union[List[int], int]] = Field(
        0,
        description="Start of the backbone section to restrain; can be a list or a comma-separated string"
    )
    bb_end: Optional[Union[List[int], int]] = Field(
        1,
        description="End of the backbone section to restrain; can be a list or a comma-separated string"
    )
    bb_equil: str = Field("no", description="Keep this backbone section rigid during equilibration")

    # Ligand anchor search definitions
    l1_x: Optional[float] = Field(None, description="X distance between P1 and center of L1 search range")
    l1_y: Optional[float] = Field(None, description="Y distance between P1 and center of L1 search range")
    l1_z: Optional[float] = Field(None, description="Z distance between P1 and center of L1 search range")
    l1_range: Optional[float] = Field(None, description="search radius for the first ligand anchor L1 ")
    min_adis: Optional[float] = Field(None, description="minimum distance between anchors")
    max_adis: Optional[float] = Field(None, description="maximum distance between anchors")
    dlambda: Optional[float] = Field(
        0.001, description="lambda width for splitting initial lambda into two close windows"
    )

    # Amber options for production simulations
    ntpr: str = Field('1000', description="print energy every ntpr steps to output file (controls DD output)")
    ntwr: str = Field('10000', description="write the restart file every ntwr steps")
    ntwe: str = Field('0', description="write the energy file every ntwe steps")
    ntwx: str = Field('2500', description="write the trajectory file every ntwx steps")
    cut: str = Field('9.0', description="nonbonded cutoff in Angstroms")
    gamma_ln: str = Field('1.0', description="collision frequency in ps^-1 for Langevin Dynamics (temperature control)")
    barostat: str = Field('2', description="type of barostat to keep the pressure constant (1 = Berendsen-default /2 - Monte Carlo)")
    dt: str = Field('0.004', description="time step in ps")
    num_fe_range: int = Field(10, description="Number of free energy simulations restarts to run for each lambda; total simulation steps will be num_fe_range * n_steps")

    # OpenMM specific options for production simulations
    itcheck: str = Field('100', description="write checkpoint file every itcheck iterations")

    # Internal usage
    weights: List[float] = Field(default_factory=list, description="Gaussian quadrature weights for TI")
    mols: List[str] = Field(default_factory=list, description="Molecules")
    rng: int = Field(0, description="Range of release_eq")
    ion_def: List[Any] = Field(default_factory=list, description="Ion definition")
    poses_def: List[str] = Field(default_factory=list, description="Poses definition")
    dic_steps1: Dict[str, int] = Field(default_factory=dict, description="Steps dictionary for stage 1")
    dic_steps2: Dict[str, int] = Field(default_factory=dict, description="Steps dictionary for stage 2")
    dic_itera1: Dict[str, int] = Field(default_factory=dict, description="Iterations dictionary for stage 1")
    dic_itera2: Dict[str, int] = Field(default_factory=dict, description="Iterations dictionary for stage 2")
    rest: List[float] = Field(default_factory=list, description="Rest definition")
    celp_st: Union[List[str], str] = Field(default_factory=list, description="Choose CELPP receptor in upper case or pdb code in lower case")
    neut: str = Field("", description="Neutralize")

    # Number of simulations, 1 equilibrium and 1 production
    apr_sim: int = Field(2, description="Number of simulations")

    @property
    def H1(self):
        return self.p1
    
    @property
    def H2(self):
        return self.p2

    @property
    def H3(self):
        return self.p3

    @model_validator(mode="after")
    def initialize_ti(self) -> "SimulationConfig":
        """
        Calculate lambdas and weights dynamically.
        """
        dec_int = self.dec_int
        ti_points = self.ti_points

        if dec_int == "ti":
            if ti_points and ti_points > 0:
                x, y = np.polynomial.legendre.leggauss(ti_points)
                lambdas = [(xi + 1) / 2 for xi in x]  # Adjust Gaussian lambdas
                weights = [wi / 2 for wi in y]       # Adjust Gaussian weights
                logger.debug(f"Lambda values: {lambdas}")
                logger.debug(f"Gaussian weights: {weights}")
                self.lambdas = lambdas
                self.weights = weights
            else:
                raise ValueError(
                    "Invalid input! ti_points must be a positive integer for the TI-GQ method."
                )

        self.rng = len(self.release_eq) - 1
        self.ion_def = [self.cation, self.anion, self.ion_conc]

        # (Deprecated CELPP logic omitted as in original)

        for comp in FEP_COMPONENTS:
            self.dic_steps1.update({f'{comp}': self.n_steps_dict[f'{comp}_steps1']})
            self.dic_steps2.update({f'{comp}': self.n_steps_dict[f'{comp}_steps2']})
            self.dic_itera1.update({f'{comp}': self.n_iter_dict[f'{comp}_itera1']})
            self.dic_itera2.update({f'{comp}': self.n_iter_dict[f'{comp}_itera2']})

        self.rest = [
            self.rec_dihcf_force,
            self.rec_discf_force,
            self.lig_distance_force,
            self.lig_angle_force,
            self.lig_dihcf_force,
            self.rec_com_force,
            self.lig_com_force
        ]
        if self.buffer_z == 0:
            logger.info('Buffer size along Z-axis is set to 0; an automatic buffer will be applied.')

        if self.num_waters != 0:
            raise ValueError("'num_waters' is removed")

        if self.rec_bb == 'no':
            self.bb_start = [1]
            self.bb_end = [0]
            self.bb_equil = 'no'
            logger.debug("No backbone dihedral restraints")
        else:
            if isinstance(self.bb_start, int):
                self.bb_start = [self.bb_start]
            if isinstance(self.bb_end, int):
                self.bb_end = [self.bb_end]
        self.neut = self.neutralize_only

        match self.fe_type:
            case 'custom':
                if self.dec_method is None:
                    logger.error('Wrong input! Please choose a decoupling method'
                                 '(dd, sdr or exchange) when using the custom option.')
                    sys.exit(1)
            case 'rest':
                self.components = ['c', 'a', 'l', 't', 'r']
                self.dec_method = 'dd'
            case 'sdr':
                self.components = ['e', 'v']
                self.dec_method = 'sdr'
            case 'dd':
                self.components = ['e', 'v', 'f', 'w']
                self.dec_method = 'dd'
            case 'sdr-rest':
                self.components = ['c', 'a', 'l', 't', 'r', 'e', 'v']
                self.dec_method = 'sdr'
            case 'express':
                self.components = ['m', 'n', 'e', 'v']
                self.dec_method = 'sdr'
            case 'dd-rest':
                self.components = ['c', 'a', 'l', 't', 'r', 'e', 'v', 'f', 'w']
                self.dec_method = 'dd'
            case 'relative':
                self.components = ['x', 'e', 'n', 'm']
                self.dec_method = 'exchange'
            case 'uno':
                self.components = ['m', 'n', 'o']
                self.dec_method = 'sdr'
            case 'uno_rest':
                self.components = ['z']
                self.dec_method = 'sdr'
            case 'uno_com':
                self.components = ['o']
                self.dec_method = 'sdr'
            case 'self':
                self.components = ['s']
                self.dec_method = 'sdr'
            case 'uno_dd':
                self.components = ['z', 'y']
                self.dec_method = 'dd'
            case _:
                raise ValueError(
                    "Invalid fe_type: {self.fe_type}. Must be one of "
                    "'rest', 'dd', 'sdr', 'dd-rest', 'sdr-rest', "
                    "'express', 'relative', 'uno', 'uno_com', 'uno_rest', 'self', 'uno_dd', or 'custom'."
                )

        for comp in self.components:
            logger.debug(f'Using component: {comp}')
            logger.debug(f'Steps for stage 1: {self.dic_steps1[comp]}')
            logger.debug(f'Steps for stage 2: {self.dic_steps2[comp]}')
            if self.dic_steps1[comp] == 0:
                raise ValueError(
                    f"Invalid input! {comp} steps for stage 1 must be greater than 0 with {comp}_steps1."
                )
            if self.dic_steps2[comp] == 0:
                raise ValueError(
                    f"Invalid input! {comp} steps for stage 2 must be greater than 0 with {comp}_steps2."
                )
        self.remd = self.remd
        
        logger.debug('------------------ Simulation Configuration ------------------')
        logger.debug(f'Software: {self.software}')
        logger.debug(f'Receptor/complex structures: {self.celp_st}')
        logger.debug(f'Ligand names: {self.mols}')
        logger.debug(f'Cobinders names: {self.other_mol}')
        logger.debug('--------------------------------------------------------------')
        logger.debug('Finished initializing simulation configuration.')
        return self

    @field_validator("calc_type", mode="before")
    def validate_calc_type(cls, value):
        if value is None:
            return value
        valid_types = {"dock", "rank", "crystal"}
        if value not in valid_types:
            raise ValueError(f"Invalid calc_type: {value}. Must be one of {valid_types}.")
        return value

    @field_validator("rec_bb", "neutralize_only", "hmr", "bb_equil", mode="before")
    def validate_yes_no(cls, value):
        if value is None:
            return value
        if isinstance(value, str) and value.lower() in {"yes", "no"}:
            return value.lower()
        raise ValueError(f"Invalid value: {value}. Must be 'yes' or 'no'.")
    
    def to_dict(self):
        return self.model_dump()


def parse_input_file(input_file: str) -> dict:
    """
    Parses an input file to extract parameters as a dictionary.
    """
    parameters: Dict[str, Any] = {}

    with open(input_file) as f_in:
        # Remove spaces, tabs, and blank lines
        lines = (line.strip(' \t\n\r') for line in f_in)
        lines = [line for line in lines if line and not line.startswith('#')]

        for line in lines:
            if '=' in line:
                key_value = line.split('#')[0].split('=')
                if len(key_value) == 2:
                    key = key_value[0].strip().lower()
                    value = key_value[1].strip()
                    # if the value is a list, split it by commas
                    if '[' in value and ']' in value:
                        value = value.strip('\'\"-,.:;#()][')
                        if key in ['poses_list', 'ligand_list', 'other_mol',
                                   'celpp_receptor', 'ligand_name',
                                   'bb_start', 'bb_end']:
                            split_sep = ','
                        else:
                            split_sep = None
                        try:
                            value = [v.strip() for v in value.split(split_sep)]
                        except ValueError:
                            pass
                    parameters[key] = value
                else:
                    raise ValueError(f"Invalid line: {line}")

    # merge FEP_COMPONENTS into a dict
    n_steps_dict: Dict[str, int] = {}
    n_iter_dict: Dict[str, int] = {}

    for comp in FEP_COMPONENTS:
        for ind in ['1', '2']:
            key = f'{comp}_steps{ind}'
            n_steps_dict[key] = int(parameters[key]) if key in parameters else 0
            key2 = f'{comp}_itera{ind}'
            n_iter_dict[key2] = int(parameters[key2]) if key2 in parameters else 0

    parameters['n_steps_dict'] = n_steps_dict
    parameters['n_iter_dict'] = n_iter_dict

    return parameters


def get_configure_from_file(file_path: str) -> Dict:
    """
    Parse the input file, validate parameters, and return a simulation configuration.
    """
    raw_params = parse_input_file(file_path)
    config = SimulationConfig(**raw_params)
    return config