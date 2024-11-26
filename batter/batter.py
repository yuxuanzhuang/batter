"""
Provide the primary functions for preparing and processing FEP systems.
"""

import os
import sys
import shutil
import subprocess as sp
import tempfile
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types

from typing import List, Tuple
import loguru
from loguru import logger

# set info level for the logger
#logger.remove()
#logger.add(sys.stdout, level='INFO')

import numpy as np
from MDAnalysis.analysis.align import alignto
from .utils import run_with_log, ANTECHAMBER, TLEAP, CPPTRAJ, PARMCHK2

class System:
    """
    A class to represent and process a Free Energy Perturbation (FEP) system.
    """

    def __init__(self,
                 complex_mae: str,
                 ligand_path: str,
                 output_dir: str,
                 retain_lig_h: bool = True,
                 ligand_ph: float = 7.4,
                 ligand_param: str = 'gaff2',
                 amberhome: str = '/home/groups/rondror/software/amber20/amber20_src',
                 ):
        """
        Initialize the FEPSystem class.

        Parameters
        ----------
        complex_mae : str
            Path to the complex file in Maestro format.
            The protein is already aligned to the reference protein structure.
        ligand_path : str
            Path to the ligand file.
        output_dir : str
            Directory where output files will be saved.
        retain_lig_h : bool, optional
            Whether to retain hydrogens in the ligand. Default is True.
        ligand_ph : float, optional
            pH value for protonating the ligand. Default is 7.4.
        ligand_param : str, optional
            Parameter set for the ligand. Default is 'gaff'.
            Options are 'gaff' and 'gaff2'.
        amberhome : str, optional
            Path to the AMBER installation directory. Default is '/home/groups/rondror/software/amber20/amber20_src'.
        """
        self.complex_mae = complex_mae
        self.ligand_path = ligand_path
        # set to absolute path
        self.output_dir = os.path.abspath(output_dir) + '/'

        self.retain_lig_h = retain_lig_h
        self.ligand_ph = ligand_ph
        self.ligand_param = ligand_param
        if self.ligand_param not in ['gaff', 'gaff2']:
            raise ValueError(f"Invalid ligand_param: {self.ligand_param}"
                                "Options are 'gaff' and 'gaff2'")
        if self.ligand_param == 'gaff':
            raise NotImplementedError("gaff is not supported yet for dabble (maybe?)")
        self.amberhome = amberhome

        # Write initial files
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/ff", exist_ok=True)

        self.ligand = mda.Universe(ligand_path)

        # Process ligand and create the complex
        self._process_ligand()
        self._prepare_ligand_parameters()

        self._prepare_system()

    def _prepare_system(self):
        # dabble the system
        logger.info('Dabbling the complex')
        export_amberhome = f'AMBERHOME={self.amberhome}'
        dabble_path = '/scratch/users/yuzhuang/miniforge3/envs/few/bin/dabble'
        input_pdb = self.complex_mae
        output_prmtop = f"{self.output_dir}/complex_dabbled.prmtop"
        ligand_lib = f"{self.output_dir}/ff/ligand.lib"
        ligand_param = f"{self.output_dir}/ligand.frcmod"
        dabble_command = f'{export_amberhome} {dabble_path} -i {input_pdb} -o {output_prmtop} -top {ligand_lib} -par {ligand_param} --hmr -w 10 -m 17.5 -O -ff amber -M water --verbose | tee {self.output_dir}/dabble.log'
        run_with_log(dabble_command)
        logger.info('Complex dabbling completed')

    def _process_ligand(self):
        """
        Process the ligand, including adding or removing hydrogens as needed.
        """

        # Ensure the ligand file is in PDB format
        logger.info(f'Processing ligand file: {self.ligand_path}')
        if not self.ligand_path.endswith('.pdb'):
            converted_path = f"{self.output_dir}/ligand.pdb"
            self.ligand.atoms.write(converted_path)
            self.ligand_path = converted_path
            self.ligand = mda.Universe(self.ligand_path)

        # retain hydrogens from the ligand
        if self.retain_lig_h:
            # convert mol2 to get charge
            shutil.copy(self.ligand_path, f"{self.output_dir}/ligand.pdb")
            run_with_log(f"obabel -i pdb {noh_path} -o mol2 -O {self.output_dir}/ligand.mol2")

        else:
            # Remove hydrogens from the ligand
            noh_path = f"{self.output_dir}/ligand_noh.pdb"
            run_with_log(f"obabel -i pdb {self.ligand_path} -o pdb -O {noh_path} -d")

            # Add hydrogens based on the specified pH
            run_with_log(f"obabel -i pdb {noh_path} -o pdb -O {self.output_dir}/ligand.pdb -p {self.ligand_ph:.2f}")
            run_with_log(f"obabel -i pdb {noh_path} -o mol2 -O {self.output_dir}/ligand.mol2 -p {self.ligand_ph:.2f}")

        self.ligand_path = f"{self.output_dir}/ligand.pdb"
        self.ligand = mda.Universe(self.ligand_path)
        self.ligand_mol2_path = f"{self.output_dir}/ligand.mol2"
        self.ligand_mol2 = mda.Universe(self.ligand_mol2_path)

        self.ligand_charge = np.round(np.sum(self.ligand_mol2.atoms.charges))
        logger.info(f'The babel protonation of the ligand is for pH {self.ligand_ph:.2f}')
        logger.info(f'The net charge of the ligand is {self.ligand_charge}')

    def _prepare_ligand_parameters(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
        logger.info('Preparing ligand parameters')
        antechamber_command = f'{ANTECHAMBER} -i {self.ligand_path} -fi pdb -o {self.output_dir}/ligand_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_param} -nc {self.ligand_charge}'
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(antechamber_command, working_dir=tmpdir)
        shutil.copy(f"{self.output_dir}/ligand_ante.mol2", f"{self.output_dir}/ff/ligand.mol2")

        if self.ligand_param == 'gaff':
            run_with_log(f'{PARMCHK2} -i {self.output_dir}/ligand_ante.mol2 -f mol2 -o {self.output_dir}/ligand.frcmod -s 1')
        elif self.ligand_param == 'gaff2':
            run_with_log(f'{PARMCHK2} -i {self.output_dir}/ligand_ante.mol2 -f mol2 -o {self.output_dir}/ligand.frcmod -s 2')
        shutil.copy(f"{self.output_dir}/ligand.frcmod", f"{self.output_dir}/ff/ligand.frcmod")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(f'{ANTECHAMBER} -i {self.ligand_path} -fi pdb -o {self.output_dir}/ligand_ante.pdb -fo pdb', working_dir=tmpdir)

        # get lib file
        tleap_script = f"""
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_param}
        lig = loadmol2 {self.output_dir}/ff/ligand.mol2
        loadamberparams {self.output_dir}/ff/ligand.frcmod
        saveoff lig {self.output_dir}/ff/ligand.lib
        saveamberparm lig {self.output_dir}/ff/ligand.prmtop {self.output_dir}/ff/ligand.inpcrd

        quit
        """
        with open(f"{self.output_dir}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{TLEAP} -f {self.output_dir}/tleap.in")

        logger.info('Ligand parameters prepared')

        
class MembraneSystem(System):
    def _prepare_system(self):
        # dabble the system
        logger.info('Dabbling the complex')
        export_amberhome = f'AMBERHOME={self.amberhome}'
        dabble_path = '/scratch/users/yuzhuang/miniforge3/envs/few/bin/dabble'
        input_mae = self.complex_mae
        output_prmtop = f"{self.output_dir}/complex_dabbled.prmtop"
        ligand_lib = f"{self.output_dir}/ff/ligand.lib"
        ligand_param = f"{self.output_dir}/ligand.frcmod"
        dabble_command = f'{export_amberhome} {dabble_path} -i {input_mae} -o {output_prmtop} -top {ligand_lib} -par {ligand_param} --hmr -w 10 -m 17.5 -O -ff amber --verbose | tee {self.output_dir}/dabble.log'
        run_with_log(dabble_command)
        logger.info('Complex dabbling completed')