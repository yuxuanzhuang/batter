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
                 protein_path: str,
                 ligand_path: str,
                 reference_path: str,
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
        protein_path : str
            Path to the protein file.
            Here we assume the protein has its surrounding water molecules.
        ligand_path : str
            Path to the ligand file.
        reference_path : str
            Path to the reference protein file.
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
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.reference_path = reference_path
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

        self.protein = mda.Universe(protein_path)

        self.ligand = mda.Universe(ligand_path)

        # Process ligand and create the complex
        self._process_ligand()
        self.complex = mda.Merge(self.protein.atoms, self.ligand.atoms)
        # remove overlapping water molecules with the ligand
        overlapped_water = self.complex.select_atoms('byres (resname WAT TIP3 and around 1 resname {self.ligand.atoms.resnames[0]})')
        logger.info(f"Removing {len(overlapped_water)} overlapping water molecules with the ligand")
        self.complex.atoms = self.complex.atoms - overlapped_water

        self.complex.atoms.write(f"{self.output_dir}/complex.pdb")
        self.complex = mda.Universe(f"{self.output_dir}/complex.pdb")

        # First align the protein and ligand to the reference protein structure
        self.reference = mda.Universe(self.reference_path)
        self._process_reference()
        self._align_system()

        self._prepare_ligand_parameters()
        self._tleap_system()

        self._prepare_system()

    def _prepare_system(self):
        # dabble the system
        logger.info('Dabbling the complex')
        export_amberhome = f'AMBERHOME={self.amberhome}'
        dabble_path = '/scratch/users/yuzhuang/miniforge3/envs/few/bin/dabble'
        input_pdb = self.complex_path
        output_prmtop = f"{self.output_dir}/complex_dabbled.prmtop"
        ligand_param = f"{self.output_dir}/ligand.frcmod"
        dabble_command = f'{export_amberhome} {dabble_path} -i {input_pdb} -o {output_prmtop} -top {ligand_param} --hmr -w 10 -m 17.5 -O -ff amber -M water --verbose | tee {self.output_dir}/dabble.log'
        run_with_log(dabble_command)
        logger.info('Complex dabbling completed')

    def _align_system(self):
        """
        Align the protein and ligand to the reference protein structure.
        """
        # Determine the root of the package
        package_root = os.path.dirname(os.path.abspath(__file__))
        usalign_path = os.path.join(package_root, 'USalign')

        logger.info('Aligning the protein and ligand to the reference protein structure')
        
        # Construct the alignment command
        complex_path = f"{self.output_dir}/complex.pdb"
        reference_path = f"{self.output_dir}/reference_amber.pdb"
        output_prefix = f"{self.output_dir}/aligned-nc"
        with tempfile.TemporaryDirectory() as tmp_dir:
            command = f"{usalign_path} {complex_path} {reference_path} -mm 0 -ter 2 -o {output_prefix}"
            run_with_log(command, working_dir=tmp_dir)

        aligned = mda.Universe(f"{output_prefix}.pdb")
        old_rmsd, new_rmsd = alignto(self.complex, aligned, select="protein and backbone")
        # separate the protein and ligand
        ligand = self.complex.select_atoms(f'resname {self.ligand.atoms.resnames[0]}')
        logger.info(f'Ligand: {ligand}')
        protein = self.complex.atoms - ligand
        logger.info(f'Protein: {protein}')

#        self.convert_charmm_2_amber(self.protein)
#        self.convert_charmm_2_amber(self.complex)

        # Write the aligned structures
        protein.atoms.write(f"{self.output_dir}/protein.pdb")
        ligand.atoms.write(f"{self.output_dir}/ligand.pdb")
        self.complex.atoms.write(f"{self.output_dir}/complex.pdb")

        self.protein = mda.Universe(f"{self.output_dir}/protein.pdb")
        self.ligand = mda.Universe(f"{self.output_dir}/ligand.pdb")
        self.complex = mda.Universe(f"{self.output_dir}/complex.pdb")
        self.protein_path = f"{self.output_dir}/protein.pdb"
        self.ligand_path = f"{self.output_dir}/ligand.pdb"
        self.complex_path = f"{self.output_dir}/complex.pdb"

        logger.info(f"RMSD before alignment: {old_rmsd:.3f} Å")
        logger.info(f"RMSD after alignment: {new_rmsd:.3f} Å")

    def _tleap_system(self):
        """
        Prepare the system using tleap. This will add hydrogens and terminals to the protein.
        """
        run_with_log(f'pdb4amber -i {self.protein_path} -o {self.output_dir}/protein_amber.pdb')
        self.protein = mda.Universe(f"{self.output_dir}/protein_amber.pdb")

        first_residue = self.protein.select_atoms('protein').residues[0].resid
        last_residue = self.protein.select_atoms('protein').residues[-1].resid
        
        # run tleap to add terminals
        tleap_script = f"""
        # Load the Amber protein force field
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_param}

        # Load the protein structure
        protein = loadPDB "{self.output_dir}/protein_amber.pdb"

        # Load the ligand parameters
        lig = loadMol2 "{self.output_dir}/ff/ligand.mol2"
        loadAmberParams "{self.output_dir}/ff/ligand.frcmod"

        # Define neutral N-terminal and C-terminal caps
        set protein.{first_residue} name "ACE"       # Neutral N-terminal (ACE cap)
        set protein.{last_residue} name "NHE"  # Neutral C-terminal (NME cap)

        complex = combine {{ protein lig }}

        # Save the modified PDB for inspection
        set default pdbwritecharges on
        savePDB complex "{self.output_dir}/complex_amber.pdb"

        # Exit tleap
        quit

        """
        with open(f"{self.output_dir}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{TLEAP} -f {self.output_dir}/tleap.in")

        self.complex = mda.Universe(f"{self.output_dir}/complex_amber.pdb",
                                    guess_bonds=True)
        guessed_elements = guess_types(self.complex.atoms.names)

        self.complex.add_TopologyAttr('elements', guessed_elements)

        self.complex.atoms.write(f"{self.output_dir}/complex_amber.pdb")
        self.complex_path = f"{self.output_dir}/complex_amber.pdb"


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

    def _process_reference(self, usalign='./USalign'):
        """Remove chain info from the reference"""
        run_with_log(f"pdb4amber -i {self.reference_path} -o {self.output_dir}/reference_amber.pdb -y")
        ref_amber = mda.Universe(f"{self.output_dir}/reference_amber.pdb")
        self.reference.del_TopologyAttr('chainIDs')
        self.reference.select_atoms('protein').write(f"{self.output_dir}/reference_amber.pdb")

    def _prepare_ligand_parameters(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
        logger.info('Preparing ligand parameters')
        antechamber_command = f'{ANTECHAMBER} -i {self.ligand_path} -fi pdb -o {self.output_dir}/ligand_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_param} -nc {self.ligand_charge}'
        run_with_log(antechamber_command)
        shutil.copy(f"{self.output_dir}/ligand_ante.mol2", f"{self.output_dir}/ff/ligand.mol2")

        if self.ligand_param == 'gaff':
            run_with_log(f'{PARMCHK2} -i {self.output_dir}/ligand_ante.mol2 -f mol2 -o {self.output_dir}/ligand.frcmod -s 1')
        elif self.ligand_param == 'gaff2':
            run_with_log(f'{PARMCHK2} -i {self.output_dir}/ligand_ante.mol2 -f mol2 -o {self.output_dir}/ligand.frcmod -s 2')
        shutil.copy(f"{self.output_dir}/ligand.frcmod", f"{self.output_dir}/ff/ligand.frcmod")

        run_with_log(f'{ANTECHAMBER} -i {self.ligand_path} -fi pdb -o {self.output_dir}/ligand_ante.pdb -fo pdb')
        logger.info('Ligand parameters prepared')

    @staticmethod
    def convert_charmm_2_amber(universe):
        """
        Convert a CHARMM universe to an AMBER universe.
        1) converting HSE to HIE
        2) converting HSD to HID
        3) converting HSP to HIP

        4) converting HD?
        5) converting H?
        """
        universe.select_atoms('resname HSE').residues.resnames = 'HIE'
        universe.select_atoms('resname HSD').residues.resnames = 'HID'
        universe.select_atoms('resname HSP').residues.resnames = 'HIP'

        return universe

        
class MembraneSystem(System):
    def _prepare_system(self):
        # dabble the system
        logger.info('Dabbling the complex')
        export_amberhome = f'AMBERHOME={self.amberhome}'
        dabble_path = '/scratch/users/yuzhuang/miniforge3/envs/few/bin/dabble'
        input_pdb = self.complex_path
        output_prmtop = f"{self.output_dir}/complex_dabbled.prmtop"
        ligand_param = f"{self.output_dir}/ligand.frcmod"
        dabble_command = f'{export_amberhome} {dabble_path} -i {input_pdb} -o {output_prmtop} -top {ligand_param} --hmr -w 10 -m 17.5 -O -ff amber -M water --verbose | tee {self.output_dir}/dabble.log'
        run_with_log(dabble_command)
        logger.info('Complex dabbling completed')