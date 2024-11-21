"""
Provide the primary functions for preparing and processing FEP systems.
"""

import os
import shutil
import subprocess as sp
import tempfile
import MDAnalysis as mda
from typing import List, Tuple
import loguru
import numpy as np
from MDAnalysis.analysis.align import alignto


class FEPSystem:
    """
    A class to represent and process a Free Energy Perturbation (FEP) system.
    """

    def __init__(self,
                 protein_path: str,
                 ligand_path: str,
                 reference_path: str,
                 output_dir: str,
                 retain_lig_h: bool = True,
                 ligand_ph: float = 7.4):
        """
        Initialize the FEPSystem class.

        Parameters
        ----------
        protein_path : str
            Path to the protein file.
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
        """
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.reference_path = reference_path
        # set to absolute path
        self.output_dir = os.path.abspath(output_dir)

        self.retain_lig_h = retain_lig_h
        self.ligand_ph = ligand_ph

        self.protein = mda.Universe(protein_path)
        self.ligand = mda.Universe(ligand_path)

        # Process ligand and create the complex
        self._process_ligand()
        self.complex = mda.Merge(self.protein.atoms, self.ligand.atoms)

        # Write initial files
        os.makedirs(self.output_dir, exist_ok=True)
        self.protein.atoms.write(f"{self.output_dir}/protein.pdb")
        self.ligand.atoms.write(f"{self.output_dir}/ligand.pdb")
        self.complex.atoms.write(f"{self.output_dir}/complex.pdb")
        
        self.reference = mda.Universe(self.reference_path)
        self._process_reference()
        self._align_system()

    def prepare_system(self):
        """
        Placeholder for further system preparation steps.
        """
        pass

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

        self.complex.atoms.write(f"{self.output_dir}/complex_aligned.pdb")
        logger.info(f"RMSD before alignment: {old_rmsd:.3f} Å")
        logger.info(f"RMSD after alignment: {new_rmsd:.3f} Å")



    def _process_ligand(self):
        """
        Process the ligand, including adding or removing hydrogens as needed.
        """

        # Ensure the ligand file is in PDB format
        loguru.logger.info(f'Processing ligand file: {self.ligand_path}')
        if not self.ligand_path.endswith('.pdb'):
            converted_path = f"{self.output_dir}/ligand.pdb"
            self.ligand.atoms.write(converted_path)
            self.ligand_path = converted_path
            self.ligand = mda.Universe(self.ligand_path)

        # retain hydrogens from the ligand
        if self.retain_lig_h:
            # convert mol2 to get charge
            run_with_log(f"obabel -i pdb {noh_path} -o mol2 -O {self.output_dir}/ligand.mol2")

        else:
            # Remove hydrogens from the ligand
            noh_path = f"{self.output_dir}/ligand_noh.pdb"
            run_with_log(f"obabel -i pdb {self.ligand_path} -o pdb -O {noh_path} -d")

            # Add hydrogens based on the specified pH
            run_with_log(f"obabel -i pdb {noh_path} -o pdb -O {self.output_dir}/ligand.pdb -p {self.ligand_ph:.2f}")
            run_with_log(f"obabel -i pdb {noh_path} -o mol2 -O {self.output_dir}/ligand.mol2 -p {self.ligand_ph:.2f}")
        self.ligand_mol2_path = f"{self.output_dir}/ligand.mol2"
        self.ligand_mol2 = mda.Universe(self.ligand_mol2_path)

        self.ligand_charge = np.round(np.sum(self.ligand_mol2.atoms.charges))
        loguru.logger.info(f'The babel protonation of the ligand is for pH {self.ligand_ph:.2f}')
        loguru.logger.info(f'The net charge of the ligand is {self.ligand_charge}')

    def _process_reference(self, usalign='./USalign'):
        """Remove chain info from the reference"""
#        run_with_log(f"pdb4amber -i {self.reference_path} -o {self.output_dir}/reference_amber.pdb -y")
#        ref_amber = mda.Universe(f"{self.output_dir}/reference_amber.pdb")
        self.reference.del_TopologyAttr('chainIDs')
        self.reference.select_atoms('protein').write(f"{self.output_dir}/reference_amber.pdb")


from loguru import logger
import subprocess as sp

def run_with_log(command, level='debug', working_dir=None):
    """
    Run a subprocess command and log its output using loguru logger.

    Parameters
    ----------
    command : str
        The command to execute.
    level : str, optional
        The log level for logging the command output. Default is 'debug'.
    working_dir : str, optional
        The working directory for the command. Default is
        the current working directory.

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    subprocess.CalledProcessError
        If the command exits with a non-zero status.
    """
    if working_dir is None:
        working_dir = os.getcwd()
    # Map log level to loguru logger methods
    log_methods = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }

    log = log_methods.get(level)
    if log is None:
        raise ValueError(f"Invalid log level: {level}")

    log(f"Running command: {command}")
    try:
        # Run the command and capture output
        result = sp.run(
            command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=True,
            cwd=working_dir
        )

        # Log stdout and stderr line by line
        if result.stdout:
            log("Command output:")
            for line in result.stdout.splitlines():
                log(line)

        if result.stderr:
            log("Command errors:")
            for line in result.stderr.splitlines():
                log(line)

    except sp.CalledProcessError as e:
        log(f"Command failed with return code {e.returncode}")
        if e.stdout:
            log("Command output before failure:")
            for line in e.stdout.splitlines():
                log(line)
        if e.stderr:
            log("Command error output:")
            for line in e.stderr.splitlines():
                log(line)
        raise