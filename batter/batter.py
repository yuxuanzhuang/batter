"""
Provide the primary functions for preparing and processing FEP systems.
"""

from .utils import (
    run_with_log,
    save_state,
    antechamber,
    tleap,
    cpptraj,
    parmchk2,
    charmmlipid2amber,
    obabel,
    vmd)

import inspect
import numpy as np
import os
import sys
import shutil
import glob
import subprocess as sp
from contextlib import contextmanager
import tempfile
import MDAnalysis as mda
from MDAnalysis.guesser import DefaultGuesser
from MDAnalysis.analysis import align
import pandas as pd
from importlib import resources
import json
from typing import Union
from pathlib import Path
import pickle
from functools import wraps
from openff.toolkit import Molecule
from rdkit import Chem

import time
from tqdm import tqdm

from typing import List, Tuple
import loguru
from loguru import logger

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.bat_lib import analysis
from batter.results import FEResult
from batter.utils.slurm_job import SLURMJob

from MDAnalysis.analysis import rms, align

from batter.builder import BuilderFactory
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from batter.utils import (
    COMPONENTS_LAMBDA_DICT,
    COMPONENTS_FOLDER_DICT,
)

DEC_FOLDER_DICT = {
    'dd': 'dd',
    'sdr': 'sdr',
    'exchange': 'sdr',
}

AVAILABLE_COMPONENTS = ['v', 'e', 'w', 'f',
                       'x', 'a', 'l', 't',
                       'r', 'c', 'm', 'n']


class System:
    """
    A class to represent and process a Free Energy Perturbation (FEP) system.

    It will prepare the input files of protein system with **one** ligand species.

    After the preparation of the equil system, run through the equilibration simulation and then
    prepare the fe system. The output of the equil system will be used as
    the input for the fe system.

    """
    def __init__(self,
                 folder: str,
                ):
        """
        Initialize the System class with the folder name.
        If the folder does not exist, a new system will be created.
        If the folder exists, the system will be loaded.

        Parameters
        ----------
        folder : str
            The folder containing the system files.
        """
        self.output_dir = os.path.abspath(folder) + '/'

        self._slurm_jobs = {}
        self._sim_finished = {}
        self._sim_failed = {}
        self._eq_prepared = False
        self._fe_prepared = False
        self._fe_results = {}
        self._max_num_jobs = 2000

        if not os.path.exists(self.output_dir):
            logger.info(f"Creating a new system: {self.output_dir}")
            os.makedirs(self.output_dir)
        else:
            logger.info(f"Loading an existing system: {self.output_dir}")
            self._load_system()

    def _load_system(self):
        """
        Load the system from the folder.
        """
        if not os.path.exists(f"{self.output_dir}/system.pkl"):
            logger.info(f"The folder does not contain system.pkl: {self.output_dir}")
            return
        
        system_file = os.path.join(self.output_dir, "system.pkl")

        try:
            with open(system_file, 'rb') as f:
                loaded_state = pickle.load(f)
                # in case the folder is moved
                loaded_state.output_dir = self.output_dir
                # Update self with loaded attributes
                self.__dict__.update(loaded_state.__dict__)
        except Exception as e:
            logger.error(f"Error loading the system: {e}")

        if not os.path.exists(f"{self.output_dir}/all-poses"):
            logger.info(f"The folder does not contain all-poses: {self.output_dir}")
            return
        if not os.path.exists(f"{self.output_dir}/equil"):
            logger.info(f"The folder does not contain equil: {self.output_dir}")
            return
        if not os.path.exists(f"{self.output_dir}/fe"):
            logger.info(f"The folder does not contain fe: {self.output_dir}")
            return

    @save_state
    def create_system(
                    self,
                    system_name: str,
                    protein_input: str,
                    system_topology: str,
                    ligand_paths: List[str],
                    receptor_segment: str = None,
                    system_coordinate: str = None,
                    protein_align: str = 'name CA and resid 60 to 250',
                    retain_lig_prot: bool = True,
                    ligand_ph: float = 7.4,
                    ligand_ff: str = 'gaff2',
                    lipid_mol: List[str] = [],
                    lipid_ff: str = 'lipid21',
                    overwrite: bool = False,
                    ):
        """
        Create a new single-ligand single-receptor system.

        Parameters
        ----------
        protein_input : str
            Path to the protein file in PDB format.
            It should be exported from Maestro,
            which means the protonation states of the protein are assigned.
            Water and ligand can be present in the file,
            but they will be removed during preparation.
        system_topology : str
            PDB file of a prepared simulation system with `dabble`.
            The ligand does not need to be present.
        system_coordinate : str
            The coordinate file for the system.
            The coordiantes and box dimensions will be used for the system.
            It can be an INPCRD file prepared from `dabble` or
            it can be a snapshot of the equilibrated system.
            If it is not provided, the coordinates from the system_topology
            will be used if available.
        ligand_paths : List[str]
            List of ligand files. It can be either PDB or mol2 format.
            It will be stored in the `all-poses` folder as `pose0.pdb`,
            `pose1.pdb`, etc.
        receptor_segment : str
            The segment of the receptor in the system_topology.
            It will be used to set the protein anchor for the ligand.
            Default is None, which means all protein atoms will be used.
            Warning: if the protein for acnhoring
            is not the first protein entry of the system_topology,
            it will cause problems when bat.py is trying to 
            get the protein anchor.
        protein_align : str
            The selection string for aligning the protein to the system.
            Default is 'name CA and resid 60 to 250'.
        retain_lig_prot : bool, optional
            Whether to retain hydrogens in the ligand. Default is True.
        ligand_ph : float, optional
            pH value for protonating the ligand. Default is 7.4.
        ligand_ff : str, optional
            Parameter set for the ligand. Default is 'gaff'.
            Options are 'gaff' and 'gaff2'.
        lipid_mol : List[str], optional
            List of lipid molecules to be included in the simulations.
            Default is an empty list.
        lipid_ff : str, optional
            Force field for lipid atoms. Default is 'lipid21'.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        """
        # Log every argument
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        
        for arg in args:
            logger.info(f"{arg}: {values[arg]}")

        self.system_name = system_name
        self._protein_input = self._convert_2_relative_path(protein_input)
        self._system_topology = self._convert_2_relative_path(system_topology)
        self._system_coordinate = self._convert_2_relative_path(system_coordinate)
        self._ligand_paths = [self._convert_2_relative_path(ligand_path) for ligand_path in ligand_paths]
        self.mols = []
        if not isinstance(self.ligand_paths, list):
            raise ValueError(f"Invalid ligand_paths: {self.ligand_paths}, "
                              "ligand_paths should be a list of ligand files")
        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.ligand_ff = ligand_ff
        self.overwrite = overwrite

        # check input existence
        if not os.path.exists(self.protein_input):
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        if not os.path.exists(self.system_topology):
            raise FileNotFoundError(f"System input file not found: {system_topology}")
        for ligand_path in self.ligand_paths:
            if not os.path.exists(ligand_path):
                raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
        
        self._process_ligands()

        if system_coordinate is None and not os.path.exists(system_coordinate):
            raise FileNotFoundError(f"System coordinate file not found: {system_coordinate}")
        
        self.u_sys = mda.Universe(self.system_topology, format='XPDB')
        if system_coordinate is not None:
            # read last line of inpcrd file to get dimensions
            with open(system_coordinate) as f:
                lines = f.readlines()
                box = np.array([float(x) for x in lines[-1].split()])
            self.system_dimensions = box
            self.u_sys.load_new(system_coordinate, format='INPCRD')
        if not self.u_sys.atoms.dimensions and self.system_coordinate is None:
            raise ValueError(f"No dimension of the box was found in the system_topology or system_coordinate")

        os.makedirs(f"{self.poses_folder}", exist_ok=True)
        os.makedirs(f"{self.ligandff_folder}", exist_ok=True)

        if self.ligand_ff not in ['gaff', 'gaff2']:
            raise ValueError(f"Invalid ligand_ff: {self.ligand_ff}"
                             "Options are 'gaff' and 'gaff2'")
        if self.ligand_ff == 'gaff':
            raise NotImplementedError("gaff is not supported yet for dabble (maybe?)")
        self.lipid_mol = lipid_mol
        if not self.lipid_mol:
            self.membrane_simulation = False
        else:
            self.membrane_simulation = True
        self.lipid_ff = lipid_ff
        if self.lipid_ff != 'lipid21':
            raise ValueError(f"Invalid lipid_ff: {self.lipid_ff}"
                             "Only 'lipid21' is available")

        # Prepare the membrane parameters
        if self.membrane_simulation:
            self._prepare_membrane()

        self._get_alignment()

        if self.overwrite or not os.path.exists(f"{self.poses_folder}/{self.system_name}_docked.pdb") or not os.path.exists(f"{self.poses_folder}/reference.pdb"):
            self._process_system()
            # Process ligand and prepare the parameters
        
        self.unique_mol_names = []
        for ind, ligand_path in enumerate(self.unique_ligand_paths, start=1):
            self._ligand_path = ligand_path
            # raise warning inf ligand is a PDB
            if not ligand_path.lower().endswith('.sdf'):
                logger.warning(f"A sdf file is preferred for the ligand: {ligand_path}")
                self._ligand = mda.Universe(ligand_path)

                if len(set(self._ligand.atoms.resnames)) > 1:
                    raise ValueError(f"Multiple ligand molecules {set(self._ligand.atoms.resnames)} found in the ligand file: {ligand_path}")
                mol_name = self._ligand.atoms.resnames[0].lower()
            else:
                # use basename of the ligand file
                molecule = Chem.MolFromMolFile(ligand_path, removeHs=False)
                if molecule.HasProp("_Name"):
                    mol_name = molecule.GetProp("_Name")
                else: 
                    mol_name = os.path.basename(ligand_path).split('.')[0].lower()


            old_mol_name = mol_name
            # if mol_name is less than 2 characters
            # add ind to the end
            # otherwise tleap will fail later
            if len(mol_name) <= 2:
                mol_name = f'{mol_name}{ind}'
                if mol_name in self.unique_mol_names:
                    mol_name = f'{mol_name[:2]}{ind}'
                    if mol_name in self.unique_mol_names:
                        raise ValueError(f"Cannot find a unique name for the ligand: {mol_name}")
                logger.warning(f"Elongating the ligand name: {old_mol_name} to {mol_name}{ind}")
            elif len(mol_name) > 3:
                mol_name = mol_name[:3]
                if mol_name in self.unique_mol_names:
                    mol_name = f'{mol_name[:2]}{ind}'
                    if mol_name in self.unique_mol_names:
                        raise ValueError(f"Cannot find a unique name for the ligand: {mol_name}")
                logger.warning(f"Shortening the ligand name: {old_mol_name} to {mol_name[:3]}")
            self._mol = mol_name
            self.mols.append(mol_name)
            self.unique_mol_names.append(mol_name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{self._mol}.frcmod"):
                logger.debug(f'Processing ligand: {self._mol}')
                if ligand_path.lower().endswith('.sdf'):
                    self._process_ligand_sdf()
                else:
                    logger.warning(f"A mol2 file is preferred for the ligand: {ligand_path}") 
                    self._process_ligand_pdb()
        self._prepare_ligand_poses()
        
        logger.info('System loaded and prepared')

    def _convert_2_relative_path(self, path):
        """
        Convert the path to a relative path to the output directory.
        """
        return os.path.relpath(path, self.output_dir)

    @property
    def protein_input(self):
        return f"{self.output_dir}/{self._protein_input}"

    @property
    def system_topology(self):
        return f"{self.output_dir}/{self._system_topology}"

    @property
    def system_coordinate(self):
        return f"{self.output_dir}/{self._system_coordinate}"

    @property
    def ligand_paths(self):
        return [f"{self.output_dir}/{ligand_path}" for ligand_path in self._ligand_paths]

    def _process_ligands(self):
        """
        Process the ligands to get the ligand paths.
        e.g., for ABFE, it will be a single ligand.
        For RBFE, it will be multiple ligands.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def _get_alignment(self):
        """
        Prepare for the alignment of the protein and ligand to the system.
        """
        logger.debug('Getting the alignment of the protein and ligand to the system')

        # translate the cog of protein to the origin
        # 
        u_prot = mda.Universe(self.protein_input)

        u_sys = self.u_sys
        cog_prot = u_sys.select_atoms('protein and name CA C N O').center_of_geometry()
        u_sys.atoms.positions -= cog_prot
        
        # get translation-rotation matrix
        mobile = u_prot.select_atoms(self.protein_align).select_atoms('name CA')
        ref = u_sys.select_atoms(self.protein_align).select_atoms('name CA')

        mobile_com = mobile.center(weights=None)
        ref_com = ref.center(weights=None)
        mobile_coord = mobile.positions - mobile_com
        ref_coord = ref.positions - ref_com

        _ = align._fit_to(
            mobile_coordinates=mobile_coord,
            ref_coordinates=ref_coord,
            mobile_atoms=u_prot.atoms,
            mobile_com=mobile_com,
            ref_com=ref_com)

        cog_prot = u_prot.select_atoms('protein and name CA C N O').center_of_geometry()
        u_prot.atoms.positions -= cog_prot
        self.translation = cog_prot

        self.u_prot = u_prot
        # store these for ligand alignment
        self.mobile_com = mobile_com
        self.ref_com = ref_com
        self.mobile_coord = mobile_coord
        self.ref_coord = ref_coord

    def _process_system(self):
        """
        Generate the protein, reference, and lipid (if applicable) files.
        We will align the protein_input to the system_topology because
        the system_topology is generated by dabble and may be shifted;
        we want to align the protein to the system so the membrane is 
        properly positioned.
        """
        logger.debug('Processing the system')
        u_prot = self.u_prot
        u_sys = self.u_sys
        try:
            u_sys.atoms.chainIDs
        except AttributeError:
            u_sys.add_TopologyAttr('chainIDs')

        memb_seg = u_sys.add_Segment(segid='MEMB')
        water_seg = u_sys.add_Segment(segid='WATR')

        membrane_ag = u_sys.select_atoms(f'resname {" ".join(self.lipid_mol)}')
        membrane_ag.chainIDs = 'M'
        membrane_ag.residues.segments = memb_seg
        logger.debug(f'Number of lipid molecules: {membrane_ag.n_residues}')
        water_ag = u_sys.select_atoms('byres (resname TIP3 and around 15 (protein or resname POPC))')
        water_ag.chainIDs = 'W'
        water_ag.residues.segments = water_seg
        logger.debug(f'Number of water molecules: {water_ag.n_residues}')

        # modify the chaininfo to be unique for each segment
        current_chain = 66
        u_prot.atoms.tempfactors = 0
        for segment in u_sys.select_atoms('protein').segments:
            resid_seg = segment.residues.resids
            resid_seq = " ".join([str(resid) for resid in resid_seg])
            chain_id = segment.atoms.chainIDs[0]
            u_prot.select_atoms(
                f'resid {resid_seq} and chainID {chain_id} and protein').atoms.tempfactors = current_chain
            current_chain += 1
        u_prot.atoms.chainIDs = [chr(int(chain_nm)) for chain_nm in u_prot.atoms.tempfactors]

        if self.receptor_segment:
            protein_anchor = u_prot.select_atoms(f'segid {self.receptor_segment} and protein')
            protein_anchor.atoms.chainIDs = 'A'
            protein_anchor.atoms.tempfactors = 65
            other_protein = u_prot.select_atoms(f'not segid {self.receptor_segment} and protein')
            u_merged = mda.Merge(protein_anchor,
                                 other_protein,
                                 membrane_ag,
                                 water_ag)
        else:
            u_merged = mda.Merge(u_prot.select_atoms('protein'),
                                 membrane_ag,
                                 water_ag)
        water = u_merged.select_atoms('resname TIP3')
        logger.debug(f'Number of water molecules in merged system: {water.n_residues}')
        logger.debug(f'Water atom names: {water.residues[0].atoms.names}')

        # Otherwise tleap cannot recognize the water molecules
        water.select_atoms('name OW').names = 'O'
        water.select_atoms('name OH2').names = 'O'

        box_dim = np.zeros(6)
        if len(self.system_dimensions) == 3:
            box_dim[:3] = self.system_dimensions
            box_dim[3:] = 90.0
        elif len(self.system_dimensions) == 6:
            box_dim = self.system_dimensions
        else:
            raise ValueError(f"Invalid system_dimensions: {self.system_dimensions}")
        u_merged.dimensions = box_dim
        # save as *_docked.pdb that matched `input-dd-amber.in`
        u_merged.atoms.write(f"{self.poses_folder}/{self.system_name}_docked.pdb")
        protein_ref = u_prot.select_atoms('protein')
        protein_ref.write(f"{self.poses_folder}/reference.pdb")

    def _process_ligand_sdf(self):
        """
        Process the ligand sdf, including adding or removing hydrogens as needed.
        """
        logger.debug(f'Processing ligand file: {self._ligand_path}')
        if not self.retain_lig_prot:
            raise NotImplementedError("Removing hydrogens from sdf is not supported yet")
        ligand_path = self._ligand_path
        mol = self._mol
        molecule = Molecule(self._ligand_path)
        molecule.assign_partial_charges(
            partial_charge_method='am1bcc'
        )

        self.ligand_charge = np.round(np.sum([charge._magnitude for charge in molecule.partial_charges]))
        logger.info(f'The net charge of the ligand {mol} in {ligand_path} is {self.ligand_charge}')

        molecule.to_file(f"{self.ligandff_folder}/{mol}.sdf", 'sdf')
        self._ligand_sdf_path = f"{self.ligandff_folder}/{mol}.sdf"
        self._prepare_ligand_parameters_sdf()

    def _prepare_ligand_parameters_sdf(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
        mol = self._mol
        logger.debug(f'Preparing ligand {mol} parameters')
        antechamber_command = f'{antechamber} -i {self._ligand_sdf_path} -fi sdf -o {self.ligandff_folder}/{mol}_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge} -rn {mol}'
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(antechamber_command, working_dir=tmpdir)
        shutil.copy(f"{self.ligandff_folder}/{mol}_ante.mol2", f"{self.ligandff_folder}/{mol}.mol2")
        self._ligand_mol2_path = f"{self.ligandff_folder}/{mol}.mol2"

        if self.ligand_ff == 'gaff':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/{mol}_ante.mol2 -f mol2 -o {self.ligandff_folder}/{mol}.frcmod -s 1')
        elif self.ligand_ff == 'gaff2':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/{mol}_ante.mol2 -f mol2 -o {self.ligandff_folder}/{mol}.frcmod -s 2')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(
                f'{antechamber} -i {self._ligand_sdf_path} -fi sdf -o {self.ligandff_folder}/{mol}_ante.pdb -fo pdb -rn {mol}', working_dir=tmpdir)

        # copy _ante.pdb to .pdb
        shutil.copy(f"{self.ligandff_folder}/{mol}_ante.pdb", f"{self.ligandff_folder}/{mol}.pdb")

        # get lib file
        tleap_script = f"""
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_ff}
        lig = loadmol2 {self.ligandff_folder}/{mol}.mol2
        loadamberparams {self.ligandff_folder}/{mol}.frcmod
        saveoff lig {self.ligandff_folder}/{mol}.lib
        saveamberparm lig {self.ligandff_folder}/{mol}.prmtop {self.ligandff_folder}/{mol}.inpcrd

        quit
        """
        with open(f"{self.ligandff_folder}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{tleap} -f tleap.in",
                        working_dir=self.ligandff_folder)

        logger.debug(f'Ligand {mol} parameters prepared')


    def _process_ligand_pdb(self):
        """
        Process the ligand PDB, including adding or removing hydrogens as needed.
        """

        # Ensure the ligand file is in PDB format
        logger.debug(f'Processing ligand file: {self._ligand_path}')

        ligand = self._ligand
        mol = self._mol
        ligand_path = f"{self.ligandff_folder}/{mol}.pdb"
        ligand.atoms.residues.resnames = mol
        ligand.atoms.write(ligand_path)

        # retain hydrogens from the ligand
        if not self.retain_lig_prot:
            # Remove hydrogens from the ligand
            noh_path = f"{self.ligandff_folder}/{mol}_noh.pdb"
            ligand.guess_TopologyAttrs(to_guess=['elements'])
            ligand.select_atoms('not hydrogen').write(noh_path)
            # Add hydrogens based on the specified pH
            logger.debug(f'The babel protonation of the ligand is for pH {self.ligand_ph:.2f}')
            run_with_log(f"{obabel} -i pdb {noh_path} -o pdb -O {self.ligandff_folder}/{mol}.pdb -p {self.ligand_ph:.2f}")
            
            ligand = mda.Universe(f"{self.ligandff_folder}/{mol}.pdb")
            
        ligand.guess_TopologyAttrs(to_guess=['elements'])
        guesser = DefaultGuesser(ligand)
        ligand.add_TopologyAttr('charges')
        ligand.atoms.charges = guesser.guess_gasteiger_charges(ligand.atoms)
        ligand.atoms.write(ligand_path)
        run_with_log(f"{obabel} -i pdb {ligand_path} -o mol2 -O {self.ligandff_folder}/{mol}.mol2")

        self._ligand_path = ligand_path
        self._ligand = mda.Universe(ligand_path)
        
        self._ligand_mol2_path = f"{self.ligandff_folder}/{mol}.mol2"

        self.ligand_charge = np.round(np.sum(ligand.atoms.charges))
        logger.info(f'The net charge of the ligand {mol} in {ligand_path} is {self.ligand_charge}')

        self._prepare_ligand_parameters_pdb()

    def _prepare_ligand_parameters_pdb(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
        # TODO: build a library of ligand parameters
        # and check if the ligand is in the library
        # if not, then prepare the ligand parameters

        mol = self._mol
        logger.debug(f'Preparing ligand {mol} parameters')

        # antechamber_command = f'{antechamber} -i {self.ligand_path} -fi pdb -o {self.ligandff_folder}/ligand_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge}'
        antechamber_command = f'{antechamber} -i {self._ligand_mol2_path} -fi mol2 -o {self.ligandff_folder}/{mol}_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge}'
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(antechamber_command, working_dir=tmpdir)
        shutil.copy(f"{self.ligandff_folder}/{mol}_ante.mol2", f"{self.ligandff_folder}/{mol}.mol2")
        self._ligand_mol2_path = f"{self.ligandff_folder}/{mol}.mol2"

        if self.ligand_ff == 'gaff':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/{mol}_ante.mol2 -f mol2 -o {self.ligandff_folder}/{mol}.frcmod -s 1')
        elif self.ligand_ff == 'gaff2':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/{mol}_ante.mol2 -f mol2 -o {self.ligandff_folder}/{mol}.frcmod -s 2')

        with tempfile.TemporaryDirectory() as tmpdir:
            #    run_with_log(f'{antechamber} -i {self.ligand_path} -fi pdb -o {self.ligandff_folder}/{mol}_ante.pdb -fo pdb', working_dir=tmpdir)
            run_with_log(
                f'{antechamber} -i {self._ligand_mol2_path} -fi mol2 -o {self.ligandff_folder}/{mol}_ante.pdb -fo pdb', working_dir=tmpdir)
        # copy _ante.pdb to .pdb
        shutil.copy(f"{self.ligandff_folder}/{mol}_ante.pdb", f"{self.ligandff_folder}/{mol}.pdb")

        # get lib file
        tleap_script = f"""
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_ff}
        lig = loadmol2 {self.ligandff_folder}/{mol}.mol2
        loadamberparams {self.ligandff_folder}/{mol}.frcmod
        saveoff lig {self.ligandff_folder}/{mol}.lib
        saveamberparm lig {self.ligandff_folder}/{mol}.prmtop {self.ligandff_folder}/{mol}.inpcrd

        quit
        """
        with open(f"{self.ligandff_folder}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{tleap} -f tleap.in",
                        working_dir=self.ligandff_folder)

        logger.debug(f'Ligand {mol} parameters prepared')

    def _prepare_ligand_poses(self):
        """
        Prepare ligand poses for the system.
        """
        logger.debug('ligand poses')

        new_pose_paths = []
        for i, pose in enumerate(self.ligand_paths):
            if len(self.unique_mol_names) > 1:
                mol_name = self.unique_mol_names[i]
            else:
                mol_name = self.unique_mol_names[0]
            # align to the system
            if not pose.lower().endswith('.sdf'):
                u = mda.Universe(pose)
            else:
                molecule = Chem.MolFromMolFile(pose, removeHs=False)
                u = mda.Universe(molecule)
                u.add_TopologyAttr('resnames')
            try:
                u.atoms.chainIDs
            except AttributeError:
                u.add_TopologyAttr('chainIDs')
            lig_seg = u.add_Segment(segid='LIG')
            u.atoms.chainIDs = 'L'
            u.atoms.residues.segments = lig_seg
            u.atoms.residues.resnames = mol_name
            
            self._align_2_system(u.atoms)
            u.atoms.write(f"{self.poses_folder}/pose{i}.pdb")
            pose = f"{self.poses_folder}/pose{i}.pdb"

            if not self.retain_lig_prot:
                noh_path = f"{self.poses_folder}/pose{i}_noh.pdb"
                run_with_log(f"{obabel} -i pdb {pose} -o pdb -O {noh_path} -d")

                # Add hydrogens based on the specified pH
                run_with_log(f"{obabel} -i pdb {noh_path} -o pdb -O {pose} -p {self.ligand_ph:.2f}")

            if not os.path.exists(f"{self.poses_folder}/pose{i}.pdb"):
                shutil.copy(pose, f"{self.poses_folder}/pose{i}.pdb")

            new_pose_paths.append(f"{self.poses_folder}/pose{i}.pdb")

        self._ligand_paths = [self._convert_2_relative_path(pose) for pose in new_pose_paths]

    def _align_2_system(self, mobile_atoms):

        _ = align._fit_to(
            mobile_coordinates=self.mobile_coord,
            ref_coordinates=self.ref_coord,
            mobile_atoms=mobile_atoms,
            mobile_com=self.mobile_com,
            ref_com=self.ref_com)

        # need to translate the mobile_atoms to the system
        mobile_atoms.positions -= self.translation

    def _prepare_membrane(self):
        """
        Prepare the membrane by converting CHARMM or 
        conventional lipid names into lipid21 names
        which e.g. for POPC, it will be PC, PA, OL.
        see: https://ambermd.org/AmberModels_lipids.php
        """
        logger.debug('Input: membrane system')

        # read charmmlipid2amber file
        charmm_csv_path = resources.files("batter") / "data/charmmlipid2amber.csv"
        charmm_amber_lipid_df = pd.read_csv(charmm_csv_path, header=1, sep=',')

        lipid_mol = self.lipid_mol
        logger.debug(f'Converting lipid input: {lipid_mol}')
        amber_lipid_mol = charmm_amber_lipid_df.query('residue in @lipid_mol')['replace']
        amber_lipid_mol = amber_lipid_mol.apply(lambda x: x.split()[1]).unique().tolist()

        # extend instead of replacing so that we can have both
        lipid_mol.extend(amber_lipid_mol)
        self.lipid_mol = lipid_mol
        logger.debug(f'New lipid_mol list: {self.lipid_mol}')

    def _get_sim_config(self,
                       input_file: Union[str, Path, SimulationConfig]
    ):
        if isinstance(input_file, (str, Path)):
            file_path = Path(input_file) if isinstance(input_file, str) else input_file
            sim_config: SimulationConfig  = get_configure_from_file(file_path)
        elif isinstance(input_file, SimulationConfig):
            sim_config = input_file
        else:
            raise ValueError(f"Invalid input_file: {input_file}")
        logger.info(f'Simulation configuration: {sim_config}')
        if sim_config.lipid_ff != self.lipid_ff:
            logger.warning(f"Different lipid_ff in the input: {sim_config.lipid_ff}\n"
                             f"System is prepared with {self.lipid_ff}")
        if sim_config.ligand_ff != self.ligand_ff:
            logger.warning(f"Different ligand_ff in the input: {sim_config.ligand_ff}\n"
                                f"System is prepared with {self.ligand_ff}")
        sim_config_retain_lig_prot = sim_config.retain_lig_prot == 'yes'
        if sim_config_retain_lig_prot != self.retain_lig_prot:
            logger.warning(f"Different retain_lig_prot in the input: {sim_config.retain_lig_prot}\n"
                            f"System is prepared with {self.retain_lig_prot}")
        
        if sim_config.fe_type == 'relative' and not isinstance(self, RBFESystem):
            raise ValueError(f"Invalid fe_type: {sim_config.fe_type}, "
                 "should be 'relative' for RBFE system")
                 
        self.sim_config = sim_config

    @save_state
    def prepare(self,
            stage: str,
            input_file: Union[str, Path, SimulationConfig],
            overwrite: bool = False,
            partition: str = 'rondror',
            ):
        """
        Prepare the system for the FEP simulation.

        Parameters
        ----------
        stage : str
            The stage of the simulation. Options are 'equil' and 'fe'.
        input_file : str
            Path to the input file for the simulation.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        partition : str, optional
            The partition to submit the job. Default is 'rondror'.
        """
        logger.debug('Preparing the system')
        self.overwrite = overwrite
        self.builders_factory = BuilderFactory()
        self.partition = partition

        self._get_sim_config(input_file)
        sim_config = self.sim_config
        
        if len(self.sim_config.poses_def) != len(self.ligand_paths):
            logger.warning(f"Number of poses in the input file: {len(self.sim_config.poses_def)} "
                           f"does not match the number of ligands: {len(self.ligand_paths)}")
            logger.warning(f"Using the ligand paths for the poses")
        self.sim_config.poses_def = [f'pose{i}' for i in range(len(self.ligand_paths))]

        if stage == 'equil':
            if self.overwrite:
                logger.debug(f'Overwriting {self.equil_folder}')
                shutil.rmtree(self.equil_folder, ignore_errors=True)
                self._eq_prepared = False
            elif self._eq_prepared:
                logger.info(f'Equilibration already prepared')
                return
            # save the input file to the equil directory
            os.makedirs(f"{self.equil_folder}", exist_ok=True)
            with open(f"{self.equil_folder}/sim_config.json", 'w') as f:
                json.dump(self.sim_config.model_dump(), f, indent=2)
            
            self._prepare_equil_system()
            logger.info('Equil System prepared')
            self._eq_prepared = True
        
        if stage == 'fe':
            if not os.path.exists(f"{self.equil_folder}"):
                raise FileNotFoundError(f"Equilibration not generated yet. Run prepare('equil') first.")
        
            if not os.path.exists(f"{self.equil_folder}/{self.sim_config.poses_def[0]}/md03.rst7"):
                raise FileNotFoundError(f"Equilibration not finished yet. First run the equilibration.")
                
            sim_config_eq = json.load(open(f"{self.equil_folder}/sim_config.json"))
            if sim_config_eq != sim_config.model_dump():
            # raise ValueError(f"Equilibration and free energy simulation configurations are different")
                warnings.warn(f"Equilibration and free energy simulation configurations are different")
                # get the difference
                diff = {k: v for k, v in sim_config_eq.items() if sim_config.model_dump().get(k) != v}
                logger.warning(f"Different configurations: {diff}")
                orig = {k: sim_config.model_dump().get(k) for k in diff.keys()}
                logger.warning(f"Original configuration: {orig}")
            if self.overwrite:
                logger.debug(f'Overwriting {self.fe_folder}')
                shutil.rmtree(self.fe_folder, ignore_errors=True)
                self._fe_prepared = False
            elif self._fe_prepared:
                logger.info(f'Free energy already prepared')
                return
            os.makedirs(f"{self.fe_folder}", exist_ok=True)

            with open(f"{self.fe_folder}/sim_config.json", 'w') as f:
                json.dump(self.sim_config.model_dump(), f, indent=2)

            self._prepare_fe_system()
            logger.info('FE System prepared')
            self._fe_prepared = True

    @save_state
    def submit(self,
               stage: str,
               cluster: str = 'slurm',
               partition=None,
               overwrite: bool = False,
               ):
        """
        Submit the simulation to the cluster.

        Parameters
        ----------
        stage : str
            The stage of the simulation. Options are 'equil' and 'fe'.
        cluster : str
            The cluster to submit the simulation.
            Options are 'slurm' and 'frontier'.
        partition : str, optional
            The partition to submit the job. Default is None,
            which means the default partition during prepartiion
            will be used.
        overwrite : bool, optional
            Whether to overwrite and re-run all the existing simulations.
        """
        n_jobs_submitted = 0
        if cluster == 'frontier':
            self._submit_frontier(stage)
            logger.info(f'Frontier {stage} job submitted!')
            return

        if stage == 'equil':
            logger.info('Submit equilibration stage')
            for pose in self.sim_config.poses_def:
                # check n_jobs_submitted is less than the max_jobs
                while n_jobs_submitted >= self._max_num_jobs:
                    time.sleep(120)
                    n_jobs_submitted = sum([1 for job in self._slurm_jobs.values() if job.is_still_running()])

                # check existing jobs
                if os.path.exists(f"{self.equil_folder}/{pose}/FINISHED") and not overwrite:
                    logger.debug(f'Equilibration for {pose} has finished; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'equil_{pose}', None)
                    continue
                if os.path.exists(f"{self.equil_folder}/{pose}/FAILED") and not overwrite:
                    logger.warning(f'Equilibration for {pose} has failed; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'equil_{pose}', None)
                    continue
                if f'equil_{pose}' in self._slurm_jobs:
                    # check if it's finished
                    slurm_job = self._slurm_jobs[f'equil_{pose}']
                    # if the job is finished but the FINISHED file is not created
                    # resubmit the job
                    if not slurm_job.is_still_running():
                        slurm_job.submit()
                        n_jobs_submitted += 1
                        continue
                    elif overwrite:
                        slurm_job.cancel()
                        slurm_job.submit()
                        n_jobs_submitted += 1
                        continue
                    else:
                        logger.info(f'Equilibration job for {pose} is still running')
                        continue

                if overwrite:
                    # remove FINISHED and FAILED
                    os.remove(f"{self.equils_folder}/{pose}/FINISHED", ignore_errors=True)
                    os.remove(f"{self.equils_folder}/{pose}/FAILED", ignore_errors=True)

                slurm_job = SLURMJob(
                                filename=f'{self.equil_folder}/{pose}/SLURMM-run',
                                partition=partition)
                slurm_job.submit()
                n_jobs_submitted += 1
                logger.info(f'Equilibration job for {pose} submitted')
                self._slurm_jobs.update(
                    {f'equil_{pose}': slurm_job}
                )

            logger.info('Equilibration systems have been submitted for all poses listed in the input file.')

        elif stage == 'fe':
            logger.info('Submit free energy stage')
            for pose in self.sim_config.poses_def:
                #shutil.copy(f'{self.fe_folder}/{pose}/rest/run_files/run-express.bash',
                #            f'{self.fe_folder}/{pose}')
                #run_with_log(f'bash run-express.bash',
                #            working_dir=f'{self.fe_folder}/{pose}')
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    windows = self.sim_config.attach_rest if comp_folder == 'rest' else self.sim_config.lambdas
                    for j in range(len(windows)):
                        if n_jobs_submitted >= self._max_num_jobs:
                            time.sleep(120)
                            n_jobs_submitted = sum([1 for job in self._slurm_jobs.values() if job.is_still_running()])
                        folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}'
                        if os.path.exists(f"{folder_2_check}/FINISHED") and not overwrite:
                            self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}{j:02d}', None)
                            logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                            continue
                        if os.path.exists(f"{folder_2_check}/FAILED") and not overwrite:
                            self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}{j:02d}', None)
                            logger.warning(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has failed; add overwrite=True to re-run the simulation')
                            continue
                        if f'fe_{pose}_{comp_folder}_{comp}{j:02d}' in self._slurm_jobs:
                            slurm_job = self._slurm_jobs[f'fe_{pose}_{comp_folder}_{comp}{j:02d}']
                            if not slurm_job.is_still_running():
                                slurm_job.submit()
                                n_jobs_submitted += 1
                                continue
                            elif overwrite:
                                slurm_job.cancel()
                                slurm_job.submit()
                                n_jobs_submitted += 1
                                continue
                            else:
                                logger.info(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} is still running')
                                continue
                        if overwrite:
                            # remove FINISHED and FAILED
                            os.remove(f"{folder_2_check}/FINISHED", ignore_errors=True)
                            os.remove(f"{folder_2_check}/FAILED", ignore_errors=True)

                        slurm_job = SLURMJob(
                                        filename=f'{folder_2_check}/SLURMM-run',
                                        partition=partition)
                        slurm_job.submit()
                        n_jobs_submitted += 1
                        logger.info(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} submitted')
                        self._slurm_jobs.update(
                            {f'fe_{pose}_{comp_folder}_{comp}{j:02d}': slurm_job}
                        )

            logger.info('Free energy systems have been submitted for all poses listed in the input file.')
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _submit_frontier(self, stage: str):
        if stage == 'equil':
            raise NotImplementedError("Frontier submission is not implemented yet")
        elif stage == 'fe':
            running_fe_equi = False
            for pose in self.sim_config.poses_def:
                if not os.path.exists(f"{self.fe_folder}/{pose}/rest/m00/eqnpt.in_04.rst7"):
                    run_with_log(f'sbatch fep_m_{pose}_eq.sbatch',
                            working_dir=f'{self.fe_folder}')
                    run_with_log(f'sbatch fep_n_{pose}_eq.sbatch',
                            working_dir=f'{self.fe_folder}')
                    run_with_log(f'sbatch fep_e_{pose}_eq.sbatch',
                            working_dir=f'{self.fe_folder}')
                    run_with_log(f'sbatch fep_v_{pose}_eq.sbatch',
                            working_dir=f'{self.fe_folder}')
                    running_fe_equi = True
            if running_fe_equi:
                return
                                  
            if not os.path.exists(f"{self.fe_folder}/current_mdin.groupfile"):
                run_with_log(f'sbatch fep_md.sbatch',
                            working_dir=f'{self.fe_folder}')
                
            run_with_log(f'sbatch fep_md_extend.sbatch',
                        working_dir=f'{self.fe_folder}')
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _prepare_equil_system(self):
        """
        Prepare the equilibration system.
        """
        sim_config = self.sim_config

        logger.info(f'Prepare for equilibration stage at {self.equil_folder}')
        if not os.path.exists(f"{self.equil_folder}/all-poses"):
            logger.debug(f'Copying all-poses folder from {self.poses_folder} to {self.equil_folder}/all-poses')
            shutil.copytree(self.poses_folder,
                        f"{self.equil_folder}/all-poses")
        if not os.path.exists(f"{self.equil_folder}/ff"):
            logger.debug(f'Copying ff folder from {self.ligandff_folder} to {self.equil_folder}/ff')
            shutil.copytree(self.ligandff_folder,
                        f"{self.equil_folder}/ff")

        for pose in self.sim_config.poses_def:
            logger.info(f'Preparing pose: {pose}')
            equil_builder = self.builders_factory.get_builder(
                stage='equil',
                system=self,
                pose=pose,
                sim_config=sim_config,
                working_dir=f'{self.equil_folder}'
            ).build()
    
        logger.info('Equilibration systems have been created for all poses listed in the input file.')

    def _prepare_fe_system(self):
        """
        Prepare the free energy system.
        """
        # raise NotImplementedError("Free energy system preparation is not implemented yet")
        sim_config = self.sim_config

        logger.info('Prepare for free energy stage')
        # molr (molecule reference) and poser (pose reference)
        # are used for exchange FE simulations.

        molr = self.mols[0]
        poser = self.sim_config.poses_def[0]


        pbar = tqdm(total=len(self.sim_config.poses_def))
        for pose in self.sim_config.poses_def:
            logger.debug(f'Preparing pose: {pose}')
            # copy ff folder
            shutil.copytree(self.ligandff_folder,
                            f"{self.fe_folder}/{pose}/ff", dirs_exist_ok=True)

            for component in sim_config.components:
                logger.debug(f'Preparing component: {component}')
                lambdas_comp = sim_config.dict()[COMPONENTS_LAMBDA_DICT[component]]
                n_sims = len(lambdas_comp)
                logger.debug(f'Number of simulations: {n_sims}')
                for i, lambdas in enumerate(lambdas_comp):
                    logger.debug(f'Preparing simulation: {lambdas}')
                    pbar.set_description(f"Preparing pose={pose}, comp={component}, win={lambdas}")
                    fe_builder = self.builders_factory.get_builder(
                        stage='fe',
                        win=i,
                        component=component,
                        system=self,
                        pose=pose,
                        sim_config=sim_config,
                        working_dir=f'{self.fe_folder}',
                        molr=molr,
                        poser=poser
                    ).build()
            pbar.update(1)

    def add_rmsf_restraints(self,
                            stage: str,
                            avg_struc: str,
                            rmsf_file: str,
                            force_constant: float = 100):
        """
        Add RMSF restraints to the system.
        Similar to https://pubs.acs.org/doi/10.1021/acs.jctc.3c00899?ref=pdf

        Steps to generate required files:

        1. Get conformational ensemble from unbiased production simulatinos
        e.g. For inactive and active states, run unbiased production simulations
        of both states. Calculate representative distances in the binding pocket,
        then use them to structure the conformational space
        into distinct clusters and to compare the two ensembles.

        Eventually, you will generate a trajectory of representative structures
        for each state/cluster.

        2. Calculate the RMSF of the representative structures

        ```python
        from MDAnalysis.analysis import rms, align
        
        u = mda.Universe('state0.pdb', 'state0.xtc')
        gpcr_sel = 'protein and name CA'
        average = align.AverageStructure(u, u, select=gpcr_sel,
                                        ref_frame=0).run()
        ref = average.results.universe
        aligner = align.AlignTraj(u, ref,
                                select=gpcr_sel,
                                in_memory=True).run()
        
        R = rms.RMSF(gpcr_atoms).run()

        ref.atoms.write('state0_avg.pdb')

        rmsf_values = R.results.rmsf
        with open('state0_rmsf.txt', 'w') as f:
            for resid, rmsf in zip(gpcr_atoms.resids, rmsf_values):
                f.write(f'{resid} {rmsf}\n')
        ```

        3. Use the RMSF values to create flat bottom restraints
        for the FEP simulations with
        `system.add_rmsf_restraints(
            avg_struc='state0_avg.pdb',
            rmsf_file='state1_rmsf.txt')`

        The restraints will be stored in `cv.in` file with a format
 
        ```bash
        &colvar 
        cv_type = 'DISTANCE_TO_COORD' 
        cv_ni = 1, cv_i = 1
        cv_nr = 4, cv_r = ref_x, ref_y, ref_z, rmsf_value
        anchor_position = 0, 0, rmsf_value, 999
        anchor_strength = 0, force_constant
        / 
        ```
        where rmsf_value is the RMSF value of the residue and 
        ref_x, ref_y, ref_z are the coordinates of the residue
        in the average structure.

        **You need to use the modified version of amber24
        (`$GROUP_HOME/software/amber24`) to use the RMSF restraints.**

        Parameters
        ----------
        stage : str
            The stage of the simulation.
            Options are 'equil' and 'fe'.
        avg_struc : str
            The path of the average structure of the
            representative conformations.
        rmsf_file : str
            The path of the RMSF file.
        force_constant : float, optional
            The force constant of the restraints. Default is 100.
        """
        logger.debug('Adding RMSF restraints')

        def generate_colvar_block(atm_index,
                                  dis_cutoff,
                                  ref_position,
                                  force_constant=100):
            colvar_block = "&colvar\n"
            colvar_block += " cv_type = 'DISTANCE_TO_COORD'\n"
            colvar_block += f" cv_ni = 1, cv_i = {atm_index}\n"
            colvar_block += f" cv_nr = 4, cv_r = {ref_position[0]:2f}, {ref_position[1]:2f}, {ref_position[2]:2f}, {dis_cutoff:2f}\n"
            colvar_block += f" anchor_position = 0, 0, {dis_cutoff}, 999\n"
            colvar_block += f" anchor_strength = 0, {force_constant:2f}\n"
            colvar_block += "/\n"
            return colvar_block

        def write_colvar_block(ref_u, cv_files):

            avg_u = mda.Universe(avg_struc)
            gpcr_sel = 'protein and name CA'

            aligner = align.AlignTraj(
                        avg_u, ref_u,
                        select=gpcr_sel,
                        match_atoms=False,
                        in_memory=True).run()

            ref_pos = np.zeros([avg_u.atoms.n_atoms, 3])

            rmsf_values = np.loadtxt(rmsf_file)
            gpcr_ref = ref_u.select_atoms(gpcr_sel)

            cv_lines = []
            for i, atm in enumerate(avg_u.atoms):
                ref_pos[i] = atm.position
                resid_i = atm.resid
                rmsf_val = rmsf_values[rmsf_values[:, 0] == resid_i, 1]
                if len(rmsf_val) == 0:
                    logger.warning(f"resid: {resid_i} not found in rmsf file")
                    continue
                # print(f"resid: {resid_i}, rmsf: {rmsf_val[0]} Å")
                # print(f"ref_pos: {ref_pos[i]}")
                atm_index = gpcr_ref[i].index + 1
        #        print(generate_colvar_block(atm_index, rmsf_val[0], ref_pos[i]))
                cv_lines.append(generate_colvar_block(atm_index, rmsf_val[0], ref_pos[i]))
            
            for cv_file in cv_files:
                
                # if .bak exists, to avoid multiple appending
                # first copy the original cv file to the backup
                if os.path.exists(cv_file + '.bak'):
                    shutil.copy(cv_file + '.bak', cv_file)
                else:
                    # copy original cv file for backup
                    shutil.copy(cv_file, cv_file + '.bak')
                
                with open(cv_file, 'r') as f:
                    lines = f.readlines()

                with open(cv_file + '.eq0', 'w') as f:
                    for line in lines:
                        f.write(line)
                    f.write("\n")
                    for line in cv_lines:
                        f.write(line)
                
                for i, line in enumerate(lines):
                    if 'anchor_strength' in line:
                        lines[i] = line.replace('anchor_strength =    10.0000,    10.0000,',
                                    f'anchor_strength =    0,   0,')
                        break
                    
                with open(cv_file, 'w') as f:
                    for line in lines:
                        f.write(line)
                    f.write("\n")
                    for line in cv_lines:
                        f.write(line)
                
        if stage == 'equil':
            for pose in self.sim_config.poses_def:
                u_ref = mda.Universe(
                        f"{self.equil_folder}/{pose}/full.pdb",
                        f"{self.equil_folder}/{pose}/full.inpcrd")

                cv_files = [f"{self.equil_folder}/{pose}/cv.in"]
                write_colvar_block(u_ref, cv_files)
                
                eqnpt0 = f"{self.equil_folder}/{pose}/eqnpt0.in"
                
                with open(eqnpt0, 'r') as f:
                    lines = f.readlines()
                with open(eqnpt0, 'w') as f:
                    for line in lines:
                        if 'cv.in' in line and 'cv.in.eq0' not in line:
                            f.write(line.replace('cv.in', 'cv.in.eq0'))
                        else:
                            f.write(line)
                
                eqnpt = f"{self.equil_folder}/{pose}/eqnpt.in"
                
                with open(eqnpt, 'r') as f:
                    lines = f.readlines()
                with open(eqnpt, 'w') as f:
                    for line in lines:
                        if 'cv.in' in line and 'cv.in.bak' not in line:
                            f.write(line.replace('cv.in', 'cv.in.bak'))
                        else:
                            f.write(line)

        elif stage == 'fe':
            for pose in self.sim_config.poses_def:
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    u_ref = mda.Universe(
                            f"{folder_comp}/{comp}00/full.pdb",
                            f"{folder_comp}/{comp}00/full.inpcrd")
                    windows = self.sim_config.attach_rest if comp_folder == 'rest' else self.sim_config.lambdas
                    cv_files = [f"{folder_comp}/{comp}{j:02d}/cv.in"
                        for j in range(0, len(windows))]
                    
                    write_colvar_block(u_ref, cv_files)
                    
                    eq_in_files = glob.glob(f"{folder_comp}/*/eqnpt0.in")
                    for eq_in_file in eq_in_files:
                        with open(eq_in_file, 'r') as f:
                            lines = f.readlines()
                        with open(eq_in_file, 'w') as f:
                            for line in lines:
                                if 'cv.in' in line and 'cv.in.eq0' not in line:
                                    f.write(line.replace('cv.in', 'cv.in.eq0'))
                                else:
                                    f.write(line)
                    
                    eq_in_files = glob.glob(f"{folder_comp}/*/eqnpt.in")
                    for eq_in_file in eq_in_files:
                        with open(eq_in_file, 'r') as f:
                            lines = f.readlines()
                        with open(eq_in_file, 'w') as f:
                            for line in lines:
                                if 'cv.in' in line and 'cv.in.bak' not in line:
                                    f.write(line.replace('cv.in', 'cv.in.bak'))
                                else:
                                    f.write(line)
        else:
            raise ValueError(f"Invalid stage: {stage}")
        logger.debug('RMSF restraints added')

    @save_state
    def analysis(
        self,
        input_file: Union[str, Path, SimulationConfig]=None,
        load=True
        ):
        """
        Analyze the simulation results.

        Parameters
        ----------
        input_file : str
            The input file for the simulation.
            Default is None and will use the input file
            used for the simulation.
        load : bool, optional
            Whether to load the system fe results from before
            Default is True.
        """
        if load:
            for pose in self.sim_config.poses_def:
                self.fe_results[pose] = FEResult(f'{self.fe_folder}/{pose}/Results/Results.dat')
                logger.info(f'{pose}:\t{self.fe_results[pose].fe:.2f} ± {self.fe_results[pose].fe_std:.2f}')
            return
        if input_file is not None:
            self._get_sim_config(input_file)
            
        blocks = self.sim_config.blocks
        components = self.sim_config.components
        temperature = self.sim_config.temperature
        attach_rest = self.sim_config.attach_rest
        lambdas = self.sim_config.lambdas
        weights = self.sim_config.weights
        dec_int = self.sim_config.dec_int
        dec_method = self.sim_config.dec_method
        rest = self.sim_config.rest
        dic_steps1 = self.sim_config.dic_steps1
        dic_steps2 = self.sim_config.dic_steps2
        dt = self.sim_config.dt
        poses_def = self.sim_config.poses_def

        with self._change_dir(self.output_dir):
            for pose in tqdm(poses_def, desc='Analyzing FE for poses'):
                fe_value, fe_std = analysis.fe_values(blocks, components, temperature, pose, attach_rest, lambdas,
                                weights, dec_int, dec_method, rest, dic_steps1, dic_steps2, dt)
                self.fe_results[pose] = FEResult('Results/Results.dat')
                os.chdir('../../')
        for i, (pose, fe) in enumerate(self.fe_results.items()):
            mol_name = self.mols[i]
            logger.info(f'{mol_name}\t{pose}\t{fe.fe:.2f} ± {fe.fe_std:.2f}')
        
    @staticmethod
    def _find_anchor_atoms(u_prot,
                           u_lig,
                           anchor_atoms,
                           ligand_anchor_atom):
        """
        Function to find the anchor atoms for the ligand
        and the protein.

        Parameters
        ----------
        u_prot : mda.Universe
            The protein universe.
        u_lig : mda.Universe
            The ligand universe.
        anchor_atoms : List[str]
            The list of three protein anchor atoms (selection strings)
            used to restrain ligand.
        ligand_anchor_atom : str
            The ligand anchor atom (selection string) used as a potential
            ligand anchor atom.
        
        Returns
        -------
        l1_x : float
            The x distance of the ligand anchor atom from the protein anchor atom.
        l1_y : float
            The y distance of the ligand anchor atom from the protein anchor atom.
        l1_z : float
            The z distance of the ligand anchor atom from the protein anchor atom.
        p1_formatted : str
            The formatted string of the first protein anchor atom.
        p2_formatted : str
            The formatted string of the second protein anchor atom.
        p3_formatted : str
            The formatted string of the third protein anchor atom.
        """

        u_merge = mda.Merge(u_prot.atoms, u_lig.atoms)
        P1_atom = u_merge.select_atoms(anchor_atoms[0])
        P2_atom = u_merge.select_atoms(anchor_atoms[1])
        P3_atom = u_merge.select_atoms(anchor_atoms[2])
        if P1_atom.n_atoms != 1 or P2_atom.n_atoms != 1 or P3_atom.n_atoms != 1:
            raise ValueError('Error: more than one atom selected in the anchor atoms')
        if P1_atom.n_atoms == 0 or P2_atom.n_atoms == 0 or P3_atom.n_atoms == 0:
            raise ValueError('Error: anchor atom not found')
        
        if ligand_anchor_atom is not None:
            lig_atom = u_merge.select_atoms(ligand_anchor_atom)
            if lig_atom.n_atoms == 0:
                raise ValueError('Error: ligand anchor atom not found')
        else:
            lig_atom = u_lig.atoms

        # get ll_x,y,z distances
        r_vect = lig_atom.center_of_mass() - P1_atom.positions
        logger.info(f'l1_x: {r_vect[0][0]:.2f}')
        logger.info(f'l1_y: {r_vect[0][1]:.2f}')
        logger.info(f'l1_z: {r_vect[0][2]:.2f}')

        p1_formatted = f':{P1_atom.resids[0]}@{P1_atom.names[0]}'
        p2_formatted = f':{P2_atom.resids[0]}@{P2_atom.names[0]}'
        p3_formatted = f':{P3_atom.resids[0]}@{P3_atom.names[0]}'
        logger.info(f'Receptor anchor atoms: P1: {p1_formatted}, P2: {p2_formatted}, P3: {p3_formatted}')
        return (r_vect[0][0], r_vect[0][1], r_vect[0][2],
                p1_formatted, p2_formatted, p3_formatted)
              

    @save_state
    def run_pipeline(self,
                     input_file: Union[str, Path, SimulationConfig],
                     overwrite: bool = False,              
                     avg_struc: str = None,
                     rmsf_file: str = None,
                     only_equil: bool = False,
                     partition: str = 'owners',
                     anchor_atoms: List[str] = None,
                     ligand_anchor_atom: str = None,
                     max_num_jobs: int = 500
                     ):
        """
        Run the whole pipeline for calculating the binding free energy
        after you `create_system`.

        Parameters
        ----------
        input_file : str
            The input file for the simulation.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        avg_struc : str
            The path of the average structure of the
            representative conformations. Default is None,
            which means no RMSF restraints are added.
        rmsf_file : str
            The path of the RMSF file. Default is None,
            which means no RMSF restraints are added.
        only_equil : bool, optional
            Whether to run only the equilibration stage.
            Default is False.
        partition : str, optional
            The partition to submit the job.
            Default is 'rondror'.
        anchor_atoms : List[str], optional
            The list of three protein anchor atoms (selection strings)
            used to restrain ligand.
            It will also be used to set l1x, l1y, l1z values that defines
            the binding pocket.
            Default is None and will use the atoms listed in the input file.
        ligand_anchor_atom : str, optional
            The ligand anchor atom (selection string) used as a potential
            ligand anchor atom.
            Default is None and will use the atom that is closed to the 
            center of mass of the ligand.
            Note only the first ligand in the ligand_paths will be used
            to create the binding pocket.
        max_num_jobs : int, optional
            The maximum number of jobs to submit at a time.
            Default is 500.
        """
        logger.info('Running the pipeline')
        self._max_num_jobs = max_num_jobs
        if avg_struc is not None and rmsf_file is not None:
            rmsf_restraints = True
        elif avg_struc is not None or rmsf_file is not None:
            raise ValueError("Both avg_struc and rmsf_file should be provided")
        else:
            rmsf_restraints = False

        start_time = time.time()
        logger.info(f'Start time: {time.ctime()}')
        self._get_sim_config(input_file)
        
        if len(self.sim_config.poses_def) != len(self.ligand_paths):
            logger.warning(f"Number of poses in the input file: {len(self.sim_config.poses_def)} "
                           f"does not match the number of ligands: {len(self.ligand_paths)}")
            logger.warning(f"Using the ligand paths for the poses")
        self.sim_config.poses_def = [f'pose{i}' for i in range(len(self.ligand_paths))]
        
        if anchor_atoms is not None:
            u_prot = mda.Universe(f'{self.output_dir}/all-poses/reference.pdb')
            u_lig = mda.Universe(f'{self.output_dir}/all-poses/pose0.pdb')
            l1_x, l1_y, l1_z, p1, p2, p3 = self._find_anchor_atoms(
                        u_prot,
                        u_lig,
                        anchor_atoms,
                        ligand_anchor_atom)
        
            self.sim_config.l1_x = l1_x
            self.sim_config.l1_y = l1_y
            self.sim_config.l1_z = l1_z

            self.sim_config.p1 = p1
            self.sim_config.p2 = p2
            self.sim_config.p3 = p3

        if self._check_equilibration():
            #1 prepare the system
            logger.info('Preparing the system')
            self.prepare(
                stage='equil',
                input_file=self.sim_config,
                overwrite=overwrite,
                partition=partition
            )
            if rmsf_restraints:
                self.add_rmsf_restraints(
                    stage='equil',
                    avg_struc=avg_struc,
                    rmsf_file=rmsf_file
                )
            logger.info('Submitting the equilibration')
            #2 submit the equilibration
            self.submit(
                stage='equil',
            )

            # Check for equilibration to finish
            logger.info('Checking the equilibration')
            while self._check_equilibration():
                logger.info('Equilibration is still running. Waiting for 0.5 hour.')
                time.sleep(30*60)
        else:
            logger.info('Equilibration is already finished')
        if only_equil:
            logger.info('only_equil is set to True. '
                        'Skipping the free energy calculation.')
            return

        #4, submit the free energy calculation
        logger.info('Running free energy calculation')

        if self._check_fe():
            #3 prepare the free energy calculation
            self.prepare(
                stage='fe',
                input_file=self.sim_config,
                overwrite=overwrite,
                partition=partition
            )
            if rmsf_restraints:
                self.add_rmsf_restraints(
                    stage='fe',
                    avg_struc=avg_struc,
                    rmsf_file=rmsf_file
                )
            logger.info('Submitting the free energy calculation')
            self.submit(
                stage='fe',
            )
            # Check the free energy calculation to finish
            logger.info('Checking the free energy calculation')
            while self._check_fe():
                logger.info('Free energy calculation is still running. Waiting for 0.5 hour.')
                time.sleep(30*60)
        else:
            logger.info('Free energy calculation is already finished')

        #5 analyze the results
        logger.info('Analyzing the results')
        self.analysis()

        logger.info('Pipeline finished')
        logger.info(f'The results are in the {self.output_dir}')
        end_time = time.time()
        logger.info(f'End time: {time.ctime()}')
        total_time = end_time - start_time
        logger.info(f'Total time: {total_time:.2f} seconds')
        logger.info(f'Results')
        logger.info(f'---------------------------------')
        logger.info(f'Mol\tPose\tFree Energy (kcal/mol)')
        logger.info(f'---------------------------------')
        for i, (pose, fe) in enumerate(self.fe_results.items()):
            mol_name = self.mols[i]
            logger.info(f'{mol_name}\t{pose}\t{fe[0]:.2f} ± {fe[1]:.2f}')
        
    def _check_equilibration(self):
        """
        Check if the equilibration is finished by checking the FINISHED file
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.sim_config.poses_def:
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                sim_finished[pose] = False
            else:
                sim_finished[pose] = True
            if os.path.exists(f"{self.equil_folder}/{pose}/FAILED"):
                sim_failed[pose] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed

        if any(self._sim_failed.values()):
            logger.error(f'Equilibration failed: {self._sim_failed}')
            raise ValueError(f'Equilibration failed: {self._sim_failed}')
            
        if all(self._sim_finished.values()):
            logger.debug('Equilibration is finished')
            return False
        else:
            not_finished = [k for k, v in self._sim_finished.items() if not v]
            logger.debug(f'Not finished: {not_finished}')
            return True

    def _check_fe(self):
        """
        Check if the free energy calculation is finished by 
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.sim_config.poses_def:
            for comp in self.sim_config.components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]
                windows = self.sim_config.attach_rest if comp_folder == 'rest' else self.sim_config.lambdas
                for j in range(0, len(windows)):
                    folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}'
                    if not os.path.exists(f"{folder_2_check}/FINISHED"):
                        sim_finished[f'{pose}/{comp_folder}/{comp}{j:02d}'] = False
                    else:
                        sim_finished[f'{pose}/{comp_folder}/{comp}{j:02d}'] = True
                    if os.path.exists(f"{folder_2_check}/FAILED"):
                        sim_failed[f'{pose}/{comp_folder}/{comp}{j:02d}'] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy calculation failed: {self._sim_failed}')
            return True
        if all(self._sim_finished.values()):
            logger.debug('Free energy calculation is finished')
            return False
        else:
            not_finished = [k for k, v in self._sim_finished.items() if not v]
            logger.debug(f'Not finished: {not_finished}')
            return True

    def check_jobs(self):
        """
        Check the status of the jobs.
        """
        logger.info('Checking the status of the jobs in')
        logger.info(f'{self.output_dir}')
        if self._check_equilibration():
            logger.info('Equilibration is still running')
        else:
            if self._check_fe():
                logger.info('Free energy calculation is still running')
        
        not_finished = [k for k, v in self._sim_finished.items() if not v]

        if len(not_finished) == 0:
            logger.info('All jobs are finished')
        else:
            for pose in self.sim_config.poses_def:
                logger.info(f'Not finished in {pose}:')
                not_finished_pose = [k for k in not_finished if pose in k]
                not_finished_pose = [job.split('/')[-1] for job in not_finished_pose]
                logger.info(not_finished_pose)

    @property
    def poses_folder(self):
        return f"{self.output_dir}/all-poses"

    @property
    def ligandff_folder(self):
        return f"{self.output_dir}/ff"

    @property
    def equil_folder(self):
        return f"{self.output_dir}/equil"
    
    @property
    def fe_folder(self):
        return f"{self.output_dir}/fe"

    @property
    def analysis_folder(self):
        return f"{self.output_dir}/analysis"
    
    @property
    def fe_results(self):
        return self._fe_results

    @property
    def ligand_poses(self):
        return self.ligand_paths
    
    @contextmanager
    def _change_dir(self, new_dir):
        cwd = os.getcwd()
        os.makedirs(new_dir, exist_ok=True)
        os.chdir(new_dir)
        logger.debug(f'Changed directory to {os.getcwd()}')
        yield
        os.chdir(cwd)
        logger.debug(f'Changed directory back to {os.getcwd()}')
    
    @property
    def slurm_jobs(self):
        return self._slurm_jobs


class ABFESystem(System):
    """
    A class to represent and process a Absolute Binding Free Energy (ABFE) Perturbation (FEP) system
    using the BAT.py methods. It gets inputs of a protein and a single ligand type
    with the possibility of providing multiple poses of the ligand.
    The ABFE of the ligand of each binding poses 
    to the provided **protein conformation** will be calculated.

    If you have multiple ligands with different protein starting conformations,
    create multiple `ABFESystem`s.

    If you have multiple ligands with the same protein starting conformation,
    create one `MABFESystem` with multiple ligands as input.
    """
    def _process_ligands(self):
        # check if they are the same ligand
        n_atoms = mda.Universe(self.ligand_paths[0]).atoms.n_atoms
        for ligand_path in self.ligand_paths:
            if mda.Universe(ligand_path).atoms.n_atoms != n_atoms:
                raise ValueError(f"Number of atoms in the ligands are different: {ligand_path}")

        # set the ligand path to the first ligand
        self.unique_ligand_paths = [self.ligand_paths[0]]


class MABFESystem(System):
    """
    A class to represent and process a Absolute Binding Free Energy Perturbation (FEP) system
    using the BAT.py methods. It gets inputs of a protein and multiple single ligand types.
    The ABFE of the ligands to the provided **protein conformation** will be calculated
    """
    def _process_ligands(self):
        # check if they are the same ligand
        self.unique_ligand_paths = self.ligand_paths


class RBFESystem(System):
    """
    A class to represent and process a Relative Binding Free Energy Perturbation (FEP) system
    using the separated topology methodology in BAT.py.
    """
    def _process_ligands(self):
        self.unique_ligand_paths = self.ligand_paths
        if len(self.unique_ligand_paths) <= 1:
            raise ValueError("RBFESystem requires at least two ligands "
                             "for the relative binding free energy calculation")
        logger.info(f'Reference ligand: {self.unique_ligand_paths[0]}')

