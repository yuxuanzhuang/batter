"""
Provide the primary functions for preparing and processing FEP systems.
"""

from .utils import (
    run_with_log,
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
import time
from tqdm import tqdm

from typing import List, Tuple
import loguru
from loguru import logger

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.bat_lib import analysis

from batter.builder import BuilderFactory
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEC_FOLDER_DICT = {
    'dd': 'dd',
    'sdr': 'sdr',
    'exchange': 'sdr',
}

AVAILABLE_COMPONENTS = ['v', 'e', 'w', 'f',
                       'x', 'a', 'l', 't',
                       'r', 'c', 'm', 'n']


COMPONENTS_FOLDER_DICT = {
    'v': 'sdr',
    'e': 'sdr',
    'w': 'sdr',
    'f': 'sdr',
    'x': 'exchange_files',
    'a': 'rest',
    'l': 'rest',
    't': 'rest',
    'r': 'rest',
    'c': 'rest',
    'm': 'rest',
    'n': 'rest',
}

COMPONENTS_LAMBDA_DICT = {
    'v': 'lambdas',
    'e': 'lambdas',
    'w': 'lambdas',
    'f': 'lambdas',
    'x': 'lambdas',
    'a': 'attach_rest',
    'l': 'attach_rest',
    't': 'attach_rest',
    'r': 'attach_rest',
    'c': 'attach_rest',
    'm': 'attach_rest',
    'n': 'attach_rest',
}

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
        try:
            system_file = os.path.join(self.output_dir, "system.pkl")
            with open(system_file, 'rb') as f:
                loaded_state = pickle.load(f)
                # Update self with loaded attributes
                self.__dict__.update(loaded_state.__dict__)
        except:
            logger.info(f"Failed to load the system state: {self.output_dir}")

        if not os.path.exists(f"{self.output_dir}/all-poses"):
            logger.info(f"The folder does not contain all-poses: {self.output_dir}")
            return
        if not os.path.exists(f"{self.output_dir}/equil"):
            logger.info(f"The folder does not contain equil: {self.output_dir}")
            return
        if not os.path.exists(f"{self.output_dir}/fe"):
            logger.info(f"The folder does not contain fe: {self.output_dir}")
            return

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
        self.protein_input = protein_input
        self.system_topology = system_topology
        self.system_coordinate = system_coordinate
        self.ligand_paths = ligand_paths
        self.mols = []
        if not isinstance(ligand_paths, list):
            raise ValueError(f"Invalid ligand_paths: {ligand_paths}, "
                              "ligand_paths should be a list of ligand files")
        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.ligand_ff = ligand_ff
        self.overwrite = overwrite

        # check input existence
        if not os.path.exists(protein_input):
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        if not os.path.exists(system_topology):
            raise FileNotFoundError(f"System input file not found: {system_topology}")
        for ligand_path in ligand_paths:
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
            self._ligand = mda.Universe(ligand_path)
            if len(set(self._ligand.atoms.resnames)) > 1:
                raise ValueError(f"Multiple ligand molecules {set(self._ligand.atoms.resnames)} found in the ligand file: {ligand_path}")
            mol_name = self._ligand.atoms.resnames[0].lower()
            # if mol_name is less than 2 characters
            # add ind to the end
            # otherwise tleap will fail later
            if len(mol_name) <= 2:
                logger.warning(f"Elongating the ligand name: {mol_name} to {mol_name}{ind}")
                mol_name = f'{mol_name}{ind}'
                if mol_name in self.unique_mol_names:
                    mol_name = f'{mol_name[:2]}{ind}'
                    if mol_name in self.unique_mol_names:
                        raise ValueError(f"Cannot find a unique name for the ligand: {mol_name}")
            elif len(mol_name) > 3:
                logger.warning(f"Shortening the ligand name: {mol_name} to {mol_name[:3]}")
                mol_name = mol_name[:3]
                if mol_name in self.unique_mol_names:
                    mol_name = f'{mol_name[:2]}{ind}'
                    if mol_name in self.unique_mol_names:
                        raise ValueError(f"Cannot find a unique name for the ligand: {mol_name}")
            self._mol = mol_name
            self.mols.append(mol_name)
            self.unique_mol_names.append(mol_name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{self._mol}.frcmod"):
                logger.debug(f'Processing ligand: {self._mol}')
                self._process_ligand()
        self._prepare_ligand_poses()
        
        # dump the system configuration
        with open(f"{self.output_dir}/system.pkl", 'wb') as f:
            pickle.dump(self, f)

        logger.info('System loaded and prepared')

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
        u_prot = mda.Universe(self.protein_input)
        u_sys = self.u_sys

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
        water_ag = u_sys.select_atoms('byres (resname TIP3 and around 10 (protein or resname POPC))')
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

    def _process_ligand(self):
        """
        Process the ligand, including adding or removing hydrogens as needed.
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

        self._prepare_ligand_parameters()

    def _prepare_ligand_parameters(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
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
            u = mda.Universe(pose)
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

        self.ligand_paths = new_pose_paths

    def _align_2_system(self, mobile_atoms):
        _ = align._fit_to(
            mobile_coordinates=self.mobile_coord,
            ref_coordinates=self.ref_coord,
            mobile_atoms=mobile_atoms,
            mobile_com=self.mobile_com,
            ref_com=self.ref_com)

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
            raise ValueError(f"Invalid lipid_ff in the input: {sim_config.lipid_ff}"
                             f"System is prepared with {self.lipid_ff}")
        if sim_config.ligand_ff != self.ligand_ff:
            raise ValueError(f"Invalid ligand_ff in the input: {sim_config.ligand_ff}"
                             f"System is prepared with {self.ligand_ff}")
        if sim_config.retain_lig_prot == 'no':
            logger.warning(f"The protonation state of the ligand will be "
                           f"reassigned to pH {self.ligand_ph:.2f}")
        self.sim_config = sim_config

    def prepare(self,
            stage: str,
            input_file: Union[str, Path, SimulationConfig],
            overwrite: bool = False):
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
        """
        logger.debug('Preparing the system')
        self.overwrite = overwrite
        self.builders_factory = BuilderFactory()

        self._get_sim_config(input_file)
        sim_config = self.sim_config
        
        if stage == 'equil':
            if self.overwrite:
                logger.debug(f'Overwriting {self.equil_folder}')
                shutil.rmtree(self.equil_folder, ignore_errors=True)
            # save the input file to the equil directory
            os.makedirs(f"{self.equil_folder}", exist_ok=True)
            with open(f"{self.equil_folder}/sim_config.json", 'w') as f:
                json.dump(sim_config.model_dump(), f, indent=2)
            
            self._prepare_equil_system()
            logger.info('Equil System prepared')
        
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
            os.makedirs(f"{self.fe_folder}", exist_ok=True)

            with open(f"{self.fe_folder}/sim_config.json", 'w') as f:
                json.dump(sim_config.model_dump(), f, indent=2)

            self._prepare_fe_system()
            logger.info('FE System prepared')

    def submit(self, stage: str, cluster: str = 'slurm'):
        """
        Submit the simulation to the cluster.

        Parameters
        ----------
        stage : str
            The stage of the simulation. Options are 'equil' and 'fe'.
        cluster : str
            The cluster to submit the simulation.
            Options are 'slurm' and 'frontier'.
        """
        if cluster == 'frontier':
            self._submit_frontier(stage)
            logger.info(f'Frontier {stage} job submitted!')
            return

        if stage == 'equil':
            logger.info('Submit equilibration stage')
            for pose in self.sim_config.poses_def:
                run_with_log(f'sbatch SLURMM-run',
                            working_dir=f'{self.equil_folder}/{pose}')
            logger.info('Equilibration systems have been submitted for all poses listed in the input file.')

        elif stage == 'fe':
            logger.info('Submit free energy stage')
            for pose in self.sim_config.poses_def:
                shutil.copy(f'{self.fe_folder}/{pose}/rest/run_files/run-express.bash',
                            f'{self.fe_folder}/{pose}')
                run_with_log(f'bash run-express.bash',
                            working_dir=f'{self.fe_folder}/{pose}')
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

        if len(self.sim_config.poses_def) != len(self.ligand_paths):
            logger.warning(f"Number of poses in the input file: {len(self.sim_config.poses_def)} "
                           f"does not match the number of ligands: {len(self.ligand_paths)}")
            logger.warning(f"Using the ligand paths for the poses")
        self.sim_config.poses_def = [f'pose{i}' for i in range(len(self.ligand_paths))]

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
        logger.debug(f'now cd equil/pose0')
        logger.debug(f'sbatch SLURMM-run')

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
        
    def analysis(self,
        input_file: Union[str, Path, SimulationConfig]=None):
        """
        Analyze the simulation results.
        """
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
            for pose in poses_def:
                analysis.fe_values(blocks, components, temperature, pose, attach_rest, lambdas,
                                weights, dec_int, dec_method, rest, dic_steps1, dic_steps2, dt)
                os.chdir('../../')
    
    def run_pipeline(self,
                     input_file: Union[str, Path, SimulationConfig],
                     overwrite: bool = False
    ):
        """
        Run the whole pipeline for calculating the binding free energy
        after you `create_system`.

        """
        logger.info('Running the pipeline')
        self._get_sim_config(input_file)

        if self._check_equilibration():
            #1 prepare the system
            logger.info('Preparing the system')
            self.prepare(
                stage='equil',
                input_file=input_file,
                overwrite=overwrite
            )
            logger.info('Submitting the equilibration')
            #2 submit the equilibration
            self.submit(
                stage='equil',
            )

            # Check for equilibration to finish
            logger.info('Checking the equilibration')
            while self._check_equilibration():
                logger.info('Equilibration is still running. Waiting for 1 hour.')
                time.sleep(60*60)
        else:
            logger.info('Equilibration is already finished')



        #4, submit the free energy calculation
        if self._check_fe():
            #3 prepare the free energy calculation
            logger.info('Preparing the free energy calculation')
            self.prepare(
                stage='fe',
                input_file=input_file,
                overwrite=overwrite
            )
            logger.info('Submitting the free energy calculation')
            self.submit(
                stage='fe',
            )
            # Check the free energy calculation to finish
            logger.info('Checking the free energy calculation')
            while self._check_fe():
                logger.info('Free energy calculation is still running. Waiting for 1 hour.')
                time.sleep(60*60)
        else:
            logger.info('Free energy calculation is already finished')

        #5 analyze the results
        logger.info('Analyzing the results')
        self.analysis()

        logger.info('Pipeline finished')
        logger.info(f'The results are in the {self.output_dir}')

    def _check_equilibration(self):
        """
        Check if the equilibration is finished by checking the FINISHED file
        """
        for pose in self.sim_config.poses_def:
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                return True
        return False

    def _check_fe(self):
        """
        Check if the free energy calculation is finished by 
        """
        for pose in self.sim_config.poses_def:
            for comp in self.sim_config.components:
                if COMPONENTS_FOLDER_DICT[comp] == 'rest':
                    for j in range(0, len(self.sim_config.attach_rest)):
                        folder_2_check = f'{self.fe_folder}/{pose}/rest/{comp}{j:02d}'
                        if not os.path.exists(f"{folder_2_check}/FINISHED"):
                            return True
                else:
                    for j in range(0, len(self.sim_config.lambdas)):
                        folder_2_check = f'{self.fe_folder}/{pose}/{self.sim_config.dec_method}/{comp}{j:02d}'
                        if not os.path.exists(f"{folder_2_check}/FINISHED"):
                            return True
        return False

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
        # check if they are the same ligand
        # set the ligand path to the first ligand
        self.unique_ligand_paths = self.ligand_paths