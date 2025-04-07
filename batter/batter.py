"""
Provide the primary functions for preparing and processing FEP systems.
"""

from batter.utils import (
    run_with_log,
    save_state,
    safe_directory,
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
try:
    from openff.toolkit import Molecule
except:
    raise ImportError("OpenFF toolkit is not installed. Please install it with `conda install -c conda-forge openff-toolkit-base`")
from rdkit import Chem

import time
from tqdm import tqdm

from typing import List, Tuple
import loguru
from loguru import logger
logger.add(sys.stdout, level='INFO')

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.bat_lib import analysis
from batter.results import FEResult
from batter.ligand_process import LigandFactory
from batter.utils.slurm_job import SLURMJob
from batter.analysis.convergence import ConvergenceValidator
from batter.analysis.sim_validation import SimValidator
from batter.data import frontier_files

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
        logger.add(f"{self.output_dir}/batter.log", level='INFO')

        self._slurm_jobs = {}
        self._sim_finished = {}
        self._sim_failed = {}
        self._eq_prepared = False
        self._fe_prepared = False
        self._fe_results = {}
        self.mols = []
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
                logger.add(f"{self.output_dir}/batter.log", level='INFO')

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

    @safe_directory
    @save_state
    def create_system(
                    self,
                    system_name: str,
                    protein_input: str,
                    system_topology: str,
                    ligand_paths: List[str],
                    anchor_atoms: List[str],
                    ligand_anchor_atom: str = None,
                    receptor_segment: str = None,
                    system_coordinate: str = None,
                    protein_align: str = 'name CA and resid 60 to 250',
                    retain_lig_prot: bool = True,
                    ligand_ph: float = 7.4,
                    ligand_ff: str = 'gaff2',
                    lipid_mol: List[str] = [],
                    lipid_ff: str = 'lipid21',
                    overwrite: bool = False,
                    verbose: bool = False,
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
        anchor_atoms : List[str], optional
            The list of three protein anchor atoms (selection strings)
            used to restrain ligand.
            It will also be used to set l1x, l1y, l1z values that defines
            the binding pocket.
        ligand_anchor_atom : str, optional
            The ligand anchor atom (selection string) used as a potential
            ligand anchor atom.
            Default is None and will use the atom that is closest to the
            center of mass of the ligand.
            Note only the first ligand in the ligand_paths will be used
            to create the binding pocket.
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
            Parameter set for the ligand. Default is 'gaff2'.
            'gaff' is not supported yet.
            Options are 'gaff' and 'gaff2'.
        lipid_mol : List[str], optional
            List of lipid molecules to be included in the simulations.
            Default is an empty list.
        lipid_ff : str, optional
            Force field for lipid atoms. Default is 'lipid21'.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        verbose : bool, optional
            The verbosity of the output. If True, it will print the debug messages.
            Default is False.
        """
        # Log every argument
        if verbose:
            logger.remove()
            logger.add(sys.stdout, level='DEBUG')
            logger.add(f"{self.output_dir}/batter.log", level='DEBUG')
            logger.debug('Verbose mode is on')
            logger.debug('Creating a new system')

        self.verbose = True
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        
        for arg in args:
            logger.info(f"{arg}: {values[arg]}")

        self.system_name = system_name
        self._protein_input = self._convert_2_relative_path(protein_input)
        self._system_topology = self._convert_2_relative_path(system_topology)
        if system_coordinate is not None:
            self._system_coordinate = self._convert_2_relative_path(system_coordinate)
        else:
            self._system_coordinate = None
        self._ligand_paths = [self._convert_2_relative_path(ligand_path) for ligand_path in ligand_paths]
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

        if system_coordinate is not None and not os.path.exists(system_coordinate):
            raise FileNotFoundError(f"System coordinate file not found: {system_coordinate}")
        
        u_sys = mda.Universe(self.system_topology, format='XPDB')
        if system_coordinate is not None:
            # read last line of inpcrd file to get dimensions
            with open(system_coordinate) as f:
                lines = f.readlines()
                box = np.array([float(x) for x in lines[-1].split()])
            self.system_dimensions = box
            u_sys.load_new(system_coordinate, format='INPCRD')
        if (u_sys.atoms.dimensions is None or not u_sys.atoms.dimensions.any()) and self.system_coordinate is None:
            raise ValueError(f"No dimension of the box was found in the system_topology or system_coordinate")

        os.makedirs(f"{self.poses_folder}", exist_ok=True)
        u_sys.atoms.write(f"{self.poses_folder}/system_input.pdb")
        self._system_input_pdb = f"{self.poses_folder}/system_input.pdb"
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
            try:
                ligand_name = self.mols[ind-1]
            except:
                ligand_name = None
            self._ligand_path = ligand_path
            ligand_factory = LigandFactory()
            ligand = ligand_factory.create_ligand(
                    ligand_file=ligand_path,
                    index=ind,
                    output_dir=self.ligandff_folder,
                    # TODO: use dictionary for ligand_paths
                    ligand_name=ligand_name,
                    retain_lig_prot=self.retain_lig_prot,
                    ligand_ff=self.ligand_ff) 
            ligand.generate_unique_name(self.unique_mol_names)
            self.mols.append(ligand.name)
            self.unique_mol_names.append(ligand.name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{ligand.name}.frcmod"):
                ligand.prepare_ligand_parameters_sdf()
            
        self._prepare_ligand_poses()

        # always get the anchor atoms from the first pose
        u_prot = mda.Universe(f'{self.output_dir}/all-poses/reference.pdb')
        u_lig = mda.Universe(f'{self.output_dir}/all-poses/pose0.pdb')
        l1_x, l1_y, l1_z, p1, p2, p3 = self._find_anchor_atoms(
                    u_prot,
                    u_lig,
                    anchor_atoms,
                    ligand_anchor_atom)

        self.anchor_atoms = anchor_atoms
        self.ligand_anchor_atom = ligand_anchor_atom

        self.l1_x = l1_x
        self.l1_y = l1_y
        self.l1_z = l1_z

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        
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

        u_sys = mda.Universe(self._system_input_pdb, format='XPDB')
        cog_prot = u_sys.select_atoms('protein and name CA C N O').center_of_geometry()
        u_sys.atoms.positions -= cog_prot
        
        # get translation-rotation matrix
        mobile = u_prot.select_atoms(self.protein_align).select_atoms('name CA and not resname NMA ACE')
        ref = u_sys.select_atoms(self.protein_align).select_atoms('name CA and not resname NMA ACE')

        if mobile.n_atoms != ref.n_atoms:
            raise ValueError(f"Number of atoms in the alignment selection is different: protein_input: "
            f"{mobile.n_atoms} and system_input {ref.n_atoms} "
            f"The selection string is {self.protein_align} and name CA and not resname NMA ACE")
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
        u_prot.atoms.write(f"{self.poses_folder}/protein_aligned.pdb")
        self._protein_aligned_pdb = f"{self.poses_folder}/protein_aligned.pdb"
        u_sys.atoms.write(f"{self.poses_folder}/system_aligned.pdb")
        self._system_aligned_pdb = f"{self.poses_folder}/system_aligned.pdb"
        
        self.translation = cog_prot

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
        u_prot = mda.Universe(self._protein_aligned_pdb)
        u_sys = mda.Universe(self._system_aligned_pdb, format='XPDB')
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
        
        # overwride l1_x, l1_y, l1_z
        sim_config.l1_x = self.l1_x
        sim_config.l1_y = self.l1_y
        sim_config.l1_z = self.l1_z

        # override the p1, p2, p3
        sim_config.p1 = self.p1
        sim_config.p2 = self.p2
        sim_config.p3 = self.p3
                 
        self.sim_config = sim_config

    @safe_directory
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
            self._fe_prepared = False
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

            self._check_equilbration_binding()
            self._prepare_fe_system()
            logger.info('FE System prepared')
            self._fe_prepared = True

    @safe_directory
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
            The stage of the simulation. Options are 'equil', 'fe', and 'fe_equil'.
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
                    logger.info(f'Equilibration for {pose} has finished; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'{pose}', None)
                    continue
                if os.path.exists(f"{self.equil_folder}/{pose}/FAILED") and not overwrite:
                    logger.warning(f'Equilibration for {pose} has failed; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'{pose}', None)
                    continue
                if f'{pose}' in self._slurm_jobs:
                    # check if it's finished
                    slurm_job = self._slurm_jobs[f'{pose}']
                    # if the job is finished but the FINISHED file is not created
                    # resubmit the job
                    if not slurm_job.is_still_running():
                        slurm_job.submit()
                        n_jobs_submitted += 1
                        continue
                    elif overwrite:
                        slurm_job.cancel()
                        slurm_job.submit(overwrite=True)
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
                slurm_job.submit(overwrite=overwrite)
                n_jobs_submitted += 1
                logger.info(f'Equilibration job for {pose} submitted: {slurm_job.jobid}')
                self._slurm_jobs.update(
                    {f'{pose}': slurm_job}
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
                            self._slurm_jobs.pop(f'{pose}/{comp_folder}/{comp}{j:02d}', None)
                            logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                            continue
                        if os.path.exists(f"{folder_2_check}/FAILED") and not overwrite:
                            self._slurm_jobs.pop(f'{pose}/{comp_folder}/{comp}{j:02d}', None)
                            logger.warning(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has failed; add overwrite=True to re-run the simulation')
                            continue
                        if f'{pose}/{comp_folder}/{comp}{j:02d}' in self._slurm_jobs:
                            slurm_job = self._slurm_jobs[f'{pose}/{comp_folder}/{comp}{j:02d}']
                            if not slurm_job.is_still_running():
                                slurm_job.submit()
                                n_jobs_submitted += 1
                                continue
                            elif overwrite:
                                slurm_job.cancel()
                                slurm_job.submit(overwrite=True)
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
                        slurm_job.submit(overwrite=overwrite)
                        n_jobs_submitted += 1
                        logger.info(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} submitted')
                        self._slurm_jobs.update(
                            {f'{pose}/{comp_folder}/{comp}{j:02d}': slurm_job}
                        )

            logger.info('Free energy systems have been submitted for all poses listed in the input file.')
        elif stage == 'fe_equil':
            logger.info('Submit NPT equilibration part of free energy stage')
            for pose in self.sim_config.poses_def:
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    # only run for window 0
                    j = 0
                    if n_jobs_submitted >= self._max_num_jobs:
                        time.sleep(120)
                        n_jobs_submitted = sum([1 for job in self._slurm_jobs.values() if job.is_still_running()])
                    folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}'
                    if os.path.exists(f"{folder_2_check}/FINISHED") and not overwrite:
                        self._slurm_jobs.pop(f'{pose}/{comp_folder}/{comp}{j:02d}', None)
                        logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                        continue
                    if os.path.exists(f"{folder_2_check}/FAILED") and not overwrite:
                        self._slurm_jobs.pop(f'{pose}/{comp_folder}/{comp}{j:02d}', None)
                        logger.warning(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has failed; add overwrite=True to re-run the simulation')
                        continue
                    if os.path.exists(f"{folder_2_check}/eqnpt04.rst7") and not overwrite:
                        logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                        continue
                    if f'{pose}/{comp_folder}/{comp}{j:02d}' in self._slurm_jobs:
                        slurm_job = self._slurm_jobs[f'{pose}/{comp_folder}/{comp}{j:02d}']
                        if not slurm_job.is_still_running():
                            slurm_job.submit(other_env={
                                'ONLY_EQ': '1'
                            }
                            )
                            n_jobs_submitted += 1
                            continue
                        elif overwrite:
                            slurm_job.cancel()
                            slurm_job.submit(overwrite=True,
                                other_env={
                                    'ONLY_EQ': '1'
                                }
                            )
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
                    slurm_job.submit(overwrite=overwrite, other_env={
                        'ONLY_EQ': '1'
                    })
                    n_jobs_submitted += 1
                    logger.info(f'FE equil job for {pose}/{comp_folder}/{comp}{j:02d} submitted')
                    self._slurm_jobs.update(
                        {f'{pose}/{comp_folder}/{comp}{j:02d}': slurm_job}
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
            if os.path.exists(f"{self.equil_folder}/{pose}") and not self.overwrite:
                logger.info(f'Pose {pose} already exists; add overwrite=True to re-build the pose')
                continue
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
            # if "UNBOUND" found in equilibration, skip
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                logger.info(f"Pose {pose} is UNBOUND in equilibration; skipping FE")
                os.makedirs(f"{self.fe_folder}/{pose}/Results", exist_ok=True)
                with open(f"{self.fe_folder}/{pose}/Results/Results.dat", 'w') as f:
                    f.write("UNBOUND\n")
                pbar.update(1)
                continue
            logger.debug(f'Preparing pose: {pose}')
            
            # load anchor_list
            with open(f"{self.equil_folder}/{pose}/anchor_list.txt", 'r') as f:
                anchor_list = f.readlines()
                l1x, l1y, l1z = [float(x) for x in anchor_list[0].split()]
                self.sim_config.l1_x = l1x
                self.sim_config.l1_y = l1y
                self.sim_config.l1_z = l1z

            # copy ff folder
            shutil.copytree(self.ligandff_folder,
                            f"{self.fe_folder}/{pose}/ff", dirs_exist_ok=True)

            for component in sim_config.components:
                logger.debug(f'Preparing component: {component}')
                lambdas_comp = sim_config.dict()[COMPONENTS_LAMBDA_DICT[component]]
                n_sims = len(lambdas_comp)
                logger.debug(f'Number of simulations: {n_sims}')
                cv_paths = []
                for i, _ in enumerate(lambdas_comp):
                    cv_path = f"{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[component]}/{component}{i:02d}/cv.in"
                    cv_paths.append(cv_path)
                if all(os.path.exists(cv_paths[i]) for i, _ in enumerate(lambdas_comp)) and not self.overwrite:
                    logger.info(f"Component {component} for pose {pose} already exists; add overwrite=True to re-build the component")
                    continue
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
                if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                    continue
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

    @safe_directory
    @save_state
    def analysis(
        self,
        input_file: Union[str, Path, SimulationConfig]=None,
        load=True,
        check_finished: bool = True,
        sim_range: Tuple[int, int] = None,
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
        check_finished : bool, optional
            Whether to check if the FINISHED file exists in FE simulation.
            Default is True.
        sim_range : Tuple[int, int], optional
            The range of simulations to analyze.
            For simulations run on Frontier, due to the time constraints,
            the simulations are run into multiple parts.
            Default is None, which will analyze all the simulations.
        """
        if input_file is not None:
            self._get_sim_config(input_file)
        
        if check_finished:
            if self._check_fe():
                logger.info('FE simulation is not finished yet')
                self.check_jobs()
                return
            
        if not sim_range:
            sim_range = (None, None)
            
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
                os.chdir(f'{self.fe_folder}/{pose}')
                if load and os.path.exists(f'{self.fe_folder}/{pose}/Results/Results.dat'):
                    self.fe_results[pose] = FEResult(f'{self.fe_folder}/{pose}/Results/Results.dat')
                    logger.info(f'FE for {pose} loaded from {self.fe_folder}/{pose}/Results/Results.dat')
                    continue
                # remove the existing Results folder
                if os.path.exists(f'{self.fe_folder}/{pose}/Results'):
                    shutil.rmtree(f'{self.fe_folder}/{pose}/Results', ignore_errors=True)
                os.makedirs(f'{self.fe_folder}/{pose}/Results', exist_ok=True)
                if False:
                    try:
                        fe_value, fe_std = analysis.fe_values(blocks, components, temperature, pose, attach_rest, lambdas,
                                        weights, dec_int, dec_method, rest, dic_steps1, dic_steps2, dt, sim_range)
                    except:
                        fe_value = np.nan
                fe_value, fe_std = analysis.fe_values(blocks, components, temperature, pose, attach_rest, lambdas,
                                        weights, dec_int, dec_method, rest, dic_steps1, dic_steps2, dt, sim_range) 

                # if failed; it will return nan
                if np.isnan(fe_value):
                    logger.warning(f'FE calculation failed for {pose}')
                    with open(f'{self.fe_folder}/{pose}/Results/Results.dat', 'w') as f:
                        f.write("UNBOUND\n")
                    self.fe_results[pose] = FEResult(f'{self.fe_folder}/{pose}/Results/Results.dat')

                    continue
                
                # validate
                for i, comp in enumerate(components):
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    windows = attach_rest if comp_folder == 'rest' else lambdas
                    Upot = np.load(f"{folder_comp}/data/Upot_{comp}_all.npy")
                    validator = ConvergenceValidator(
                        Upot=Upot,
                        lambdas=windows,
                        temperature=temperature,
                    )
                    try:
                        validator.plot_convergence(save_path=f"{self.fe_folder}/{pose}/Results/convergence_{pose}_{comp}.png",
                                               title=f'Convergence of {pose} {comp}')
                    except:
                        logger.warning(f'Convergence failed for {pose} {comp}')
                        continue

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
        if P1_atom.n_atoms == 0 or P2_atom.n_atoms == 0 or P3_atom.n_atoms == 0:
            raise ValueError('Error: anchor atom not found')
        if P1_atom.n_atoms != 1 or P2_atom.n_atoms != 1 or P3_atom.n_atoms != 1:
            raise ValueError('Error: more than one atom selected in the anchor atoms')
        
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
              
    def _check_equilbration_binding(self):
        """
        Check if the ligand is bound after equilibration
        """
        bound_poses = []
        for pose_i, pose in enumerate(self.sim_config.poses_def):
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                raise FileNotFoundError(f"Equilibration not finished yet")
            if os.path.exists(f"{self.equil_folder}/{pose}/FAILED"):
                raise FileNotFoundError(f"Equilibration failed")
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                logger.warning(f"Pose {pose} is UNBOUND in equilibration")
                continue
            if os.path.exists(f"{self.equil_folder}/{pose}/representative.pdb"):
                bound_poses.append([pose_i, pose])
                logger.info(f"Representative snapshot found for pose {pose}")
                continue
            with self._change_dir(f"{self.equil_folder}/{pose}"):
                pdb = "full.pdb"
                trajs = ["md-01.nc", "md-02.nc", "md-03.nc"]
                universe = mda.Universe(pdb, trajs)
                sim_val = SimValidator(universe)
                sim_val.plot_ligand_bs()
                sim_val.plot_rmsd()
                if sim_val.results['ligand_bs'][-1] > 5:
                    logger.warning(f"Ligand is not bound for pose {pose}")
                    # write "UNBOUND" file
                    with open(f"{self.equil_folder}/{pose}/UNBOUND", 'w') as f:
                        f.write(f"UNBOUND with ligand_bs = {sim_val.results['ligand_bs'][-1]}")
                else:
                    bound_poses.append([pose_i, pose])
                    rep_snapshot = sim_val.find_representative_snapshot()
                    logger.info(f"Representative snapshot: {rep_snapshot}")
                    cpptraj_command = f"""cpptraj -p full.prmtop <<EOF
trajin md-01.nc
trajin md-02.nc
trajin md-03.nc
trajout representative.pdb pdb onlyframes {rep_snapshot+1}
trajout md03.rst7 restart onlyframes {rep_snapshot+1}
EOF"""
                    run_with_log(cpptraj_command,
                                working_dir=f"{self.equil_folder}/{pose}")
        logger.info(f"Bound poses: {bound_poses} will be used for the production stage")

        # get new l1x, l1y, l1z distances
        for pose_i, pose in bound_poses:
            u_sys = mda.Universe(f'{self.equil_folder}/{pose}/representative.pdb')
            u_lig = u_sys.select_atoms(f'resname {self.mols[pose_i]}')

            anchor_file = f'{self.equil_folder}/{pose}/build_files/protein_anchors.txt'
            with open(anchor_file, 'r') as f:
                anchor_atoms_lines = f.readlines()
            # convert amber selection to mda selection
            # amber: :84@CA
            # mda :resid 84 and name CA
            anchor_atoms = []
            for line in anchor_atoms_lines:
                resid = line.split(':')[1].split('@')[0]
                atom_name = line.split('@')[1].strip()
                anchor_atoms.append(f'resid {resid} and name {atom_name}')

            ligand_anchor_atom = self.ligand_anchor_atom

            logger.info(f'Finding anchor atoms for pose {pose}')
            l1_x, l1_y, l1_z, p1, p2, p3 = self._find_anchor_atoms(
                        u_sys,
                        u_lig,
                        anchor_atoms,
                        ligand_anchor_atom)
            with open(f'{self.equil_folder}/{pose}/anchor_list.txt', 'w') as f:
                f.write(f'{l1_x} {l1_y} {l1_z}')

    @safe_directory
    @save_state
    def run_pipeline(self,
                     input_file: Union[str, Path, SimulationConfig],
                     overwrite: bool = False,              
                     avg_struc: str = None,
                     rmsf_file: str = None,
                     only_equil: bool = False,
                     only_fe_preparation: bool = False,
                     partition: str = 'owners',
                     max_num_jobs: int = 500,
                     verbose: bool = False
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
        only_fe_preparation : bool, optional
            Whether to prepare the files for the production stage
            without running the production stage.
            Default is False.
        partition : str, optional
            The partition to submit the job.
            Default is 'rondror'.
        max_num_jobs : int, optional
            The maximum number of jobs to submit at a time.
            Default is 500.
        verbose : bool, optional
            Whether to print the verbose output. Default is False.
        """
        if verbose:
            logger.remove()
            logger.add(sys.stdout, level='DEBUG')
            logger.add(f'{self.output_dir}/batter.log', level='DEBUG')
            logger.info('Verbose output is set to True')
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
                n_finished = len([k for k, v in self._sim_finished.items() if v])
                logger.info(f'Finished jobs: {n_finished} / {len(self._sim_finished)}')
                not_finished = [k for k, v in self._sim_finished.items() if not v]
                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
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
            if only_fe_preparation:
                logger.info('only_fe_preparation is set to True. '
                            'Skipping the free energy calculation.')
                return
            logger.info('Submitting the free energy calculation')
            self.submit(
                stage='fe',
            )
            # Check the free energy calculation to finish
            logger.info('Checking the free energy calculation')
            while self._check_fe():
                # get finishd jobs
                n_finished = len([k for k, v in self._sim_finished.items() if v])
                logger.info(f'Finished jobs: {n_finished} / {len(self._sim_finished)}')
                not_finished = [k for k, v in self._sim_finished.items() if not v]
                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
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
            logger.info(f'{mol_name}\t{pose}\t{fe.fe:.2f} ± {fe.fe_std:.2f}')
        
    @save_state
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

    @save_state
    def _check_fe(self):
        """
        Check if the free energy calculation is finished by 
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.sim_config.poses_def:
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                sim_finished[pose] = True
                continue
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

    @staticmethod
    def _continue_job(job: SLURMJob):
        """
        Continue the job if it is not finished.
        """
        if not job.is_still_running():
            job.submit()
            logger.info(f'Job {job.jobid} is resubmitted')

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
    
    @safe_directory
    @save_state
    def generate_frontier_files(self, version=24):
        """
        Generate the frontier files for the system
        to run them in a bundle.
        """
        poses_def = self.sim_config.poses_def
        components = self.sim_config.components
        attach_rest = self.sim_config.attach_rest
        lambdas = self.sim_config.lambdas
        weights = self.sim_config.weights
        dec_int = self.sim_config.dec_int
        dec_method = self.sim_config.dec_method
        rest = self.sim_config.rest


        dec_method_folder_dict = {
            'dd': 'dd',
            'sdr': 'sdr',
            'exchange': 'sdr',
        }
        component_2_folder_dict = {
            'v': dec_method_folder_dict[dec_method],
            'e': dec_method_folder_dict[dec_method],
            'w': dec_method_folder_dict[dec_method],
            'f': dec_method_folder_dict[dec_method],
            'x': 'exchange_files',
            'a': 'rest',
            'l': 'rest',
            't': 'rest',
            'r': 'rest',
            'c': 'rest',
            'm': 'rest',
            'n': 'rest',
        }
        sim_stages = {
            'rest': [
                'mini.in',
            #    'therm1.in', 'therm2.in',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
                'mdin.in', 'mdin.in.extend'
            ],
            'sdr': [
                'mini.in',
            #    'heat.in_00',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
                'mdin.in', 'mdin.in.extend'
            ],
        }
        # write a groupfile for each component

        def write_2_pose(pose, components):
            """
            Write a groupfile for each component in the pose
            """
            all_replicates = {comp: [] for comp in components}

            pose_name = f'fe/{pose}/'
            os.makedirs(pose_name, exist_ok=True)
            os.makedirs(f'{pose_name}/groupfiles', exist_ok=True)
            for component in components:
                folder_name = component_2_folder_dict[component]
                sim_folder_temp = f'{pose}/{folder_name}/{component}'
                if component in ['x', 'e', 'v', 'w', 'f']:
                    n_sims = len(lambdas)
                else:
                    n_sims = len(attach_rest)

                stage_previous = f'{sim_folder_temp}REPXXX/full.inpcrd'

                for stage in sim_stages[component_2_folder_dict[component]]:
                    groupfile_name = f'{pose_name}/groupfiles/{component}_{stage}.groupfile'
                    with open(groupfile_name, 'w') as f:
                        for i in range(n_sims):
                            #stage_previous_temp = stage_previous.replace('00', f'{i:02d}')
                            sim_folder_name = f'{sim_folder_temp}{i:02d}'
                            prmtop = f'{sim_folder_name}/full.hmr.prmtop'
                            inpcrd = f'{sim_folder_name}/full.inpcrd'
                            mdinput = f'{sim_folder_name}/{stage.split("_")[0]}'
                            # Read and modify the MD input file to update the relative path
                            if stage in ['mdin.in', 'mdin.in.extend']:
                                mdinput = mdinput.replace(stage, 'mdin-02')
                            with open(f'fe/{mdinput}', 'r') as infile:
                                input_lines = infile.readlines()

                            new_mdinput = f'fe/{sim_folder_name}/{stage.split("_")[0]}_frontier'
                            with open(new_mdinput, 'w') as outfile:
                                for line in input_lines:
                                    if 'cv_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"cv_file = '{sim_folder_name}/{file_name}'\n"
                                    if 'output_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"output_file = '{sim_folder_name}/{file_name}'\n"
                                    if 'disang' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"DISANG={sim_folder_name}/{file_name}\n"
                                    # update the number of steps
                                    if 'nstlim = 10000' in line:
                                        line = '  nstlim = 50000,\n'
                                    # do not only write the ntwprt atoms
                                    if 'ntwprt' in line:
                                        line = '\n'
                                    if 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "")
                                        # replace :1-2
                                        restraint_mask = restraint_mask.replace(':1-2', ':2')
                                        restraint_mask = restraint_mask.replace(':1', '@CA')
                                        line = f"restraintmask = '@CA | {restraint_mask}' \n"
                                    if stage == 'mdin.in' or stage == 'mdin.in.extend':
                                        if 'nstlim' in line:
                                            inpcrd_file = f'fe/{sim_folder_name}/full.inpcrd'
                                            # read the second line of the inpcrd file
                                            with open(inpcrd_file, 'r') as infile:
                                                lines = infile.readlines()
                                                n_atoms = int(lines[1])
                                            performance = calculate_performance(n_atoms, component)
                                            n_steps = int(20 / 60 / 24 * performance * 1000 * 1000 / 4)
                                            n_steps = int(n_steps // 100000 * 100000)
                                            line = f'  nstlim = {n_steps},\n'
                                        if 'ntp = ' in line:
                                            line = '  ntp = 1,\n'
                                        if 'csurften' in line:
                                            line = '\n'
                                    outfile.write(line)

                            f.write(f'# {component} {i} {stage}\n')
                            if stage == 'mdin.in':
                                f.write(f'-O -i {sim_folder_name}/mdin.in_frontier -p {sim_folder_name}/full.hmr.prmtop -c {sim_folder_name}/eqnpt.in_04.rst7 '
                                        f'-o {sim_folder_name}/mdin-00.out -r {sim_folder_name}/mdin-00.rst7 -x {sim_folder_name}/mdin-00.nc '
                                        f'-ref {sim_folder_name}/eqnpt.in_04.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-00.log '
                                        f'-e {sim_folder_name}/mdin-00.mden\n')
                            elif stage == 'mdin.in.extend':
                                f.write(f'-O -i {sim_folder_name}/mdin.in_frontier -p {sim_folder_name}/full.hmr.prmtop -c {sim_folder_name}/mdin-CURRNUM.rst7 '
                                        f'-o {sim_folder_name}/mdin-NEXTNUM.out -r {sim_folder_name}/mdin-NEXTNUM.rst7 -x {sim_folder_name}/mdin-NEXTNUM.nc '
                                        f'-ref {sim_folder_name}/mdin-CURRNUM.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-NEXTNUM.log '
                                        f'-e {sim_folder_name}/mdin-NEXTNUM.mden\n')
                            else:
                                f.write(
                                    f'-O -i {sim_folder_name}/{stage.split("_")[0]}_frontier -p {prmtop} -c {stage_previous.replace("REPXXX", f"{i:02d}")} '
                                    f'-o {sim_folder_name}/{stage}.out -r {sim_folder_name}/{stage}.rst7 -x {sim_folder_name}/{stage}.nc '
                                    f'-ref {stage_previous.replace("REPXXX", f"{i:02d}")} -inf {sim_folder_name}/{stage}.mdinfo -l {sim_folder_name}/{stage}.log '
                                    f'-e {sim_folder_name}/{stage}.mden\n'
                                )
                            if stage == 'mdin.in':
                                all_replicates[component].append(f'{sim_folder_name}')
                        stage_previous = f'{sim_folder_temp}REPXXX/{stage}.rst7'
            logger.debug(f'all_replicates: {all_replicates}')
            return all_replicates

        def write_sbatch_file(pose, components):
            file_temp = f'{frontier_files}/fep_run.sbatch'
            lines_temp = open(file_temp).readlines()
            sbatch_all_file = f'fe/fep_{pose}_eq_all.sbatch'
            lines_all = []
            for line in lines_temp:
                lines_all.append(line)
            for component in components:
                lines = []
                for line in lines_temp:
                    lines.append(line)
                folder = os.getcwd()
                folder = '_'.join(folder.split(os.sep)[-4:])
                # write the sbatch file for equilibration

                lines.append(f'\n\n\n')
                lines.append(f'# {pose} {component}\n')
                lines_all.append(f'\n\n\n')
                lines_all.append(f'# {pose} {component}\n')

                sbatch_file = f'fe/fep_{component}_{pose}_eq.sbatch'
                groupfile_names = [
                    f'{pose}/groupfiles/{component}_{stage}.groupfile' for stage in sim_stages[component_2_folder_dict[component]]
                ]
                logger.debug(f'groupfile_names: {groupfile_names}')
                for g_name in groupfile_names:
                    if 'mdin.in' in g_name:
                        continue
                    if component in ['x', 'e', 'v', 'w', 'f']:
                        n_sims = len(lambdas)
                    else:
                        n_sims = len(attach_rest)
                    n_nodes = int(np.ceil(n_sims / 8))
                    if 'mini' in g_name:
                        # run with pmemd.mpi for minimization
                        lines.append(
                            f'srun -N {n_nodes} -n {n_sims * 8} pmemd.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                        lines_all.append(
                            f'srun -N {n_nodes} -n {n_sims * 8} pmemd.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                    else:
                        lines.append(
                            f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                        lines_all.append(
                            f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                lines = [line
                        .replace('NUM_NODES', str(n_nodes))
                        .replace('FEP_SIM_XXX', f'{folder}_{component}_{pose}') for line in lines]
                lines_all = [line
                        .replace('NUM_NODES', str(n_nodes))
                        .replace('FEP_SIM_XXX', f'{folder}_{component}_{pose}') for line in lines_all]
                with open(sbatch_file, 'w') as f:
                    f.writelines(lines)
            with open(sbatch_all_file, 'w') as f:
                f.writelines(lines_all)

                
        def calculate_performance(n_atoms, comp):
            # Very rough estimate of the performance of the simulations
            # for 200000-atom systems: rest: 100 ns/day, sdr: 50 ns/day
            # for 70000-atom systems: rest: 200 ns/day, sdr: 100 ns/day
            # run 30 mins for each simulation
            if comp not in ['e', 'v', 'w', 'f', 'x']:
                if n_atoms < 80000:
                    return 150
                else:
                    return 80
            else:
                if n_atoms < 80000:
                    return 80
                else:
                    return 40

        def write_production_sbatch(all_replicates):
            sbatch_file = f'fe/fep_md.sbatch'
            sbatch_extend_file = f'fe/fep_md_extend.sbatch'

            file_temp = f'{frontier_files}/fep_run.sbatch'
            temp_lines = open(file_temp).readlines()
            temp_lines.append(f'\n\n\n')
                    # run the production
            folder = os.getcwd()
            folder = '_'.join(folder.split(os.sep)[-4:]) 
            n_sims = 0
            for replicates_pose in all_replicates:
                for comp, rep in replicates_pose.items():
                    n_sims += len(rep)
            n_nodes = int(np.ceil(n_sims / 8))
            temp_lines = [
                line.replace('NUM_NODES', str(n_nodes))
                    .replace('FEP_SIM_XXX', f'fep_md_{folder}')
                for line in temp_lines]

            with open(sbatch_file, 'w') as f:
                f.writelines(temp_lines)

            with open(sbatch_extend_file, 'w') as f:
                f.writelines(temp_lines)
                f.writelines(
                [
                    '# Make sure it\'s extending\n',
                    'latest_file=$(ls pose0/sdr/e00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)\n\n',
                    '# Check if any mdin-xx.rst7 files exist\n',
                    'if [[ -z "$latest_file" ]]; then\n',
                    'echo "No old production files found in the current directory."\n',
                    'echo "Run sbatch fep_md.sbatch."\n',
                    'exit 1\n',
                    'fi\n\n',
                    # 'poses=(pose0 pose1 pose2 pose3 pose4)
                    f'poses=({" ".join(poses_def)})\n',
                    # groups=(m n e v)
                    f'groups=({" ".join(components)})\n',
                    'for pose in "${poses[@]}"; do\n',
                    '  for group in "${groups[@]}"; do\n',
                    '    latest_file=$(ls $pose/*/${group}00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)\n',
                    '    echo "Last file for $pose/$group: $latest_file"\n',
                    '    latest_num=$(echo "$latest_file" | grep -oP "(?<=-)[0-9]{2}(?=\.rst7)")\n',
                    '    next_num=$(printf "%02d" $((10#$latest_num + 1)))\n',
                    '    echo "Last file for $pose/$group: $latest_file"\n',
                    '    echo "Next number: $next_num"\n',
                    '    sed "s/CURRNUM/$latest_num/g" ${pose}/groupfiles/${group}_mdin.in.extend.groupfile > ${pose}/groupfiles/${group}_temp_mdin.groupfile\n',
                    '    sed "s/NEXTNUM/$next_num/g" ${pose}/groupfiles/${group}_temp_mdin.groupfile > ${pose}/groupfiles/${group}_current_mdin.groupfile\n',
                            
                    '    case "$group" in\n',
                    '        m|n) \n',
                    '        srun -N 2 -n 16 pmemd.hip_DPFP.MPI -ng 16 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile &\n',
                    '        echo "srun -N 2 -n 16 pmemd.hip_DPFP.MPI -ng 16 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile"\n',
                    '        ;;\n',
                    '        e|v|x)\n',
                    '        srun -N 3 -n 24 pmemd.hip_DPFP.MPI -ng 24 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile &\n',
                    '        echo "srun -N 3 -n 24 pmemd.hip_DPFP.MPI -ng 24 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile"\n',
                    '        ;;\n',
                    '        *)\n',
                    '        echo "Invalid group"\n',
                    '        ;;\n',
                    '    esac\n',
                    '    sleep 0.5\n',
                    '    done\n',
                    'done\n',
                    'wait\n\n',
                ]
                )

            for replicates_pose in all_replicates:
                for comp, rep in replicates_pose.items():
                    pose = rep[0].split('/')[0]
                    groupfile_name_prod = f'{pose}/groupfiles/{comp}_mdin.in.groupfile'
                    groupfile_name_prod_extend = f'{pose}/groupfiles/{comp}_mdin.in.extend.groupfile'

                    n_nodes = int(np.ceil(len(rep) / 8))
                    n_sims = len(rep)
                    with open(sbatch_file, 'a') as f:
                        f.writelines(
                            [
                        f'# {pose} {comp}\n',
                        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {groupfile_name_prod} &\n',
                        f'sleep 0.5\n'
                            ]
                        )
                    #with open(sbatch_extend_file, 'a') as f:
                    #    f.writelines(
                    #        [ 
                    #        f'sed "s/CURRNUM/$latest_num/g" {pose}/groupfiles/{comp}_mdin.in.extend.groupfile > {pose}/groupfiles/{comp}_temp_mdin.groupfile\n',
                    #        f'sed "s/NEXTNUM/$next_num/g" {pose}/groupfiles/{comp}_temp_mdin.groupfile > {pose}/groupfiles/{comp}_current_mdin.groupfile\n',
                    #        f'# {pose} {comp}\n',
                    #        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {pose}/groupfiles/{comp}_current_mdin.groupfile &\n',
                    #        f'sleep 0.5\n\n'
                    #        ]
                    #    )
            # append wait
            with open(sbatch_file, 'a') as f:
                f.write('wait\n')

            #with open(sbatch_extend_file, 'a') as f:
            #    f.write('wait\n')

        all_replicates = []

        with self._change_dir(self.output_dir):

            for pose in poses_def:
                if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                    continue
                all_replicates.append(write_2_pose(pose, components))
                write_sbatch_file(pose, components)
                logger.debug(f'Generated groupfiles for {pose}')
            logger.debug(all_replicates)
            write_production_sbatch(all_replicates)
            # copy env.amber.24
            env_amber_file = f'{frontier_files}/env.amber.{version}'
            shutil.copy(env_amber_file, 'fe/env.amber')
            logger.info('Generated groupfiles for all poses')

    @safe_directory
    @save_state
    def generate_frontier_files_nvt(self,
                                    remd=False,
                                    version=24):
        """
        Generate the frontier files for the system
        to run them in a bundle.
        """
        poses_def = self.sim_config.poses_def
        components = self.sim_config.components
        attach_rest = self.sim_config.attach_rest
        lambdas = self.sim_config.lambdas
        weights = self.sim_config.weights
        dec_int = self.sim_config.dec_int
        dec_method = self.sim_config.dec_method
        rest = self.sim_config.rest
        # if remd:
        len_lambdas = len(lambdas)
        # even spacing from 0 to 1
        #revised_lambdas = np.linspace(0, 1, len_lambdas)
        revised_lambdas = np.asarray([
            0.00000000,0.02756000,0.05417000,0.08003000,0.10729000,0.13769000,0.17041000,0.21174000,0.25756000,0.30552000,0.36274000,0.42362000,0.48726000,0.55589000,0.62235000,0.68323000,0.74207000,0.79496000,0.82904000,0.86500000,0.90368000,0.94077000,0.97151000,1.00000000
        ])

        lambdas = revised_lambdas

        dec_method_folder_dict = {
            'dd': 'dd',
            'sdr': 'sdr',
            'exchange': 'sdr',
        }
        component_2_folder_dict = {
            'v': dec_method_folder_dict[dec_method],
            'e': dec_method_folder_dict[dec_method],
            'w': dec_method_folder_dict[dec_method],
            'f': dec_method_folder_dict[dec_method],
            'x': 'exchange_files',
            'a': 'rest',
            'l': 'rest',
            't': 'rest',
            'r': 'rest',
            'c': 'rest',
            'm': 'rest',
            'n': 'rest',
        }
        sim_stages = {
            'rest': [
                'mini.in',
            #    'therm1.in', 'therm2.in',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
                'mdin.in', 'mdin.in.extend'
            ],
            'sdr': [
                'mini.in',
            #    'heat.in_00',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
                'mdin.in', 'mdin.in.extend'
            ],
        }
        # write a groupfile for each component

        def write_2_pose(pose, components):
            """
            Write a groupfile for each component in the pose
            """
            all_replicates = {comp: [] for comp in components}

            pose_name = f'fe/{pose}/'
            os.makedirs(pose_name, exist_ok=True)
            os.makedirs(f'{pose_name}/groupfiles', exist_ok=True)
            for component in components:
                folder_name = component_2_folder_dict[component]
                sim_folder_temp = f'{pose}/{folder_name}/{component}'
                if component in ['x', 'e', 'v', 'w', 'f']:
                    n_sims = len(lambdas)
                else:
                    n_sims = len(attach_rest)

                stage_previous = f'{sim_folder_temp}REPXXX/equilibrated.rst7'

                for stage in sim_stages[component_2_folder_dict[component]]:
                    groupfile_name = f'{pose_name}/groupfiles/{component}_{stage}.groupfile'
                    with open(groupfile_name, 'w') as f:
                        for i in range(n_sims):
                            #stage_previous_temp = stage_previous.replace('00', f'{i:02d}')
                            win00_sim_folder_name = f'{sim_folder_temp}00'
                            sim_folder_name = f'{sim_folder_temp}{i:02d}'
                            prmtop = f'{win00_sim_folder_name}/full.hmr.prmtop'
                            inpcrd = f'{win00_sim_folder_name}/full.inpcrd'
                            mdinput = f'{sim_folder_name}/{stage.split("_")[0]}'
                            # Read and modify the MD input file to update the relative path
                            if stage in ['mdin.in', 'mdin.in.extend']:
                                mdinput = mdinput.replace(stage, 'mdin-02')
                            with open(f'fe/{mdinput}', 'r') as infile:
                                input_lines = infile.readlines()

                            new_mdinput = f'fe/{sim_folder_name}/{stage.split("_")[0]}_frontier'
                            with open(new_mdinput, 'w') as outfile:
                                for line in input_lines:
                                    if 'imin' in line:
                                        # add MC-MD water exchange
                                        if stage == 'mdin.in' or stage == 'mdin.in.extend':
                                            #if component in ['e', 'v',]:
                                            # do not use MC-MD water exchange
                                            if component in ['non']:
                                                outfile.write(
                                                    '  mcwat = 1,\n'
                                                    '  nmd = 1000,\n'
                                                    '  nmc = 1000,\n'
                                                    '  mcwatmask = ":1",\n'
                                                    '  mcligshift = 20,\n'
                                                    '  mcresstr = "WAT",\n'
                                                    #'  numexchg = 1000,\n'
                                                )
                                        if component in ['x', 'e', 'v', 'w', 'f']:
                                            outfile.write(
                                                #'scalpha = 0.5,\n'
                                                #'scbeta = 1.0,\n'
                                                'gti_cut         = 1,\n'
                                                'gti_output      = 1,\n'
                                                'gti_add_sc      = 25,\n'
                                                'gti_chg_keep   = 1,\n'
                                                'gti_scale_beta  = 1,\n'
                                                'gti_cut_sc_on   = 7,\n'
                                                'gti_cut_sc_off  = 9,\n'
                                                #'gti_lam_sch     = 1,\n'
                                                #'gti_ele_sc      = 1,\n'
                                                #'gti_vdw_sc      = 1,\n'
                                                #'gti_cut_sc      = 2,\n'
                                                'gti_ele_exp     = 2,\n'
                                                'gti_vdw_exp     = 2,\n'
                                                f'clambda         = {lambdas[i]:.2f},\n'
                                                f'mbar_lambda     = {", ".join([f"{l:.2f}" for l in lambdas])},\n'
                                            )
                                            if remd and stage != 'mini.in':
                                                outfile.write(
                                                    '  numexchg = 3000,\n'
                                                )
                                                outfile.write(
                                                    'bar_intervall = 100,\n'
                                                )
                                    if 'cv_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"cv_file = '{sim_folder_name}/{file_name}'\n"
                                    if 'output_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"output_file = '{sim_folder_name}/{file_name}'\n"
                                    if 'disang' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"DISANG={sim_folder_name}/{file_name}\n"
                                    # update the number of steps
                                    # if 'nstlim = 50000' in line:
                                    #    line = '  nstlim = 5,\n'
                                    # do not only write the ntwprt atoms
                                    if 'irest' in line:
                                        #if remd and component in ['x', 'e', 'v', 'w', 'f']:
                                        if stage == 'mdin.in':
                                            line = '  irest = 0,\n'
                                    if 'ntx' in line:
                                        #if remd:
                                        if stage == 'mdin.in':
                                            line = '  ntx = 1,\n'
                                    if 'ntwprt' in line:
                                        line = '\n'
                                    if 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "")
                                        # do not restraint the first dummy atom
                                        # line = f"restraintmask = '(!:1 & ({restraint_mask}))' \n"
                                        # alter
                                        # replace :1-2
                                        restraint_mask = restraint_mask.replace(':1-2', ':2')
                                        # placeholder that does not exist in the system
                                        restraint_mask = restraint_mask.replace(':1', '@ZYX')
                                        if stage == 'mdin.in.extend':
                                            line = f"restraintmask = '{restraint_mask}'\n"
                                        else:
                                            line = f"restraintmask = '@CA | {restraint_mask}' \n"
                                    if 'ntp' in line:
                                        # nvt simulation
                                        line = '  ntp = 0,\n'
                                    if 'gti_add_sc' in line:
                                        line = '\n'
                                    if 'gti_chg_keep' in line:
                                        line = '\n'
                                    if 'mbar_lambda' in line:
                                        line = '\n'
                                    if 'dt' in line:
                                        if stage == 'mdin.in':
                                            line = '  dt = 0.001,\n'
                                    if 'clambda' in line:
                                        final_line = []
                                        para_line = line.split(',')
                                        for i in range(len(para_line)):
                                            if 'clambda' in para_line[i]:
                                                continue
                                            if 'scalpha' in para_line[i]:
                                                continue
                                            if 'scbeta' in para_line[i]:
                                                continue
                                            final_line.append(para_line[i])
                                        line = ',\n'.join(final_line)
                                    if stage == 'mdin.in' or stage == 'mdin.in.extend':
                                        if 'nstlim' in line:
                                            inpcrd_file = f'fe/{win00_sim_folder_name}/full.inpcrd'
                                            # read the second line of the inpcrd file
                                            with open(inpcrd_file, 'r') as infile:
                                                lines = infile.readlines()
                                                n_atoms = int(lines[1])
                                            performance = calculate_performance(n_atoms, component)
                                            n_steps = int(50 / 60 / 24 * performance * 1000 * 1000 / 4)
                                            n_steps = int(n_steps // 100000 * 100000)
                                            line = f'  nstlim = {n_steps},\n'
                                            if remd and component in ['x', 'e', 'v', 'w', 'f'] and stage != 'mini.in':
                                                line = f'  nstlim = 100,\n'
                                    outfile.write(line)
                            f.write(f'# {component} {i} {stage}\n')
                            if stage == 'mdin.in':
                                f.write(f'-O -i {sim_folder_name}/mdin.in_frontier -p {prmtop} -c {sim_folder_name}/mini.in.rst7 '
                                        f'-o {sim_folder_name}/mdin-00.out -r {sim_folder_name}/mdin-00.rst7 -x {sim_folder_name}/mdin-00.nc '
                                        f'-ref {sim_folder_name}/mini.in.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-00.log '
                                        f'-e {sim_folder_name}/mdin-00.mden\n')
                            elif stage == 'mdin.in.extend':
                                f.write(f'-O -i {sim_folder_name}/mdin.in.extend_frontier -p {prmtop} -c {sim_folder_name}/mdin-CURRNUM.rst7 '
                                        f'-o {sim_folder_name}/mdin-NEXTNUM.out -r {sim_folder_name}/mdin-NEXTNUM.rst7 -x {sim_folder_name}/mdin-NEXTNUM.nc '
                                        f'-ref {sim_folder_name}/mini.in.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-NEXTNUM.log '
                                        f'-e {sim_folder_name}/mdin-NEXTNUM.mden\n')
                            else:
                                f.write(
                                    f'-O -i {sim_folder_name}/{stage.split("_")[0]}_frontier -p {prmtop} -c {stage_previous.replace("REPXXX", f"{i:02d}")} '
                                    f'-o {sim_folder_name}/{stage}.out -r {sim_folder_name}/{stage}.rst7 -x {sim_folder_name}/{stage}.nc '
                                    f'-ref {stage_previous.replace("REPXXX", f"{i:02d}")} -inf {sim_folder_name}/{stage}.mdinfo -l {sim_folder_name}/{stage}.log '
                                    f'-e {sim_folder_name}/{stage}.mden\n'
                                )
                            if stage == 'mdin.in':
                                all_replicates[component].append(f'{sim_folder_name}')
                        stage_previous = f'{sim_folder_temp}REPXXX/{stage}.rst7'
            logger.debug(f'all_replicates: {all_replicates}')
            return all_replicates

        def write_sbatch_file(pose, components):
            file_temp = f'{frontier_files}/fep_run.sbatch'
            lines_temp = open(file_temp).readlines()
            sbatch_all_file = f'fe/fep_{pose}_eq_all.sbatch'
            lines_all = []
            for line in lines_temp:
                lines_all.append(line)
            for component in components:
                lines = []
                for line in lines_temp:
                    lines.append(line)
                folder = os.getcwd()
                folder = '_'.join(folder.split(os.sep)[-4:])
                # write the sbatch file for equilibration

                lines.append(f'\n\n\n')
                lines.append(f'# {pose} {component}\n')
                lines_all.append(f'\n\n\n')
                lines_all.append(f'# {pose} {component}\n')

                sbatch_file = f'fe/fep_{component}_{pose}_eq.sbatch'
                groupfile_names = [
                    f'{pose}/groupfiles/{component}_{stage}.groupfile' for stage in sim_stages[component_2_folder_dict[component]]
                ]
                logger.debug(f'groupfile_names: {groupfile_names}')
                for g_name in groupfile_names:
                    if 'mdin.in' in g_name:
                        continue
                    if component in ['x', 'e', 'v', 'w', 'f']:
                        n_sims = len(lambdas)
                    else:
                        n_sims = len(attach_rest)
                    n_nodes = int(np.ceil(n_sims / 8))
                    if 'mini' in g_name:
                        # run with pmemd.mpi for minimization
                        lines.append(
                            f'srun -N {n_nodes} -n {n_sims * 8} pmemd.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                        lines_all.append(
                            f'srun -N {n_nodes} -n {n_sims * 8} pmemd.MPI -ng {n_sims} -groupfile {g_name}\n'
                        )
                    else:
                        if component in ['x', 'e', 'v', 'w', 'f'] and remd:
                            lines.append(
                                f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -rem 3 -groupfile {g_name}\n'
                            )
                            lines_all.append(
                                f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -rem 3 -groupfile {g_name}\n'
                            )
                        else:
                            lines.append(
                                f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {g_name}\n'
                            )
                            lines_all.append(
                                f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {g_name}\n'
                            )
                lines = [line
                        .replace('NUM_NODES', str(n_nodes))
                        .replace('FEP_SIM_XXX', f'{folder}_{component}_{pose}') for line in lines]
                lines_all = [line
                        .replace('NUM_NODES', str(n_nodes))
                        .replace('FEP_SIM_XXX', f'{folder}_{component}_{pose}') for line in lines_all]
                with open(sbatch_file, 'w') as f:
                    f.writelines(lines)
            with open(sbatch_all_file, 'w') as f:
                f.writelines(lines_all)

                
        def calculate_performance(n_atoms, comp):
            # Very rough estimate of the performance of the simulations
            # for 200000-atom systems: rest: 100 ns/day, sdr: 50 ns/day
            # for 70000-atom systems: rest: 200 ns/day, sdr: 100 ns/day
            # run 30 mins for each simulation
            if comp not in ['e', 'v', 'w', 'f', 'x']:
                if n_atoms < 80000:
                    return 150
                else:
                    return 80
            else:
                if n_atoms < 80000:
                    return 80
                else:
                    return 40

        def write_production_sbatch(all_replicates):
            sbatch_file = f'fe/fep_md.sbatch'
            sbatch_extend_file = f'fe/fep_md_extend.sbatch'

            file_temp = f'{frontier_files}/fep_run.sbatch'
            temp_lines = open(file_temp).readlines()
            temp_lines.append(f'\n\n\n')
                    # run the production
            folder = os.getcwd()
            folder = '_'.join(folder.split(os.sep)[-4:]) 
            n_sims = 0
            for replicates_pose in all_replicates:
                for comp, rep in replicates_pose.items():
                    n_sims += len(rep)
            n_nodes = int(np.ceil(n_sims / 8))
            temp_lines = [
                line.replace('NUM_NODES', str(n_nodes))
                    .replace('FEP_SIM_XXX', f'fep_md_{folder}')
                for line in temp_lines]

            with open(sbatch_file, 'w') as f:
                f.writelines(temp_lines)

            with open(sbatch_extend_file, 'w') as f:
                f.writelines(temp_lines)
                f.writelines(
                [
                    '# Make sure it\'s extending\n',
                    'latest_file=$(ls pose0/sdr/e00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)\n\n',
                    '# Check if any mdin-xx.rst7 files exist\n',
                    'if [[ -z "$latest_file" ]]; then\n',
                    'echo "No old production files found in the current directory."\n',
                    'echo "Run sbatch fep_md.sbatch."\n',
                    'exit 1\n',
                    'fi\n\n',
                    # 'poses=(pose0 pose1 pose2 pose3 pose4)
                    f'poses=({" ".join(poses_def)})\n',
                    # groups=(m n e v)
                    f'groups=({" ".join(components)})\n',
                    'for pose in "${poses[@]}"; do\n',
                    '  for group in "${groups[@]}"; do\n',
                    '    latest_file=$(ls $pose/*/${group}00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)\n',
                    '    echo "Last file for $pose/$group: $latest_file"\n',
                    '    latest_num=$(echo "$latest_file" | grep -oP "(?<=-)[0-9]{2}(?=\.rst7)")\n',
                    '    next_num=$(printf "%02d" $((10#$latest_num + 1)))\n',
                    '    echo "Last file for $pose/$group: $latest_file"\n',
                    '    echo "Next number: $next_num"\n',
                    '    sed "s/CURRNUM/$latest_num/g" ${pose}/groupfiles/${group}_mdin.in.extend.groupfile > ${pose}/groupfiles/${group}_temp_mdin.groupfile\n',
                    '    sed "s/NEXTNUM/$next_num/g" ${pose}/groupfiles/${group}_temp_mdin.groupfile > ${pose}/groupfiles/${group}_current_mdin.groupfile\n',
                            
                    '    case "$group" in\n',
                    '        m|n) \n',
                    '        srun -N 2 -n 16 pmemd.hip_DPFP.MPI -ng 16 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile &\n',
                    '        echo "srun -N 2 -n 16 pmemd.hip_DPFP.MPI -ng 16 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile"\n',
                    '        ;;\n',
                    '        e|v|x)\n',
                    '        srun -N 3 -n 24 pmemd.hip_DPFP.MPI -ng 24 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile &\n',
                    '        echo "srun -N 3 -n 24 pmemd.hip_DPFP.MPI -ng 24 -groupfile ${pose}/groupfiles/${group}_current_mdin.groupfile"\n',
                    '        ;;\n',
                    '        *)\n',
                    '        echo "Invalid group"\n',
                    '        ;;\n',
                    '    esac\n',
                    '    sleep 0.5\n',
                    '    done\n',
                    'done\n',
                    'wait\n\n',
                ]
                )

            for replicates_pose in all_replicates:
                for comp, rep in replicates_pose.items():
                    pose = rep[0].split('/')[0]
                    groupfile_name_prod = f'{pose}/groupfiles/{comp}_mdin.in.groupfile'
                    groupfile_name_prod_extend = f'{pose}/groupfiles/{comp}_mdin.in.extend.groupfile'

                    n_nodes = int(np.ceil(len(rep) / 8))
                    n_sims = len(rep)
                    with open(sbatch_file, 'a') as f:
                        f.writelines(
                            [
                        f'# {pose} {comp}\n',
                        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {groupfile_name_prod} &\n',
                        f'sleep 0.5\n'
                            ]
                        )
                    #with open(sbatch_extend_file, 'a') as f:
                    #    f.writelines(
                    #        [ 
                    #        f'sed "s/CURRNUM/$latest_num/g" {pose}/groupfiles/{comp}_mdin.in.extend.groupfile > {pose}/groupfiles/{comp}_temp_mdin.groupfile\n',
                    #        f'sed "s/NEXTNUM/$next_num/g" {pose}/groupfiles/{comp}_temp_mdin.groupfile > {pose}/groupfiles/{comp}_current_mdin.groupfile\n',
                    #        f'# {pose} {comp}\n',
                    #        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_DPFP.MPI -ng {n_sims} -groupfile {pose}/groupfiles/{comp}_current_mdin.groupfile &\n',
                    #        f'sleep 0.5\n\n'
                    #        ]
                    #    )
            # append wait
            with open(sbatch_file, 'a') as f:
                f.write('wait\n')

            #with open(sbatch_extend_file, 'a') as f:
            #    f.write('wait\n')

        all_replicates = []

        with self._change_dir(self.output_dir):

            for pose in poses_def:
                if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                    continue
                all_replicates.append(write_2_pose(pose, components))
                write_sbatch_file(pose, components)
                logger.debug(f'Generated groupfiles for {pose}')
            logger.debug(all_replicates)
            write_production_sbatch(all_replicates)
            # copy env.amber.24
            env_amber_file = f'{frontier_files}/env.amber.{version}'
            shutil.copy(env_amber_file, 'fe/env.amber')
            logger.info('Generated groupfiles for all poses')
    
    def check_sim_stage(self):
        stage_sims = {}
        for pose in self.sim_config.poses_def:
            stage_sims[pose] = {}
            for comp in self.sim_config.components:
                if comp in ['m', 'n']:
                    sim_type = 'rest'
                elif comp in ['e', 'v', 'x']:
                    sim_type = 'sdr'
                folder = f'{self.fe_folder}/{pose}/{sim_type}/{comp}00'
                mdin_files = glob.glob(f'{folder}/mdin-*.rst7')
                # make sure the size is not empty
                mdin_files = [f for f in mdin_files if os.path.getsize(f) > 100]
                mdin_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
                if len(mdin_files) > 0:
                    stage_sims[pose][comp] = int(mdin_files[-1].split('-')[-1].split('.')[0])
                else:
                    stage_sims[pose][comp] = -1
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        stage_sims_df = pd.DataFrame(stage_sims)
        fig, ax = plt.subplots(figsize=(1* len(self.sim_config.poses_def), 5))
        sns.heatmap(stage_sims_df, ax=ax, annot=True, cmap='viridis')
        plt.show()


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

