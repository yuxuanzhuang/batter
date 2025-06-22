"""
Provide the primary functions for preparing and processing FEP systems.
"""
import inspect
from collections.abc import MutableMapping
import numpy as np
import os
import sys
import shutil
import glob
from contextlib import contextmanager
import MDAnalysis as mda
from MDAnalysis.analysis import align
import re
import pandas as pd
from importlib import resources
import json
from typing import Union
from pathlib import Path
import pickle
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from loguru import logger
from batter.input_process import SimulationConfig, get_configure_from_file
from batter.results import FEResult, NewFEResult
from batter.utils.utils import tqdm_joblib
from batter.utils.slurm_job import SLURMJob, get_squeue_job_count
from batter.data import frontier_files
from batter.data import run_files as run_files_orig
from batter.data import build_files as build_files_orig
from batter.builder import BuilderFactory
from batter.utils import (
    run_with_log,
    save_state,
    safe_directory,
    natural_keys

)

from batter.utils import (
    COMPONENTS_LAMBDA_DICT,
    COMPONENTS_FOLDER_DICT,
    COMPONENTS_DICT,
    DEC_FOLDER_DICT,
)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


AVAILABLE_COMPONENTS = COMPONENTS_DICT['dd'] + COMPONENTS_DICT['rest']

# the direction of the components that are used in the simulation
COMPONENT_DIRECTION_DICT = {
    'm': -1,
    'n': 1,
    'e': -1,
    'v': -1,
    'o': -1,
    'z': -1,
    'Boresch': -1,
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
                 create_new: bool = False,
                ):
        """
        Initialize the System class with the folder name.
        The system will be loaded from the folder.
        If create_new is True, a new system will be created when
        the folder does not exist.

        Parameters
        ----------
        folder : str
            The folder containing the system files.
        """
        self.output_dir = os.path.abspath(folder) + '/'
        if not os.path.exists(self.output_dir) and not create_new:
            raise FileNotFoundError(f"System folder does not exist: {self.output_dir}; add create_new=True to create a new system")

        #logger.add(f"{self.output_dir}/batter.log", level='INFO')
        logger.add(f"{self.output_dir}/batter.log", level='DEBUG')

        self._slurm_jobs = {}
        self._sim_finished = {}
        self._sim_failed = {}
        self._eq_prepared = False
        self._fe_prepared = False
        self._fe_results = {}
        self.mols = []

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
                    ligand_paths: Union[List[str], dict[str, str]],
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
        ligand_paths : List[str] or Dict[str, str]
            List of ligand files. It can be either PDB, mol2, or sdf format.
            It will be stored in the `all-poses` folder as `pose0.pdb`,
            `pose1.pdb`, etc.
            If it's a dictionary, the keys will be used as the ligand names
            and the values will be the ligand files.
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
        # fail late on import openff
        try:
            from openff.toolkit import Molecule
        except:
            raise ImportError("OpenFF toolkit is not installed. Please install it with `conda install -c conda-forge openff-toolkit-base`")

        # Log every argument
        if verbose:
            logger.remove()
            logger.add(sys.stdout, level='DEBUG')
            logger.add(f"{self.output_dir}/batter.log", level='DEBUG')
            logger.debug('Verbose mode is on')
            logger.debug('Creating a new system')

        self.verbose = verbose
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
        
        # always store a unique identifier for the ligand
        if isinstance(ligand_paths, list):
            self._ligand_list = {
                f'lig{i}': self._convert_2_relative_path(path)
                for i, path in enumerate(ligand_paths)
            }
        elif isinstance(ligand_paths, dict):
            self._ligand_list = {ligand_name: self._convert_2_relative_path(ligand_path) for ligand_name, ligand_path in ligand_paths.items()}
        self.receptor_segment = receptor_segment
        self._protein_align = protein_align
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
        
        # copy dummy atom parameters to the ligandff folder
        os.system(f"cp {build_files_orig}/dum.mol2 {self.ligandff_folder}")
        os.system(f"cp {build_files_orig}/dum.frcmod {self.ligandff_folder}")

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
        
        from batter.ligand_process import LigandFactory
        
        self.unique_mol_names = []
        mols = []
        # only process the unique ligand paths
        # for ABFESystem, it will be a single ligand
        # for MBABFE and RBFE, it will be multiple ligands
        for ind, (ligand_path, ligand_names) in enumerate(self._unique_ligand_paths.items(), start=1):
            logger.debug(f'Processing ligand {ind}: {ligand_path} for {ligand_names}')
            # first if self.mols is not empty, then use it as the ligand name
            try:
                ligand_name = self.mols[ind-1]
            except:
                ligand_name = ligand_names[0]

            ligand_factory = LigandFactory()
            ligand = ligand_factory.create_ligand(
                    ligand_file=ligand_path,
                    index=ind,
                    output_dir=self.ligandff_folder,
                    ligand_name=ligand_name,
                    retain_lig_prot=self.retain_lig_prot,
                    ligand_ff=self.ligand_ff,
                    unique_mol_names=self.unique_mol_names
            )

            mols.append(ligand.name)
            self.unique_mol_names.append(ligand.name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{ligand.name}.frcmod"):
                ligand.prepare_ligand_parameters()
            for ligand_name in ligand_names:
                self.ligand_list[ligand_name] = self._convert_2_relative_path(f'{self.ligandff_folder}/{ligand.name}.pdb')

        logger.debug( f"Unique ligand names: {self.unique_mol_names} ")
        logger.debug('updating the ligand paths')
        logger.debug(self.ligand_list)

        self.mols = mols
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
    def protein_align(self):
        try:
            return self._protein_align
        except AttributeError:
            return 'name CA'
    
    @protein_align.setter
    def protein_align(self, value):
        self._protein_align = value


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
        """
        The paths to the ligand files.
        """
        return [f"{self.output_dir}/{ligand_path}" for ligand_path in self._ligand_list.values()]
    
    @property
    def ligand_list(self):
        """
        A dictionary of ligands.
        """
        return self._ligand_list
    
    @property
    def pose_ligand_dict(self):
        """
        A dictionary of ligands with pose names as keys.
        """
        try:
            return self._pose_ligand_dict
        except AttributeError:
            return {pose.split('/')[-1].split('.')[0]: ligand
                    for ligand, pose in self.ligand_list.items()}

    
    @property
    def ligand_names(self):
        """
        The names of the ligands.
        """
        return list(self._ligand_list.keys())
    
    @property
    def all_poses(self):
        """
        The path to the all-poses folder.
        """
        try:
            return self._all_poses
        except AttributeError:
            return [f'pose{i}' for i in range(len(self.ligand_paths))]
        
    @property
    def bound_poses(self):
        """
        The bound poses of the ligands. It will be estimated
        from equilibration simulation.
        """
        try:
            return self._bound_poses
        except AttributeError:
            self._check_equilbration_binding()
            return self._bound_poses

    @property
    def bound_mols(self):
        """
        The bound molecules of the ligands. It will be estimated
        from equilibration simulation.
        """
        try:
            return self._bound_mols
        except AttributeError:
            self._check_equilbration_binding()
            return self._bound_mols

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
        logger.debug(f'Number of water molecules: {water_ag.n_residues}')
        # also include ions (in CHARMM name) to water_ag
        ion_ag = u_sys.select_atoms('byres (resname SOD POT CLA and around 5 (protein))')
        logger.debug(f'Number of ion molecules: {ion_ag.n_residues}')
        # replace SOD with Na+ and POT with K+ and CLA with Cl-
        ion_ag.select_atoms('resname SOD').names = 'Na+'
        ion_ag.select_atoms('resname SOD').residues.resnames = 'Na+'
        ion_ag.select_atoms('resname POT').names = 'K+'
        ion_ag.select_atoms('resname POT').residues.resnames = 'K+'
        ion_ag.select_atoms('resname CLA').names = 'Cl-'
        ion_ag.select_atoms('resname CLA').residues.resnames = 'Cl-'

        water_ag = water_ag + ion_ag
        water_ag.chainIDs = 'W'
        water_ag.residues.segments = water_seg

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
        logger.debug('prepare ligand poses')
        with self._change_dir(self.output_dir):
            new_ligand_list = {}
            for i, (name, pose) in enumerate(self.ligand_list.items()):
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
                
                logger.debug(f"Processing ligand {i}: {pose}")
                self._align_2_system(u.atoms)
                u.atoms.write(f"{self.poses_folder}/pose{i}.pdb")
                pose = f"{self.poses_folder}/pose{i}.pdb"

                if not os.path.exists(f"{self.poses_folder}/pose{i}.pdb"):
                    shutil.copy(pose, f"{self.poses_folder}/pose{i}.pdb")

                new_ligand_list[name] = pose
            self._ligand_list = new_ligand_list

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
        logger.debug(f'Simulation configuration: {sim_config}')
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
            input_file: Union[str, Path, SimulationConfig] = None,
            overwrite: bool = False,
            partition: str = 'rondror',
            n_workers: int = 12,
            win_info_dict: dict = None,
            avg_struc: str = None,
            rmsf_file: str = None,
            extra_restraints: str = None,
            extra_restraints_fc: float = 10,
            ):
        """
        Prepare the system for the FEP simulation.

        Parameters
        ----------
        stage : str
            The stage of the simulation. Options are 'equil' and 'fe'.
        input_file : str
            Path to the input file for the simulation. If None,
            the loaded SimulationConfig will be used.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        partition : str, optional
            The partition to submit the job. Default is 'rondror'.
        n_workers : int, optional
            The number of workers to use for the simulation. Default is 6.
        win_info_dict : dict, optional
            The lambda / restraint values for components.
            It should be in a format of e.g. for `e` component:
            {
                'e': [0, 0.5, 1.0],
            }
        """
        logger.debug('Preparing the system')
        self.overwrite = overwrite
        self.partition = partition
        self._n_workers = n_workers
        if avg_struc is not None and rmsf_file is not None:
            rmsf_restraints = True
        elif avg_struc is not None or rmsf_file is not None:
            raise ValueError("Both avg_struc and rmsf_file should be provided")
        else:
            rmsf_restraints = False
        
        if input_file is not None:
            self._get_sim_config(input_file)
            self._component_windows_dict = ComponentWindowsDict(self)
        
        try:
            sim_config = self.sim_config
        except AttributeError:
            raise ValueError("Simulation configuration is not set. "
                             "Please provide an input file")
        if win_info_dict is not None:
            for key, value in win_info_dict.items():
                if key not in self._component_windows_dict:
                    raise ValueError(f"Invalid component: {key}. Available components are: {self._component_windows_dict.keys()}")
                self._component_windows_dict[key] = value
        
        if len(self.sim_config.poses_def) != len(self.ligand_paths):
            logger.debug(f"Number of poses in the input file: {len(self.sim_config.poses_def)} "
                           f"does not match the number of ligands: {len(self.ligand_paths)}")
            logger.debug("Using the ligand paths for the poses")
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]
        self._pose_ligand_dict = {pose: ligand for pose, ligand in zip(self._all_poses, self.ligand_names)}
        self.sim_config.poses_def = self._all_poses 

        if stage == 'equil':
            if self.overwrite:
                logger.debug(f'Overwriting {self.equil_folder}')
                shutil.rmtree(self.equil_folder, ignore_errors=True)
                self._eq_prepared = False
            elif self._eq_prepared and os.path.exists(f"{self.equil_folder}"):
                logger.info('Equilibration already prepared')
                return
            self._slurm_jobs = {}
            # save the input file to the equil directory
            os.makedirs(f"{self.equil_folder}", exist_ok=True)
            with open(f"{self.equil_folder}/sim_config.json", 'w') as f:
                json.dump(self.sim_config.model_dump(), f, indent=2)
            
            self._prepare_equil_system()
            if rmsf_restraints:
                self.add_rmsf_restraints(
                        stage='equil',
                        avg_struc=avg_struc,
                        rmsf_file=rmsf_file
                    )
            if extra_restraints is not None:
                self.add_extra_restraints(
                        stage='equil',
                        extra_restraints=extra_restraints,
                        extra_restraints_fc=extra_restraints_fc
                    )
            logger.info('Equil System prepared')
            self._eq_prepared = True
        
        if stage == 'fe':
            if not os.path.exists(f"{self.equil_folder}"):
                raise FileNotFoundError("Equilibration not generated yet. Run prepare('equil') first.")
        
            if not os.path.exists(f"{self.equil_folder}/{self.all_poses[0]}/md03.rst7"):
                raise FileNotFoundError("Equilibration not finished yet. First run the equilibration.")
                
            sim_config_eq = json.load(open(f"{self.equil_folder}/sim_config.json"))
            if sim_config_eq != sim_config.model_dump():
            # raise ValueError(f"Equilibration and free energy simulation configurations are different")
                warnings.warn("Equilibration and free energy simulation configurations are different")
                # get the difference
                diff = {k: v for k, v in sim_config_eq.items() if sim_config.model_dump().get(k) != v}
                logger.warning(f"Different configurations: {diff}")
                orig = {k: sim_config.model_dump().get(k) for k in diff.keys()}
                logger.warning(f"Original configuration: {orig}")

            #self._fe_prepared = False
            if self.overwrite:
                logger.debug(f'Overwriting {self.fe_folder}')
                shutil.rmtree(self.fe_folder, ignore_errors=True)
                self._fe_prepared = False
            elif self._fe_prepared and os.path.exists(f"{self.fe_folder}"):
                logger.info('Free energy already prepared')
                return
            self._slurm_jobs = {}
            self._fe_prepared = False
            os.makedirs(f"{self.fe_folder}", exist_ok=True)
            
            if not os.path.exists(f"{self.fe_folder}/ff"):
                logger.debug(f'Copying ff folder from {self.ligandff_folder} to {self.fe_folder}/ff')
                # shutil.copytree(self.ligandff_folder,
                #             f"{self.fe_folder}/ff")
                # use os.copy instead
                os.makedirs(f"{self.fe_folder}/ff", exist_ok=True)
                for file in os.listdir(self.ligandff_folder):
                    os.system(f"cp {self.ligandff_folder}/{file} {self.fe_folder}/ff/{file}")


            
            with open(f"{self.fe_folder}/sim_config.json", 'w') as f:
                json.dump(self.sim_config.model_dump(), f, indent=2)

            self._check_equilbration_binding()
            self._find_new_anchor_atoms()
            self._prepare_fe_system()
            if rmsf_restraints:
                self.add_rmsf_restraints(
                        stage='fe',
                        avg_struc=avg_struc,
                        rmsf_file=rmsf_file
                    )
            if extra_restraints is not None:
                self.add_extra_restraints(
                        stage='fe',
                        extra_restraints=extra_restraints,
                        extra_restraints_fc=extra_restraints_fc
                    )
            logger.info('FE System prepared')
            self._fe_prepared = True
        
    @safe_directory
    @save_state
    def submit(self,
               stage: str,
               cluster: str = 'slurm',
               partition=None,
               time_limit=None,
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
        time_limit: str, optional
            The time limit for the job. Default is None,
        overwrite : bool, optional
            Whether to overwrite and re-run all the existing simulations.
        """
        if cluster == 'frontier':
            raise NotImplementedError('run with `batter run-in-batch` instead')
            return

        if stage == 'equil':
            logger.debug('Submit equilibration stage')
            pbar = tqdm(total=len(self.all_poses), desc='Submitting equilibration jobs')
            for pose in self.all_poses:
                # check n_jobs_submitted is less than the max_jobs
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit equilibration jobs')

                # check existing jobs
                if os.path.exists(f"{self.equil_folder}/{pose}/FINISHED") and not overwrite:
                    logger.debug(f'Equilibration for {pose} has finished; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'eq_{pose}', None)
                    continue
                if os.path.exists(f"{self.equil_folder}/{pose}/FAILED") and not overwrite:
                    logger.warning(f'Equilibration for {pose} has failed; add overwrite=True to re-run the simulation')
                    self._slurm_jobs.pop(f'eq_{pose}', None)
                    continue
                if f'eq_{pose}' in self._slurm_jobs:
                    # check if it's finished
                    slurm_job = self._slurm_jobs[f'eq_{pose}']
                    # if the job is finished but the FINISHED file is not created
                    # resubmit the job
                    if not slurm_job.is_still_running():
                        slurm_job.submit(time=time_limit)
                        continue
                    elif overwrite:
                        slurm_job.cancel()
                        slurm_job.submit(overwrite=True,
                                         time=time_limit)
                        continue
                    else:
                        logger.debug(f'Equilibration job for {pose} is still running')
                        continue

                if overwrite:
                    # remove FINISHED and FAILED
                    os.remove(f"{self.equils_folder}/{pose}/FINISHED", ignore_errors=True)
                    os.remove(f"{self.equils_folder}/{pose}/FAILED", ignore_errors=True)

                slurm_job = SLURMJob(
                                filename=f'{self.equil_folder}/{pose}/SLURMM-run',
                                partition=partition,
                                jobname=f'fep_{self.equil_folder}/{pose}_equil')
                slurm_job.submit(overwrite=overwrite, time=time_limit)
                pbar.update(1)
                pbar.set_description(f'Equilibration job for {pose} submitted: {slurm_job.jobid}')
                self._slurm_jobs.update(
                    {f'eq_{pose}': slurm_job}
                )
                # make sure the system is saved every time when a job is submitted
                with open(f"{self.output_dir}/system.pkl", 'wb') as f:
                    pickle.dump(self, f)

            pbar.close()
            logger.info('Equilibration systems have been submitted for all poses listed in the input file.')
        elif stage == 'fe_equil':
            logger.info('Submit NPT equilibration part of free energy stage')
            pbar = tqdm(total=len(self.bound_poses), desc='Submitting free energy equilibration jobs')
            for pose in self.bound_poses:
                # only check for each pose to reduce frequently checking SLURM 
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit FE equilibration jobs')
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    # only run for window -1 (eq)
                    j = -1
                    # check n_jobs_submitted is less than the max_jobs
                    folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}'
                    #if os.path.exists(f"{folder_2_check}/FINISHED") and not overwrite:
                    #    self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}{j:02d}', None)
                    #    logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                    #    continue
                    if os.path.exists(f"{folder_2_check}/FAILED") and not overwrite:
                        self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}{j:02d}', None)
                        logger.warning(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has failed; add overwrite=True to re-run the simulation')
                        continue
                    if os.path.exists(f"{folder_2_check}/EQ_FINISHED") and not overwrite:
                        logger.debug(f'FE for {pose}/{comp_folder}/{comp}{j:02d} has finished; add overwrite=True to re-run the simulation')
                        continue
                    if f'fe_{pose}_{comp_folder}_{comp}{j:02d}' in self._slurm_jobs:
                        slurm_job = self._slurm_jobs[f'fe_{pose}_{comp_folder}_{comp}{j:02d}']
                        if not slurm_job.is_still_running():
                            slurm_job.submit(
                                requeue=True,
                                time=time_limit,
                                other_env={
                                    'ONLY_EQ': '1',
                                    'INPCRD': 'full.inpcrd'
                            }
                            )
                            continue
                        elif overwrite:
                            slurm_job.cancel()
                            slurm_job.submit(overwrite=True,
                                time=time_limit,
                                other_env={
                                    'ONLY_EQ': '1',
                                    'INPCRD': 'full.inpcrd'
                                }
                            )
                            continue
                        else:
                            logger.debug(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} is still running')
                            continue

                    if overwrite:
                        # remove FINISHED and FAILED
                        os.remove(f"{folder_2_check}/FINISHED", ignore_errors=True)
                        os.remove(f"{folder_2_check}/FAILED", ignore_errors=True)

                    slurm_job = SLURMJob(
                                    filename=f'{folder_2_check}/SLURMM-run',
                                    partition=partition,
                                    jobname=f'fep_{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}_equil')
                    slurm_job.submit(
                        overwrite=overwrite,
                        time=time_limit,
                        other_env={
                        'ONLY_EQ': '1',
                        'INPCRD': 'full.inpcrd'
                    })
                    pbar.set_description(f'FE equil job for {pose}/{comp_folder}/{comp}{j:02d} submitted')
                    self._slurm_jobs.update(
                        {f'fe_{pose}_{comp_folder}_{comp}{j:02d}': slurm_job}
                    )
                    with open(f"{self.output_dir}/system.pkl", 'wb') as f:
                        pickle.dump(self, f)

                pbar.update(1)
            logger.info('Free energy systems have been submitted for all poses listed in the input file.')        
            pbar.close()
        elif stage == 'fe':
            logger.info('Submit free energy stage')
            pbar = tqdm(total=len(self.bound_poses), desc='Submitting free energy jobs')
            priorities = np.arange(1, len(self.bound_poses) + 1)[::-1] * 10000
            for i, pose in enumerate(self.bound_poses):
                # set gradually lower priority for jobs
                priority = priorities[i]
                # only check for each pose to reduce frequently checking SLURM 
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit FE jobs')
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    windows = self.component_windows_dict[comp]
                    for j in range(len(windows)):
                        # check n_jobs_submitted is less than the max_jobs
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
                            slurm_job.priority = priority
                            if not slurm_job.is_still_running():
                                slurm_job.submit(
                                    requeue=True,
                                    time=time_limit,
                                    other_env={
                                        'INPCRD': f'../{comp}-1/eqnpt04.rst7'
                                    }
                                )
                                continue
                            elif overwrite:
                                slurm_job.cancel()
                                slurm_job.submit(overwrite=True,
                                    time=time_limit,
                                    other_env={
                                        'INPCRD': f'../{comp}-1/eqnpt04.rst7'
                                    }
                                )
                                continue
                            else:
                                logger.debug(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} is still running')
                                continue

                        if overwrite:
                            # remove FINISHED and FAILED
                            os.remove(f"{folder_2_check}/FINISHED", ignore_errors=True)
                            os.remove(f"{folder_2_check}/FAILED", ignore_errors=True)

                        slurm_job = SLURMJob(
                                        filename=f'{folder_2_check}/SLURMM-run',
                                        partition=partition,
                                        jobname=f'fep_{folder_2_check}_fe',
                                        priority=priority)
                        slurm_job.submit(overwrite=overwrite,
                                    time=time_limit,
                                    other_env={
                                            'INPCRD': f'../{comp}-1/eqnpt04.rst7'
                                        }
                        )

                        pbar.set_description(f'FE job for {pose}/{comp_folder}/{comp}{j:02d} submitted')
                        self._slurm_jobs.update(
                            {f'fe_{pose}_{comp_folder}_{comp}{j:02d}': slurm_job}
                        )
                        with open(f"{self.output_dir}/system.pkl", 'wb') as f:
                            pickle.dump(self, f)
                pbar.update(1)
            pbar.close()

            logger.info('Free energy systems have been submitted for all poses listed in the input file.')
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
            #shutil.copytree(self.poses_folder,
            #            f"{self.equil_folder}/all-poses")
            # use os.copy instead
            os.makedirs(f"{self.equil_folder}/all-poses", exist_ok=True)
            for file in os.listdir(self.poses_folder):
                os.system(f"cp {self.poses_folder}/{file} {self.equil_folder}/all-poses/{file}")
        if not os.path.exists(f"{self.equil_folder}/ff"):
            logger.debug(f'Copying ff folder from {self.ligandff_folder} to {self.equil_folder}/ff')
            #shutil.copytree(self.ligandff_folder,
            #            f"{self.equil_folder}/ff")
            # use os.copy instead
            os.makedirs(f"{self.equil_folder}/ff", exist_ok=True)
            for file in os.listdir(self.ligandff_folder):
                os.system(f"cp {self.ligandff_folder}/{file} {self.equil_folder}/ff/{file}")

        # copy run_files
        if not os.path.exists(f"{self.equil_folder}/run_files"):
            logger.debug(f'Copying run_files folder from {self.ligandff_folder} to {self.equil_folder}/run_files')
            #shutil.copytree(run_files_orig,
            #            f"{self.equil_folder}/run_files")
            # use os.copy instead
            os.makedirs(f"{self.equil_folder}/run_files", exist_ok=True)
            for file in os.listdir(run_files_orig):
                os.system(f"cp {run_files_orig}/{file} {self.equil_folder}/run_files/{file}")
        
        hmr = self.sim_config.hmr
        if hmr == 'no':
            replacement = 'full.prmtop'
            for dname, dirs, files in os.walk(f'{self.equil_folder}/run_files'):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.hmr.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)
        elif hmr == 'yes':
            replacement = 'full.hmr.prmtop'
            for dname, dirs, files in os.walk(f'{self.equil_folder}/run_files'):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)

        builders = []
        builders_factory = BuilderFactory()
        for pose in self.all_poses:
            #logger.info(f'Preparing pose: {pose}')
            if os.path.exists(f"{self.equil_folder}/{pose}/cv.in") and not self.overwrite:
                logger.info(f'Pose {pose} already exists; add overwrite=True to re-build the pose')
                continue
            equil_builder = builders_factory.get_builder(
                stage='equil',
                system=self,
                pose=pose,
                sim_config=sim_config,
                working_dir=f'{self.equil_folder}'
            )
            builders.append(equil_builder)

        # run builders.build in parallel
        logger.info(f'Building equilibration systems for {len(builders)} poses')
        Parallel(n_jobs=self.n_workers, backend='loky')(
            delayed(builder.build)() for builder in builders
        )
        
        logger.info('Equilibration systems have been created for all poses listed in the input file.')

    def _prepare_fe_equil_system(self):

        # molr (molecule reference) and poser (pose reference)
        # are used for exchange FE simulations.
        sim_config = self.sim_config
        molr = self.mols[0]
        poser = self.bound_poses[0]
        builders = []
        builders_factory = BuilderFactory()
        for pose in sim_config.poses_def:
            # if "UNBOUND" found in equilibration, skip
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                logger.info(f"Pose {pose} is UNBOUND in equilibration; skipping FE")
                os.makedirs(f"{self.fe_folder}/{pose}/Results", exist_ok=True)
                with open(f"{self.fe_folder}/{pose}/Results/Results.dat", 'w') as f:
                    f.write("UNBOUND\n")
                continue
            logger.debug(f'Preparing pose: {pose}')
            
            # load anchor_list
            with open(f"{self.equil_folder}/{pose}/anchor_list.txt", 'r') as f:
                anchor_list = f.readlines()
                l1x, l1y, l1z = [float(x) for x in anchor_list[0].split()]
                sim_config.l1_x = l1x
                sim_config.l1_y = l1y
                sim_config.l1_z = l1z

            # copy ff folder
            #shutil.copytree(self.ligandff_folder,
            #                f"{self.fe_folder}/{pose}/ff", dirs_exist_ok=True)
            os.makedirs(f"{self.fe_folder}/{pose}/ff", exist_ok=True)
            for file in os.listdir(self.ligandff_folder):
                shutil.copy(f"{self.ligandff_folder}/{file}",
                            f"{self.fe_folder}/{pose}/ff/{file}")
            
            for component in sim_config.components:
                logger.debug(f'Preparing component: {component}')
                lambdas_comp = sim_config.dict()[COMPONENTS_LAMBDA_DICT[component]]
                n_sims = len(lambdas_comp)
                logger.debug(f'Number of simulations: {n_sims}')
                cv_path = f"{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[component]}/{component}-1/cv.in"
                if os.path.exists(cv_path) and not self.overwrite:
                    logger.info(f"Component {component} for pose {pose} already exists; add overwrite=True to re-build the component")
                    continue
                fe_eq_builder = builders_factory.get_builder(
                    stage='fe',
                    win=-1,
                    component=component,
                    system=self,
                    pose=pose,
                    sim_config=sim_config,
                    working_dir=f'{self.fe_folder}',
                    molr=molr,
                    poser=poser
                )
                builders.append(fe_eq_builder)
        with tqdm_joblib(tqdm(
            total=len(builders),
            desc="Preparing FE equilibration",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")) as pbar:
            Parallel(n_jobs=self.n_workers, backend='loky')(
                delayed(builder.build)() for builder in builders
        )
            
    def _prepare_fe_windows(self, regenerate: bool = False):
        sim_config = self.sim_config
        molr = self.mols[0]
        poser = self.bound_poses[0]

        builders = []
        builders_factory = BuilderFactory()
        for pose in self.bound_poses:
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                continue

            for component in sim_config.components:
                if regenerate:
                    # delete existing windows
                    windows = glob.glob(f"{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[component]}/{component}[0-9][0-9]")
                    for window in windows:
                        shutil.rmtree(window, ignore_errors=True)
                lambdas_comp = self.component_windows_dict[component]
                cv_paths = []
                for i, _ in enumerate(lambdas_comp):
                    cv_path = f"{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[component]}/{component}{i:02d}/cv.in"
                    cv_paths.append(cv_path)
                # check if the cv.in file exists
                # it should be created in the last step
                if all(os.path.exists(cv_paths[i]) for i, _ in enumerate(lambdas_comp)) and not self.overwrite:
                    continue

                for i, lambdas in enumerate(lambdas_comp):
                    fe_builder = builders_factory.get_builder(
                        stage='fe',
                        win=i,
                        component=component,
                        system=self,
                        pose=pose,
                        sim_config=sim_config,
                        working_dir=f'{self.fe_folder}',
                        molr=molr,
                        poser=poser
                    )
                    builders.append(fe_builder)

        with tqdm_joblib(tqdm(
            total=len(builders),
            desc="Preparing FE windows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")) as pbar:
            Parallel(n_jobs=self.n_workers, backend='loky')(
                delayed(builder.build)() for builder in builders
            )

    def _prepare_fe_system(self):
        """
        Prepare the free energy system.
        """
        logger.info('Prepare for free energy stage')
        logger.info('Prepare FE equilibration')
        self._prepare_fe_equil_system()

        logger.info('Prepare FE windows')
        self._prepare_fe_windows()
        logger.info('Free energy systems have been created for all poses listed in the input file.')


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
                    #shutil.copy(cv_file + '.bak', cv_file)
                    os.system(f"cp {cv_file} {cv_file}.bak")
                else:
                    # copy original cv file for backup
                    #shutil.copy(cv_file, cv_file + '.bak')
                    os.system(f"cp {cv_file} {cv_file}.bak")
                
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
                
        logger.info(f'Adding RMSF restraints for {stage} stage')
        if stage == 'equil':
            for pose in self.all_poses:
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
            for pose in self.bound_poses:
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    u_ref = mda.Universe(
                            f"{folder_comp}/{comp}00/full.pdb",
                            f"{folder_comp}/{comp}00/full.inpcrd")
                    windows = self.component_windows_dict[comp]
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

    def add_extra_restraints(self,
                            stage: str,
                            extra_restraints: str,
                            extra_restraints_fc: float = 10):
        """
        Add Harmonic position restraints to the system.

        Parameters
        ----------
        stage : str
            The stage of the simulation.
            Options are 'equil' and 'fe'.
        extra_restraints : str
            The selection string of the atoms to be restrained.
        extra_restraints_fc : float, optional
            The force constant of the restraints. Default is 10.
        """
        logger.debug('Adding Harmonic postion restraints')

        num_eq_sim = len(self.sim_config.release_eq)
        num_fe_sim = self.sim_config.num_fe_range

        def write_restraint_block(files, folder_2_write):
            for file in files:
                with open(f'{folder_2_write}/{file}', 'r') as f:
                    lines = f.readlines()

                # if new_mask_component is already in the file, skip
                if any(new_mask_component in line for line in lines):
                    logger.debug(f"Restraint mask {new_mask_component} already exists in {file}; skipping")
                    continue
                
                with open(f'{folder_2_write}/{file}', 'w') as f:
                    for line in lines:
                        if 'ntr' in line and 'cntr' not in line:
                            line = '  ntr = 1,\n'
                        elif 'restraintmask' in line:
                            current_mask = re.search(r'restraintmask\s*=\s*["\']([^"\']*)["\']', line)
                            if current_mask:
                                base_mask = current_mask.group(1).strip().rstrip(',')
                                # Remove previously appended CA-residue parts if present
                                base_mask = re.sub(r'\|\s*\(\(:.*?\)&\s*@CA\)', '', base_mask).strip()
                            else:
                                base_mask = ""
                            
                            if base_mask != "":
                                restraint_mask = f'({base_mask}) | ({new_mask_component})'
                            else:
                                restraint_mask = new_mask_component

                            line = f'  restraintmask = "{restraint_mask}",\n'
                            if len(restraint_mask) > 256:
                                raise ValueError(f"Restraint mask is too long (>256 characters): {restraint_mask}")
                        
                        elif 'restraint_wt' in line:
                            line = f' restraint_wt = {extra_restraints_fc},\n'
                        f.write(line)
                    f.write("\n")

        logger.info(f'Adding extra restraints for {stage} stage')
        if stage == 'equil':
            # only need to do it once as all poses have the same protein
            ref_u = mda.Universe(
                    f"{self.equil_folder}/{self.all_poses[0]}/full.pdb",
                    f"{self.equil_folder}/{self.all_poses[0]}/full.inpcrd")
            
            selection = ref_u.select_atoms(f'({extra_restraints}) and name CA')
            logger.debug(f"Selection: {selection} to be restrained")

            atm_resids = selection.residues.resids

            for pose in self.all_poses:
                renum_txt = f'{self.equil_folder}/{pose}/build_files/protein_renum.txt'

                renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_chain', 'old_resid',
                            'new_resname', 'new_resid'])

                # map atm_resids to new_resid
                amber_resids = renum_data.loc[renum_data['old_resid'].isin(atm_resids), 'new_resid']

                formatted_resids = format_ranges(amber_resids)
                logger.debug(f"Restraint atoms in amber format: {formatted_resids}")
                new_mask_component = f'(:{formatted_resids}) & @CA'

                files = ['eqnpt.in']
                for i in range(num_eq_sim):
                    files.append(f'mdin-{i:02d}')

                write_restraint_block(
                                      files=files,
                                      folder_2_write=f'{self.equil_folder}/{pose}/')

        elif stage == 'fe':
            # only need to do it once as all poses have the same protein
            ref_u = mda.Universe(
                    f"{self.equil_folder}/{self.all_poses[0]}/full.pdb",
                    f"{self.equil_folder}/{self.all_poses[0]}/full.inpcrd")
            
            selection = ref_u.select_atoms(f'({extra_restraints}) and name CA')
            logger.debug(f"Selection: {selection} to be restrained")

            atm_resids = selection.residues.resids
            for pose in self.bound_poses:
                renum_txt = f'{self.equil_folder}/{pose}/build_files/protein_renum.txt'

                renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_chain', 'old_resid',
                            'new_resname', 'new_resid'])

                # map atm_resids to new_resid
                amber_resids = renum_data.loc[renum_data['old_resid'].isin(atm_resids), 'new_resid']

                formatted_resids = format_ranges(amber_resids)
                logger.debug(f"Restraint atoms in amber format: {formatted_resids}")
                new_mask_component = f'(:{formatted_resids}) & @CA'

                for comp in self.sim_config.components:
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    windows = self.component_windows_dict[comp]
                    files = ['eqnpt.in']
                    for i in range(num_fe_sim+1):
                        files.append(f'mdin-{i:02d}')
                    for j in range(-1, len(windows)):
                        write_restraint_block(
                            files=files,
                            folder_2_write=f'{folder_comp}/{comp}{j:02d}/')
        else:
            raise ValueError(f"Invalid stage: {stage}")
        logger.debug('Extra position restraints added')

    def analyze_pose(self,
                    pose: str = None,
                    sim_range: Tuple[int, int] = None,
                    raise_on_error: bool = True,
                    n_workers: int = 4):
        """
        Analyze the free energy results for one pose
        Parameters
        ----------
        pose : str
            The pose to analyze.
        sim_range : tuple
            The range of simulations to analyze.
            If files are missing from the range, the analysis will fail.
        raise_on_error : bool
            Whether to raise an error if the analysis fails.
        n_workers : int
            The number of workers to use for parallel processing.
            Default is 4.
        """
        from batter.analysis.analysis import BoreschAnalysis, MBARAnalysis, RESTMBARAnalysis

        pose_path = f'{self.fe_folder}/{pose}'
        os.makedirs(f'{pose_path}/Results', exist_ok=True)

        try:
            results_entries = []

            fe_values = []
            fe_stds = []

            # first get analytical results from Boresch restraint

            if 'v' in self.sim_config.components:
                disangfile = f'{self.fe_folder}/{pose}/sdr/v-1/disang.rest'
            elif 'o' in self.sim_config.components:
                disangfile = f'{self.fe_folder}/{pose}/sdr/o-1/disang.rest'
            elif 'z' in self.sim_config.components:
                disangfile = f'{self.fe_folder}/{pose}/sdr/z-1/disang.rest'

            rest = self.sim_config.rest
            k_r = rest[2]
            k_a = rest[3]
            bor_ana = BoreschAnalysis(
                                disangfile=disangfile,
                                k_r=k_r, k_a=k_a,
                                temperature=self.sim_config.temperature,)
            bor_ana.run_analysis()
            fe_values.append(COMPONENT_DIRECTION_DICT['Boresch'] * bor_ana.results['fe'])
            fe_stds.append(bor_ana.results['fe_error'])
            results_entries.append(
                f'Boresch\t{COMPONENT_DIRECTION_DICT["Boresch"] * bor_ana.results["fe"]:.2f}\t{bor_ana.results["fe_error"]:.2f}'
            )
            
            for comp in self.sim_config.components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]
                comp_path = f'{pose_path}/{comp_folder}'
                windows = self.component_windows_dict[comp]

                # skip n if no conformational restraints are applied
                if comp == 'n' and rest[1] == 0 and rest[4] == 0:
                    logger.debug(f'Skipping {comp} component as no conformational restraints are applied')
                    continue

                if comp in COMPONENTS_DICT['dd']:
                    # directly read energy files
                    mbar_ana = MBARAnalysis(
                        pose_folder=pose_path,
                        component=comp,
                        windows=windows,
                        temperature=self.sim_config.temperature,
                        sim_range=sim_range,
                        load=False,
                        n_jobs=n_workers
                    )
                    mbar_ana.run_analysis()
                    mbar_ana.plot_convergence(save_path=f'{pose_path}/Results/{comp}_convergence.png',
                                            title=f'Convergence for {comp} {pose}',
                    )

                    fe_values.append(COMPONENT_DIRECTION_DICT[comp] * mbar_ana.results['fe'])
                    fe_stds.append(mbar_ana.results['fe_error'])
                    results_entries.append(
                        f'{comp}\t{COMPONENT_DIRECTION_DICT[comp] * mbar_ana.results["fe"]:.2f}\t{mbar_ana.results["fe_error"]:.2f}'
                    )
                elif comp in COMPONENTS_DICT['rest']:
                    rest_mbar_ana = RESTMBARAnalysis(
                        pose_folder=pose_path,
                        component=comp,
                        windows=windows,
                        temperature=self.sim_config.temperature,
                        sim_range=sim_range,
                        load=False,
                        n_jobs=n_workers,
                    )
                    rest_mbar_ana.run_analysis()
                    rest_mbar_ana.plot_convergence(save_path=f'{pose_path}/Results/{comp}_convergence.png',
                                            title=f'Convergence for {comp} {pose}',
                    )

                    fe_values.append(COMPONENT_DIRECTION_DICT[comp] *rest_mbar_ana.results['fe'])
                    fe_stds.append(rest_mbar_ana.results['fe_error'])
                    results_entries.append(
                        f'{comp}\t{COMPONENT_DIRECTION_DICT[comp] * rest_mbar_ana.results["fe"]:.2f}\t{rest_mbar_ana.results["fe_error"]:.2f}'
                    )
        
            # calculate total free energy
            fe_value = np.sum(fe_values)
            fe_std = np.sqrt(np.sum(np.array(fe_stds)**2))
        except Exception as e:
            logger.error(f'Error during FE analysis for {pose}: {e}')
            if raise_on_error:
                raise e
            fe_value = np.nan
            fe_std = np.nan

        results_entries.append(
            f'Total\t{fe_value:.2f}\t{fe_std:.2f}'
        )
        with open(f'{self.fe_folder}/{pose}/Results/Results.dat', 'w') as f:
            f.write('\n'.join(results_entries))
        
        return
                
    @safe_directory
    @save_state
    def analysis(
        self,
        input_file: Union[str, Path, SimulationConfig]=None,
        load: bool = True,
        check_finished: bool = False,
        sim_range: Tuple[int, int] = None,
        overwrite: bool = False,
        raise_on_error: bool = False,
        run_with_slurm: bool = False,
        run_with_slurm_kwargs: dict = None,
        ):
        """
        Analyze the free energy results for all poses.
        Parameters
        ----------
        input_file : str or Path or SimulationConfig, optional
            The input file or SimulationConfig object to load the simulation configuration.
        load : bool, optional
            Whether to load the existing results from the output directory. Default is True.
        check_finished : bool, optional
            Whether to check if the simulation is finished before analyzing. Default is False.
        sim_range : tuple, optional
            The range of simulations to analyze. Default is None, which means all simulations.
            If less simulations were found than specified, it will analyze all found simulations.
        overwrite : bool, optional
            Whether to overwrite the existing results. Default is False.
        raise_on_error : bool, optional
            Whether to raise an error if the analysis fails. Default is False.
        run_with_slurm : bool, optional
            Whether to dispatch the analysis jobs using SLURM. Default is False.
        run_with_slurm_kwargs : dict, optional
            Additional keyword arguments for SLURM job submission. Default is None.
            Check https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
            for available options.
        """
        if input_file is not None:
            self._get_sim_config(input_file)
        
        if check_finished:
            if self._check_fe():
                logger.info('FE simulation is not finished yet')
                self.check_jobs()
                return
        
        if not os.path.exists(f'{self.output_dir}/Results'):
            os.makedirs(f'{self.output_dir}/Results', exist_ok=True)
            self._generate_aligned_pdbs()

        # gather existing results
        if load:
            self.load_results()

            # get unfinished or failed poses
            unfinished_poses = []
            for pose in self.all_poses:
                if self.fe_results[pose] is None:
                    unfinished_poses.append(pose)
                elif np.isnan(self.fe_results[pose].fe):
                    unfinished_poses.append(pose)
        else:
            unfinished_poses = self.all_poses
        
        if run_with_slurm:
            logger.info('Running analysis with SLURM Cluster')
            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client, as_completed

            log_dir = os.path.expanduser('~/.batter_jobs')
            os.makedirs(log_dir, exist_ok=True)
            slurm_kwargs = {
                # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
                'queue': self.partition,
                'cores': 6,
                'memory': '30GB',
                'walltime': '00:30:00',
                'processes': 1,
                'job_extra_directives': [
                    f'--job-name=batter-analysis',
                    f'--output={log_dir}/dask-%j.out',
                    f'--error={log_dir}/dask-%j.err',
                ],
                'worker_extra_args': [
                    "--resources",
                    "analysis=1",
                    "--no-dashboard",
                    f"--local-directory={log_dir}/scratch"
                ],
                # 'account': 'your_slurm_account',
            }
            if run_with_slurm_kwargs is not None:
                slurm_kwargs.update(run_with_slurm_kwargs)
            cluster = SLURMCluster(
                **slurm_kwargs,
            )
            cluster.scale(jobs=len(self.all_poses))
            #cluster.scale(jobs=10)
            logger.info(f'SLURM Cluster created with {len(self.all_poses)} workers')

            client = Client(cluster)
            logger.info(f'Dask Client connected to SLURM Cluster: {client}')
            logger.info(f'Link: {client.dashboard_link}')
            futures = []
            input_dict = {
                'fe_folder': self.fe_folder,
                'components': self.sim_config.components,
                'rest': self.sim_config.rest,
                'temperature': self.sim_config.temperature,
                'component_windows_dict': self.component_windows_dict,
                'sim_range': sim_range,
                'raise_on_error': raise_on_error,
                'n_workers': 6,
            }
            for pose in unfinished_poses:
                logger.debug(f'Submitting analysis for pose: {pose}')
                fut = client.submit(
                    analyze_single_pose_wrapper,
                    pose,
                    input_dict,
                    pure=True,
                    resources={'analysis': 1},
                    key=f'analyze_{pose}'
                )
                futures.append(fut)
            
            logger.info(f'{len(futures)} analysis jobs submitted to SLURM Cluster')
            logger.info('Waiting for analysis jobs to complete...')
            results = client.gather(futures)
            
            logger.info('Analysis with SLURM Cluster completed')
            client.close()
            cluster.close()
            
        else:
            pbar = tqdm(
                unfinished_poses,
                desc='Analyzing FE for poses',
            )
            for pose in unfinished_poses:
                pbar.set_postfix(pose=pose)
                self.analyze_pose(
                    pose=pose,
                    sim_range=sim_range,
                    raise_on_error=raise_on_error,
                    n_workers=self.n_workers
                )
    
        # sort self.fe_sults by pose
        self.load_results()
    
        with open(f'{self.output_dir}/Results/Results.dat', 'w') as f:
            for i, (pose, fe) in enumerate(self.fe_results.items()):
                if fe is None:
                    logger.warning(f'FE for {pose} is None; skipping')
                    continue
                mol_name = self.mols[i]
                ligand_name = self.pose_ligand_dict[pose]
                logger.debug(f'{ligand_name}\t{mol_name}\t{pose}\t{fe.fe:.2f} ± {fe.fe_std:.2f} kcal/mol')
                f.write(f'{ligand_name}\t{mol_name}\t{pose}\t{fe.fe:.2f} ± {fe.fe_std:.2f} kcal/mol\n')
        logger.info(f'self.fe_results: {self.fe_results}')
            
    @safe_directory
    @save_state
    def analysis_new(
        self,
        input_file: Union[str, Path, SimulationConfig]=None,
        load=True,
        check_finished: bool = False,
        sim_range: Tuple[int, int] = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
        ):
        """
        doublegangler for analysis method
        """
        self.analysis(input_file=input_file,
                      load=load,
                      check_finished=check_finished,
                      sim_range=sim_range,
                      overwrite=overwrite,
                      raise_on_error=raise_on_error)

    
    @safe_directory
    @save_state
    def load_results(self):
        """
        Load the results from the output directory.
        """
        loaded_poses = []
        for pose in self.all_poses:
            results_file = f'{self.fe_folder}/{pose}/Results/Results.dat'
            if os.path.exists(results_file):
                self.fe_results[pose] = NewFEResult(results_file)
                if np.isnan(self.fe_results[pose].fe):
                    logger.debug(f'FE for {pose} is invalid (None or NaN); rerun `analysis`.')
                else:
                    loaded_poses.append(pose)
                    logger.debug(f'FE for {pose} loaded from {results_file}')
            else:
                logger.debug(f'FE results file {results_file} not found for pose {pose}')
                self.fe_results[pose] = None
        if not self.fe_results:
            raise ValueError('No results found in the output directory. Please run the analysis first.')
        logger.info(f'Results for {loaded_poses} loaded successfully')
        

    def _generate_aligned_pdbs(self):
        # generate aligned pdbs
        logger.info('Generating aligned pdbs')
        reference_pdb_file = f'{self.poses_folder}/{self.system_name}_docked.pdb'
        u_ref = mda.Universe(reference_pdb_file)
        os.makedirs(f'{self.output_dir}/Results', exist_ok=True)
        os.system(f'cp {reference_pdb_file} {self.output_dir}/Results/reference.pdb')

        for pose in tqdm(self.bound_poses, desc='Generating aligned pdbs'):
            pdb_file = f'{self.equil_folder}/{pose}/representative.pdb'
            u = mda.Universe(pdb_file)
            align.alignto(u,
                          u_ref,
                          select=f'(({self.protein_align}) and not resname NME ACE and protein)',
                          weights="mass")
            saved_ag = u.select_atoms(f'not resname WAT DUM Na+ Cl- and not resname {" ".join(self.lipid_mol)}')
            saved_ag.write(f'{self.output_dir}/Results/protein_{pose}.pdb')

            initial_pose = f'{self.poses_folder}/{pose}.pdb'
            os.system(f'cp {initial_pose} {self.output_dir}/Results/init_{pose}.pdb')
           
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
                logger.warning(f"Provided ligand anchor atom {ligand_anchor_atom} not found in the ligand."
                               "Using all ligand atoms instead.")
                lig_atom = u_lig.atoms
        else:
            lig_atom = u_lig.atoms

        # get ll_x,y,z distances
        r_vect = lig_atom.center_of_mass() - P1_atom.positions
        logger.debug(f'l1_x: {r_vect[0][0]:.2f}; l1_y: {r_vect[0][1]:.2f}; l1_z: {r_vect[0][2]:.2f}')

        p1_formatted = f':{P1_atom.resids[0]}@{P1_atom.names[0]}'
        p2_formatted = f':{P2_atom.resids[0]}@{P2_atom.names[0]}'
        p3_formatted = f':{P3_atom.resids[0]}@{P3_atom.names[0]}'
        logger.debug(f'Receptor anchor atoms: P1: {p1_formatted}, P2: {p2_formatted}, P3: {p3_formatted}')
        return (r_vect[0][0], r_vect[0][1], r_vect[0][2],
                p1_formatted, p2_formatted, p3_formatted)
              
    def _check_equilbration_binding(self):
        """
        Check if the ligand is bound after equilibration
        """
        from batter.analysis.sim_validation import SimValidator

        logger.info("Checking if ligands are bound after equilibration")
        UNBOUND_THRESHOLD = 8.0
        bound_poses = []
        num_eq_sims = len(self.sim_config.release_eq)
        for pose_i, pose in enumerate(self.all_poses):
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                raise FileNotFoundError(f"Equilibration not finished yet")
            if os.path.exists(f"{self.equil_folder}/{pose}/FAILED"):
                raise FileNotFoundError(f"Equilibration failed")
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                logger.warning(f"Pose {pose} is UNBOUND in equilibration")
                continue
            if os.path.exists(f"{self.equil_folder}/{pose}/representative.rst7",) and not self.overwrite:
                bound_poses.append([pose_i, pose])
                logger.debug(f"Representative snapshot found for pose {pose}")
                continue
            with self._change_dir(f"{self.equil_folder}/{pose}"):
                pdb = "full.pdb"
                # exclude the first equilibration simulation
                trajs = [f"md-{i:02d}.nc" for i in range(1, num_eq_sims)]
                universe = mda.Universe(pdb, trajs)
                sim_val = SimValidator(universe)
                sim_val.plot_analysis()
                if sim_val.results['ligand_bs'][-1] > UNBOUND_THRESHOLD:
                    logger.warning(f"Ligand is not bound for pose {pose}")
                    # write "UNBOUND" file
                    with open(f"{self.equil_folder}/{pose}/UNBOUND", 'w') as f:
                        f.write(f"UNBOUND with ligand_bs = {sim_val.results['ligand_bs'][-1]}")
                else:
                    bound_poses.append([pose_i, pose])
                    rep_snapshot = sim_val.find_representative_snapshot()
                    logger.debug(f"Representative snapshot: {rep_snapshot}")
                    
                    cpptraj_command = "cpptraj -p full.prmtop <<EOF\n"
                    for i in range(1, num_eq_sims):
                        cpptraj_command += f"trajin md-{i:02d}.nc\n"
                    cpptraj_command += f"trajout representative.pdb pdb onlyframes {rep_snapshot+1}\n"
                    cpptraj_command += f"trajout representative.rst7 restart onlyframes {rep_snapshot+1}\n"
                    cpptraj_command += "EOF\n"
                    run_with_log(cpptraj_command,
                                working_dir=f"{self.equil_folder}/{pose}")
                    
                    # convert representative.pdb resid info to old residues.
                    # instead of start from 1.
                    renum_txt = f"{self.equil_folder}/{pose}/build_files/protein_renum.txt"
                    renum_data = pd.read_csv(
                        renum_txt,
                        sep=r'\s+',
                        header=None,
                        names=['old_resname', 'old_chain', 'old_resid',
                                'new_resname', 'new_resid'])
                    u = mda.Universe(f"{self.equil_folder}/{pose}/representative.pdb")
                    u.select_atoms('protein').residues.resids = renum_data['old_resid'].values
                    u.atoms.write(f"{self.equil_folder}/{pose}/representative.pdb")
                    
        self._bound_poses = [pose for _, pose in bound_poses]
        self._bound_mols = [self.mols[pose_i] for pose_i, _ in bound_poses]
        logger.debug(f"Bound poses: {bound_poses} will be used for the production stage")

    def _find_new_anchor_atoms(self):
        """
        Find the new anchor atoms for the ligand and the protein after equilibration.
        """
        # get new l1x, l1y, l1z distances
        for i, pose in enumerate(self.bound_poses):
            u_sys = mda.Universe(f'{self.equil_folder}/{pose}/representative.pdb')
            u_lig = u_sys.select_atoms(f'resname {self.bound_mols[i]}')

            ligand_anchor_atom = self.ligand_anchor_atom

            logger.debug(f'Finding anchor atoms for pose {pose}')
            l1_x, l1_y, l1_z, p1, p2, p3 = self._find_anchor_atoms(
                        u_sys,
                        u_lig,
                        self.anchor_atoms,
                        ligand_anchor_atom)
            with open(f'{self.equil_folder}/{pose}/anchor_list.txt', 'w') as f:
                f.write(f'{l1_x} {l1_y} {l1_z}')
        

    @safe_directory
    @save_state
    def run_pipeline(self,
                     input_file: Union[str, Path, SimulationConfig] = None,
                     overwrite: bool = False,              
                     avg_struc: str = None,
                     rmsf_file: str = None,
                     only_equil: bool = False,
                     only_fe_preparation: bool = False,
                     dry_run: bool = False,
                     extra_restraints: str = None,
                     extra_restraints_fc: float = 10,
                     partition: str = 'owners',
                     max_num_jobs: int = 2000,
                     time_limit: str = '6:00:00',
                     verbose: bool = False
                     ):
        """
        Run the whole pipeline for calculating the binding free energy
        after you `create_system`.

        Parameters
        ----------
        input_file : str or SimulationConfig
            The input file for the simulation
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        avg_struc : str, optional
            The path of the average structure of the
            representative conformations. Default is None,
            which means no RMSF restraints are added.
        rmsf_file : str, optional
            The path of the RMSF file. Default is None,
            which means no RMSF restraints are added.
        only_equil : bool, optional
            Whether to run only the equilibration stage.
            Default is False.
        only_fe_preparation : bool, optional
            Whether to prepare the files for the production stage
            without running the production stage.
            Default is False.
        dry_run : bool, optional
            Whether to run the pipeline until performing any
            simulation submissions. Default is False.
        extra_restraints : str, optional
            The selection string for the extra position restraints.
            Default is None, which means no extra restraints are added.
        extra_restraints_fc : float, optional
            The force constant for the extra position restraints.
        partition : str, optional
            The partition to submit the job.
            Default is 'rondror'.
        max_num_jobs : int, optional
            The maximum number of jobs to submit at a time.
            Default is 2000.
        time_limit : str, optional
            The time limit for the job submission.
            Default is '6:00:00'.
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

        start_time = time.time()
        logger.info(f'Start time: {time.ctime()}')
        if input_file is not None:
            self._get_sim_config(input_file)
        else:
            if not hasattr(self, 'sim_config'):
                raise ValueError('Input file is not provided and sim_config is not set.')
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]

        if self._check_equilibration():
            #1 prepare the system
            logger.info('Preparing the equilibration stage')
            logger.info('If you want to have a fresh start, set overwrite=True')
            self.prepare(
                stage='equil',
                input_file=self.sim_config,
                overwrite=overwrite,
                partition=partition,
                avg_struc=avg_struc,
                rmsf_file=rmsf_file,
                extra_restraints=extra_restraints,
                extra_restraints_fc=extra_restraints_fc
            )
            logger.info(f'Equilibration folder: {self.equil_folder} prepared for equilibration')
            logger.info('Submitting the equilibration')
            #2 submit the equilibration
            if dry_run:
                logger.info('Dry run is set to True. '
                            'Skipping the equilibration submission.')
                return
            self.submit(
                stage='equil',
                partition=partition,
                time_limit=time_limit
            )
            logger.info('Equilibration jobs submitted')

            # Check for equilibration to finish
            pbar = tqdm(total=len(self.all_poses),
                        desc="Equilibration sims finished",
                        unit="job")
            while self._check_equilibration():
                n_finished = len([k for k, v in self._sim_finished.items() if v])
                #logger.info(f'{time.ctime()} - Finished jobs: {n_finished} / {len(self._sim_finished)}')
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – Equilibration sims finished"
                pbar.set_description(desc)

                not_finished = [k for k, v in self._sim_finished.items() if not v]
                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                # sleep for 10 minutes to avoid overwhelming the scheduler
                time.sleep(10 * 60)
            pbar.update(len(self.all_poses) - pbar.n)  # update to total
            pbar.set_description('Equilibration finished')
            pbar.close()
            
            self._generate_aligned_pdbs()

        else:
            logger.info('Equilibration simulations are already finished')
            logger.info(f'If you want to have a fresh start, remove {self.equil_folder} manually')
        
        if only_equil:
            logger.info('only_equil is set to True. '
                        'Skipping the free energy calculation.')
            return

        #4.0, submit the free energy equilibration
        logger.info('Running equilibrations before final FE simulations')
        if self._check_fe_equil():
            #3 prepare the free energy calculation
            logger.info('Preparing the free energy equilibration stage')
            logger.info('If you want to have a fresh start, set overwrite=True')
            self.prepare(
                stage='fe',
                input_file=self.sim_config,
                overwrite=overwrite,
                partition=partition,
                avg_struc=avg_struc,
                rmsf_file=rmsf_file,
                extra_restraints=extra_restraints,
                extra_restraints_fc=extra_restraints_fc
            )

            logger.info(f'Free energy folder: {self.fe_folder} prepared for free energy equilibration')
            logger.info('Submitting the free energy equilibration')
            if dry_run:
                logger.info('Dry run is set to True. '
                            'Skipping the free energy equilibration submission.')
                return
            self.submit(
                stage='fe_equil',
                partition=partition,
                time_limit=time_limit,
                )
            logger.info('Free energy equilibration jobs submitted')

            # Check the free energy eq calculation to finish
            pbar = tqdm(total=len(self.bound_poses) * len(self.sim_config.components),
                        desc="FE Equilibration sims finished",
                        unit="job")
            while self._check_fe_equil():
                # get finishd jobs
                n_finished = len([k for k, v in self._sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE Equilibration sims finished"
                pbar.set_description(desc)

                #logger.info(f'{time.ctime()} - Finished jobs: {n_finished} / {len(self._sim_finished)}')
                not_finished = [k for k, v in self._sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]
                # name f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                # exclude the failed jobs
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(5*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config.components) - pbar.n)  # update to total
            pbar.set_description('FE equilibration finished')
            pbar.close()

        else:
            logger.info('Free energy equilibration simulations are already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        # copy last equilibration snapshot to the free energy folder
        for pose in self.bound_poses:
            for comp in self.sim_config.components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]
                folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                eq_rst7 = f'{folder_comp}/{comp}-1/eqnpt04.rst7'
                
                windows = self.component_windows_dict[comp]
                for j in range(0, len(windows)):
                    if os.path.exists(f'{folder_comp}/{comp}{j:02d}/eqnpt04.rst7'):
                        continue
                    #os.system(f'cp {eq_rst7} {folder_comp}/{comp}{j:02d}/eqnpt04.rst7')
                    # get relative path of eq_rst7
                    eq_rst7_rel = os.path.relpath(eq_rst7, start=f'{folder_comp}/{comp}{j:02d}')
                    os.system(f'ln -s {eq_rst7_rel} {folder_comp}/{comp}{j:02d}/eqnpt04.rst7')


        if only_fe_preparation:
            logger.info('only_fe_preparation is set to True. '
                        'Skipping the free energy calculation.')
            logger.info('Move the prepared and equilibrated system to HPC center for further simulations')
            return

        #4, submit the free energy calculation
        logger.info('Running free energy calculation')

        if self._check_fe():
            logger.info('Submitting the free energy calculation')
            if dry_run:
                logger.info('Dry run is set to True. '
                            'Skipping the free energy submission.')
                return
            self.submit(
                stage='fe',
                partition=partition,
                time_limit=time_limit,
            )
            logger.info('Free energy jobs submitted')
            
            # Check the free energy calculation to finish
            pbar = tqdm(
                desc="FE simsulations finished",
                unit="job"
            )
            while self._check_fe():
                # get finishd jobs
                n_finished = len([k for k, v in self._sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE simulations finished"
                pbar.set_description(desc)
                #logger.info(f'{time.ctime()} Finished jobs: {n_finished} / {len(self._sim_finished)}')
                not_finished = [k for k, v in self._sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(10*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config.components) - pbar.n)  # update to total
            pbar.set_description('FE calculation finished')
            pbar.close()
        else:
            logger.info('Free energy calculation is already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        #5 analyze the results
        logger.info('Analyzing the results')
        self.analysis_new(
            load=True,
            check_finished=False,
        )
        logger.info(f'The results are in the {self.output_dir}')
        logger.info(f'Results')
        logger.info(f'---------------------------------')
        logger.info(f'Mol\tPose\tFree Energy (kcal/mol)')
        logger.info(f'---------------------------------')
        for i, (pose, fe) in enumerate(self.fe_results.items()):
            mol_name = self.mols[i]
            logger.info(f'{mol_name}\t{pose}\t{fe.fe:.2f} ± {fe.fe_std:.2f}')
        logger.info('Pipeline finished')
        end_time = time.time()
        logger.info(f'End time: {time.ctime()}')
        total_time = end_time - start_time
        logger.info(f'Total time: {total_time:.2f} seconds')

    @save_state
    def _check_equilibration(self):
        """
        Check if the equilibration is finished by checking the FINISHED file
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.all_poses:
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                sim_finished[f'eq_{pose}'] = False
            else:
                sim_finished[f'eq_{pose}'] = True
            if os.path.exists(f"{self.equil_folder}/{pose}/FAILED"):
                sim_failed[f'eq_{pose}'] = True

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
    def _check_fe_equil(self):
        """
        Check if the eq stage of free energy calculation is finished.
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.bound_poses:
            for comp in self.sim_config.components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]

                win = -1
                folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{win:02d}'
                if not os.path.exists(f"{folder_2_check}/EQ_FINISHED"):
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = False
                else:
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = True
                if os.path.exists(f"{folder_2_check}/FAILED"):
                    sim_failed[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy EQ calculation failed: {self._sim_failed}')
            raise ValueError(f'Free energy EQ calculation failed: {self._sim_failed}')
            
        if all(self._sim_finished.values()):
            logger.debug('Free energy EQ calculation is finished')
            return False
        else:
            not_finished = [k for k, v in self._sim_finished.items() if not v]
            logger.debug(f'Not finished: {not_finished}')
            return True

    @save_state
    def _check_fe(self):
        """
        Check if the free energy calculation is finished.
        """
        sim_finished = {}
        sim_failed = {}
        for pose in self.bound_poses:
            for comp in self.sim_config.components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]
                windows = self.component_windows_dict[comp]
                for j in range(0, len(windows)):
                    folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}'
                    if not os.path.exists(f"{folder_2_check}/FINISHED"):
                        sim_finished[f'fe_{pose}_{comp_folder}_{comp}{j:02d}'] = False
                    else:
                        sim_finished[f'fe_{pose}_{comp_folder}_{comp}{j:02d}'] = True
                    if os.path.exists(f"{folder_2_check}/FAILED"):
                        sim_failed[f'fe_{pose}_{comp_folder}_{comp}{j:02d}'] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy calculation failed: {self._sim_failed}')
            raise ValueError(f'Free energy calculation failed: {self._sim_failed}')
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
            job.submit(requeue=True)
            logger.debug(f'Job {job.jobid} is resubmitted')


    def check_jobs(self):
        """
        Check the status of the jobs and print the not finished jobs.
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
            self._slurm_jobs = {}
        else:
            for pose in self.all_poses:
                logger.info(f'Not finished in {pose}:')
                not_finished_pose = [k for k in not_finished if pose in k]
                not_finished_pose = [job.split('/')[-1] for job in not_finished_pose]
                logger.info(not_finished_pose)
    

    @safe_directory
    @save_state
    def generate_frontier_files(self,
                                remd=False,
                                version=24,
                                ):
        """
        Generate the frontier files for the system
        to run them in a bundle.
        """
        self._generate_frontier_equilibration()
        self._generate_frontier_fe_equilibration(version=version)
        self._generate_frontier_fe(remd=remd, version=version)

    def _generate_frontier_equilibration(self):
        """
        Generate the frontier files for the equilibration stage
        to run them in a bundle.
        """
        # TODO: implement this function
        # The problem is that there's a time restriction of 2 hours
        # to run jobs in Frontier.
        # Thus the equilibration need to be split into smaller jobs
        pass


    def _generate_frontier_fe_equilibration(self, version=24):
        """
        Generate the frontier files for the free energy calculation equilibration stage.
        """
        poses_def = self.bound_poses
        components = self.sim_config.components

        sim_stages = [
                'mini.in',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
        ]

        def write_2_pose(pose):
            """
            Write a groupfile for each component in the pose
            """
            os.makedirs(f'fe/{pose}/groupfiles', exist_ok=True)
            n_sims = len(components)
            stage_previous_template = f'{pose}/{{}}/{{}}-1/full.inpcrd'

            for stage in sim_stages:
                groupfile_name = f'fe/{pose}/groupfiles/fe_eq_{stage}.groupfile'
                with open(groupfile_name, 'w') as f:
                    for component in components:
                        stage_previous = stage_previous_template.format(
                            COMPONENTS_FOLDER_DICT[component],
                            component
                        )
                        sim_folder_temp = f'{pose}/{COMPONENTS_FOLDER_DICT[component]}/{component}'
                        win_eq_sim_folder_name = f'{sim_folder_temp}-1'
                        prmtop = f'{win_eq_sim_folder_name}/full.hmr.prmtop'
                        mdinput = f'fe/{win_eq_sim_folder_name}/{stage.split("_")[0]}'
                        with open(mdinput, 'r') as infile:
                            input_lines = infile.readlines()
                            new_mdinput = f'{mdinput}_frontier'
                            with open(new_mdinput, 'w') as outfile:
                                for line in input_lines:
                                    if 'cv_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        line = f"cv_file = '{win_eq_sim_folder_name}/{file_name}'\n"
                                    if 'output_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        line = f"output_file = '{win_eq_sim_folder_name}/{file_name}'\n"
                                    if 'disang' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        line = f"DISANG={win_eq_sim_folder_name}/{file_name}\n"
                                    outfile.write(line)
                            f.write(f'#fe_eq {component} {stage}\n')
                            file_name_map = {
                                'mini.in': 'mini',
                                'eqnpt0.in': 'eqnpt_pre',
                                'eqnpt.in_00': 'eqnpt00',
                                'eqnpt.in_01': 'eqnpt01',
                                'eqnpt.in_02': 'eqnpt02',
                                'eqnpt.in_03': 'eqnpt03',
                                'eqnpt.in_04': 'eqnpt04',
                            }
                            f.write(
                                f'-O -i {win_eq_sim_folder_name}/{stage.split("_")[0]}_frontier -p {prmtop} -c {stage_previous} '
                                f'-o {win_eq_sim_folder_name}/{file_name_map[stage]}.out -r {win_eq_sim_folder_name}/{file_name_map[stage]}.rst7 -x {win_eq_sim_folder_name}/{file_name_map[stage]}.nc '
                                f'-ref {stage_previous} -inf {win_eq_sim_folder_name}/{file_name_map[stage]}.mdinfo -l {win_eq_sim_folder_name}/{file_name_map[stage]}.log '
                                f'-e {win_eq_sim_folder_name}/{file_name_map[stage]}.mden\n'
                            )
                    stage_previous_template = f'{pose}/{{}}/{{}}-1/{file_name_map[stage]}.rst7'

        with self._change_dir(self.output_dir):
            for i, pose in enumerate(poses_def):
                if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                    continue
                write_2_pose(pose)
                logger.debug(f'Generated groupfiles for {pose}')
            # copy env.amber.24
            env_amber_file = f'{frontier_files}/env.amber.{version}'
            #shutil.copy(env_amber_file, 'fe/env.amber')
            os.system(f'cp {env_amber_file} fe/env.amber')
            logger.info('FE EQ groupfiles generated for all poses')


    def _generate_frontier_fe(self,
                              remd=False,
                              version=24,
                              ):
        """
        Generate the frontier files for the free energy calculation production stage.
        """
        poses_def = self.bound_poses
        components = self.sim_config.components

        sim_stages = [
                'mini.in',
                'mdin.in', 'mdin.in.extend'
        ]
        
        def calculate_performance(n_atoms, comp):
            if comp not in COMPONENTS_DICT['dd']:
                return 150 if n_atoms < 80000 else 80
            else:
                return 80 if n_atoms < 80000 else 40
        

        def write_2_pose(pose):
            """
            Write a groupfile for each component in the pose
            """
            all_replicates = {comp: [] for comp in components}

            pose_name = f'fe/{pose}/'
            logger.debug(f'Creating groupfiles for {pose}')
            
            for component in components:
                lambdas = self.component_windows_dict[component]
                folder_name = COMPONENTS_FOLDER_DICT[component]
                sim_folder_temp = f'{pose}/{folder_name}/{component}'
                n_sims = len(lambdas)

                stage_previous = f'{sim_folder_temp}-1/eqnpt04.rst7'

                for stage in sim_stages:
                    groupfile_name = f'{pose_name}/groupfiles/{component}_{stage}.groupfile'
                    with open(groupfile_name, 'w') as f:
                        for i in range(n_sims):
                            win_eq_sim_folder_name = f'{sim_folder_temp}-1'
                            sim_folder_name = f'{sim_folder_temp}{i:02d}'
                            prmtop = f'{win_eq_sim_folder_name}/full.hmr.prmtop'
                            mdinput_path = f'{sim_folder_name}/{stage.split("_")[0]}'
                            
                            # Read and modify the MD input file to update the relative path
                            if stage == 'mdin.in':
                                mdinput_path = mdinput_path.replace(stage, 'mdin-00')
                            elif stage == 'mdin.in.extend':
                                mdinput_path = mdinput_path.replace(stage, 'mdin-01')
                            with open(f'fe/{mdinput_path}', 'r') as infile:
                                input_lines = infile.readlines()

                            new_mdinput_path = f'fe/{sim_folder_name}/{stage.split("_")[0]}_frontier'

                            with open(new_mdinput_path, 'w') as outfile:
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
                                        if component in COMPONENTS_DICT['dd'] and remd and stage != 'mini.in':
                                            outfile.write(
                                                'numexchg = 3000,\n'
                                            )
                                            outfile.write(
                                                'bar_intervall = 100,\n'
                                            )
                                    elif 'cv_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"cv_file = '{sim_folder_name}/{file_name}'\n"
                                    elif 'output_file' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"output_file = '{sim_folder_name}/{file_name}'\n"
                                    elif 'disang' in line:
                                        file_name = line.split('=')[1].strip().replace("'", "")
                                        line = f"DISANG={sim_folder_name}/{file_name}\n"
                                    # update the number of steps
                                    # if 'nstlim = 50000' in line:
                                    #    line = '  nstlim = 5,\n'
                                    # do not only write the ntwprt atoms
                                    elif 'irest' in line:
                                        #if remd and component in ['x', 'e', 'v', 'w', 'f']:
                                        if stage == 'mdin.in':
                                            line = '  irest = 0,\n'
                                    elif 'ntx =' in line:
                                        #if remd:
                                        if stage == 'mdin.in':
                                            line = '  ntx = 1,\n'
                                    elif 'nmropt' in line:
                                        if stage == 'mdin.in':
                                            line = '  nmropt = 0,\n'
                                    elif 'ntxo' in line:
                                        line = '  ntxo = 2,\n'
                                    elif 'ntwprt' in line:
                                        line = '\n'
                                    elif 'maxcyc' in line:
                                        line = '  maxcyc = 5000,\n'
                                    if stage == 'mdin.in' or stage == 'mdin.in.extend':
                                        if 'nstlim' in line:
                                            if remd and component in COMPONENTS_DICT['dd'] and stage != 'mini.in':
                                                line = '  nstlim = 100,\n'
                                            else:
                                               # hard estimation of the size to be 100000 atoms 
                                                n_atoms = 100000
                                                performance = calculate_performance(n_atoms, component)
                                                n_steps = int(50 / 60 / 24 * performance * 1000 * 1000 / 4)
                                                n_steps = int(n_steps // 100000 * 100000)
                                                line = f'  nstlim = {n_steps},\n'
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
            #logger.debug(f'all_replicates: {all_replicates}')
            return all_replicates

        all_replicates = []

        with self._change_dir(self.output_dir):
            filtered_poses = [pose for pose in poses_def if not os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND")]

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(tqdm(executor.map(write_2_pose, filtered_poses),
                                    total=len(filtered_poses),
                                    desc='Generating production groupfiles'))
                all_replicates.extend(results)

        #with self._change_dir(self.output_dir):
        #    for i, pose in tqdm(enumerate(poses_def),
        #                        desc='Generating production groupfiles',
        #                        total=len(poses_def)):
        #        if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
        #            continue
        #        all_replicates.append(write_2_pose(pose, components))
        #        logger.debug(f'Generated groupfiles for {pose}')
            logger.debug(all_replicates)
            # copy env.amber.24
            env_amber_file = f'{frontier_files}/env.amber.{version}'
            #shutil.copy(env_amber_file, 'fe/env.amber')
            os.system(f'cp {env_amber_file} fe/env.amber')
            logger.info('FE production groupfiles generated for all poses')
    
    def check_sim_stage(self):
        """
        Check the status of running of all the simulations
        """
        stage_sims = {}
        for pose in self.bound_poses:
            stage_sims[pose] = {}
            for comp in self.sim_config.components:
                if comp in ['m', 'n']:
                    sim_type = 'rest'
                elif comp in COMPONENTS_DICT['dd']:
                    sim_type = 'sdr'
                min_stage = float('inf')
                for win in range(0, len(self.component_windows_dict[comp])):
                    folder = f'{self.fe_folder}/{pose}/{sim_type}/{comp}{win:02d}'
                    mdin_files = glob.glob(f'{folder}/md*.rst7')
                    # make sure the size is not empty
                    mdin_files = [f for f in mdin_files if os.path.getsize(f) > 100]
                    # only base name
                    # sort_key = lambda x: int(x.split('-')[-1].split('.')[0])
                    def sort_key(x):
                        return int(os.path.splitext(os.path.basename(x))[0][-2:])
                    mdin_files.sort(key=sort_key)
                    min_stage = min(min_stage, sort_key(mdin_files[-1]) if mdin_files else -1)
                stage_sims[pose][comp] = min_stage
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        stage_sims_df = pd.DataFrame(stage_sims)
        fig, ax = plt.subplots(figsize=(1* len(self.bound_poses), 5))
        sns.heatmap(stage_sims_df, ax=ax, annot=True, cmap='viridis')
        plt.title('Simulation Stages for each Pose and Component')
        plt.show()


    def copy_2_new_folder(self,
                          folder_name,
                          only_equil=True,
                          symlink=True,):
        """
        Copy the system to a new folder
        """
        if symlink:
            cp_cmd = 'ln -s'
        else:
            cp_cmd = 'cp -r'
        if os.path.exists(folder_name):
            raise ValueError(f"Folder {folder_name} already exists")
        os.makedirs(folder_name, exist_ok=True)

        with self._change_dir(folder_name):
            os.system(f'cp {self.output_dir}/system.pkl .')
        
            all_pose_folder = os.path.relpath(self.poses_folder, os.getcwd())
            os.system(f'{cp_cmd} {all_pose_folder} .')

            ligandff_folder = os.path.relpath(self.ligandff_folder, os.getcwd())
            os.system(f'{cp_cmd} {ligandff_folder} .')

            equil_folder = os.path.relpath(self.equil_folder, os.getcwd())
            os.system(f'{cp_cmd} {equil_folder} .')

            if only_equil:
                logger.info(f'Copied equilibration files to {folder_name}')
                return
            
            # for fe, only copy necessary files
            os.makedirs('fe', exist_ok=True)
            # for fe, only copy necessary files
            folder_names = ['ff', 'groupfiles']
            for pose in self.all_poses:
                os.makedirs(f'fe/{pose}', exist_ok=True)
                for folder_name in folder_names:
                    if os.path.exists(f'{self.fe_folder}/{pose}/{folder_name}'):
                        os.system(f'cp -r {self.fe_folder}/{pose}/{folder_name} fe/{pose}/')

            sim_files = ['full.pdb', 'full.hmr.prmtop', 'full.inpcrd', 'vac.pdb',
                    'eqnpt04.rst7',
                    'mini.in', 'mdin-00', 'mdin-01', 'SLURMM-run', 'run-local.bash',
                    'cv.in', 'disang.rest', 'restraints.in']

            for pose in tqdm(self.all_poses, desc='Copying files'):
                for comp in self.sim_config.components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]

                    win_folder = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}'
                    os.makedirs(f'fe/{pose}/{comp_folder}', exist_ok=True)
                    
                    os.system(f'cp -r {self.fe_folder}/{pose}/{comp_folder}/{comp}_build_files fe/{pose}/{comp_folder}/')
                    os.system(f'cp -r {self.fe_folder}/{pose}/{comp_folder}/{comp}_amber_files fe/{pose}/{comp_folder}/')
                    os.system(f'cp -r {self.fe_folder}/{pose}/{comp_folder}/{comp}_run_files fe/{pose}/{comp_folder}/')
                    windows = self.component_windows_dict[comp]
                    folder_name = f'{win_folder}-1'
                    new_folder = f'fe/{pose}/{comp_folder}/{comp}-1'
                    if os.path.exists(folder_name):
                        os.makedirs(new_folder, exist_ok=True)
                        files = os.listdir(folder_name)
                        for file in files:
                            os.system(f'cp {folder_name}/{file} {new_folder}')
                    else:
                        logger.warning(f'Folder {folder_name} does not exist.')


                    for i, window in enumerate(windows):
                        folder_name = f'{win_folder}{i:02d}'
                        new_folder = f'fe/{pose}/{comp_folder}/{comp}{i:02d}'
                        os.makedirs(new_folder, exist_ok=True)
                        # list all files in the folder
                        files = os.listdir(folder_name)
                        for file in files:
                            if file in sim_files:
                                os.system(f'cp {folder_name}/{file} {new_folder}')
                            # ignore rst7 files
                            elif file.endswith('.rst7'):
                                continue
                            elif file.endswith('.nc'):
                                continue
                            elif file.endswith('.log'):
                                continue
                            elif file.endswith('.out'):
                                continue
                            elif file == 'amber_files':
                                continue
                            else:
                                #os.system(f'cp {folder_name}/{file} {new_folder}')
                                continue
            logger.info(f'Copied system to {folder_name}')

    def load_window_json(self, window_json):
        """
        Load the window json file and regenerate all the windows
        
        window_json : str
            The path to the window json file.

            The json file should includes lambda / restraint
            values for components.
            e.g. it can be in a format of `e` component:
            {
                'e': [0, 0.5, 1.0],
            }
        """
        win_json = json.load(open(window_json, 'r'))
        for comp, windows in win_json.items():
            self.component_windows_dict[comp] = windows
        self._prepare_fe_windows(
            regenerate=True,
        )

    @property
    def component_windows_dict(self):
        """
        Get the component windows dictionary
        """
        if not hasattr(self, '_component_windows_dict'):
            self._component_windows_dict = ComponentWindowsDict(self)

        return self._component_windows_dict


    @property
    def n_workers(self):
        """
        Get the number of workers
        """
        if not hasattr(self, '_n_workers'):
            self._n_workers = 12
        return self._n_workers
    
    @n_workers.setter
    def n_workers(self, n_workers):
        """
        Set the number of workers
        """
        if n_workers <= 0:
            raise ValueError("Number of workers must be positive")
        self._n_workers = n_workers
    
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
    
    @property
    def max_num_jobs(self):
        try:
            return self._max_num_jobs
        except AttributeError:
            return 2000


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
        from rdkit import Chem

        # check if they are the same ligand
        if self.ligand_paths[0].lower().endswith('.sdf'):
            suppl = Chem.SDMolSupplier(self.ligand_paths[0], removeHs=False)
            mol = suppl[0]
        elif self.ligand_paths[0].lower().endswith('.pdb'):
            mol = self.ligand_paths[0]
        n_atoms = mda.Universe(mol).atoms.n_atoms
        for ligand_path in self.ligand_paths:
            if ligand_path.lower().endswith('.sdf'):
                suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
                ligand_path = suppl[0]
            if mda.Universe(ligand_path).atoms.n_atoms != n_atoms:
                raise ValueError(f"Number of atoms in the ligands are different: {ligand_path}")

        # set the ligand path to the first ligand
        self._unique_ligand_paths = {self.ligand_paths[0]: self.ligand_names}


class MABFESystem(System):
    """
    A class to represent and process a Absolute Binding Free Energy Perturbation (FEP) system
    using the BAT.py methods. It gets inputs of a protein and multiple single ligand types.
    The ABFE of the ligands to the provided **protein conformation** will be calculated
    """
    def _process_ligands(self):
        self._unique_ligand_paths = {}
        for ligand_path, ligand_name in zip(self.ligand_paths, self.ligand_names):
            if ligand_path not in self._unique_ligand_paths:
                self._unique_ligand_paths[ligand_path] = []
            self._unique_ligand_paths[ligand_path].append(ligand_name)
        logger.debug(f' Unique ligand paths: {self._unique_ligand_paths}')


class RBFESystem(System):
    """
    A class to represent and process a Relative Binding Free Energy Perturbation (FEP) system
    using the separated topology methodology in BAT.py.
    """
    def _process_ligands(self):
        self._unique_ligand_paths = {}
        for ligand_path, ligand_name in zip(self.ligand_paths, self.ligand_names):
            if ligand_path not in self._unique_ligand_paths:
                self._unique_ligand_paths[ligand_path] = set()
            self._unique_ligand_paths[ligand_path].add(ligand_name)

        if len(self._unique_ligand_paths.keys()) < 2:
            raise ValueError("RBFESystem requires at least two ligands "
                             "for the relative binding free energy calculation")
        logger.info(f'Reference ligand: {self._unique_ligand_paths.keys()[0]}')


class ComponentWindowsDict(MutableMapping):
    def __init__(self, system):
        self._data = {}
        self.sim_config = system.sim_config

    def __getitem__(self, key):
        if key in COMPONENTS_DICT['dd']:
            return self._data.get(key, self.sim_config.lambdas)
        elif key in COMPONENTS_DICT['rest']:
            return self._data.get(key, self.sim_config.attach_rest)
        else:
            raise ValueError(f"Component {key} not in the system")
    
    def __setitem__(self, key, value):
        if key in AVAILABLE_COMPONENTS:
            self._data[key] = value
        else:
            raise ValueError(f"Component {key} not in the system")
    
    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
    

def format_ranges(numbers):
    """
    Convert a list of numbers into a string of ranges.
    For example, [1, 2, 3, 5, 6] will be converted to "1-3,5-6".

    This is to avoid the nasty AMBER issue that restraintmask can
    only be 256 characters long. -.-
    """
    from itertools import groupby
    numbers = sorted(set(numbers))
    ranges = []

    for _, group in groupby(enumerate(numbers), key=lambda x: x[1] - x[0]):
        group = list(group)
        start = group[0][1]
        end = group[-1][1]
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
    
    return ','.join(ranges)


def analyze_single_pose_wrapper(pose, input_dict):
    from distributed import get_worker
    logger.info(f"Running on worker: {get_worker().name}")
    logger.info(f'Analyzing pose: {pose}')
    logger.info(f'Input: {input_dict}')
    fe_folder = input_dict.get('fe_folder')
    components = input_dict.get('components')
    rest = input_dict.get('rest')
    temperature = input_dict.get('temperature')
    component_windows_dict = input_dict.get('component_windows_dict')
    sim_range = input_dict.get('sim_range', None)
    raise_on_error = input_dict.get('raise_on_error', True)
    n_workers = input_dict.get('n_workers', 4)
    
    analyze_pose_task(
        fe_folder=fe_folder,
        pose=pose,
        components=components,
        rest=rest,
        temperature=temperature,
        component_windows_dict=component_windows_dict,
        sim_range=sim_range,
        raise_on_error=raise_on_error,
        n_workers=n_workers
    )


def analyze_pose_task(
                fe_folder: str,
                pose: str,
                components: List[str],
                rest: Tuple[float, float, float, float, float],
                temperature: float,
                component_windows_dict: "ComponentWindowsDict",
                sim_range: Tuple[int, int] = None,
                raise_on_error: bool = True,
                n_workers: int = 4):
    """
    Analyze the free energy results for one pose as an independent task.

    Parameters
    ----------
    fe_folder : str
        The folder containing the free energy results for the pose.
    pose : str
        The pose to analyze.
    components : List[str]
        The components to analyze.
        This should be a list of strings, e.g. ['e', 'v', 'z', 'n', 'o'].
    rest : tuple
        The restraint values for the pose.
    temperature : float
        The temperature of the simulation in Kelvin.
    sim_range : tuple
        The range of simulations to analyze.
        If files are missing from the range, the analysis will fail.
    raise_on_error : bool
        Whether to raise an error if the analysis fails.
    n_workers : int
        The number of workers to use for parallel processing.
        Default is 4.
    """
    from batter.analysis.analysis import BoreschAnalysis, MBARAnalysis, RESTMBARAnalysis

    pose_path = f'{fe_folder}/{pose}'
    os.makedirs(f'{pose_path}/Results', exist_ok=True)

    try:
        results_entries = []

        fe_values = []
        fe_stds = []

        # first get analytical results from Boresch restraint

        if 'v' in components:
            disangfile = f'{fe_folder}/{pose}/sdr/v-1/disang.rest'
        elif 'o' in components:
            disangfile = f'{fe_folder}/{pose}/sdr/o-1/disang.rest'
        elif 'z' in components:
            disangfile = f'{fe_folder}/{pose}/sdr/z-1/disang.rest'

        k_r = rest[2]
        k_a = rest[3]
        bor_ana = BoreschAnalysis(
                            disangfile=disangfile,
                            k_r=k_r, k_a=k_a,
                            temperature=temperature)
        bor_ana.run_analysis()
        fe_values.append(COMPONENT_DIRECTION_DICT['Boresch'] * bor_ana.results['fe'])
        fe_stds.append(bor_ana.results['fe_error'])
        results_entries.append(
            f'Boresch\t{COMPONENT_DIRECTION_DICT["Boresch"] * bor_ana.results["fe"]:.2f}\t{bor_ana.results["fe_error"]:.2f}'
        )
        
        for comp in components:
            comp_folder = COMPONENTS_FOLDER_DICT[comp]
            comp_path = f'{pose_path}/{comp_folder}'
            windows = component_windows_dict[comp]

            # skip n if no conformational restraints are applied
            if comp == 'n' and rest[1] == 0 and rest[4] == 0:
                logger.debug(f'Skipping {comp} component as no conformational restraints are applied')
                continue

            if comp in COMPONENTS_DICT['dd']:
                # directly read energy files
                mbar_ana = MBARAnalysis(
                    pose_folder=pose_path,
                    component=comp,
                    windows=windows,
                    temperature=temperature,
                    sim_range=sim_range,
                    load=False,
                    n_jobs=n_workers
                )
                mbar_ana.run_analysis()
                mbar_ana.plot_convergence(save_path=f'{pose_path}/Results/{comp}_convergence.png',
                                        title=f'Convergence for {comp} {pose}',
                )

                fe_values.append(COMPONENT_DIRECTION_DICT[comp] * mbar_ana.results['fe'])
                fe_stds.append(mbar_ana.results['fe_error'])
                results_entries.append(
                    f'{comp}\t{COMPONENT_DIRECTION_DICT[comp] * mbar_ana.results["fe"]:.2f}\t{mbar_ana.results["fe_error"]:.2f}'
                )
            elif comp in COMPONENTS_DICT['rest']:
                rest_mbar_ana = RESTMBARAnalysis(
                    pose_folder=pose_path,
                    component=comp,
                    windows=windows,
                    temperature=temperature,
                    sim_range=sim_range,
                    load=False,
                    n_jobs=n_workers,
                )
                rest_mbar_ana.run_analysis()
                rest_mbar_ana.plot_convergence(save_path=f'{pose_path}/Results/{comp}_convergence.png',
                                        title=f'Convergence for {comp} {pose}',
                )

                fe_values.append(COMPONENT_DIRECTION_DICT[comp] *rest_mbar_ana.results['fe'])
                fe_stds.append(rest_mbar_ana.results['fe_error'])
                results_entries.append(
                    f'{comp}\t{COMPONENT_DIRECTION_DICT[comp] * rest_mbar_ana.results["fe"]:.2f}\t{rest_mbar_ana.results["fe_error"]:.2f}'
                )
    
        # calculate total free energy
        fe_value = np.sum(fe_values)
        fe_std = np.sqrt(np.sum(np.array(fe_stds)**2))
    except Exception as e:
        logger.error(f'Error during FE analysis for {pose}: {e}')
        if raise_on_error:
            raise e
        fe_value = np.nan
        fe_std = np.nan

    results_entries.append(
        f'Total\t{fe_value:.2f}\t{fe_std:.2f}'
    )
    with open(f'{fe_folder}/{pose}/Results/Results.dat', 'w') as f:
        f.write('\n'.join(results_entries))
    
    return
            