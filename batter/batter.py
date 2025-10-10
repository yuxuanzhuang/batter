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
from pathlib import Path
import pickle
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Union
from loguru import logger
from collections import defaultdict
from batter import __version__
from batter.input_process import SimulationConfig, get_configure_from_file
from batter.analysis.results import FEResult
from batter.utils.utils import tqdm_joblib
from batter.utils.slurm_job import SLURMJob, get_squeue_job_count
from batter.data import run_files as run_files_orig
from batter.data import build_files as build_files_orig
from batter.data import batch_files as batch_files_orig
from batter.builder import BuilderFactory, get_ligand_candidates
from batter.utils import (
    run_with_log,
    save_state,
    safe_directory,
    natural_keys,
)

from batter.utils import (
    COMPONENTS_LAMBDA_DICT,
    COMPONENTS_FOLDER_DICT,
    COMPONENTS_DICT,
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
    'y': 1,
    'Boresch': -1,
}

class ProxyAttr:
    """Descriptor that proxies attribute access to obj.<target>.<name>."""
    def __init__(self, target: str, name: str):
        self.target = target
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(getattr(obj, self.target), self.name)

    def __set__(self, obj, value):
        setattr(getattr(obj, self.target), self.name, value)


def proxy_to(target: str, names: list[str]):
    """Class decorator to add ProxyAttr descriptors for each name."""
    def deco(cls):
        for n in names:
            setattr(cls, n, ProxyAttr(target, n))
        return cls
    return deco

@proxy_to('sim_config', SimulationConfig.__annotations__.keys())
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
        self._pose_failed = {}
        self._eq_prepared = False
        self._fe_prepared = False
        self._ligand_objects = {}
        self._fe_results = {}

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
                    receptor_ff: str = 'protein.ff14SB',
                    retain_lig_prot: bool = True,
                    ligand_ph: float = 7.4,
                    ligand_ff: str = 'gaff2',
                    existing_ligand_db: str = None,
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
        receptor_ff: str
            Force field for the protein atoms.
            Default is 'protein.ff14SB'.
        retain_lig_prot : bool, optional
            Whether to retain hydrogens in the ligand. Default is True.
        ligand_ph : float, optional
            pH value for protonating the ligand. Default is 7.4.
        ligand_ff : str, optional
            Parameter set for the ligand. Default is 'gaff2'.
            Options are 'gaff' and 'gaff2' and openff force fields.
            See https://github.com/openforcefield/openff-forcefields for full list.
        existing_ligand_db : str, optional
            Path to an existing ligand database to fetch parameters.
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

        if self._eq_prepared or self._fe_prepared:
            if not overwrite:
                raise ValueError("The system has been prepared for equilibration or free energy simulations. "
                                 "Set overwrite=True to overwrite the existing system or skip `create_system` step.")
                                                 
        self._system_name = system_name
        self._protein_input = self._convert_2_relative_path(protein_input)
        self._system_topology = self._convert_2_relative_path(system_topology)
        if system_coordinate is not None:
            self._system_coordinate = self._convert_2_relative_path(system_coordinate)
        else:
            self._system_coordinate = None
        
        # always store a unique identifier for the ligand
        if isinstance(ligand_paths, list):
            self.ligand_dict = {
                f'lig{i}': self._convert_2_relative_path(path)
                for i, path in enumerate(ligand_paths)
            }
        elif isinstance(ligand_paths, dict):
            self.ligand_dict = {ligand_name: self._convert_2_relative_path(ligand_path) for ligand_name, ligand_path in ligand_paths.items()}
        else:
            raise ValueError("ligand_paths must be a list or a dictionary")
        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.receptor_ff = receptor_ff
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.ligand_ff = ligand_ff
        self.overwrite = overwrite

        self.lipid_mol = lipid_mol
        if not self.lipid_mol:
            self._membrane_simulation = False
        else:
            self._membrane_simulation = True

        # check input existence
        if not os.path.exists(self.protein_input):
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        if not os.path.exists(self.system_topology):
            raise FileNotFoundError(f"System input file not found: {system_topology}")
        for ligand_path in self.ligand_paths:
            if not os.path.exists(ligand_path):
                raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
                
        logger.info(f"# {len(self.ligand_paths)} ligands.")
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
        else:
            try:
                self.system_dimensions = u_sys.dimensions[:3]
            except TypeError:
                if self.membrane_simulation:
                    raise ValueError("No box dimensions found in the system_topology. "
                                     "For membrane systems, the box dimensions must be provided.")
                else:
                    # set a suitable box with padding of 10 A
                    protein = u_sys.select_atoms('protein')
                    padding = 10.0
                    box_x = protein.positions[:,0].max() - protein.positions[:,0].min() + 2 * padding
                    box_y = protein.positions[:,1].max() - protein.positions[:,1].min() + 2 * padding
                    box_z = protein.positions[:,2].max() - protein.positions[:,2].min() + 2 * padding
                    self.system_dimensions = np.array([box_x, box_y, box_z])

                    logger.warning("No box dimensions found in the system_topology. "
                                   "Setting a default box with 10 A padding around the protein. "
                                   f"Box dimensions: {self.system_dimensions}"
                                   )
        if (u_sys.atoms.dimensions is None or not u_sys.atoms.dimensions.any()) and self.system_coordinate is None:
            raise ValueError(f"No dimension of the box was found in the system_topology or system_coordinate")

        os.makedirs(f"{self.poses_folder}", exist_ok=True)
        u_sys.atoms.write(f"{self.poses_folder}/system_input.pdb")
        self._system_input_pdb = f"{self.poses_folder}/system_input.pdb"
        os.makedirs(f"{self.ligandff_folder}", exist_ok=True)
        
        # copy dummy atom parameters to the ligandff folder
        os.system(f"cp {build_files_orig}/dum.mol2 {self.ligandff_folder}")
        os.system(f"cp {build_files_orig}/dum.frcmod {self.ligandff_folder}")

        from openff.toolkit.typing.engines.smirnoff.forcefield import get_available_force_fields
        available_amber_ff = ['gaff', 'gaff2']
        available_openff_ff = [ff.removesuffix(".offxml") for ff in get_available_force_fields() if 'openff' in ff]
        if ligand_ff not in available_amber_ff + available_openff_ff:
            raise ValueError(f"Unsupported force field: {ligand_ff}. "
                             f"Supported force fields are: {available_amber_ff + available_openff_ff}")

        self.lipid_ff = lipid_ff
        if self.lipid_ff != 'lipid21':
            raise ValueError(f"Invalid lipid_ff: {self.lipid_ff}"
                             "Only 'lipid21' is available")

        # Prepare the membrane parameters
        if self.membrane_simulation:
            self._prepare_membrane()

        self._process_protein()
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
            self._ligand_objects[ligand_name] = ligand

            mols.append(ligand.name)
            self.unique_mol_names.append(ligand.name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{ligand.name}.frcmod"):
                # try to fetch from existing ligand db
                if existing_ligand_db is not None:
                    fetched = ligand.fetch_from_existing_db(existing_ligand_db)
                    if fetched:
                        logger.info(f"Fetched parameters for {ligand.name} from existing ligand database {existing_ligand_db}")
                    else:
                        logger.info(f"No parameters found for {ligand.name} in existing ligand database {existing_ligand_db}. Generating new parameters.")
                        ligand.prepare_ligand_parameters()
                else:
                    logger.info(f"Generating parameters for {ligand.name} using {self.ligand_ff} force field.")
                    ligand.prepare_ligand_parameters()
            for ligand_name in ligand_names:
                self.ligand_dict[ligand_name] = self._convert_2_relative_path(f'{self.ligandff_folder}/{ligand.name}.pdb')

        logger.debug( f"Unique ligand names: {self.unique_mol_names} ")
        logger.debug('updating the ligand paths')
        logger.debug(self.ligand_dict)

        self._mols = mols
        # update self.mols to output_dir/mols.txt
        with open(f"{self.output_dir}/mols.txt", 'w') as f:
            for ind, (ligand_path, ligand_names) in enumerate(self._unique_ligand_paths.items()):
                f.write(f"pose{ind}\t{self._mols[ind]}\t{ligand_path}\t{ligand_names}\n")
        self._prepare_ligand_poses()

        # always get the anchor atoms from the first pose
        u_prot = mda.Universe(f'{self.output_dir}/all-poses/reference.pdb')
        u_lig = mda.Universe(f'{self.output_dir}/all-poses/pose0.pdb')
        lig_sdf = f'{self.ligandff_folder}/{self.mols[0]}.sdf'
        l1_x, l1_y, l1_z, p1, p2, p3, l1_range = self._find_anchor_atoms(
                    u_prot,
                    u_lig,
                    lig_sdf,
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

        self.l1_range = l1_range
        
        logger.info('System loaded and prepared')

    def _convert_2_relative_path(self, path):
        """
        Convert the path to a relative path to the output directory.
        """
        return os.path.relpath(path, self.output_dir)

    @property
    def sim_config(self):
        try:
            return self._sim_config
        except AttributeError:
            self._sim_config = SimulationConfig(system_name=self._system_name,
                            fe_type='uno_rest')
            return self._sim_config
    
    @sim_config.setter
    def sim_config(self, config: SimulationConfig):
        if not isinstance(config, SimulationConfig):
            raise ValueError("sim_config must be an instance of SimulationConfig")
        self._sim_config = config    

    @property
    def extra_restraints(self):
        try:
            return self._extra_restraints
        except AttributeError:
            return None

    @property
    def extra_conformation_restraints(self):
        try:
            return self._extra_conformation_restraints
        except AttributeError:
            return None

    @property
    def rmsf_restraints(self):
        try:
            return self._rmsf_restraints
        except AttributeError:
            return None

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
        return [f"{self.output_dir}/{ligand_path}" for ligand_path in self.ligand_dict.values()]
    
    @property
    def pose_ligand_dict(self):
        """
        A dictionary of ligands with pose names as keys.
        """
        try:
            return self._pose_ligand_dict
        except AttributeError:
            return {pose.split('/')[-1].split('.')[0]: ligand
                    for ligand, pose in self.ligand_dict.items()}

    @property
    def ligand_pose_dict(self):
        """
        A dictionary of poses with ligand names as keys.
        """
        return {v: k for k, v in self.pose_ligand_dict.items()}
       
    @property
    def ligand_names(self):
        """
        The names of the ligands.
        """
        return list(self.ligand_dict.keys())
    
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
        
    @property
    def membrane_simulation(self):
        try:
            return self._membrane_simulation
        except (AttributeError, TypeError):
            if not self.lipid_mol:
                return False
            else:
                return True

        
    def _process_ligands(self):
        """
        Process the ligands to get the ligand paths.
        e.g., for ABFE, it will be a single ligand.
        For RBFE, it will be multiple ligands.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def _process_protein(self):
        """
        Process the protein input file to fix issues so that it can be processed
        by AMBER; it includes 
        - removing capping groups
        - renaming ILE CD to CD1
        # - removing alternate locations?
        - unset segid (if any)
        """
        u = mda.Universe(self.protein_input)
        # remove capping groups
        saved_atoms = u.atoms
        cap_resnames = ['ACE', 'NME', 'NMA', 'CT3', 'CT2', 'CT1', 'NH2', 'NH3']
        cap_atoms = u.select_atoms(f'resname {" ".join(cap_resnames)}')
        if len(cap_atoms) > 0:
            logger.debug(f"Removing capping groups: {cap_resnames}")
            saved_atoms -= cap_atoms
        # rename ILE CD to CD1
        ile_cd = saved_atoms.select_atoms('resname ILE and name CD')
        ile_cd.names = 'CD1'
        # unset segid
        saved_atoms.segments.segids = 'A'
        saved_atoms.write(f"{self.poses_folder}/protein_processed.pdb")
        self._protein_input = "all-poses/protein_processed.pdb"    

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
            f"{mobile.n_atoms} and system_input {ref.n_atoms} \n"
            f"The selection string is {self.protein_align} and name CA and not resname NMA ACE\n"
            f"protein selected resids: {mobile.residues.resids}\n"
            f"system selected resids: {ref.residues.resids}\n"
            "set `protein_align` to a selection string that has the same number of atoms in both files"
            "when running `create_system`."
            )
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

        # modify the chaininfo to be unique for each segment
        current_chain = 65
        u_prot.atoms.tempfactors = 0

        # read the validate the correct segments
        # sometimes the segments can be messed up.
        n_segments = len(u_sys.select_atoms('protein').segments)
        n_segment_name = np.unique(u_sys.select_atoms('protein').segids)
        if len(n_segment_name) != n_segments:
            logger.warning(f"Number of segments in the system is {n_segments} "
                                f"but the segment names are {n_segment_name}. "
                                f"Setting all segments to 'A' for the protein."
                                "If you want to use different segments, "
                                "modify the segments column in the system_topology file manually."
                                )
            protein_seg = u_sys.add_Segment(segid='A')
            u_sys.select_atoms('protein').residues.segments = protein_seg

        for segment in u_sys.select_atoms('protein').segments:
            resid_seg = segment.residues.resids
            resid_seq = " ".join([str(resid) for resid in resid_seg])
            chain_id = segment.atoms.chainIDs[0]
            u_prot.select_atoms(
                f'resid {resid_seq} and chainID {chain_id} and protein').atoms.tempfactors = current_chain
            current_chain += 1
        u_prot.atoms.chainIDs = [chr(int(chain_nm)) for chain_nm in u_prot.atoms.tempfactors]

        comp_2_combined = []

        if self.receptor_segment:
            protein_anchor = u_prot.select_atoms(f'segid {self.receptor_segment} and protein')
            protein_anchor.atoms.chainIDs = 'A'
            protein_anchor.atoms.tempfactors = 65
            other_protein = u_prot.select_atoms(f'not segid {self.receptor_segment} and protein')
            
            comp_2_combined.append(protein_anchor)
            comp_2_combined.append(other_protein)
        else:
            comp_2_combined.append(u_prot.select_atoms('protein'))
        

        if self.membrane_simulation:
            membrane_ag = u_sys.select_atoms(f'resname {" ".join(self.lipid_mol)}')
            if len(membrane_ag) == 0:
                logger.warning(f"No membrane atoms found with resname {self.lipid_mol}. \n"
                                f"Available resnames are {np.unique(u_sys.atoms.resnames)}"
                                f"Please check the lipid_mol parameter.")
            else:
                with open(f'{build_files_orig}/memb_opls2charmm.json', 'r') as f:
                    MEMB_OPLS_2_CHARMM_DICT = json.load(f)
                if np.any(membrane_ag.names == 'O1'):
                    if np.any(membrane_ag.resnames != 'POPC'):
                        raise ValueError(f"Found OPLS lipid name {membrane_ag.residues.resnames}, only 'POPC' is supported. ")
                    # convert the lipid names to CHARMM names
                    membrane_ag.names = [MEMB_OPLS_2_CHARMM_DICT.get(name, name) for name in membrane_ag.names]
                    logger.info(f"Converting OPLS lipid names to CHARMM names.")
                
                membrane_ag.chainIDs = 'M'
                membrane_ag.residues.segments = memb_seg
                logger.debug(f'Number of lipid molecules: {membrane_ag.n_residues}')
                comp_2_combined.append(membrane_ag)
        else:
            # empty selection
            membrane_ag = u_sys.atoms[[]]

        # maestro generated pdb doesn't have SPC water H and O in the same place.
        # probably a potential bug
        water_ag = u_sys.select_atoms('byres (((resname SPC and name O) or water) and around 15 (protein or group memb))', memb=membrane_ag)
        logger.debug(f'Number of water molecules: {water_ag.n_residues}')
        
        # also include ions to water_ag
        ion_ag = u_sys.select_atoms('byres (resname SOD POT CLA NA CL and around 5 (protein))')
        logger.debug(f'Number of ion molecules: {ion_ag.n_residues}')
        # replace SOD with Na+ and POT with K+ and CLA with Cl-
        ion_ag.select_atoms('resname SOD').names = 'Na+'
        ion_ag.select_atoms('resname SOD').residues.resnames = 'Na+'
        ion_ag.select_atoms('resname NA').names = 'Na+'
        ion_ag.select_atoms('resname NA').residues.resnames = 'Na+'
        ion_ag.select_atoms('resname POT').names = 'K+'
        ion_ag.select_atoms('resname POT').residues.resnames = 'K+'
        ion_ag.select_atoms('resname CLA').names = 'Cl-'
        ion_ag.select_atoms('resname CLA').residues.resnames = 'Cl-'
        ion_ag.select_atoms('resname CL').names = 'Cl-'
        ion_ag.select_atoms('resname CL').residues.resnames = 'Cl-'

        water_ag = water_ag + ion_ag
        water_ag.chainIDs = 'W'
        water_ag.residues.segments = water_seg
        if len(water_ag) == 0:
            logger.warning(f"No water molecules found in the system. "
                             f"Available resnames are {np.unique(u_sys.atoms.resnames)}"
                             f"Please check the system_topology and system_coordinate files.")
        else:
            comp_2_combined.append(water_ag)

        u_merged = mda.Merge(*comp_2_combined)

        water = u_merged.select_atoms('water or resname SPC')
        if len(water) != 0:
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
            new_ligand_dict = {}
            for i, (name, pose) in enumerate(self.ligand_dict.items()):
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

                new_ligand_dict[name] = pose
            self.ligand_dict = new_ligand_dict

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

    @save_state
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
        if sim_config.fe_type == 'relative' and not isinstance(self, RBFESystem):
            raise ValueError(f"Invalid fe_type: {sim_config.fe_type}, "
                 "should be 'relative' for RBFE system")
        
        try:
            # overwride l1_x, l1_y, l1_z
            sim_config.l1_x = self.l1_x
            sim_config.l1_y = self.l1_y
            sim_config.l1_z = self.l1_z

            # override the p1, p2, p3
            sim_config.p1 = self.p1
            sim_config.p2 = self.p2
            sim_config.p3 = self.p3
        except:
            logger.debug('cannot set l1_x, l1_y, l1_z, p1, p2, p3 in sim_config')
        
        sim_config.system_name = self.system_name
        sim_config.ligand_dict = self.ligand_dict
        sim_config._membrane_simulation = self.membrane_simulation
        sim_config.protein_align = self.protein_align
        sim_config.receptor_segment = self.receptor_segment
        sim_config.receptor_ff = self.receptor_ff
        sim_config.lipid_ff = self.lipid_ff
        sim_config.ligand_ff = self.ligand_ff
        sim_config.lipid_mol = self.lipid_mol
        sim_config.poses_list = self.all_poses
                 
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
            extra_conformation_restraints: str = None,
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
        avg_struc : str, optional
            The average structure file for adding RMSF restraints.
            Default is None, which means no RMSF restraints will be added.
        rmsf_file : str, optional
            The RMSF file for adding RMSF restraints.
            Default is None, which means no RMSF restraints will be added.
        extra_restraints : str, optional
            The extra restraints file for adding extra restraints.
            Default is None, which means no extra restraints will be added.
        extra_restraints_fc : float, optional
            The force constant for the extra restraints.
            Default is 10 kcal/mol/A^2.
        extra_conformation_restraints : str, optional
            The extra conformation restraints file for adding extra conformation restraints.
            Default is None, which means no extra conformation restraints will be added.
        """
        logger.debug('Preparing the system')
        self.overwrite = overwrite
        self.partition = partition
        self._n_workers = n_workers

        if avg_struc is not None or rmsf_file is not None:
            raise ValueError("Both avg_struc and rmsf_file should be provided")
        if extra_conformation_restraints is not None and not os.path.exists(extra_conformation_restraints):
            raise FileNotFoundError(f"Extra conformation restraints file not found: {extra_conformation_restraints}")
        
        if input_file is not None:
            self._get_sim_config(input_file)
            self._component_windows_dict = ComponentWindowsDict(self)
        
        if win_info_dict is not None:
            for key, value in win_info_dict.items():
                if key not in self._component_windows_dict:
                    raise ValueError(f"Invalid component: {key}. Available components are: {self._component_windows_dict.keys()}")
                self._component_windows_dict[key] = value
        
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]
        self._pose_ligand_dict = {pose: ligand for pose, ligand in zip(self._all_poses, self.ligand_names)}
        self.sim_config.poses_list = self._all_poses 

        if stage == 'equil':
            if self.overwrite:
                logger.debug(f'Overwriting {self.equil_folder}')
                shutil.rmtree(self.equil_folder, ignore_errors=True)
                self._eq_prepared = False
            elif self._eq_prepared and os.path.exists(f"{self.equil_folder}"):
                logger.info('Equilibration already prepared')
                return
            self._eq_prepared = False
            self._slurm_jobs = {}
            # save the input file to the equil directory
            os.makedirs(f"{self.equil_folder}", exist_ok=True)
            with open(f"{self.equil_folder}/sim_config.json", 'w') as f:
                json.dump(self.sim_config.model_dump(), f, indent=2)
            
            self._prepare_equil_system()
            if rmsf_file is not None:
                self._rmsf_restraints = {
                    'avg_struc': avg_struc,
                    'rmsf_file': rmsf_file
                }
                self.add_rmsf_restraints_new(
                        stage='equil',
                        avg_struc=avg_struc,
                        rmsf_file=rmsf_file
                    )
            if extra_restraints is not None:
                self._extra_restraints = extra_restraints
                self.add_extra_restraints(
                        stage='equil',
                        extra_restraints=extra_restraints,
                        extra_restraints_fc=extra_restraints_fc
                    )
            if extra_conformation_restraints is not None:
                self._extra_conformation_restraints = extra_conformation_restraints
                self.add_extra_conformation_restraints(
                        stage='equil',
                        extra_conformation_restraints=extra_conformation_restraints,
                    )
            logger.info('Equil System prepared')
            self._eq_prepared = True
        
        if stage == 'fe':
            if not os.path.exists(f"{self.equil_folder}"):
                raise FileNotFoundError("Equilibration not generated yet. Run prepare('equil') first.")
        
            if not all(os.path.exists(f"{self.equil_folder}/{pose}/FINISHED") for pose in self.all_poses):
                raise FileNotFoundError("Equilibration not finished yet. First run the equilibration.")
                
            sim_config_eq = json.load(open(f"{self.equil_folder}/sim_config.json"))
            sim_config = self.sim_config
            if sim_config_eq != sim_config.model_dump():
            # raise ValueError(f"Equilibration and free energy simulation configurations are different")
                warnings.warn("Equilibration and free energy simulation configurations are different")
                # get the difference
                diff = {k: v for k, v in sim_config_eq.items() if sim_config.model_dump().get(k) != v}
                logger.warning(f"Different configurations: {diff}")
                orig = {k: sim_config.model_dump().get(k) for k in diff.keys()}
                logger.warning(f"Original configuration: {orig}")

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
            if rmsf_file is not None:
                self._rmsf_restraints = {
                    'avg_struc': avg_struc,
                    'rmsf_file': rmsf_file
                }
                self.add_rmsf_restraints_new(
                        stage='fe',
                        avg_struc=avg_struc,
                        rmsf_file=rmsf_file
                    )
            if extra_restraints is not None:
                self._extra_restraints = extra_restraints
                self.add_extra_restraints(
                        stage='fe',
                        extra_restraints=extra_restraints,
                        extra_restraints_fc=extra_restraints_fc
                    )
            if extra_conformation_restraints is not None:
                self._extra_conformation_restraints = extra_conformation_restraints
                self.add_extra_conformation_restraints(
                        stage='fe',
                        extra_conformation_restraints=extra_conformation_restraints,
                    )
            logger.info('FE System prepared')
            self._fe_prepared = True
        
    @safe_directory
    @save_state
    def submit(self,
               stage: str,
               batch_mode: bool = False,
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
        batch_mode : str
            Whether to submit the job in batch mode or not.
        partition : str, optional
            The partition to submit the job. Default is None,
            which means the default partition during prepartiion
            will be used.
        time_limit: str, optional
            The time limit for the job. Default is None,
        overwrite : bool, optional
            Whether to overwrite and re-run all the existing simulations.
        """
        if batch_mode:
            if stage != 'fe':
                raise NotImplementedError("Batch mode is only implemented for 'fe' stage")
            # submit at fe folder
            logger.info('Submit free energy stage in batch mode')
            pbar = tqdm(total=len(self.bound_poses), desc='Submitting free energy jobs')
            for i, pose in enumerate(self.bound_poses):
                if self.pose_failed.get(pose, False):
                    logger.warning(f"Pose {pose} has failed in previous runs. Skipping submission.")
                    continue
                # only check for each pose to reduce frequently checking SLURM 
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit FE jobs')
                for comp in self.sim_config._components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}'
                    if os.path.exists(f"{folder_2_check}/{comp}_FINISHED") and not overwrite:
                        self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}batch', None)
                        logger.debug(f'FE for {pose}/{comp_folder}/{comp}batch has finished; add overwrite=True to re-run the simulation')
                        continue
                    if os.path.exists(f"{folder_2_check}/{comp}_FAILED") and not overwrite:
                        self._slurm_jobs.pop(f'fe_{pose}_{comp_folder}_{comp}batch', None)
                        logger.warning(f'FE for {pose}/{comp_folder}/{comp}batch has failed; add overwrite=True to re-run the simulation')
                        continue
                    if f'fe_{pose}_{comp_folder}_{comp}batch' in self._slurm_jobs:
                        slurm_job = self._slurm_jobs[f'fe_{pose}_{comp_folder}_{comp}batch']
                        if not slurm_job.is_still_running():
                            slurm_job.submit(
                                requeue=True,
                                time=time_limit,
                            )
                            continue
                        elif overwrite:
                            slurm_job.cancel()
                            slurm_job.submit(overwrite=True,
                                time=time_limit,
                            )
                            continue
                        else:
                            logger.debug(f'FE job for {pose}/{comp_folder}/{comp}batch is still running')
                            continue

                    if overwrite:
                        # remove FINISHED and FAILED
                        os.remove(f"{folder_2_check}/{comp}_FINISHED", ignore_errors=True)
                        os.remove(f"{folder_2_check}/{comp}_FAILED", ignore_errors=True)

                    slurm_job = SLURMJob(
                                    filename=f'{self.fe_folder}/batch_run/SLURMM-run-{pose}-{comp}',
                                    path=f'{self.fe_folder}',
                                    partition=partition,
                                    jobname=f'fep_{folder_2_check}_{comp}_fe',
                                    )
                    slurm_job.submit(overwrite=overwrite,
                                time=time_limit,
                    )

                    pbar.set_description(f'FE job for {pose}/{comp_folder}/{comp}batch submitted')
                    self._slurm_jobs.update(
                        {f'fe_{pose}_{comp_folder}_{comp}batch': slurm_job}
                    )
                    self._save_state()
                pbar.update(1)
            pbar.close()

            logger.info('Free energy systems have been submitted in batch for all poses listed in the input file.')
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
                self._save_state()

            pbar.close()
            logger.info('Equilibration systems have been submitted for all poses listed in the input file.')
        elif stage == 'fe_equil':
            logger.info('Submit NPT equilibration part of free energy stage')
            pbar = tqdm(total=len(self.bound_poses), desc='Submitting free energy equilibration jobs')
            for pose in self.bound_poses:
                if self.pose_failed.get(pose, False):
                    logger.warning(f"Pose {pose} has failed in previous runs. Skipping submission.")
                    continue
                # only check for each pose to reduce frequently checking SLURM 
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit FE equilibration jobs')
                for comp in self.sim_config._components:
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
                    self._save_state()

                pbar.update(1)
            logger.info('Free energy systems have been submitted for all poses listed in the input file.')        
            pbar.close()
        elif stage == 'fe':
            logger.info('Submit free energy stage')
            pbar = tqdm(total=len(self.bound_poses), desc='Submitting free energy jobs')
            priorities = np.arange(1, len(self.bound_poses) + 1)[::-1] * 10000
            for i, pose in enumerate(self.bound_poses):
                if self.pose_failed.get(pose, False):
                    logger.warning(f"Pose {pose} has failed in previous runs. Skipping submission.")
                    continue
                # set gradually lower priority for jobs
                priority = priorities[i]
                # only check for each pose to reduce frequently checking SLURM 
                while get_squeue_job_count(partition=partition) >= self.max_num_jobs:
                    time.sleep(120)
                    pbar.set_description(f'Waiting to submit FE jobs')
                for comp in self.sim_config._components:
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
                        self._save_state()
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
                pose=pose,
                sim_config=sim_config,
                component_windows_dict=self.component_windows_dict,
                working_dir=f'{self.equil_folder}',
                infe = (self.rmsf_restraints is not None) or (self.extra_conformation_restraints is not None)
            )
            builders.append(equil_builder)

        # run builders.build in parallel
        logger.info(f'Building equilibration systems for {len(builders)} poses')
        n_workers = min(self.n_workers, len(builders))
        Parallel(n_jobs=n_workers, backend='loky')(
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
        for pose in sim_config.poses_list:
            # if "UNBOUND" found in equilibration, skip
            if os.path.exists(f"{self.equil_folder}/{pose}/UNBOUND"):
                logger.info(f"Pose {pose} is UNBOUND in equilibration; skipping FE")
                os.makedirs(f"{self.fe_folder}/{pose}/Results", exist_ok=True)
                with open(f"{self.fe_folder}/{pose}/Results/Results.dat", 'w') as f:
                    f.write("UNBOUND\n")
                continue
            logger.debug(f'Preparing pose: {pose}')
            
            sim_config_pose = sim_config.copy(deep=True)
            # load anchor_list
            with open(f"{self.equil_folder}/{pose}/anchor_list.txt", 'r') as f:
                anchor_list = f.readlines()
                l1x, l1y, l1z, l1_range = [float(x) for x in anchor_list[0].split()]
                sim_config_pose.l1_x = l1x
                sim_config_pose.l1_y = l1y
                sim_config_pose.l1_z = l1z
                sim_config_pose.l1_range = l1_range

            # copy ff folder
            #shutil.copytree(self.ligandff_folder,
            #                f"{self.fe_folder}/{pose}/ff", dirs_exist_ok=True)
            os.makedirs(f"{self.fe_folder}/{pose}/ff", exist_ok=True)
            for file in os.listdir(self.ligandff_folder):
                shutil.copy(f"{self.ligandff_folder}/{file}",
                            f"{self.fe_folder}/{pose}/ff/{file}")
            
            for component in sim_config._components:
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
                    pose=pose,
                    sim_config=sim_config_pose,
                    component_windows_dict=self.component_windows_dict,
                    working_dir=f'{self.fe_folder}',
                    molr=molr,
                    poser=poser,
                    infe = (self.rmsf_restraints is not None) or (self.extra_conformation_restraints is not None)
                )
                builders.append(fe_eq_builder)
        if len(builders) == 0:
            logger.info('No new FE equilibration systems to build.')
            return
        with tqdm_joblib(tqdm(
            total=len(builders),
            desc="Preparing FE equilibration",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")) as pbar:
            n_workers = min(self.n_workers, len(builders))
            Parallel(n_jobs=n_workers, backend='loky')(
                delayed(builder.build)() for builder in builders
        )
            
    def _prepare_fe_windows(self, regenerate: bool = False):
        sim_config = self.sim_config
        molr = self.mols[0]
        poser = self.bound_poses[0]

        builders = []
        builders_factory = BuilderFactory()
        for pose in self.bound_poses:
            for component in sim_config._components:
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
                        pose=pose,
                        sim_config=sim_config,
                        component_windows_dict=self.component_windows_dict,
                        working_dir=f'{self.fe_folder}',
                        molr=molr,
                        poser=poser,
                        infe = (self.rmsf_restraints is not None) or (self.extra_conformation_restraints is not None)
                    )
                    builders.append(fe_builder)

        if len(builders) == 0:
            logger.info('No new FE window systems to build.')
            return
        if True:
            with tqdm_joblib(tqdm(
                total=len(builders),
                desc="Preparing FE windows",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")) as pbar:
                n_workers = min(self.n_workers, len(builders))
                Parallel(n_jobs=n_workers, backend='loky')(
                    delayed(builder.build)() for builder in builders
                )
        else:
            for builder in tqdm(builders, desc="Preparing FE windows"):
                builder.build()

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


    def add_extra_conformation_restraints(
                            self,
                            stage: str,
                            extra_conformation_restraints: str,
                            ):
        """
        Additional distance restraints to be added to the system.

        Parameters
        ----------
        stage : str
            The stage of the simulation. Options are 'equil' and 'fe'.
        extra_conformation_restraints : str
            The json file containing the extra distance restraints.
            The file should be in the format of:
            direction, res1, res2, cutoff, force_constant
        """
        COMP_MODIFIED = ['z', 'o']
        def generate_restraint_to_cv(pose_folder, restraints):
            colvar_block_all = []
            for restraint in restraints:
                direction, res1, res2, cutoff, force_constant = restraint
                u = mda.Universe(f'{pose_folder}/full.pdb')
                atom1 = u.select_atoms(f'resid {res1} and name CA')[0].index + 1
                atom2 = u.select_atoms(f'resid {res2} and name CA')[0].index + 1
                colvar_block = "&colvar\n"
                colvar_block += " cv_type = 'DISTANCE'\n"
                colvar_block += f" cv_ni = 2, cv_i = {atom1},{atom2}\n"
                if direction == '>=':
                    cutoff_2 = cutoff - 0.3
                    colvar_block += f" anchor_position = {cutoff_2:03f}, {cutoff:03f}, 999, 999\n"
                elif direction == '<=':
                    cutoff_2 = cutoff + 0.3
                    colvar_block += f" anchor_position = 0, 0, {cutoff:03f}, {cutoff_2:03f}\n"
                else:
                    raise ValueError(f"Invalid direction: {direction}")
                colvar_block += f" anchor_strength = {force_constant:2f}, {force_constant:2f}\n"
                colvar_block += "/\n"
                colvar_block_all.extend(colvar_block)
            return colvar_block_all

        with open(extra_conformation_restraints, 'r') as f:
            extra_restraints = json.load(f)
        
        logger.debug(f'Adding extra conformation restraints from {extra_conformation_restraints}')
        if stage == 'equil':
            pose_folder_base = self.equil_folder
            poses = self.all_poses
            # The colvar block is the same for all poses in equilibration
            colvar_block_all = generate_restraint_to_cv(f"{pose_folder_base}/{poses[0]}", extra_restraints)
            for pose in poses:
                # modify cv.in file in each pose folder
                cv_file = f"{pose_folder_base}/{pose}/cv.in"
                # if .bak exists, to avoid multiple appending
                # first copy the original cv file to the backup
                if os.path.exists(cv_file + '.bak'):
                    os.system(f"cp {cv_file}.bak {cv_file}")
                else:
                    # copy original cv file for backup
                    os.system(f"cp {cv_file} {cv_file}.bak")
                
                with open(cv_file, 'r') as f:
                    lines = f.readlines()

                with open(cv_file, 'w') as f:
                    for line in lines:
                        f.write(line)
                    f.write("\n")
                    for line in colvar_block_all:
                        f.write(line)
                        
        elif stage == 'fe':
            pose_folder_base = self.fe_folder
            poses = self.bound_poses
            for pose in self.bound_poses:
                for comp in self.sim_config._components:
                    if comp not in COMP_MODIFIED:
                        continue
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    colvar_block = generate_restraint_to_cv(f"{folder_comp}/{comp}-1", extra_restraints)
                    windows = self.component_windows_dict[comp]
                    cv_files = [f"{folder_comp}/{comp}{j:02d}/cv.in"
                        for j in range(-1, len(windows))]
                    for cv_file in cv_files:
                        # if .bak exists, to avoid multiple appending
                        # first copy the original cv file to the backup
                        if os.path.exists(cv_file + '.bak'):
                            os.system(f"cp {cv_file}.bak {cv_file}")
                        else:
                            # copy original cv file for backup
                            os.system(f"cp {cv_file} {cv_file}.bak")
                        
                        with open(cv_file, 'r') as f:
                            lines = f.readlines()

                        with open(cv_file, 'w') as f:
                            for line in lines:
                                f.write(line)
                            f.write("\n")
                            for line in colvar_block:
                                f.write(line)
        else:
            raise ValueError(f"Invalid stage: {stage}")


    def add_rmsf_restraints_new(
                            self,
                            stage: str,
                            avg_struc: str,
                            rmsf_file: str,
                            force_constant: float = 100):
        """
        Add RMSF restraints to the system.
        Similar to https://pubs.acs.org/doi/10.1021/acs.jctc.3c00899?ref=pdf

        The restraint is added by adding extra dummy atoms representing the
        center of restraints for each C-alpha atom in the protein.
        The restraints are added to the cv.in file in the form of
        ```bash
        &colvar
         cv_type = 'DISTANCE'
         cv_ni = 2, cv_i = index_ca_i, index_dum_i,
         anchor_position = 0, 0, rmsf_val, 999
         anchror_strength = force_constant, force_constant
        /
        ```
        where index_ca_i is the index of the C-alpha atom in the protein,
        index_dum_i is the index of the dummy atom representing the center
        of the restraint, rmsf_val is the RMSF value of the residue, and
        force_constant is the force constant of the restraint.

        """
        logger.debug('Adding RMSF restraints')
        COMP_MODIFIED = ['z', 'o']

        def generate_colvar_block(atm_index,
                                  dum_index,
                                  dis_cutoff,
                                  force_constant=100):
            colvar_block = "&colvar\n"
            colvar_block += " cv_type = 'DISTANCE'\n"
            colvar_block += f" cv_ni = 2, cv_i = {atm_index},{dum_index}\n"
            colvar_block += f" anchor_position = 0, 0, {dis_cutoff}, 999\n"
            colvar_block += f" anchor_strength = {force_constant:2f}, {force_constant:2f}\n"
            colvar_block += "/\n"
            return colvar_block

        def add_dum_atoms(pose_folder, ref_pos):
            """
            Files to be modified:
            - full.pdb
            - full.inpcrd
            - full.prmtop
            - full.hmr.prmtop
            with parmed
            dum atom prmtop: solvate_dum.prmtop
            """
            import parmed as pmd
            
            # move old full.pdb and full.inpcrd to full_old.pdb and full_old.inpcrd
            if os.path.exists(f"{pose_folder}/full.pdb.bak"):
                os.system(f'cp {pose_folder}/full.pdb.bak {pose_folder}/full.pdb')
            else:
                os.system(f'cp {pose_folder}/full.pdb {pose_folder}/full.pdb.bak')
            if os.path.exists(f"{pose_folder}/full.inpcrd.bak"):
                os.system(f'cp {pose_folder}/full.inpcrd.bak {pose_folder}/full.inpcrd')
            else:
                os.system(f'cp {pose_folder}/full.inpcrd {pose_folder}/full.inpcrd.bak')
            if os.path.exists(f"{pose_folder}/full.hmr.prmtop.bak"):
                os.system(f'cp {pose_folder}/full.hmr.prmtop.bak {pose_folder}/full.hmr.prmtop')
            else:
                os.system(f'cp {pose_folder}/full.hmr.prmtop {pose_folder}/full.hmr.prmtop.bak')
            if os.path.exists(f"{pose_folder}/full.prmtop.bak"):
                os.system(f'cp {pose_folder}/full.prmtop.bak {pose_folder}/full.prmtop')
            else:
                os.system(f'cp {pose_folder}/full.prmtop {pose_folder}/full.prmtop.bak')
            
            
            full_prmtop = f"{pose_folder}/full.hmr.prmtop" if self.sim_config.hmr == 'yes' else f"{pose_folder}/full.prmtop"
            combined = pmd.load_file(full_prmtop, f"{pose_folder}/full.inpcrd")
            n_atoms = ref_pos.shape[0]
            all_ps = []
            for atm in range(n_atoms):
                dum_p = pmd.load_file(f'{pose_folder}/dum.prmtop', f'{pose_folder}/dum.inpcrd')
                dum_p.coordinates = ref_pos[atm]
                all_ps.append(dum_p)
            
            all_dums = all_ps[0]
            for dum in all_ps[1:]:
                all_dums += dum
            combined += all_dums

            # set dum_p resname ANC
            # can only do it after merging
            # becasue the modification on dum_p will not be reflected
            # for some reason??
            for res in combined.residues[-len(all_dums.residues):]:
                res.name = 'ANC'
            
            combined.save(f"{pose_folder}/full.pdb", overwrite=True)
            combined.save(f"{pose_folder}/full.inpcrd", overwrite=True)
            if self.sim_config.hmr == 'yes':
                combined.save(f"{pose_folder}/full.hmr.prmtop", overwrite=True)
            else:
                combined.save(f"{pose_folder}/full.prmtop", overwrite=True)

        def write_colvar_block(ref_u, avg_u, cv_files, pose_folder):
            n_atoms = ref_u.atoms.n_atoms
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
                # new DUM atom added to the system
                dum_index = n_atoms + i + 1
                ref_pos[i] = atm.position
                resid_i = atm.resid
                rmsf_val = rmsf_values[rmsf_values[:, 0] == resid_i, 1]
                if len(rmsf_val) == 0:
                    logger.warning(f"resid: {resid_i} not found in rmsf file")
                    continue
                atm_index = gpcr_ref[i].index + 1
                cv_lines.append(generate_colvar_block(
                        atm_index=atm_index,
                        dum_index=dum_index,
                        dis_cutoff=rmsf_val[0],
                        force_constant=force_constant
                ))
            add_dum_atoms(pose_folder,
                          ref_pos)
            
            for cv_file in cv_files:
                
                # if .bak exists, to avoid multiple appending
                # first copy the original cv file to the backup
                if os.path.exists(cv_file + '.bak'):
                    os.system(f"cp {cv_file}.bak {cv_file}")
                else:
                    # copy original cv file for backup
                    os.system(f"cp {cv_file} {cv_file}.bak")
                
                with open(cv_file, 'r') as f:
                    lines = f.readlines()

                with open(f'{cv_file}.eq0', 'w') as f:
                    for line in lines:
                        f.write(line)
                    f.write("\n")
                    for line in cv_lines:
                        f.write(line)
                
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
                avg_u = mda.Universe(avg_struc)

                write_colvar_block(u_ref, avg_u, cv_files,
                                  f"{self.equil_folder}/{pose}")
                n_atoms_before = u_ref.atoms.n_atoms
                n_dums = avg_u.atoms.n_atoms
                new_mask_component = f'@{n_atoms_before + 1}-{n_atoms_before + n_dums + 1}'
                

                eqnvt = f"{self.equil_folder}/{pose}/eqnvt.in"
                
                with open(eqnvt, 'r') as f:
                    lines = f.readlines()
                with open(eqnvt, 'w') as f:
                    for line in lines:
                        if 'restraintmask' in line:
                            line = modify_restraint_mask(line, new_mask_component)
                        elif 'ntr = ' in line:
                            line = ' ntr = 1,\n'
                        elif 'cv.in' in line and 'cv.in.eq0' not in line:
                            line = line.replace('cv.in', 'cv.in.eq0')
                        f.write(line)

                eqnpt0 = f"{self.equil_folder}/{pose}/eqnpt0.in"
                
                with open(eqnpt0, 'r') as f:
                    lines = f.readlines()
                with open(eqnpt0, 'w') as f:
                    for line in lines:
                        if 'restraintmask' in line:
                            line = modify_restraint_mask(line, new_mask_component)
                        elif 'ntr = ' in line:
                            line = ' ntr = 1,\n'
                        elif 'cv.in' in line and 'cv.in.eq0' not in line:
                            line = line.replace('cv.in', 'cv.in.eq0')
                        f.write(line)
                
                eqnpt = f"{self.equil_folder}/{pose}/eqnpt.in"
                
                with open(eqnpt, 'r') as f:
                    lines = f.readlines()
                with open(eqnpt, 'w') as f:
                    for line in lines:
                        if 'restraintmask' in line:
                            line = modify_restraint_mask(line, new_mask_component)
                        elif 'ntr = ' in line:
                            line = ' ntr = 1,\n'
                        if 'cv.in' in line and 'cv.in.bak' not in line:
                            line = line.replace('cv.in', 'cv.in.bak')
                        f.write(line)
                
                md_in_files = glob.glob(f"{self.equil_folder}/{pose}/mdin-*")
                for md_in_file in md_in_files:
                    with open(md_in_file, 'r') as f:
                        lines = f.readlines()
                    with open(md_in_file, 'w') as f:
                        for line in lines:
                            if 'restraintmask' in line:
                                line = modify_restraint_mask(line, new_mask_component)
                            elif 'barostat' in line:
                                # bug when using Berendesen barostat
                                # with NFE module
                                # https://github.com/yuxuanzhuang/nfe_berendsen
                                # need to switch to Monte Carlo barostat
                                line = 'barostat = 2,\n'
                            elif 'ntr = ' in line:
                                line = ' ntr = 1,\n'
                            f.write(line)


        elif stage == 'fe':
            # this should be done after fe_equil
            for pose in self.bound_poses:
                avg_u = mda.Universe(avg_struc)

                for comp in self.sim_config._components:
                    if comp not in COMP_MODIFIED:
                        continue
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    u_ref = mda.Universe(
                            f"{folder_comp}/{comp}-1/full.pdb",
                    )
                    windows = self.component_windows_dict[comp]
                    cv_files = [f"{folder_comp}/{comp}{j:02d}/cv.in"
                        for j in range(-1, len(windows))]
                    n_atoms_before = u_ref.atoms.n_atoms
                    n_dums = avg_u.atoms.n_atoms
                    write_colvar_block(u_ref, avg_u, cv_files,
                                        f"{folder_comp}/{comp}-1")
                    
                    eq_in_files = glob.glob(f"{folder_comp}/*/eqnpt0.in")
                    for eq_in_file in eq_in_files:
                        with open(eq_in_file, 'r') as f:
                            lines = f.readlines()
                        with open(eq_in_file, 'w') as f:
                            for line in lines:
                                if 'restraintmask' in line:
                                    new_mask_component = f'@{n_atoms_before + 1}-{n_atoms_before + n_dums + 1}'
                                    line = modify_restraint_mask(line, new_mask_component)
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
                                if 'restraintmask' in line:
                                    new_mask_component = f'@{n_atoms_before + 1}-{n_atoms_before + n_dums + 1}'
                                    line = modify_restraint_mask(line, new_mask_component)
                                if 'cv.in' in line and 'cv.in.bak' not in line:
                                    f.write(line.replace('cv.in', 'cv.in.bak'))
                                else:
                                    f.write(line)
                    
                    md_in_files = glob.glob(f"{folder_comp}/*/mdin-*")
                    for md_in_file in md_in_files:
                        with open(md_in_file, 'r') as f:
                            lines = f.readlines()
                        with open(md_in_file, 'w') as f:
                            for line in lines:
                                if 'restraintmask' in line:
                                    new_mask_component = f'@{n_atoms_before + 1}-{n_atoms_before + n_dums + 1}'
                                    line = modify_restraint_mask(line, new_mask_component)
                                if 'ntr = ' in line:
                                    line = ' ntr = 1,\n'
                                f.write(line)
        else:
            raise ValueError(f"Invalid stage: {stage}")
        logger.debug('RMSF restraints added')


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
                atm_index = gpcr_ref[i].index + 1
                cv_lines.append(generate_colvar_block(atm_index, rmsf_val[0], ref_pos[i]))
            
            for cv_file in cv_files:
                
                # if .bak exists, to avoid multiple appending
                # first copy the original cv file to the backup
                if os.path.exists(cv_file + '.bak'):
                    os.system(f"cp {cv_file}.bak {cv_file}")
                else:
                    # copy original cv file for backup
                    os.system(f"cp {cv_file} {cv_file}.bak")
                
                with open(cv_file, 'r') as f:
                    lines = f.readlines()

                with open(f'{cv_file}.eq0', 'w') as f:
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
            # this should be done after fe_equil
            for pose in self.bound_poses:
                for comp in self.sim_config._components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    folder_comp = f'{self.fe_folder}/{pose}/{COMPONENTS_FOLDER_DICT[comp]}'
                    u_ref = mda.Universe(
                            f"{folder_comp}/{comp}-1/full.pdb",
                    )
                    windows = self.component_windows_dict[comp]
                    cv_files = [f"{folder_comp}/{comp}{j:02d}/cv.in"
                        for j in range(-1, len(windows))]
                    
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
                        if 'ntr =' in line:
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
                    f"{self.equil_folder}/{self.all_poses[0]}/full.pdb")
            
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
            )
            
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

                for comp in self.sim_config._components:
                    # only add to components containing protein
                    if comp in ['y']:
                        continue
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
                    sim_range: Optional[Tuple[int, int]] = None,
                    raise_on_error: bool = True,
                    mol: str = 'lig',
                    n_workers: int = 4,
                    input_dict: dict = None):
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
        mol : str
            The molecule to analyze. Default is 'lig'.
            This is used to set the legend of the plot.
        n_workers : int
            The number of workers to use for parallel processing.
            Default is 4.
        """
        input_dict_pose = {
            'fe_folder': self.fe_folder,
            'components': self.sim_config._components,
            'rest': self.sim_config.rest,
            'temperature': self.sim_config.temperature,
            'component_windows_dict': self.component_windows_dict,
            'sim_range': sim_range,
            'raise_on_error': raise_on_error,
            'n_workers': n_workers,
        }
        if input_dict is not None:
            input_dict_pose.update(input_dict)
        analyze_pose_task(
            pose=pose,
            mol=mol,
            **input_dict_pose
        )
        
                
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
        job_extra_directives: list = None,
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
        job_extra_directives : list, optional
            Additional SLURM directives to update the job submission.
            Default is None.
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
                elif self.fe_results[pose].fe == 'unbound':
                    continue
                elif np.isnan(self.fe_results[pose].fe):
                    unfinished_poses.append(pose)
                    continue
        else:
            unfinished_poses = self.all_poses

        input_dict = {
            'fe_folder': self.fe_folder,
            'components': self.sim_config._components,
            'rest': self.sim_config.rest,
            'temperature': self.sim_config.temperature,
            'water_model': self.sim_config.water_model,
            'rocklin_correction': self.sim_config.rocklin_correction,
            'component_windows_dict': self.component_windows_dict,
            'sim_range': sim_range,
            'raise_on_error': raise_on_error,
        }

        if run_with_slurm and len(unfinished_poses) > 0:
            logger.info('Running analysis with SLURM Cluster')
            from dask_jobqueue import SLURMCluster
            from dask.distributed import Client
            from distributed.utils import TimeoutError

            log_dir = os.path.expanduser('~/.batter_jobs')
            os.makedirs(log_dir, exist_ok=True)
            slurm_kwargs = {
                # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
                'n_workers': len(unfinished_poses),
                'queue': self.partition,
                'cores': 6,
                'memory': '30GB',
                'walltime': '00:30:00',
                'processes': 1,
                'nanny': False,
                'job_extra_directives': [
                    '--job-name=batter-analysis',
                    f'--output={log_dir}/dask-%j.out',
                    f'--error={log_dir}/dask-%j.err',
                ],
                'worker_extra_args': [
                    "--no-dashboard",
                    "--resources analysis=1"
                ],
                # 'account': 'your_slurm_account',
            }
            input_dict['n_workers'] = slurm_kwargs['cores']
            if run_with_slurm_kwargs is not None:
                slurm_kwargs.update(run_with_slurm_kwargs)
            if job_extra_directives is not None:
                slurm_kwargs['job_extra_directives'].extend(job_extra_directives)

            cluster = SLURMCluster(
                **slurm_kwargs,
            )
            logger.info(f'SLURM Cluster created.')

            client = Client(cluster)
            logger.info(f'Dask Dashboard Link: {client.dashboard_link}')
            # Wait for all expected workers
            try:
                logger.info(f'Waiting for {len(unfinished_poses)} workers to start...')
                client.wait_for_workers(n_workers=len(unfinished_poses), timeout=200)
            except TimeoutError:
                logger.warning(f"Timeout: Only {len(client.scheduler_info()['workers'])} workers started.")
                # scale down the cluster to the number of available workers
                if len(client.scheduler_info()['workers']) == 0:
                    client.close()
                    cluster.close()
                    
                    raise TimeoutError("No workers started in 200 sec. Check SLURM job status or run without SLURM.")
                cluster.scale(jobs=len(client.scheduler_info()['workers']))

            futures = []
            for pose in unfinished_poses:
                bound_ind = self.bound_poses.index(pose)
                mol = self.bound_mols[bound_ind]
                logger.debug(f'Submitting analysis for pose: {pose}')
                fut = client.submit(
                    analyze_single_pose_dask_wrapper,
                    pose,
                    mol,
                    input_dict,
                    pure=True,
                    resources={'analysis': 1},
                    key=f'analyze_{pose}',
                    retries=3,
                )
                futures.append(fut)
            
            logger.info(f'{len(futures)} analysis jobs submitted to SLURM Cluster')
            logger.info('Waiting for analysis jobs to complete...')
            _ = client.gather(futures, errors='skip')
            
            logger.info('Analysis with SLURM Cluster completed')
            client.close()
            cluster.close()
        elif len(unfinished_poses) >= 0:
            pbar = tqdm(
                unfinished_poses,
                desc='Analyzing FE for poses',
            )
            for pose in unfinished_poses:
                bound_ind = self.bound_poses.index(pose)
                mol = self.bound_mols[bound_ind]
                pbar.set_postfix(pose=pose)
                self.analyze_pose(
                    pose=pose,
                    mol=mol,
                    sim_range=sim_range,
                    raise_on_error=raise_on_error,
                    n_workers=self.n_workers,
                    input_dict=input_dict
                )
    
        # sort self.fe_sults by pose
        self.load_results()
    
        with open(f'{self.output_dir}/Results/Results.dat', 'w') as f:
            for i, (pose, fe) in enumerate(self.fe_results.items()):
                mol_name = self.mols[i]
                ligand_name = self.pose_ligand_dict[pose]
                if fe is None:
                    logger.warning(f'FE for {pose} is None; skipping')
                elif fe.fe == 'unbound':
                    f.write(f'{ligand_name}\t{mol_name}\t{pose}\tunbound\n')
                else:
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
            fe_timeseries_file = f'{self.fe_folder}/{pose}/Results/fe_timeseries.json'
            if os.path.exists(results_file):
                with open(fe_timeseries_file) as f:
                    fe_timeseries = json.load(f)
                self.fe_results[pose] = FEResult(results_file, fe_timeseries)
                if self.fe_results[pose].fe == 'unbound':
                    logger.debug(f'FE for {pose} is unbound.')
                    loaded_poses.append(pose)
                elif np.isnan(self.fe_results[pose].fe):
                    logger.debug(f'FE for {pose} is invalid (None or NaN); rerun `analysis`.')
                else:
                    loaded_poses.append(pose)
                    logger.debug(f'FE for {pose} loaded from {results_file}')
            else:
                logger.debug(f'FE results file {results_file} not found for pose {pose}')
                self.fe_results[pose] = None
        if not self.fe_results:
            raise ValueError('No results found in the output directory. Please run the analysis first.')
        logger.debug(f'Results for {loaded_poses} loaded successfully')
        

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
            if len(self.lipid_mol) > 0:
                saved_ag = u.select_atoms(f'not resname WAT DUM Na+ Cl- ANC and not resname {" ".join(self.lipid_mol)}')
            else:
                saved_ag = u.select_atoms(f'not resname WAT DUM Na+ Cl- ANC')
            saved_ag.write(f'{self.output_dir}/Results/protein_{pose}.pdb')

            initial_pose = f'{self.poses_folder}/{pose}.pdb'
            os.system(f'cp {initial_pose} {self.output_dir}/Results/init_{pose}.pdb')
           
    @staticmethod
    def _find_anchor_atoms(u_prot,
                           u_lig,
                           lig_sdf,
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
        lig_sdf : str
            The ligand sdf file.
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
        r_dist : float
            The distance between the ligand anchor atom and the first protein anchor atom.
        """

        u_merge = mda.Merge(u_prot.atoms, u_lig.atoms)

        P1_atom = u_merge.select_atoms(anchor_atoms[0])
        P2_atom = u_merge.select_atoms(anchor_atoms[1])
        P3_atom = u_merge.select_atoms(anchor_atoms[2])
        if P1_atom.n_atoms == 0 or P2_atom.n_atoms == 0 or P3_atom.n_atoms == 0:
            raise ValueError('Error: anchor atom not found with the provided selection string.\n'
                             f'p1: {anchor_atoms[0]}, p2: {anchor_atoms[1]}, p3: {anchor_atoms[2]}\n'
                             f'P1_atom.n_atoms: {P1_atom.n_atoms}, \n P2_atom.n_atoms: {P2_atom.n_atoms}, \n P3_atom.n_atoms: {P3_atom.n_atoms}')
        if P1_atom.n_atoms != 1 or P2_atom.n_atoms != 1 or P3_atom.n_atoms != 1:
            raise ValueError('Error: more than one atom selected in the anchor atoms')
        
        if ligand_anchor_atom is not None:
            lig_atom = u_merge.select_atoms(ligand_anchor_atom)
            if lig_atom.n_atoms == 0:
                logger.warning(f"Provided ligand anchor atom {ligand_anchor_atom} not found in the ligand."
                               "Using all ligand atoms instead.")
                lig_atom = u_lig.atoms
        else:
            candidates = get_ligand_candidates(lig_sdf)
            lig_atom = u_lig.atoms[candidates]

        # get ll_x,y,z distances
        r_vect = lig_atom.center_of_mass() - P1_atom.positions
        logger.debug(f'l1_x: {r_vect[0][0]:.2f}; l1_y: {r_vect[0][1]:.2f}; l1_z: {r_vect[0][2]:.2f}')

        p1_formatted = f':{P1_atom.resids[0]}@{P1_atom.names[0]}'
        p2_formatted = f':{P2_atom.resids[0]}@{P2_atom.names[0]}'
        p3_formatted = f':{P3_atom.resids[0]}@{P3_atom.names[0]}'
        logger.debug(f'Receptor anchor atoms: P1: {p1_formatted}, P2: {p2_formatted}, P3: {p3_formatted}')
        r_dist = np.linalg.norm(r_vect) + 1
        return (r_vect[0][0], r_vect[0][1], r_vect[0][2],
                p1_formatted, p2_formatted, p3_formatted,
                r_dist)
              
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
                    
                    if self.sim_config.hmr == 'no':
                        prmtop_f = 'full.prmtop'
                    else:
                        prmtop_f = 'full.hmr.prmtop'
                    cpptraj_command = f"cpptraj -p {prmtop_f} <<EOF\n"
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
            if u_lig.n_atoms == 0:
                raise ValueError(f'Ligand {self.bound_mols[i]} not found in the system for pose {pose}'
                f'all resnames: {np.unique(u_sys.atoms.resnames)}')
            lig_sdf = f'{self.ligandff_folder}/{self.bound_mols[i]}.sdf'

            ligand_anchor_atom = self.ligand_anchor_atom

            logger.debug(f'Finding anchor atoms for pose {pose}')
            l1_x, l1_y, l1_z, p1, p2, p3, l1_range = self._find_anchor_atoms(
                        u_sys,
                        u_lig,
                        lig_sdf,
                        self.anchor_atoms,
                        ligand_anchor_atom)
            with open(f'{self.equil_folder}/{pose}/anchor_list.txt', 'w') as f:
                f.write(f'{l1_x} {l1_y} {l1_z} {l1_range}')
        

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
                     extra_conformation_restraints: str = None,
                     partition: str = 'owners',
                     max_num_jobs: int = 2000,
                     time_limit: str = '6:00:00',
                     fail_on_error: bool = False,
                     vmd: str = None,
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
        extra_conformation_restraints: str, optional
            The extra conformation restraints json file to be read from
        partition : str, optional
            The partition to submit the job.
            Default is 'rondror'.
        max_num_jobs : int, optional
            The maximum number of jobs to submit at a time.
            Default is 2000.
        time_limit : str, optional
            The time limit for the job submission.
            Default is '6:00:00'.
        fail_on_error : bool, optional
            Whether to fail the pipeline on error during simulations.
            Default is False with the failed simulations marked as 'FAILED'.
        vmd : str, optional
            The path to the VMD executable. If not provided,
            the code will use `vmd`.
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
        self._fail_on_error = fail_on_error

        start_time = time.time()
        logger.info(f'Start time: {time.ctime()}')
        if input_file is not None:
            self._get_sim_config(input_file)
        else:
            if not hasattr(self, 'sim_config'):
                raise ValueError('Input file is not provided and sim_config is not set.')
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]
        if vmd:
            if not os.path.exists(vmd):
                raise FileNotFoundError(f'VMD executable {vmd} not found')
            # set batter.utils.vmd to the provided path
            import batter.utils
            batter.utils.vmd = vmd
            logger.info(f'Setting VMD path to {vmd}')

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
                extra_restraints_fc=extra_restraints_fc,
                extra_conformation_restraints=extra_conformation_restraints,
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
                n_finished = len([k for k, v in self.sim_finished.items() if v])
                #logger.info(f'{time.ctime()} - Finished jobs: {n_finished} / {len(self.sim_finished)}')
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – Equilibration sims finished"
                pbar.set_description(desc)

                not_finished = [k for k, v in self.sim_finished.items() if not v]
                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                # sleep for 10 minutes to avoid overwhelming the scheduler
                time.sleep(10 * 60)
            pbar.update(len(self.all_poses) - pbar.n)  # update to total
            pbar.set_description('Equilibration finished')
            pbar.close()
            

        else:
            logger.info('Equilibration simulations are already finished')
            logger.info(f'If you want to have a fresh start, remove {self.equil_folder} manually')
        
        if only_equil:
            logger.info('only_equil is set to True. '
                        'Skipping the free energy calculation.')
            return

        #4.0, submit the free energy equilibration
        # check if all poses failed or unbound
        if len(self.bound_poses) == 0:
            logger.warning('No bound poses found after equilibration. '
                           'Please check the equilibration results.')
            return
        if all([self.pose_failed.get(pose, False) for pose in self.bound_poses]):
            logger.warning('All bound poses failed in equilibration. '
                           'Please check the equilibration results.')
            return

        remd = self.sim_config.remd == 'yes'
        self.batch_mode = remd
        num_fe_sim = self.sim_config.num_fe_range

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
                extra_restraints_fc=extra_restraints_fc,
                extra_conformation_restraints=extra_conformation_restraints,
            )
            if not os.path.exists(f'{self.fe_folder}/pose0/groupfiles') or overwrite:
                logger.info('Generating batch run files...')
                self.generate_batch_files(remd=remd, num_fe_sim=num_fe_sim)

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
            pbar = tqdm(total=len(self.bound_poses) * len(self.sim_config._components),
                        desc="FE Equilibration sims finished",
                        unit="job")
            while self._check_fe_equil():
                # get finishd jobs
                n_finished = len([k for k, v in self.sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE Equilibration sims finished"
                pbar.set_description(desc)

                #logger.info(f'{time.ctime()} - Finished jobs: {n_finished} / {len(self.sim_finished)}')
                not_finished = [k for k, v in self.sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]
                # name f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                # exclude the failed jobs
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(5*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config._components) - pbar.n)  # update to total
            pbar.set_description('FE equilibration finished')
            pbar.close()

        else:
            logger.info('Free energy equilibration simulations are already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        # copy last equilibration snapshot to the free energy folder
        for pose in self.bound_poses:
            for comp in self.sim_config._components:
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

        if all([self.pose_failed.get(pose, False) for pose in self.bound_poses]):
            logger.warning('All bound poses failed in FE equilibration. '
                           'Please check the FE equilibration results.')
            return
        if not os.path.exists(f'{self.fe_folder}/pose0/groupfiles') or overwrite:
            logger.info('Generating batch run files...')
            self.generate_batch_files(remd=remd, num_fe_sim=num_fe_sim)

        if self._check_fe():
            logger.info('Submitting the free energy calculation')
            if dry_run:
                logger.info('Dry run is set to True. '
                            'Skipping the free energy submission.')
                return
            
            if self.batch_mode:
                logger.info('Running free energy calculation with REMD in batch mode')
                self.submit(
                    stage='fe',
                    batch_mode=True,
                    partition=partition,
                    time_limit=time_limit,
                )
            else:
                logger.info('Running free energy calculation without REMD')
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
                n_finished = len([k for k, v in self.sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE simulations finished"
                pbar.set_description(desc)
                #logger.info(f'{time.ctime()} Finished jobs: {n_finished} / {len(self.sim_finished)}')
                not_finished = [k for k, v in self.sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(10*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config._components) - pbar.n)  # update to total
            pbar.set_description('FE calculation finished')
            pbar.close()
        else:
            logger.info('Free energy calculation is already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        #5 analyze the results
        logger.info('Analyzing the results')
        self._generate_aligned_pdbs()

        num_sim = self.sim_config.num_fe_range
        # exclude the first few simulations for analysis
        start_sim = 3 if num_sim >= 6 else 1
        self.analysis(
            load=True,
            check_finished=False,
            sim_range=(start_sim, -1),
        )
        logger.info(f'The results are in the {self.output_dir}')
        logger.info(f'Results')
        # print out self.output_dir/Results/Results.dat
        with open(f'{self.output_dir}/Results/Results.dat', 'r') as f:
            results = f.readlines()
            for line in results:
                logger.info(line.strip())
        logger.info('Pipeline finished')
        end_time = time.time()
        logger.info(f'End time: {time.ctime()}')
        total_time = end_time - start_time
        logger.info(f'Total time: {total_time:.2f} seconds')
        # dump the state at the end of the pipeline
        # to the system folder
        self.dump()

    @save_state
    def _check_equilibration(self):
        """
        Check if the equilibration is finished by checking the FINISHED file
        """
        sim_finished = {}
        sim_failed = {}
        pose_failed = {}
        for pose in self.all_poses:
            if not os.path.exists(f"{self.equil_folder}/{pose}/FINISHED"):
                sim_finished[f'eq_{pose}'] = False
            else:
                sim_finished[f'eq_{pose}'] = True
            if os.path.exists(f"{self.equil_folder}/{pose}/FAILED"):
                sim_failed[f'eq_{pose}'] = True
                pose_failed[pose] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        self._pose_failed = pose_failed

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
        pose_failed = {}
        for pose in self.bound_poses:
            for comp in self.sim_config._components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]

                win = -1
                folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{win:02d}'
                if not os.path.exists(f"{folder_2_check}/EQ_FINISHED"):
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = False
                else:
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = True
                if os.path.exists(f"{folder_2_check}/FAILED"):
                    sim_failed[f'fe_{pose}_{comp_folder}_{comp}{win:02d}'] = True
                    pose_failed[pose] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        self._pose_failed = pose_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy EQ calculation failed: {self._sim_failed}')
            if self._fail_on_error:
                raise RuntimeError(f'Free energy EQ calculation failed in pose {self._pose_failed}')
            else:
                # add failed runs to finished runs
                for k in self._sim_failed.keys():
                    self._sim_finished[k] = True
            
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
        if self.batch_mode:
            return self._check_fe_batch()
        sim_finished = {}
        sim_failed = {}
        pose_failed = {}
        for pose in self.bound_poses:
            for comp in self.sim_config._components:
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
                        pose_failed[pose] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        self._pose_failed = pose_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy calculation failed: {self._sim_failed}')
            if self._fail_on_error:
                raise RuntimeError(f'Free energy calculation failed in pose {self._pose_failed}')
            else:
                # add failed runs to finished runs
                for k in self._sim_failed.keys():
                    self._sim_finished[k] = True
        if all(self._sim_finished.values()):
            logger.debug('Free energy calculation is finished')
            return False
        else:
            not_finished = [k for k, v in self._sim_finished.items() if not v]
            logger.debug(f'Not finished: {not_finished}')
            return True

    @save_state
    def _check_fe_batch(self):
        """
        Check if the free energy calculation is finished in batch mode.
        """
        sim_finished = {}
        sim_failed = {}
        pose_failed = {}
        for pose in self.bound_poses:
            for comp in self.sim_config._components:
                comp_folder = COMPONENTS_FOLDER_DICT[comp]
                folder_2_check = f'{self.fe_folder}/{pose}/{comp_folder}/'
                if not os.path.exists(f"{folder_2_check}/{comp}_FINISHED"):
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}batch'] = False
                else:
                    sim_finished[f'fe_{pose}_{comp_folder}_{comp}batch'] = True
                if os.path.exists(f"{folder_2_check}/{comp}_FAILED"):
                    sim_failed[f'fe_{pose}_{comp_folder}_{comp}batch'] = True
                    pose_failed[pose] = True

        self._sim_finished = sim_finished
        self._sim_failed = sim_failed
        self._pose_failed = pose_failed
        # if all are finished, return False
        if any(self._sim_failed.values()):
            logger.error(f'Free energy calculation failed: {self._sim_failed}')
            if self._fail_on_error:
                raise RuntimeError(f'Free energy calculation failed in pose {self._pose_failed}')
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
        
        not_finished = [k for k, v in self.sim_finished.items() if not v]

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
    def generate_batch_files(self,
                            remd=False,
                            time_limit=50, # 50 minutes
                            num_fe_sim=10, # run each window for 10 times (restart).
                            run_mcmd=False, # whether to run mcmd for water dynamics
                            ):
        """
        Generate the batch-run files for the system
        with the option to run them with ACES (H-REMD).

        Specially, it will generate input files so that they can
        run from the fe folder.
        """
        self._generate_batch_fe_equilibration()
        self._generate_batch_fe(remd=remd,
                                time_limit=time_limit,
                                num_fe_sim=num_fe_sim,
                                run_mcmd=run_mcmd)

    def _generate_batch_equilibration(self):
        """
        Generate the batch files for the equilibration stage
        to run them in a bundle.
        """
        # TODO: implement this function
        # The problem is that there's a time restriction of 2 hours
        # to run jobs in clusters e.g. Frontier.
        # Thus the equilibration need to be split into smaller jobs
        pass

    def _generate_batch_fe_equilibration(self):
        """
        Generate the batch files for the free energy calculation equilibration stage.
        """
        poses_def = self.bound_poses
        components = self.sim_config._components

        sim_stages = [
                'mini_eq.in',
                'eqnpt0.in',
                'eqnpt.in_00',
                'eqnpt.in_01', 'eqnpt.in_02',
                'eqnpt.in_03', 'eqnpt.in_04',
        ]
        file_name_map = {
            'mini_eq.in': 'mini',
            'eqnpt0.in': 'eqnpt_pre',
            'eqnpt.in_00': 'eqnpt00',
            'eqnpt.in_01': 'eqnpt01',
            'eqnpt.in_02': 'eqnpt02',
            'eqnpt.in_03': 'eqnpt03',
            'eqnpt.in_04': 'eqnpt04',
        }
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
                        stage_basename = stage.split('.')[0]
                        mdinput = f'fe/{win_eq_sim_folder_name}/{stage_basename}.in'
                        with open(mdinput, 'r') as infile:
                            input_lines = infile.readlines()
                            new_mdinput = f'{mdinput}_batch'
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
                            f.write(
                                f'-O -i {win_eq_sim_folder_name}/{stage_basename}.in_batch -p {prmtop} -c {stage_previous} '
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
            logger.info('FE EQ groupfiles generated for all poses')


    def _generate_batch_fe(self,
                           remd: bool = False,
                           time_limit: int = 50, # 50 minutes
                           num_fe_sim: int = 10, # run each window for 10 times (restart).
                           num_gpus: int = 4, # number of GPUs per node to be used for REMD simulations
                           num_nodes: int = 1, # number of nodes to be used for REMD simulations
                           run_mcmd: bool = False, # whether to run MC-MD water exchange
                           ):
        """
        Generate the batch files for the free energy calculation production stage.
        """
        if not remd and run_mcmd:
            raise ValueError("MC-MD water exchange can only be used with REMD simulations.")
        
        components = self.sim_config._components

        sim_stages = [
                'mini.in',
                'mdin.in', 'mdin.in.extend'
        ]
        
        def calculate_performance(n_atoms, comp, n_gpus_per_job):
            if comp not in COMPONENTS_DICT['dd']:
                return 150 * n_gpus_per_job if n_atoms < 80000 else 80 * n_gpus_per_job
            else:
                return 80 * n_gpus_per_job if n_atoms < 80000 else 40 * n_gpus_per_job
        

        def write_2_pose(pose, mol):
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

                            new_mdinput_path = f'fe/{sim_folder_name}/{stage.split("_")[0]}_batch'

                            with open(new_mdinput_path, 'w') as outfile:
                                for line in input_lines:
                                    if 'imin' in line:
                                        # add MC-MD water exchange
                                        #if stage == 'mdin.in' or stage == 'mdin.in.extend':
                                        if stage == 'mdin.in.extend':
                                            if run_mcmd:
                                                outfile.write(
                                                    '  mcwat = 1,\n'
                                                    '  nmd = 100,\n'
                                                    '  nmc = 1000,\n'
                                                    f"  mcwatmask = ':{mol}',\n"
                                                    '  mcligshift = 30,\n'
                                                    '  mcresstr = "WAT",\n'
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
                                                performance = calculate_performance(n_atoms, component, n_gpus_per_job=num_gpus / n_sims)
                                                n_steps = int(time_limit / 60 / 24 * performance * 1000 * 1000 / 4)
                                                n_steps = int(n_steps // 100000 * 100000)
                                                n_steps = 500000
                                                if stage == 'mdin.in':
                                                    n_steps = n_steps // 10
                                                line = f'  nstlim = {n_steps},\n'
                                    outfile.write(line)

                            f.write(f'# {component} {i} {stage}\n')
                            if stage == 'mdin.in':
                                f.write(f'-O -i {sim_folder_name}/mdin.in_batch -p {prmtop} -c {sim_folder_name}/mini.in.rst7 '
                                        f'-o {sim_folder_name}/mdin-00.out -r {sim_folder_name}/mdin-00.rst7 -x {sim_folder_name}/mdin-00.nc '
                                        f'-ref {sim_folder_name}/mini.in.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-00.log '
                                        f'-e {sim_folder_name}/mdin-00.mden\n')
                            elif stage == 'mdin.in.extend':
                                f.write(f'-O -i {sim_folder_name}/mdin.in.extend_batch -p {prmtop} -c {sim_folder_name}/mdin-CURRNUM.rst7 '
                                        f'-o {sim_folder_name}/mdin-NEXTNUM.out -r {sim_folder_name}/mdin-NEXTNUM.rst7 -x {sim_folder_name}/mdin-NEXTNUM.nc '
                                        f'-ref {sim_folder_name}/mini.in.rst7 -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-NEXTNUM.log '
                                        f'-e {sim_folder_name}/mdin-NEXTNUM.mden\n')
                            else:
                                f.write(
                                    f'-O -i {sim_folder_name}/{stage.split("_")[0]}_batch -p {prmtop} -c {stage_previous.replace("REPXXX", f"{i:02d}")} '
                                    f'-o {sim_folder_name}/{stage}.out -r {sim_folder_name}/{stage}.rst7 -x {sim_folder_name}/{stage}.nc '
                                    f'-ref {stage_previous.replace("REPXXX", f"{i:02d}")} -inf {sim_folder_name}/{stage}.mdinfo -l {sim_folder_name}/{stage}.log '
                                    f'-e {sim_folder_name}/{stage}.mden\n'
                                )
                            if stage == 'mdin.in':
                                all_replicates[component].append(f'{sim_folder_name}')
                        stage_previous = f'{sim_folder_temp}REPXXX/{stage}.rst7'

                # generate the mdin.in.extend groupfiles for num_fe_sim files
                temp_file = f'{pose_name}/groupfiles/{component}_mdin.in.extend.groupfile'
                with open(temp_file, 'r') as infile:
                    input_lines = infile.readlines()
                for i in range(num_fe_sim):
                    # replace temp_file NEXTNUM to i+1
                    # replace CURRNUM to i
                    new_temp_file = f'{pose_name}/groupfiles/{component}_mdin.in.stage{i+1:02d}.groupfile'
                    with open(new_temp_file, 'w') as outfile:
                        for line in input_lines:
                            if 'mdin-NEXTNUM' in line:
                                line = line.replace('mdin-NEXTNUM', f'mdin-{i+1:02d}')
                            if 'mdin-CURRNUM' in line:
                                line = line.replace('mdin-CURRNUM', f'mdin-{i:02d}')
                            outfile.write(line)
            return all_replicates

        all_replicates = []

        with self._change_dir(self.output_dir):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(tqdm(executor.map(write_2_pose, self.bound_poses, self.bound_mols),
                                    total=len(self.bound_poses),
                                    desc='Generating production groupfiles'))
                all_replicates.extend(results)

            logger.debug(all_replicates)

            # Write SLURM run files into FE folder
            os.makedirs(f'{self.fe_folder}/batch_run', exist_ok=True)
            shutil.copytree(
                batch_files_orig,
                f'{self.fe_folder}/batch_run',
                dirs_exist_ok=True
            )
            with open(f'{self.fe_folder}/batch_run/run-local-batch.bash', "rt") as f:
                fin = f.readlines()
            
            with open(f'{self.fe_folder}/batch_run/run-local-batch.bash', "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_fe_sim)))
            with open(f'{self.fe_folder}/batch_run/SLURMM-BATCH-Am', "rt") as f:
                fin = f.readlines()
            for pose in self.bound_poses:
                for comp in self.sim_config._components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
                    num_windows = len(self.component_windows_dict.get(comp, []))
                    with open(f"{self.fe_folder}/batch_run/SLURMM-run-{pose}-{comp}", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('STAGE', 'fe').replace(
                                            'SYSTEMNAME', self.system_name).replace(
                                            'POSE', pose).replace(
                                            'PARTITIONNAME', self.partition).replace(
                                            'NGPUS', str(num_gpus)).replace(
                                            'NNODES', str(num_nodes)).replace(
                                            'PFOLDERXXX', pose).replace(
                                            'CFOLDERXXX', comp_folder).replace(
                                            'COMPXXX', comp).replace(
                                            'NWINDOWSXXX', str(num_windows)).replace(
                                            'REMDXXX', '1' if remd else '0')
                            )

            for pose in self.bound_poses:
                for comp in self.sim_config._components:
                    comp_folder = COMPONENTS_FOLDER_DICT[comp]
            
            if 'z' in self.sim_config._components or 'o' in self.sim_config._components:
                # add lambda.sch to the folder
                with open(f'{self.fe_folder}/lambda.sch', 'w') as f:
                    f.write('TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n')
  
            logger.info('FE production groupfiles generated for all poses')
    
    def check_sim_stage(self, output='image', min_file_size=100, max_workers=16):
        """
        Check the simulation stage for each pose and component.

        Parameters
        ----------
        output : str
            'text', 'image', or 'both'
        min_file_size : int
            Minimum file size (in bytes) to count a .rst7 file as valid
        max_workers : int
            Number of threads to use for directory scanning
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def sort_key(x):
            return int(os.path.splitext(os.path.basename(x))[0][-2:])
        def latest_stage(folder):
            try:
                entries = os.listdir(folder)
            except FileNotFoundError:
                return -1

            max_idx = -1
            for name in entries:
                if name.endswith('.rst7') and name.startswith('md') and len(name) >= 8:
                    try:
                        stage_idx = sort_key(name)
                        full_path = os.path.join(folder, name)
                        if os.path.getsize(full_path) > min_file_size:
                            max_idx = max(max_idx, stage_idx)
                    except (ValueError, FileNotFoundError, OSError):
                        continue
            return max_idx

        def stage_task(pose, comp, win):
            folder = f'{self.fe_folder}/{pose}/{comp_type[comp]}/{comp}{win:02d}'
            return latest_stage(folder)

        comp_type = {
            comp: ('rest' if comp in ['m', 'n'] else 'sdr')
            for comp in self.sim_config._components
            if comp in ['m', 'n'] or comp in COMPONENTS_DICT['dd']
        }

        # Prepare all jobs
        jobs = []
        for pose in self.bound_poses:
            for comp in comp_type:
                for win in range(len(self.component_windows_dict.get(comp, []))):
                    jobs.append((pose, comp, win))

        results = defaultdict(lambda: defaultdict(list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(stage_task, pose, comp, win): (pose, comp)
                for (pose, comp, win) in jobs
            }
            for future in as_completed(future_to_job):
                pose, comp = future_to_job[future]
                stage = future.result()
                results[pose][comp].append(stage)

        # Collect min stage per component per pose
        final_stages = {
            pose: {comp: min(vals) if vals else -1 for comp, vals in comps.items()}
            for pose, comps in results.items()
        }

        stage_df = pd.DataFrame(final_stages)

        # Output
        if output == 'text':
            print(stage_df.to_string())

        if output == 'image':
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(max(6, len(self.bound_poses)), 5))
            sns.heatmap(stage_df, ax=ax, annot=True, cmap='viridis')
            plt.title('Simulation Stages for Each Pose and Component')
            plt.tight_layout()
            plt.show()

        #return stage_df

    def _save_state(self):
        """
        Save the state of the system to a file.
        """
        with open(f"{self.output_dir}/system.pkl", 'wb') as f:
            pickle.dump(self, f)

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
            os.system(f'cp {self.output_dir}/mols.txt .')
        
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
                    'cv.in', 'disang.rest', 'restraints.in', 'mini.rst7']

            for pose in tqdm(self.all_poses, desc='Copying files'):
                for comp in self.sim_config._components:
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
    
    @component_windows_dict.setter
    def component_windows_dict(self, value):
        """
        Set the component windows dictionary
        """
        if not isinstance(value, ComponentWindowsDict):
            raise ValueError("component_windows_dict must be an instance of ComponentWindowsDict")
        self._component_windows_dict = value
        self._save_state()

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
    
    @property
    def mols(self):
        try:
            return self._mols
        except AttributeError:
            mols = []
            with open(f'{self.output_dir}/mols.txt', 'r') as f:
                for line in f:
                    mols.append(line.strip().split('\t')[1])
            self._mols = mols
            return self._mols
    
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
        
    @property
    def sim_finished(self):
        """
        Get the simulation finished status.
        """
        if not hasattr(self, '_sim_finished'):
            self._sim_finished = {}
        return self._sim_finished
    
    @property
    def sim_failed(self):
        """
        Get the simulation failures status.
        """
        if not hasattr(self, '_sim_failed'):
            self._sim_failed = {}
        return self._sim_failed
    
    @property
    def pose_failed(self):
        """
        Get the pose failures status.
        """
        if not hasattr(self, '_pose_failed'):
            self._pose_failed = {}
        return self._pose_failed
    
    def dump(self, location=None):
        """
        Dump the system information to a json file.
        It includes
        - batter version
        - system info (protein and ligand)
        - simulation config
        - results
        """
        json_dict = {}
        json_dict['fe_results'] = {name: result.to_dict() if result is not None else None for name, result in self.fe_results.items()}
        json_dict['batter_version'] = __version__
        json_dict['system_name'] = self.system_name
        json_dict['protein_input'] = self.protein_input
        json_dict['pose_ligand_dict'] = self.pose_ligand_dict
        json_dict['ligands_process'] = {name: ligand.to_dict() for name, ligand in self._ligand_objects.items()}
        json_dict['output_dir'] = self.output_dir
        json_dict['sim_config'] = self.sim_config.to_dict()
        # dump restraints
        if self.rmsf_restraints is not None:
            json_dict['rmsf_restraints'] = {name: rest.to_dict() for name, rest in self.rmsf_restraints.items()}
        if self.extra_restraints is not None:
            json_dict['extra_restraints'] = self.extra_restraints
        if self.extra_conformation_restraints is not None:
            json_dict['extra_conformation_restraints'] = self.extra_conformation_restraints

        output_dir = self.output_dir if location is None else location
        if location is not None:
            os.makedirs(location, exist_ok=True)
        logger.info(f'Dumping system information to {output_dir}/manifest.json')
        with open(f'{output_dir}/manifest.json', 'w') as f:
            json.dump(json_dict, f, indent=4)

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
        raise NotImplementedError(
            'RBFESystem is not implemented yet. '
        )
        self._unique_ligand_paths = {}
        for ligand_path, ligand_name in zip(self.ligand_paths, self.ligand_names):
            if ligand_path not in self._unique_ligand_paths:
                self._unique_ligand_paths[ligand_path] = set()
            self._unique_ligand_paths[ligand_path].add(ligand_name)

        if len(self._unique_ligand_paths.keys()) < 2:
            raise ValueError("RBFESystem requires at least two ligands "
                             "for the relative binding free energy calculation")
        logger.info(f'Reference ligand: {self._unique_ligand_paths.keys()[0]}')


class MASFESystem(System):
    """
    A class to represent and process an Absolute solvation free energy (ASFE) system
    It doesn't include equil stage and no input of protein.
    """
    def _process_ligands(self):
        self._unique_ligand_paths = {}
        for ligand_path, ligand_name in zip(self.ligand_paths, self.ligand_names):
            if ligand_path not in self._unique_ligand_paths:
                self._unique_ligand_paths[ligand_path] = []
            self._unique_ligand_paths[ligand_path].append(ligand_name)
        logger.debug(f' Unique ligand paths: {self._unique_ligand_paths}')

    @safe_directory
    @save_state
    def create_system(
                    self,
                    system_name: str,
                    ligand_paths: Union[List[str], dict[str, str]],
                    retain_lig_prot: bool = True,
                    ligand_ph: float = 7.4,
                    ligand_ff: str = 'gaff2',
                    existing_ligand_db: Optional[str] = None,
                    overwrite: bool = False,
                    verbose: bool = False,
                    ):
        """
        Create a new single-ligand single-receptor system.

        Parameters
        ----------
        system_name : str
            The name of the system. It will be used to name the output folder.
        ligand_paths : List[str] or Dict[str, str]
            List of ligand files. It can be either PDB, mol2, or sdf format.
            It will be stored in the `all-poses` folder as `pose0.pdb`,
            `pose1.pdb`, etc.
            If it's a dictionary, the keys will be used as the ligand names
            and the values will be the ligand files.
        retain_lig_prot : bool, optional
            Whether to retain hydrogens in the ligand. Default is True.
        ligand_ph : float, optional
            pH value for protonating the ligand. Default is 7.4.
        ligand_ff : str, optional
            Parameter set for the ligand. Default is 'gaff2'.
            Options are 'gaff' and 'gaff2' and openff force fields.
            See https://github.com/openforcefield/openff-forcefields for full list.
        existing_ligand_db : str, optional
            The path to an existing ligand database to directly fetch
            ligand parameters. Default is None.
        overwrite : bool, optional
            Whether to overwrite the existing files. Default is False.
        verbose : bool, optional
            The verbosity of the output. If True, it will print the debug messages.
            Default is False.
        """
        self._eq_prepared = True
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

        if self._fe_prepared:
            if not overwrite:
                raise ValueError("The system has been prepared for equilibration or free energy simulations. "
                                 "Set overwrite=True to overwrite the existing system or skip `create_system` step.")
                                                 
        self._system_name = system_name
        
        # always store a unique identifier for the ligand
        if isinstance(ligand_paths, list):
            self.ligand_dict = {
                f'lig{i}': self._convert_2_relative_path(path)
                for i, path in enumerate(ligand_paths)
            }
        elif isinstance(ligand_paths, dict):
            self.ligand_dict = {ligand_name: self._convert_2_relative_path(ligand_path) for ligand_name, ligand_path in ligand_paths.items()}
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.ligand_ff = ligand_ff
        self.overwrite = overwrite

        self._membrane_simulation = False
        self._protein_input = 'no_protein'

        for ligand_path in self.ligand_paths:
            if not os.path.exists(ligand_path):
                raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
                
        logger.info(f"# {len(self.ligand_paths)} ligands.")
        self._process_ligands()

        os.makedirs(f"{self.poses_folder}", exist_ok=True)
        os.makedirs(f"{self.ligandff_folder}", exist_ok=True)
        
        # copy dummy atom parameters to the ligandff folder
        os.system(f"cp {build_files_orig}/dum.mol2 {self.ligandff_folder}")
        os.system(f"cp {build_files_orig}/dum.frcmod {self.ligandff_folder}")

        from openff.toolkit.typing.engines.smirnoff.forcefield import get_available_force_fields
        available_amber_ff = ['gaff', 'gaff2']
        available_openff_ff = [ff.removesuffix(".offxml") for ff in get_available_force_fields() if 'openff' in ff]
        if ligand_ff not in available_amber_ff + available_openff_ff:
            raise ValueError(f"Unsupported force field: {ligand_ff}. "
                             f"Supported force fields are: {available_amber_ff + available_openff_ff}")

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
            self._ligand_objects[ligand_name] = ligand

            mols.append(ligand.name)
            self.unique_mol_names.append(ligand.name)
            if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/{ligand.name}.frcmod"):
                # try to fetch from existing ligand db
                if existing_ligand_db is not None:
                    fetched = ligand.fetch_from_existing_db(existing_ligand_db)
                    if fetched:
                        logger.info(f"Fetched parameters for {ligand.name} from existing ligand database {existing_ligand_db}")
                    else:
                        logger.info(f"No parameters found for {ligand.name} in existing ligand database {existing_ligand_db}. Generating new parameters.")
                        ligand.prepare_ligand_parameters()
                else:
                    logger.info(f"Generating parameters for {ligand.name} using {self.ligand_ff} force field.")
                    ligand.prepare_ligand_parameters()
            for ligand_name in ligand_names:
                self.ligand_dict[ligand_name] = self._convert_2_relative_path(f'{self.ligandff_folder}/{ligand.name}.pdb')

        logger.debug( f"Unique ligand names: {self.unique_mol_names} ")
        logger.debug('updating the ligand paths')
        logger.debug(self.ligand_dict)

        self._mols = mols
        # update self.mols to output_dir/mols.txt
        with open(f"{self.output_dir}/mols.txt", 'w') as f:
            for ind, (ligand_path, ligand_names) in enumerate(self._unique_ligand_paths.items()):
                f.write(f"pose{ind}\t{self._mols[ind]}\t{ligand_path}\t{ligand_names}\n")
        
        # mock alignment to the 0,0,0
        self.mobile_coord = np.array([0.0, 0.0, 0.0])
        self.ref_coord = np.array([0.0, 0.0, 0.0])
        self.mobile_com = np.array([0.0, 0.0, 0.0])
        self.ref_com = np.array([0.0, 0.0, 0.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        self._prepare_ligand_poses()

        self.lipid_mol = []
        self.receptor_ff = 'protein.ff14SB'
        self.lipid_ff = 'lipid21' 
        logger.info('System loaded and prepared')

    @safe_directory
    @save_state
    def run_pipeline(self,
                     input_file: Union[str, Path, SimulationConfig] = None,
                     overwrite: bool = False,              
                     only_fe_preparation: bool = False,
                     dry_run: bool = False,
                     partition: str = 'owners',
                     max_num_jobs: int = 2000,
                     time_limit: str = '6:00:00',
                     fail_on_error: bool = False,
                     vmd: str = None,
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
        only_fe_preparation : bool, optional
            Whether to prepare the files for the production stage
            without running the production stage.
            Default is False.
        dry_run : bool, optional
            Whether to run the pipeline until performing any
            simulation submissions. Default is False.
        partition : str, optional
            The partition to submit the job.
            Default is 'rondror'.
        max_num_jobs : int, optional
            The maximum number of jobs to submit at a time.
            Default is 2000.
        time_limit : str, optional
            The time limit for the job submission.
            Default is '6:00:00'.
        fail_on_error : bool, optional
            Whether to fail the pipeline on error during simulations.
            Default is False with the failed simulations marked as 'FAILED'.
        vmd : str, optional
            The path to the VMD executable. If not provided,
            the code will use `vmd`.
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
        self._fail_on_error = fail_on_error

        start_time = time.time()
        logger.info(f'Start time: {time.ctime()}')
        if input_file is not None:
            self._get_sim_config(input_file)
        else:
            if not hasattr(self, 'sim_config'):
                raise ValueError('Input file is not provided and sim_config is not set.')
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]
        if vmd:
            if not os.path.exists(vmd):
                raise FileNotFoundError(f'VMD executable {vmd} not found')
            # set batter.utils.vmd to the provided path
            import batter.utils
            batter.utils.vmd = vmd
            logger.info(f'Setting VMD path to {vmd}')

        self._bound_poses = self._all_poses
        self._bound_mols = self.mols

        #4.0, submit the free energy equilibration

        remd = self.sim_config.remd == 'yes'
        self.batch_mode = remd
        num_fe_sim = self.sim_config.num_fe_range

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
            )
            if not os.path.exists(f'{self.fe_folder}/pose0/groupfiles') or overwrite:
                logger.info('Generating batch run files...')
                self.generate_batch_files(remd=remd, num_fe_sim=num_fe_sim)

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
            pbar = tqdm(total=len(self.bound_poses) * len(self.sim_config._components),
                        desc="FE Equilibration sims finished",
                        unit="job")
            while self._check_fe_equil():
                # get finishd jobs
                n_finished = len([k for k, v in self.sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE Equilibration sims finished"
                pbar.set_description(desc)

                #logger.info(f'{time.ctime()} - Finished jobs: {n_finished} / {len(self.sim_finished)}')
                not_finished = [k for k, v in self.sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]
                # name f'{self.fe_folder}/{pose}/{comp_folder}/{comp}{j:02d}

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                # exclude the failed jobs
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(5*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config._components) - pbar.n)  # update to total
            pbar.set_description('FE equilibration finished')
            pbar.close()

        else:
            logger.info('Free energy equilibration simulations are already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        # copy last equilibration snapshot to the free energy folder
        for pose in self.bound_poses:
            for comp in self.sim_config._components:
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

        if all([self.pose_failed.get(pose, False) for pose in self.bound_poses]):
            logger.warning('All bound poses failed in FE equilibration. '
                           'Please check the FE equilibration results.')
            return
        if not os.path.exists(f'{self.fe_folder}/pose0/groupfiles') or overwrite:
            logger.info('Generating batch run files...')
            self.generate_batch_files(remd=remd, num_fe_sim=num_fe_sim)

        if self._check_fe():
            logger.info('Submitting the free energy calculation')
            if dry_run:
                logger.info('Dry run is set to True. '
                            'Skipping the free energy submission.')
                return
            
            if self.batch_mode:
                logger.info('Running free energy calculation with REMD in batch mode')
                self.submit(
                    stage='fe',
                    batch_mode=True,
                    partition=partition,
                    time_limit=time_limit,
                )
            else:
                logger.info('Running free energy calculation without REMD')
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
                n_finished = len([k for k, v in self.sim_finished.items() if v])
                pbar.update(n_finished - pbar.n)
                now = time.strftime("%m-%d %H:%M:%S")
                desc = f"{now} – FE simulations finished"
                pbar.set_description(desc)
                #logger.info(f'{time.ctime()} Finished jobs: {n_finished} / {len(self.sim_finished)}')
                not_finished = [k for k, v in self.sim_finished.items() if not v]
                failed = [k for k, v in self._sim_failed.items() if v]

                not_finished_slurm_jobs = [job for job in self._slurm_jobs.keys() if job in not_finished]
                not_finished_slurm_jobs = [job for job in not_finished_slurm_jobs if job not in failed]
                for job in not_finished_slurm_jobs:
                    self._continue_job(self._slurm_jobs[job])
                time.sleep(10*60)
            pbar.update(len(self.bound_poses) * len(self.sim_config._components) - pbar.n)  # update to total
            pbar.set_description('FE calculation finished')
            pbar.close()
        else:
            logger.info('Free energy calculation is already finished')
            logger.info(f'If you want to have a fresh start, remove {self.fe_folder} manually')

        #5 analyze the results
        logger.info('Analyzing the results')
        self._generate_aligned_pdbs()

        num_sim = self.sim_config.num_fe_range
        # exclude the first few simulations for analysis
        start_sim = 3 if num_sim >= 6 else 1
        self.analysis(
            load=True,
            check_finished=False,
            sim_range=(start_sim, -1),
        )
        logger.info(f'The results are in the {self.output_dir}')
        logger.info(f'Results')
        # print out self.output_dir/Results/Results.dat
        with open(f'{self.output_dir}/Results/Results.dat', 'r') as f:
            results = f.readlines()
            for line in results:
                logger.info(line.strip())
        logger.info('Pipeline finished')
        end_time = time.time()
        logger.info(f'End time: {time.ctime()}')
        total_time = end_time - start_time
        logger.info(f'Total time: {total_time:.2f} seconds')
        # dump the state at the end of the pipeline
        # to the system folder
        self.dump()

    @safe_directory
    @save_state
    def prepare(self,
            stage: str,
            input_file: Union[str, Path, SimulationConfig] = None,
            overwrite: bool = False,
            partition: str = 'rondror',
            n_workers: int = 12,
            win_info_dict: dict = None,
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
        
        self._all_poses = [f'pose{i}' for i in range(len(self.ligand_paths))]
        self._pose_ligand_dict = {pose: ligand for pose, ligand in zip(self._all_poses, self.ligand_names)}
        self.sim_config.poses_list = self._all_poses 

        if stage == 'equil':
            raise ValueError("Equilibration stage is not needed for ASFE system.")
        if stage == 'fe':
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

            self._prepare_fe_system()
            logger.info('FE System prepared')
            self._fe_prepared = True
    
    def _prepare_fe_equil_system(self):

        # molr (molecule reference) and poser (pose reference)
        # are used for exchange FE simulations.
        sim_config = self.sim_config
        molr = self.mols[0]
        poser = self.bound_poses[0]
        builders = []
        builders_factory = BuilderFactory()
        for pose in sim_config.poses_list:
            logger.debug(f'Preparing pose: {pose}')
            
            sim_config_pose = sim_config.copy(deep=True)
            os.makedirs(f"{self.fe_folder}/{pose}", exist_ok=True)
            #os.makedirs(f"{self.fe_folder}/{pose}/ff", exist_ok=True)
            #for file in os.listdir(self.ligandff_folder):
            #    shutil.copy(f"{self.ligandff_folder}/{file}",
            #                f"{self.fe_folder}/{pose}/ff/{file}")
            # create softlink to the ff folder
            if not os.path.exists(f"{self.fe_folder}/{pose}/ff"):
                os.symlink(f"{self.fe_folder}/ff",
                           f"{self.fe_folder}/{pose}/ff")
            
            for component in sim_config._components:
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
                    pose=pose,
                    sim_config=sim_config_pose,
                    component_windows_dict=self.component_windows_dict,
                    working_dir=f'{self.fe_folder}',
                    molr=molr,
                    poser=poser,
                    infe = (self.rmsf_restraints is not None) or (self.extra_conformation_restraints is not None)
                )
                builders.append(fe_eq_builder)
        if len(builders) == 0:
            logger.info('No new FE equilibration systems to build.')
            return
        with tqdm_joblib(tqdm(
            total=len(builders),
            desc="Preparing FE equilibration",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")) as pbar:
            n_workers = min(self.n_workers, len(builders))
            Parallel(n_jobs=n_workers, backend='loky')(
                delayed(builder.build)() for builder in builders
        )

    def _generate_aligned_pdbs(self):
        os.makedirs(f'{self.output_dir}/Results', exist_ok=True)

        for pose in tqdm(self.bound_poses, desc='Generating aligned pdbs'):
            initial_pose = f'{self.poses_folder}/{pose}.pdb'
            os.system(f'cp {initial_pose} {self.output_dir}/Results/init_{pose}.pdb')
            

class ComponentWindowsDict(MutableMapping):
    def __init__(self, system):
        self._data = {}
        self._sim_config = system.sim_config

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
    
    @property
    def sim_config(self):
        """
        Get the simulation configuration from the system.
        """
        try:
            return self._sim_config
        except AttributeError:
            # old API use system instead
            return self.system.sim_config
        
    @sim_config.setter
    def sim_config(self, sim_config):
        """
        Set the simulation configuration for the component windows.
        """
        self._get_sim_config(sim_config)
    

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


def analyze_single_pose_dask_wrapper(pose, mol, input_dict):
    from distributed import get_worker
    logger.info(f"Running on worker: {get_worker().name}")
    logger.info(f'Analyzing pose: {pose}')
    logger.info(f'Input: {input_dict}')
    fe_folder = input_dict.get('fe_folder')
    components = input_dict.get('components')
    rest = input_dict.get('rest')
    temperature = input_dict.get('temperature')
    water_model = input_dict.get('water_model')
    rocklin_correction = input_dict.get('rocklin_correction', False)
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
        water_model=water_model,
        rocklin_correction=rocklin_correction,
        component_windows_dict=component_windows_dict,
        sim_range=sim_range,
        raise_on_error=raise_on_error,
        mol=mol,
        n_workers=n_workers
    )
    logger.info(f'Finished analyzing pose: {pose}')
    return pose


def analyze_pose_task(
                fe_folder: str,
                pose: str,
                components: List[str],
                rest: Tuple[float, float, float, float, float],
                temperature: float,
                water_model: str,
                component_windows_dict: "ComponentWindowsDict",
                rocklin_correction: bool = False,
                sim_range: Tuple[int, int] = None,
                raise_on_error: bool = True,
                mol: str = 'LIG',
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
    water_model : str
        The water model used in the simulation.
    rocklin_correction : bool
        Whether to apply the Rocklin correction for charged ligands.
    sim_range : tuple
        The range of simulations to analyze.
        If files are missing from the range, the analysis will fail.
    raise_on_error : bool
        Whether to raise an error if the analysis fails.
    mol : str
        The name of the ligand molecule.
    n_workers : int
        The number of workers to use for parallel processing.
        Default is 4.
    """
    from batter.analysis.analysis import BoreschAnalysis, MBARAnalysis, RESTMBARAnalysis

    pose_path = f'{fe_folder}/{pose}'
    os.makedirs(f'{pose_path}/Results', exist_ok=True)
    
    results_entries = []
    LEN_FE_TIMESERIES = 10
    try:

        fe_values = []
        fe_stds = []
        fe_timeseries = {}

        # first get analytical results from Boresch restraint
        try:
            if 'v' in components:
                disangfile = f'{fe_folder}/{pose}/sdr/v-1/disang.rest'
            elif 'o' in components:
                disangfile = f'{fe_folder}/{pose}/sdr/o-1/disang.rest'
            elif 'z' in components:
                disangfile = f'{fe_folder}/{pose}/sdr/z-1/disang.rest'
            else:
                raise ValueError('No Boresch needed')

            k_r = rest[2]
            k_a = rest[3]
            bor_ana = BoreschAnalysis(
                                disangfile=disangfile,
                                k_r=k_r, k_a=k_a,
                                temperature=temperature)
            bor_ana.run_analysis()
            fe_values.append(COMPONENT_DIRECTION_DICT['Boresch'] * bor_ana.results['fe'])
            fe_stds.append(bor_ana.results['fe_error'])

            # constant Boresch restraint value
            fe_timeseries['Boresch'] = np.asarray([bor_ana.results['fe'], 0])

            results_entries.append(
                f'Boresch\t{COMPONENT_DIRECTION_DICT["Boresch"] * bor_ana.results["fe"]:.2f}\t{bor_ana.results["fe_error"]:.2f}'
            )
        except ValueError as e:
            pass
        
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
                                        title=f'Convergence for {comp} {mol}',
                )

                fe_values.append(COMPONENT_DIRECTION_DICT[comp] * mbar_ana.results['fe'])
                fe_stds.append(mbar_ana.results['fe_error'])
                fe_timeseries[comp] = mbar_ana.results['fe_timeseries']

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
                                        title=f'Convergence for {comp} {mol}',
                )

                fe_values.append(COMPONENT_DIRECTION_DICT[comp] * rest_mbar_ana.results['fe'])
                fe_stds.append(rest_mbar_ana.results['fe_error'])
                fe_timeseries[comp] = rest_mbar_ana.results['fe_timeseries']

                results_entries.append(
                    f'{comp}\t{COMPONENT_DIRECTION_DICT[comp] * rest_mbar_ana.results["fe"]:.2f}\t{rest_mbar_ana.results["fe_error"]:.2f}'
                )
    
        # calculate total free energy
        fe_value = np.sum(fe_values)
        fe_std = np.sqrt(np.sum(np.array(fe_stds)**2))
        # get the time series for the total free energy

        fe_timeseries_fe_value = np.zeros(LEN_FE_TIMESERIES)
        fe_timeseries_std = np.zeros(LEN_FE_TIMESERIES)
        for comp, timeseries in fe_timeseries.items():
            direction = COMPONENT_DIRECTION_DICT[comp]
            if timeseries.ndim == 1:
                fe_timeseries_fe_value += timeseries[0] * direction
                fe_timeseries_std += np.zeros(LEN_FE_TIMESERIES)
            else:
                fe_timeseries_fe_value += timeseries[:, 0] * direction
                fe_timeseries_std += timeseries[:, 1] ** 2

        fe_timeseries_std = np.sqrt(fe_timeseries_std)

    except Exception as e:
        logger.error(f'Error during FE analysis for {pose}: {e}')
        if raise_on_error:
            raise e
        fe_value = np.nan
        fe_std = np.nan
        fe_timeseries_fe_value = np.zeros(LEN_FE_TIMESERIES) * np.nan
        fe_timeseries_std = np.zeros(LEN_FE_TIMESERIES) * np.nan

    if rocklin_correction == 'yes':
        from batter.analysis.rocklin import run_rocklin_correction
        # only implement for single charged ligand system
        # in component `y`
        if 'y' not in components:
            raise ValueError('Rocklin correction is only implemented for single charged ligand system in component `y`')
        
        universe = mda.Universe(
            f'{fe_folder}/{pose}/sdr/y-1/full.prmtop',
            f'{fe_folder}/{pose}/sdr/y-1/eq_output.pdb'
        )
        box = universe.dimensions[:3]
        lig_ag = universe.select_atoms(f'resname {mol}')
        if len(lig_ag) == 0:
            raise ValueError(f'No ligand atoms found in the system for Rocklin correction with resname {mol}')
        lig_netq = int(round(lig_ag.total_charge()))
        other_ag = universe.atoms - lig_ag
        other_netq = int(round(other_ag.total_charge()))
        if lig_netq == 0:
            logger.info('Neutral ligand found; skipping Rocklin correction.')
        else:
            corr = run_rocklin_correction(
                universe=universe,
                mol_name=mol,
                box=box,
                lig_netq=lig_netq,
                other_netq=other_netq,
                temp=temperature,
                water_model=water_model
            )
            # fe_value is negative of solvation free energy
            fe_value += corr
            results_entries.append(f'Rocklin\t{corr:.2f}\t0.00')
            fe_timeseries_fe_value += corr

    results_entries.append(
        f'Total\t{fe_value:.2f}\t{fe_std:.2f}'
    )
    with open(f'{fe_folder}/{pose}/Results/Results.dat', 'w') as f:
        f.write('\n'.join(results_entries))
    
    with open(f'{fe_folder}/{pose}/Results/fe_timeseries.json', 'w') as f:
        json.dump({
            'fe_value': fe_timeseries_fe_value.tolist(),
            'fe_std': fe_timeseries_std.tolist(),
        }, f)
    
    # plot fe_timeseries
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        np.arange(1, LEN_FE_TIMESERIES + 1) / LEN_FE_TIMESERIES * 100,
        fe_timeseries_fe_value,
        yerr=fe_timeseries_std,
        fmt='-o',
        capsize=5,
    )
    # plot horizontal line at fe_value with shaded area for 1 kcal/mol
    ax.axhline(fe_value, color='red', linestyle='--', label='FE value (±1 kcal/mol)')
    ax.fill_between(
            x=np.arange(1, LEN_FE_TIMESERIES + 1) / LEN_FE_TIMESERIES * 100,
            y1=fe_value - 1.0,
            y2=fe_value + 1.0,
            color='red',
            alpha=0.2,
        )
    ax.set_xlabel('Simulation Progress (%)')
    ax.set_ylabel('Free Energy (kcal/mol)')
    ax.set_title(f'Free Energy Timeseries for {mol}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{fe_folder}/{pose}/Results/fe_timeseries.png')
    plt.close(fig)
    return


def modify_restraint_mask(line, new_mask_component):
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
    return line