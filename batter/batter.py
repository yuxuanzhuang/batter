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
import numpy as np
import os
import sys
import shutil
import subprocess as sp
import tempfile
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types
from MDAnalysis.analysis import align
import pandas as pd
from importlib import resources
import json
from typing import Union
from pathlib import Path

from typing import List, Tuple
import loguru
from loguru import logger
from batter.input_process import SimulationConfig, get_configure_from_file
from batter.builder import EquilibrationBuilder
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

    TODO: what if there are multiple ligands from the input file?

    After the preparation of the equil system, run through the equilibration simulation and then
    prepare the fe system. The output of the equil system will be used as
    the input for the fe system.
    """

    def __init__(self,
                 system_name: str,
                 protein_input: str,
                 system_topology: str,
                 ligand_path: str,
                 receptor_segment: str = None,
                 system_coordinate: str = None,
                 protein_align: str = 'name CA and resid 60 to 250',
                 ligand_poses: List[str] = [],
                 output_dir: str = 'FEP',
                 retain_lig_prot: bool = True,
                 ligand_ph: float = 7.4,
                 ligand_ff: str = 'gaff2',
                 lipid_mol: List[str] = [],
                 lipid_ff: str = 'lipid21',
                 overwrite: bool = False,
                 ):
        """
        Initialize the FEPSystem class.

        Parameters
        ----------
        system_name : str
            The name of the system.
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
        ligand_path : str
            Path to the ligand file.
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
        ligand_poses : List[str], optional
            List of ligand poses to be included in the simulations.
            If it is empty, the pose from ligand_path will be used.
        output_dir : str
            Directory where output files will be saved.
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
        self.system_name = system_name
        self.protein_input = protein_input
        self.system_topology = system_topology
        self.system_coordinate = system_coordinate
        self.ligand_path = ligand_path
        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.ligand_poses = ligand_poses
        self.overwrite = overwrite

        # check input existence
        if not os.path.exists(protein_input):
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        if not os.path.exists(system_topology):
            raise FileNotFoundError(f"System input file not found: {system_topology}")
        if not os.path.exists(ligand_path):
            raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
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

        # set to absolute path
        self.output_dir = os.path.abspath(output_dir) + '/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        os.makedirs(f"{self.poses_folder}", exist_ok=True)
        os.makedirs(f"{self.ligandff_folder}", exist_ok=True)

        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.ligand_ff = ligand_ff
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

        if self.overwrite or not os.path.exists(f"{self.poses_folder}/{self.system_name}_docked.pdb") or not os.path.exists(f"{self.poses_folder}/reference.pdb"):
            self._get_alignment()
            self._process_system()
            # Process ligand and prepare the parameters
        if self.overwrite or not os.path.exists(f"{self.ligandff_folder}/ligand.frcmod"):
            self._process_ligand()
            self._prepare_ligand_poses()
        logger.info('System loaded and prepared')

    def _get_alignment(self):
        """
        Prepare for the alignment of the protein and ligand to the system.
        """
        logger.info('Getting the alignment of the protein and ligand to the system')
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
        logger.info('Processing the system')
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
        logger.info(f'Processing ligand file: {self.ligand_path}')

        ligand = mda.Universe(self.ligand_path)
        converted_path = f"{self.ligandff_folder}/ligand.pdb"
        ligand.atoms.write(converted_path)
        self.ligand_path = converted_path

        # retain hydrogens from the ligand
        if self.retain_lig_prot:
            # convert mol2 to get charge
            run_with_log(f"{obabel} -i pdb {self.ligand_path} -o mol2 -O {self.ligandff_folder}/ligand.mol2")

        else:
            # Remove hydrogens from the ligand
            noh_path = f"{self.ligandff_folder}/ligand_noh.pdb"
            run_with_log(f"{obabel} -i pdb {self.ligand_path} -o pdb -O {noh_path} -d")

            # Add hydrogens based on the specified pH
            run_with_log(f"{obabel} -i pdb {noh_path} -o pdb -O {self.ligandff_folder}/ligand.pdb -p {self.ligand_ph:.2f}")
            run_with_log(f"{obabel} -i pdb {noh_path} -o mol2 -O {self.ligandff_folder}/ligand.mol2 -p {self.ligand_ph:.2f}")

        self.ligand_path = f"{self.ligandff_folder}/ligand.pdb"
        self.ligand = mda.Universe(self.ligand_path)
        self.ligand_mol2_path = f"{self.ligandff_folder}/ligand.mol2"
        ligand_mol2 = mda.Universe(self.ligand_mol2_path)

        self.ligand_charge = np.round(np.sum(ligand_mol2.atoms.charges))
        logger.info(f'The babel protonation of the ligand is for pH {self.ligand_ph:.2f}')
        logger.info(f'The net charge of the ligand is {self.ligand_charge}')

        self._prepare_ligand_parameters()

    def _prepare_ligand_parameters(self):
        """Prepare ligand parameters for the system"""
        # Get ligand parameters
        logger.info('Preparing ligand parameters')
        # antechamber_command = f'{antechamber} -i {self.ligand_path} -fi pdb -o {self.ligandff_folder}/ligand_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge}'
        antechamber_command = f'{antechamber} -i {self.ligand_mol2_path} -fi mol2 -o {self.ligandff_folder}/ligand_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge}'
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(antechamber_command, working_dir=tmpdir)
        shutil.copy(f"{self.ligandff_folder}/ligand_ante.mol2", f"{self.ligandff_folder}/ligand.mol2")
        self.ligand_mol2_path = f"{self.ligandff_folder}/ligand.mol2"

        if self.ligand_ff == 'gaff':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/ligand_ante.mol2 -f mol2 -o {self.ligandff_folder}/ligand.frcmod -s 1')
        elif self.ligand_ff == 'gaff2':
            run_with_log(f'{parmchk2} -i {self.ligandff_folder}/ligand_ante.mol2 -f mol2 -o {self.ligandff_folder}/ligand.frcmod -s 2')

        with tempfile.TemporaryDirectory() as tmpdir:
            #    run_with_log(f'{antechamber} -i {self.ligand_path} -fi pdb -o {self.ligandff_folder}/ligand_ante.pdb -fo pdb', working_dir=tmpdir)
            run_with_log(
                f'{antechamber} -i {self.ligand_mol2_path} -fi mol2 -o {self.ligandff_folder}/ligand_ante.pdb -fo pdb', working_dir=tmpdir)

        # get lib file
        tleap_script = f"""
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_ff}
        lig = loadmol2 {self.ligandff_folder}/ligand.mol2
        loadamberparams {self.ligandff_folder}/ligand.frcmod
        saveoff lig {self.ligandff_folder}/ligand.lib
        saveamberparm lig {self.ligandff_folder}/ligand.prmtop {self.ligandff_folder}/ligand.inpcrd

        quit
        """
        with open(f"{self.ligandff_folder}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{tleap} -f tleap.in",
                        working_dir=self.ligandff_folder)

        logger.info('Ligand parameters prepared')

    def _prepare_ligand_poses(self):
        """
        Prepare ligand poses for the system.
        """
        logger.info('Preparing ligand poses')

        if not self.ligand_poses:
            self.ligand_poses = [self.ligand_path]

        new_pose_paths = []
        for i, pose in enumerate(self.ligand_poses):
            # align to the system
            u = mda.Universe(pose)
            try:
                u.atoms.chainIDs
            except AttributeError:
                u.add_TopologyAttr('chainIDs')
            lig_seg = u.add_Segment(segid='LIG')
            u.atoms.chainIDs = 'L'
            u.atoms.residues.segments = lig_seg
            
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

        self.ligand_poses = new_pose_paths

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
        logger.info('Input: membrane system')

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
        logger.info('Preparing the system')
        self.overwrite = overwrite

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
        
        if stage == 'equil':
            self.sim_config = sim_config
            # save the input file to the equil directory
            os.makedirs(f"{self.equil_folder}", exist_ok=True)
            with open(f"{self.equil_folder}/sim_config.json", 'w') as f:
                json.dump(sim_config.model_dump(), f, indent=2)
            
            self._prepare_equil_system()
            logger.info('Equil System prepared')
        
        if stage == 'fe':
            self.sim_config = sim_config
            if not os.path.exists(f"{self.equil_folder}"):
                raise FileNotFoundError(f"Equilibration not generated yet. Run prepare('equil') first.")
        
            if not os.path.exists(f"{self.equil_folder}/{self.sim_config.poses_def[0]}/md03.rst7"):
                raise FileNotFoundError(f"Equilibration not finished yet. First run the equilibration.")

            sim_config_eq = json.load(open(f"{self.equil_folder}/sim_config.json"))
            if sim_config_eq != sim_config.model_dump():
#                raise ValueError(f"Equilibration and free energy simulation configurations are different")
                warnings.warn(f"Equilibration and free energy simulation configurations are different")
                # get the difference
                diff = {k: v for k, v in sim_config_eq.items() if sim_config.model_dump().get(k) != v}
                logger.warning(f"Different configurations: {diff}")
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
        if self.overwrite:
            logger.info('Overwrite is set. Removing existing equilibration files')
            shutil.rmtree(self.equil_folder, ignore_errors=True)
        os.makedirs(f"{self.equil_folder}", exist_ok=True)

        logger.info('Prepare for equilibration stage')
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
            equil_builder = EquilibrationBuilder(
                system=self,
                pose_name=pose,
                sim_config=sim_config,
                working_dir=f'{self.equil_folder}',
            ).build()
    
        logger.info('Equilibration systems have been created for all poses listed in the input file.')
        logger.info(f'now cd equil/pose0')
        logger.info(f'sbatch SLURMM-run')

    def _prepare_fe_system(self):
        """
        Prepare the free energy system.
        """
        raise NotImplementedError("Free energy system preparation is not implemented yet")
        sim_config = self.sim_config
        if self.overwrite:
            logger.info('Overwrite is set. Removing existing free energy files')
            shutil.rmtree(self.fe_folder, ignore_errors=True)
        os.makedirs(f"{self.fe_folder}", exist_ok=True)

        logger.info('Prepare for free energy stage')

        for pose in self.sim_config.poses_def:
            logger.info(f'Preparing pose: {pose}')
            # copy ff folder
            shutil.copytree(self.ligandff_folder,
                            f"{self.fe_folder}/{pose}/ff")
            for component in sim_config.components:
                logger.info(f'Preparing component: {component}')
                lambdas_comp = sim_config[COMPONENTS_LAMBDA_DICT[component]]
                n_sims = len(lambdas_comp)
                for i, lambdas in enumerate(lambdas_comp):
                    fe_builder = FreeEnergyBuilder(
                        win=i,
                        component=component,
                        system=self,
                        pose_name=pose,
                        sim_config=sim_config,
                        working_dir=f'{self.fe_folder}',
                    ).build()
            

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