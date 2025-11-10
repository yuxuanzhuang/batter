"""
Convert list of trajectories into a single trajecotry and strip e.g. water molecules
"""

import numpy as np
import MDAnalysis as mda
import os
from loguru import logger
import glob
import click
import tempfile
from batter.utils import natural_keys
from MDAnalysis.analysis import align
from MDAnalysis.transformations.base import TransformationBase
import MDAnalysis.transformations as trans
from tqdm import tqdm
from joblib import Parallel, delayed



# From https://groups.google.com/g/mdnalysis-discussion/c/umDpvbCmQiE/m/FKtNClazAwAJ
# Author: Richard Gowers
class GroupHug(TransformationBase):
    def __init__(self, center, *others):
        super().__init__(max_threads=1, parallelizable=True)
        self.c = center
        self.o = others

    @staticmethod
    def calc_restoring_vec(ag1, ag2):
        box = ag1.dimensions[:3]
        dist = ag1.center_of_mass() - ag2.center_of_mass()

        return box * np.rint(dist / box)

    def _transform(self, ts):
        # loop over other atomgroups and shunt them into nearest image to
        # center
        for i in self.o:
            rvec = self.calc_restoring_vec(self.c, i)
            i.translate(+rvec)
        return ts


def combine_trajectories(
    topology,
    bonded_topology,
    trajectory_files,
    output_folder,
    protein_selection='not resname DUM TIP3 SOD CLA POPC WAT Na+ Cl- PA PC OL',
    skip=5,
    fix_pbc=True,
    ):
    """
    Combine multiple trajectory files into a single trajectory and strip unwanted molecules.
    The final trajectory will be saved into `output_folder/skip{skip}` folder.
    
    Parameters
    ----------
    topology : str
        Path to the topology file (e.g., PDB, PSF, PRMTOP).
    bonded_topology : str
        Path to the bonded topology file (e.g., PRMTOP for AMBER).
    trajectory_files : list of str
        List of paths to trajectory files (e.g., DCD, XTC).
    output_file : str
        Path to the output combined trajectory file.
    protein_selection : str
        Selection string to specify which atoms to keep (default excludes common solvent and ions).
    skip : int
        Number of frames to skip in each trajectory (default is 1).
    fix_pbc : bool
        Whether to fix pbc and align the trajectory frames (default is True).
    """

    u = mda.Universe(topology, trajectory_files)
    protein = u.select_atoms(protein_selection)

    if len(protein) == 0:
        raise ValueError("No protein atoms found with the given selection.")

    if fix_pbc:
        u_bond = mda.Universe(bonded_topology)
        u.add_bonds(u_bond.bonds.to_indices())
        
        u_prot = u.select_atoms(protein_selection)
        # protein chain could include some other molecules
        prot_chain_list = []
        for chain in u_prot.segments:
            prot_chain_list.append(chain.atoms)

        # reorder to make protein the first in the list
        for i, chain in enumerate(prot_chain_list):
            if chain.atoms.select_atoms('name CA').n_atoms > 0:
                prot_index = i
                prot_chain = chain
                break
        
        # if no protein chain found, raise an error
        if prot_index is None:
            raise ValueError("No protein chain found in the selection.")
        
        prot_chain_list.pop(prot_index)  # remove protein chain from the list

        prot_group = GroupHug(prot_chain, *prot_chain_list)
        unwrap = trans.unwrap(u.atoms)
        center_in_box = trans.center_in_box(u_prot)

        rot_fit_trans = trans.fit_rot_trans(
            u.select_atoms(f"name CA and ({protein_selection})"),
            u.select_atoms(f"name CA and ({protein_selection})")
        )

        non_prot = u.select_atoms(f"not {protein_selection}")
        wrap = trans.wrap(non_prot)
        u.trajectory.add_transformations(
            *[unwrap, prot_group, center_in_box, rot_fit_trans]
        )

    output_folder = f'{output_folder}/skip{skip}'
    os.makedirs(f'{output_folder}', exist_ok=True)
    with mda.Writer(f'{output_folder}/protein.xtc', protein.n_atoms) as W_prot, \
            mda.Writer(f'{output_folder}/system.xtc', u.atoms.n_atoms) as W_sys:
        for ts in tqdm(u.trajectory[::skip],
                          desc='Combining trajectories',
                          unit='frames',
                          total=len(u.trajectory[::skip])):
            W_prot.write(protein)
            W_sys.write(u.atoms)
    
    # write the first frame
    protein.write(f'{output_folder}/protein.pdb')
    u.atoms.write(f'{output_folder}/system.pdb')

    logger.info(f"Combined trajectory written to {output_folder}/protein.xtc and {output_folder}/system.xtc")


@click.command()
@click.option('--folder', type=click.Path(exists=True, file_okay=False), multiple=True, required=True,
              help='One or more folders containing trajectory files to process')
@click.option('--fix_pbc', is_flag=True, default=True, help='Fix pbc and align (default: True)')
@click.option('--skip', type=int, default=5, help='Number of frames to skip in each trajectory (default: 1)')
@click.option('--n_jobs', type=int, default=4, help='Number of parallel jobs to run (default: 4)')
def preprocess(folder, fix_pbc=True, skip=5, n_jobs=4):
    """
    Process trajectory files in one or more specified folders and combine them into single trajectories.
    """
    def process_one_folder(folder_path):
        logger.info(f"Processing trajectories in folder: {folder_path}")

        topology = os.path.join(folder_path, 'full.pdb')
        if not os.path.exists(topology):
            logger.error(f"Topology file not found: {topology}")
            return

        bonded_topology = os.path.join(folder_path, 'full.prmtop')
        if not os.path.exists(bonded_topology):
            logger.error(f"Bonded topology file not found: {bonded_topology}")
            return

        trajectories = glob.glob(f"{folder_path}/md*.nc")
        if not trajectories:
            logger.warning(f"No trajectory files found in {folder_path}")
            return

        trajectories.sort(key=natural_keys)
        logger.info(f"[{folder_path}] Found {len(trajectories)} trajectory files.")

        combine_trajectories(
            topology=topology,
            bonded_topology=bonded_topology,
            trajectory_files=trajectories,
            output_folder=folder_path,
            protein_selection='not resname DUM TIP3 SOD CLA POPC WAT Na+ Cl- PA PC OL',
            skip=skip,
            fix_pbc=fix_pbc
        )

        logger.info(f"[{folder_path}] Trajectory processing completed successfully.")

    Parallel(n_jobs=n_jobs)(delayed(process_one_folder)(f) for f in folder)
