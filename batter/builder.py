from abc import ABC, abstractmethod
from loguru import logger

import os
import shutil
import re
import glob
import pandas as pd
import numpy as np
import MDAnalysis as mda
from contextlib import contextmanager
import tempfile

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.data import build_files as build_files_orig
from batter.data import amber_files as amber_files_orig
from batter.data import run_files as run_files_orig
from batter.bat_lib import setup, analysis, scripts

from batter.utils import (
    run_with_log,
    antechamber,
    tleap,
    cpptraj,
    parmchk2,
    charmmlipid2amber,
    obabel
)


class SystemBuilder(ABC):
    stage = None
    win = None

    def __init__(self,
                 system: 'batter.System',
                 pose_name: str,
                 sim_config: SimulationConfig,
                 working_dir: str,
                 overwrite: bool = False
                 ):
        """
        The base class for all system builders.

        Parameters
        ----------
        system : batter.System
            The system to build.
        pose_name : str
            The name of the pose
        sim_config : batter.input_process.SimulationConfig
            The simulation configuration.
        working_dir : str
            The working directory.
        """
        self.system = system
        self.pose_name = pose_name
        self.sim_config = sim_config
        self.working_dir = working_dir
        self.overwrite = overwrite
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    def build(self):
        with self._change_dir(self.working_dir):
            logger.info(f'Building {self.pose_name}...')
            logger.debug(f'Working directory: {os.getcwd()}')
            with self._change_dir('build_files'):
                logger.debug(f'Copying build files to {os.getcwd()}')
                anchor_found = self._build_complex()
            if not anchor_found:
                return None
            with self._change_dir(self.pose_name):
                print(f'Building the system in {os.getcwd()}')
                self._create_simulation_dir()
                self._create_box()
            self._restraints()
            self._sim_files()
        return self

    @abstractmethod
    def _build_complex(self):
        """
        Build the complex.
        It involves 
        1. Cleanup the system.
        2. Set the parameters of the ligand (TODO: has it been done already?)
        3. Find anchor atoms.
        4. Add dummy atoms
        """
        raise NotImplementedError()

    @abstractmethod
    def _create_box(self):
        """
        Create the box.
        It involves
        1. Add ligand (that differs for different systems)
        2. Solvate the system.
        3. Add ions.
        4. For membrane systems, add lipids.
        """
        raise NotImplementedError()

    @abstractmethod
    def _restraints(self):
        """
        Add restraints.
        """
        raise NotImplementedError()

    @abstractmethod
    def _sim_files(self):
        """
        Create simulation files, e.g. input files form AMBER.
        """
        raise NotImplementedError()

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
    def ff_folder(self):
        return f'{self.working_dir}/ff'


class EquilibrationBuilder(SystemBuilder):
    stage = 'equil'
    win = 0

    def _build_complex(self):
        sdr_dist = 0
        self.sdr_dist = sdr_dist
        H1 = self.sim_config.H1
        H2 = self.sim_config.H2
        H3 = self.sim_config.H3
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis

        if os.path.exists(f'build_files') and not self.overwrite:
            return False
        else:
            shutil.rmtree(f'build_files', ignore_errors=True)
            shutil.copytree(build_files_orig, '.', dirs_exist_ok=True)

        # copy dum param to ff
        shutil.copy(f'dum.mol2', f'../ff/dum.mol2')
        shutil.copy(f'dum.frcmod', f'../ff/dum.frcmod')

        all_pose_folder = self.system.poses_folder
        system_name = self.system.system_name

        shutil.copy(f'{all_pose_folder}/reference.pdb',
                    f'reference.pdb')
        shutil.copy(f'{all_pose_folder}/{system_name}_docked.pdb',
                    f'rec_file.pdb')
        shutil.copy(f'{all_pose_folder}/{self.pose_name}.pdb',
                    f'.')

        other_mol = self.sim_config.other_mol
        lipid_mol = self.sim_config.lipid_mol
        solv_shell = self.sim_config.solv_shell
        mol_u = mda.Universe(f'{self.pose_name}.pdb')
        if len(set(mol_u.residues.resnames)) > 1:
            raise ValueError(f'The ligand {self.pose_name} has more than one residue: '
                             f'{mol_u.atoms.resnames}')
        self.mol = mol_u.residues[0].resname
        mol = self.mol
        if mol in other_mol:
            raise ValueError(f'The ligand {mol}'
                             f'cannot be in the other_mol list: '
                             f'{other_mol}')
        # copy pose
        shutil.copy(f'{self.pose_name}.pdb', f'{mol.lower()}.pdb')

        # rename pose param
        shutil.copy(f'../ff/ligand.frcmod', f'../ff/{mol.lower()}.frcmod')
        shutil.copy(f'../ff/ligand.mol2', f'../ff/{mol.lower()}.mol2')

        # Split initial receptor file
        with open("split-ini.tcl", "rt") as fin:
            with open("split.tcl", "wt") as fout:
                if other_mol:
                    other_mol_vmd = " ".join(other_mol)
                else:
                    other_mol_vmd = 'XXX'
                if lipid_mol:
                    lipid_mol_vmd = " ".join(lipid_mol)
                else:
                    lipid_mol_vmd = 'XXX'
                for line in fin:
                    if 'lig' not in line:
                        fout.write(line.replace('SHLL', '%4.2f' % solv_shell)
                                   .replace('OTHRS', str(other_mol_vmd))
                                   .replace('LIPIDS', str(lipid_mol_vmd))
                                   .replace('MMM', mol.upper()))
        run_with_log('vmd -dispdev text -e split.tcl')

        # Remove possible remaining molecules
        if not other_mol:
            open('others.pdb', 'w').close()
        if not lipid_mol:
            open('lipids.pdb', 'w').close()

        shutil.copy('./protein.pdb', './protein_vmd.pdb')
        run_with_log('pdb4amber -i protein_vmd.pdb -o protein.pdb -y')
        renum_txt = 'protein_renum.txt'

        renum_data = pd.read_csv(
            renum_txt,
            sep='\s+',
            header=None,
            names=['old_resname', 'old_chain', 'old_resid',
                   'new_resname', 'new_resid'])

        # Get beginning and end of protein and save first residue as global variable
        u_vmd = mda.Universe('protein_vmd.pdb')
        first_res_vmd = int(u_vmd.residues[0].resid)

        # use the original receptor file to get the residue numbering
        # because terminal residues are removed during split.
        u_original = mda.Universe('rec_file.pdb')
        first_res = int(u_original.residues[0].resid)
        resid_list = u_original.residues.resids

        u_pdb4amber = mda.Universe('protein.pdb')
        recep_resid_num = len(u_pdb4amber.residues)
        self.first_res = first_res
        self.recep_resid_num = recep_resid_num

        logger.debug('Receptor first residue: %s' % first_res)
        logger.debug('Receptor total length: %s' % recep_resid_num)

        # Adjust protein anchors to the new residue numbering
        h1_resid = int(H1.split('@')[0][1:])
        h2_resid = int(H2.split('@')[0][1:])
        h3_resid = int(H3.split('@')[0][1:])

        h1_atom = H1.split('@')[1]
        h2_atom = H2.split('@')[1]
        h3_atom = H3.split('@')[1]

        protein_chain = 'A'
        h1_entry = (renum_data
                    .query('old_resid == @h1_resid')
                    .query('old_chain == @protein_chain')
                    )
        h2_entry = (renum_data
                    .query('old_resid == @h2_resid')
                    .query('old_chain == @protein_chain')
                    )
        h3_entry = (renum_data
                    .query('old_resid == @h3_resid')
                    .query('old_chain == @protein_chain')
                    )
        if h1_entry.empty or h2_entry.empty or h3_entry.empty:
            logger.warning('Receptor is not set as chain A; '
                           'trying to fetch the first residue '
                           'from the protein and it may be wrong.')
            h1_entry = (renum_data
                        .query('old_resid == @h1_resid')
                        )
            h2_entry = (renum_data
                        .query('old_resid == @h2_resid')
                        )
            h3_entry = (renum_data
                        .query('old_resid == @h3_resid')
                        )
        # check not empty
        if h1_entry.empty or h2_entry.empty or h3_entry.empty:
            renum_data.to_csv('protein_renum_err.txt', sep='\t', index=False)
            logger.error('Could not find the receptor anchors in the protein sequence')
            logger.error('Please check the residue numbering in the receptor file')
            logger.error(f'The renum is stored in {os.getcwd()}/protein_renum_err.txt')
            raise ValueError('Could not find the receptor anchors in the protein sequence')

        # +1 because a dummy atom is added at the beginning
        p1_resid = h1_entry['new_resid'].values[0] + 1
        p2_resid = h2_entry['new_resid'].values[0] + 1
        p3_resid = h3_entry['new_resid'].values[0] + 1
        p1_vmd = f'{p1_resid}'

        P1 = f':{p1_resid}@{h1_atom}'
        P2 = f':{p2_resid}@{h2_atom}'
        P3 = f':{p3_resid}@{h3_atom}'

        logger.debug('Receptor anchors:')
        logger.info(f'P1: {P1}')
        logger.info(f'P2: {P2}')
        logger.info(f'P3: {P3}')
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3

        # Replace names in initial files and VMD scripts
        # Here we convert all four letter residue names to three letter residue names
        if any(mol[:3] != mol for mol in other_mol):
            logger.warning(
                'The residue names of the co-binders are four-letter names.'
                'They were truncated to three-letter names'
                'for compatibility with AMBER.')
            other_mol = [mol[:3] for mol in other_mol]
        self.other_mol = other_mol

        if any(mol[:3] != mol for mol in lipid_mol):
            logger.warning(
                'The residue names of the lipids are four-letter names.'
                'They were truncated to three-letter names'
                'for compatibility with AMBER.')
        self.lipid_mol = [mol[:3] for mol in lipid_mol]
        # Convert CHARMM lipid into lipid21
        run_with_log(f'{charmmlipid2amber} -i lipids.pdb -o lipids_amber.pdb')
        u = mda.Universe('lipids_amber.pdb')
        lipid_resnames = set([resname for resname in u.residues.resnames])
        old_lipid_mol = list(lipid_mol)
        lipid_mol = list(lipid_resnames)
        self.lipid_mol = lipid_mol
        logger.debug(f'Converting CHARMM lipids: {old_lipid_mol} to AMBER format: {lipid_mol}')

        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                other_mol_vmd = " ".join(other_mol)
                lipid_mol_vmd = " ".join(lipid_mol)
                for line in fin:
                    fout.write(line.replace('MMM', mol).replace('mmm', mol.lower())
                               .replace('NN', h1_atom)
                               .replace('P1A', p1_vmd)
                               .replace('FIRST', '1')
                               .replace('LAST', str(recep_resid_num))
                               .replace('STAGE', self.stage)
                               .replace('XDIS', '%4.2f' % l1_x)
                               .replace('YDIS', '%4.2f' % l1_y)
                               .replace('ZDIS', '%4.2f' % l1_z)
                               .replace('RANG', '%4.2f' % l1_range)
                               .replace('DMAX', '%4.2f' % max_adis)
                               .replace('DMIN', '%4.2f' % min_adis)
                               .replace('SDRD', '%4.2f' % sdr_dist)
                               .replace('OTHRS', str(other_mol_vmd))
                               .replace('LIPIDS', str(lipid_mol_vmd))
                               )

        # Create raw complex and clean it
        filenames = ['protein.pdb',
                     '%s.pdb' % mol.lower(),
                     'others.pdb',
                     'lipids_amber.pdb',
                     'crystalwat.pdb']
        with open('./complex-merge.pdb', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        with open('complex-merge.pdb') as oldfile, open('complex.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'CRYST1' in line and not 'CONECT' in line and not 'END' in line:
                    newfile.write(line)

        # New work around to avoid chain swapping during alignment
        run_with_log('pdb4amber -i reference.pdb -o reference_amber.pdb -y')
        run_with_log('vmd -dispdev text -e nochain.tcl')
        run_with_log('./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc')
        run_with_log('vmd -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 4:
                    newfile.write(line)

        # Note: this command will convert four-letter residue names to three-letter residue names
        # This will also turn lipid partial group (PC, PA, OL)
        # that belong to the same residue number
        # into different residue number
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # add box info back if lipid is present
        # also renumber the residue number
        # so that PA, PC, OL (for POPC) are in the same residue number
        if lipid_mol:
            u = mda.Universe('aligned_amber.pdb')
            box_origin = u_original.dimensions
            u.dimensions = box_origin
            logger.debug(f'Adding box info back to aligned_amber.pdb: {box_origin}')

            renum_txt = 'aligned_amber_renum.txt'

            renum_data = pd.read_csv(
                renum_txt,
                sep='\s+',
                header=None,
                names=['old_resname', 'old_chain', 'old_resid',
                       'new_resname', 'new_resid'])

            revised_resids = []
            resid_counter = 1
            prev_resid = 0
            for i, row in renum_data.iterrows():
                if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                    revised_resids.append(resid_counter)
                    resid_counter += 1
                else:
                    revised_resids.append(resid_counter - 1)
                prev_resid = row['old_resid']
            u.atoms.residues.resids = revised_resids

            u.atoms.write('aligned_amber.pdb')

        run_with_log('vmd -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            logger.warning('\n')
            logger.warning(f'WARNING: Could not find the ligand first anchor L1 for {self.pose_name}')
            logger.warning('The ligand is most likely not in the defined binding site in these systems.')
            return False

        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                logger.warning('\n')
                logger.warning(f'WARNING: Could not find the ligand L2 or L3 anchors for {self.pose_name}')
                logger.warning('Try reducing the min_adis parameter in the input file.')
                return False

        os.rename('./anchors.txt',
                  f'anchors-{self.pose_name}.txt')
        return True

    def _create_simulation_dir(self):
        dum_coords = []
        recep_coords = []
        lig_coords = []
        oth_coords = []
        dum_atomlist = []
        lig_atomlist = []
        recep_atomlist = []
        oth_atomlist = []
        dum_rsnmlist = []
        recep_rsnmlist = []
        lig_rsnmlist = []
        oth_rsnmlist = []
        dum_rsidlist = []
        recep_rsidlist = []
        lig_rsidlist = []
        oth_rsidlist = []
        dum_chainlist = []
        recep_chainlist = []
        lig_chainlist = []
        oth_chainlist = []
        dum_atom = 0
        lig_atom = 0
        recep_atom = 0
        oth_atom = 0
        total_atom = 0
        resid_lig = 0
        resname_lig = self.mol
        mol = self.mol
        pose = self.pose_name
        other_mol = self.other_mol
        lipid_mol = self.lipid_mol

        first_res = self.first_res
        recep_resid_num = self.recep_resid_num
        P1 = self.P1
        P2 = self.P2
        P3 = self.P3

        # Copy a few files
        shutil.copy('../build_files/equil-%s.pdb' % mol.lower(), './')

        # Use equil-reference.pdb to retrieve the box size
        shutil.copy('../build_files/equil-%s.pdb' % mol.lower(), './equil-reference.pdb')
        shutil.copy('../build_files/%s-noh.pdb' % mol.lower(), './%s.pdb' % mol.lower())
        shutil.copy('../build_files/anchors-'+pose+'.txt', './anchors.txt')

        # Read coordinates for dummy atoms
        for i in range(1, 2):
            shutil.copy('../build_files/dum'+str(i)+'.pdb', './')
            with open('dum'+str(i)+'.pdb') as dum_in:
                lines = (line.rstrip() for line in dum_in)
                lines = list(line for line in lines if line)
                dum_coords.append((float(lines[1][30:38].strip()), float(
                    lines[1][38:46].strip()), float(lines[1][46:54].strip())))
                dum_atomlist.append(lines[1][12:16].strip())
                dum_rsnmlist.append(lines[1][17:20].strip())
                dum_rsidlist.append(float(lines[1][22:26].strip()))
                dum_chainlist.append(lines[1][21].strip())
                dum_atom += 1
                total_atom += 1

        # Read coordinates from aligned system
        with open('equil-%s.pdb' % mol.lower()) as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Count atoms of receptor and ligand
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                molecule = lines[i][17:21].strip()
                if molecule not in {mol, 'DUM', 'WAT'} and molecule not in other_mol and molecule not in lipid_mol:
                    recep_coords.append((
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip())))
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(molecule)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1
                elif molecule == mol:
                    lig_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(molecule)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1
                elif (molecule == 'WAT') or (molecule in other_mol):
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1
                elif molecule in lipid_mol:
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resid_list = [resid if resid < 10000 else (resid % 10000) + 1 for resid in resid_list]

        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
        lig_resid = str(recep_last + dum_atom + 1)
        chain_tmp = 'None'
        resid_tmp = 'None'

        # Read ligand anchors obtained from VMD
        anchor_file = 'anchors.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        logger.info('Ligand anchors:')
        logger.info(f'L1: {L1}')
        logger.info(f'L2: {L2}')
        logger.info(f'L3: {L3}')

        # Write the new pdb file
        build_file = open('build.pdb', 'w')

        # Positions for the dummy atoms
        for i in range(0, 1):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        # Positions of the receptor atoms
        for i in range(dum_atom, dum_atom + recep_atom):
            if chain_list[i] != chain_tmp:
                if resname_list[i] not in other_mol and resname_list[i] != 'WAT':
                    build_file.write('TER\n')
            chain_tmp = chain_list[i]
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))
        build_file.write('TER\n')

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')

        # Positions of the other atoms
        for i in range(dum_atom + recep_atom + lig_atom, total_atom):
            if resid_list[i] != resid_tmp:
                build_file.write('TER\n')
            resid_tmp = resid_list[i]
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')
        build_file.write('END\n')

        # Write anchors and last protein residue to original pdb file
        with open('equil-%s.pdb' % mol.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('equil-%s.pdb' % mol.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])

        # Check for missing residues in receptor structure
        if recep_last != recep_resid_num:
            logger.warning(
                'WARNING: Missing residues in the receptor protein sequence. Unless the protein is engineered this is not recommended,')
            logger.warning('a protein modeling tool might be required before running equilibration.')

        f_in.close()
        build_file.close()

        # Write dry build file

        with open('build.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        with open('./build-dry.pdb', 'w') as outfile:
            for i in range(0, len(lines)):
                if lines[i][17:20].strip() == 'WAT':
                    break
                outfile.write(lines[i]+'\n')
        outfile.close()

    def _create_box(self):
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        mol = self.mol
        comp = 'q'
        self.comp = comp
        molr = self.mol
        buffer_x = self.sim_config.buffer_x
        buffer_y = self.sim_config.buffer_y
        buffer_z = self.sim_config.buffer_z
        dt = self.sim_config.dt
        ntpr = self.sim_config.ntpr
        ntwr = self.sim_config.ntwr
        ntwe = self.sim_config.ntwe
        ntwx = self.sim_config.ntwx
        cut = self.sim_config.cut
        gamma_ln = self.sim_config.gamma_ln
        barostat = self.sim_config.barostat
        receptor_ff = self.sim_config.receptor_ff
        ligand_ff = self.sim_config.ligand_ff
        lipid_ff = self.sim_config.lipid_ff
        water_model = self.sim_config.water_model
        num_waters = self.sim_config.num_waters
        ion_def = self.sim_config.ion_def
        ion_conc = self.sim_config.ion_conc
        neut = self.sim_config.neut

        if lipid_mol:
            amber_files_path = 'amber_files'
            p_coupling = '3'
            c_surften = '3'
        else:
            amber_files_path = 'amber_files_no_lipid'
            p_coupling = '1'
            c_surften = '0'

        if os.path.exists(f'../{amber_files_path}') and not self.overwrite:
            raise ValueError(f'../{amber_files_path} already exists. Set overwrite=True to overwrite it.')
        else:
            shutil.rmtree(f'../{amber_files_path}', ignore_errors=True)
            shutil.copytree(
                amber_files_orig,
                f'../{amber_files_path}',
                dirs_exist_ok=True)

        for dname, dirs, files in os.walk(f'../{amber_files_path}'):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (s
                         .replace('_step_', dt)
                         .replace('_ntpr_', ntpr)
                         .replace('_ntwr_', ntwr)
                         .replace('_ntwe_', ntwe)
                         .replace('_ntwx_', ntwx)
                         .replace('_cutoff_', cut)
                         .replace('_gamma_ln_', gamma_ln)
                         .replace('_barostat_', barostat)
                         .replace('_receptor_ff_', receptor_ff)
                         .replace('_ligand_ff_', ligand_ff)
                         .replace('_lipid_ff_', lipid_ff)
                         .replace('_p_coupling_', p_coupling)
                         .replace('_c_surften_', c_surften)
                         )
                with open(fpath, "w") as f:
                    f.write(s)
            # if building a lipid system
        # use x, y box dimensions from the lipid system
        if lipid_mol:
            buffer_x = 0
            buffer_y = 0

        # copy all the files from the ff directory
        for file in glob.glob('../ff/*'):
            if file.endswith('.in'):
                continue
            shutil.copy(file, './')

        # Copy tleap files that are used for restraint generation and analysis
        shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap_vac.in')
        shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap_vac_ligand.in')
        shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap.in')

        # Append tleap file for vacuum
        with open('tleap_vac.in', 'a') as tleap_vac:
            tleap_vac.write('# Load the necessary parameters\n')
            for mol in other_mol:
                tleap_vac.write(f'loadamberparams ../ff/{mol.lower()}.frcmod\n')
                tleap_vac.write(f'{mol.upper()} = loadmol2 ../ff/{mol.lower()}.mol2\n')
            tleap_vac.write(f'loadamberparams ../ff/{mol.lower()}.frcmod\n')
            tleap_vac.write(f'{mol.upper()} = loadmol2 ../ff/{mol.lower()}.mol2\n\n')
            if comp == 'x':
                tleap_vac.write(f'loadamberparams ../ff/{molr.lower()}.frcmod\n')
                tleap_vac.write(f'{molr.upper()} = loadmol2 ../ff/{molr.lower()}.mol2\n\n')
            tleap_vac.write('# Load the water parameters\n')
            if water_model.lower() != 'tip3pf':
                tleap_vac.write(f'source leaprc.water.{water_model.lower()}\n\n')
            else:
                tleap_vac.write('source leaprc.water.fb3\n\n')
            tleap_vac.write('model = loadpdb build-dry.pdb\n\n')
            tleap_vac.write('check model\n')
            tleap_vac.write('savepdb model vac.pdb\n')
            tleap_vac.write('saveamberparm model vac.prmtop vac.inpcrd\n')
            tleap_vac.write('quit\n')

        # Append tleap file for ligand only
        tleap_vac_ligand = open('tleap_vac_ligand.in', 'a')
        tleap_vac_ligand.write('# Load the ligand parameters\n')
        tleap_vac_ligand.write('loadamberparams %s.frcmod\n' % (mol.lower()))
        tleap_vac_ligand.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
        tleap_vac_ligand.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
        tleap_vac_ligand.write('check model\n')
        tleap_vac_ligand.write('savepdb model vac_ligand.pdb\n')
        tleap_vac_ligand.write('saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n')
        tleap_vac_ligand.write('quit\n')
        tleap_vac_ligand.close()

        # Generate complex in vacuum
        p = run_with_log(tleap + ' -s -f tleap_vac.in > tleap_vac.log')

        # Generate ligand structure in vacuum
        p = run_with_log(tleap + ' -s -f tleap_vac_ligand.in > tleap_vac_ligand.log')

        # Find out how many cations/anions are needed for neutralization
        neu_cat = 0
        neu_ani = 0
        f = open('tleap_vac.log', 'r')
        for line in f:
            if "The unperturbed charge of the unit" in line:
                splitline = line.split()
                if float(splitline[6].strip('\'\",.:;#()][')) < 0:
                    neu_cat = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
                elif float(splitline[6].strip('\'\",.:;#()][')) > 0:
                    neu_ani = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
        f.close()

        # Get ligand removed charge when doing LJ calculations
        lig_cat = 0
        lig_ani = 0
        f = open('tleap_vac_ligand.log', 'r')
        for line in f:
            if "The unperturbed charge of the unit" in line:
                splitline = line.split()
                if float(splitline[6].strip('\'\",.:;#()][')) < 0:
                    lig_cat = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
                elif float(splitline[6].strip('\'\",.:;#()][')) > 0:
                    lig_ani = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
        f.close()

        # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
        if (comp == 'v' and dec_method == 'sdr') or comp == 'x':
            charge_neut = neu_cat - neu_ani - 2*lig_cat + 2*lig_ani
            neu_cat = 0
            neu_ani = 0
            if charge_neut > 0:
                neu_cat = abs(charge_neut)
            if charge_neut < 0:
                neu_ani = abs(charge_neut)
        if comp == 'e' and dec_method == 'sdr':
            charge_neut = neu_cat - neu_ani - 3*lig_cat + 3*lig_ani
            neu_cat = 0
            neu_ani = 0
            if charge_neut > 0:
                neu_cat = abs(charge_neut)
            if charge_neut < 0:
                neu_ani = abs(charge_neut)

        # Define volume density for different water models
        ratio = 0.060
        if water_model == 'TIP3P':
            water_box = water_model.upper()+'BOX'
        elif water_model == 'SPCE':
            water_box = 'SPCBOX'
        elif water_model == 'TIP4PEW':
            water_box = water_model.upper()+'BOX'
        elif water_model == 'OPC':
            water_box = water_model.upper()+'BOX'
        elif water_model == 'TIP3PF':
            water_box = water_model.upper()+'BOX'

        # Fixed number of water molecules
        if num_waters != 0:

            # Create the first box guess to get the initial number of waters and cross sectional area
            buff = 50.0
            scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
            num_added = scripts.check_tleap()
            cross_area = scripts.cross_sectional_area()

            # First iteration to estimate box volume and number of ions
            res_diff = num_added - num_waters
            buff_diff = res_diff/(ratio*cross_area)
            buff -= buff_diff
            if buff < 0:
                logger.error(
                    'Not enough water molecules to fill the system in the z direction, please increase the number of water molecules')
                sys.exit(1)
            # Get box volume and number of added ions
            scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
            box_volume = scripts.box_volume()
            logger.debug(f'Box volume {box_volume}')
            # box volume already takes into account system shrinking during equilibration
            num_cations = round(ion_def[2]*6.02e23*box_volume*1e-27)

            # A rough reduction of the number of cations
            # for lipid systems
            if lipid_mol:
                num_cations = num_cations // 2

            # Number of cations and anions
            num_cat = num_cations
            num_ani = num_cations - neu_cat + neu_ani
            # If there are not enough chosen cations to neutralize the system
            if num_ani < 0:
                num_cat = neu_cat
                num_cations = neu_cat
                num_ani = 0
            logger.debug(f'Number of cations: {num_cat}')
            logger.debug(f'Number of anions: {num_ani}')

            # Update target number of residues according to the ion definitions and vacuum waters
            vac_wt = 0
            with open('./build.pdb') as myfile:
                for line in myfile:
                    if 'WAT' in line and ' O ' in line:
                        vac_wt += 1
            if (neut == 'no'):
                target_num = int(num_waters - neu_cat + neu_ani + 2*int(num_cations) - vac_wt)
            elif (neut == 'yes'):
                target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

            # Define a few parameters for solvation iteration
            buff = 50.0
            count = 0
            max_count = 10
            rem_limit = 16
            factor = 1
            ind = 0.90
            buff_diff = 1.0

            # Iterate to get the correct number of waters
            while num_added != target_num:
                count += 1
                if count > max_count:
                    # Try different parameters
                    rem_limit += 4
                    if ind > 0.5:
                        ind = ind - 0.02
                    else:
                        ind = 0.90
                    factor = 1
                    max_count = max_count + 10
                tleap_remove = None
                # Manually remove waters if inside removal limit
                if num_added > target_num and (num_added - target_num) < rem_limit:
                    difference = num_added - target_num
                    tleap_remove = [target_num + 1 + i for i in range(difference)]
                    scripts.write_tleap(mol, molr, comp, water_model, water_box, buff,
                                        buffer_x, buffer_y, other_mol, tleap_remove)
                    scripts.check_tleap()
                    break
                # Set new buffer size based on chosen water density
                res_diff = num_added - target_num - (rem_limit/2)
                buff_diff = res_diff/(ratio*cross_area)
                buff -= (buff_diff * factor)
                if buff < 0:
                    logger.error(
                        'Not enough water molecules to fill the system in the z direction, please increase the number of water molecules')
                    sys.exit(1)
                # Set relaxation factor
                factor = ind * factor
                # Get number of waters
                scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
                num_added = scripts.check_tleap()
            logger.debug(f'{count} iterations for fixed water number')
        # Fixed z buffer
        elif buffer_z != 0:
            buff = buffer_z
            tleap_remove = None
            # Get box volume and number of added ions
            scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
            box_volume = scripts.box_volume()
            logger.debug(f'Box volume {box_volume}')
            # box volume already takes into account system shrinking during equilibration
            num_cations = round(ion_def[2]*6.02e23*box_volume*1e-27)

            # A rough reduction of the number of cations
            # for lipid systems
            if lipid_mol:
                num_cations = num_cations // 2
            # Number of cations and anions
            num_cat = num_cations
            num_ani = num_cations - neu_cat + neu_ani
            # If there are not enough chosen cations to neutralize the system
            if num_ani < 0:
                num_cat = neu_cat
                num_cations = neu_cat
                num_ani = 0
            logger.debug(f'Number of cations: {num_cat}')
            logger.debug(f'Number of anions: {num_ani}')

        # First round just solvate the system
        shutil.copy('tleap.in', 'tleap_solvate_pre.in')
        tleap_solvate = open('tleap_solvate_pre.in', 'a')
        tleap_solvate.write('# Load the necessary parameters\n')
        for i in range(0, len(other_mol)):
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
            tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i].upper(), other_mol[i].lower()))
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
        if comp == 'x':
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
        if comp == 'x':
            tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr.upper(), molr.lower()))
        tleap_solvate.write('# Load the water and jc ion parameters\n')
        if water_model.lower() != 'tip3pf':
            tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
        else:
            tleap_solvate.write('source leaprc.water.fb3\n\n')
        tleap_solvate.write('model = loadpdb build.pdb\n\n')
        tleap_solvate.write('# Create water box with chosen model\n')
        tleap_solvate.write('solvatebox model ' + water_box +
                            ' {' + str(buffer_x) + ' ' + str(buffer_y) + ' ' + str(buff) + '} 1.5\n\n')
        if tleap_remove is not None:
            tleap_solvate.write('# Remove a few waters manually\n')
            for water in tleap_remove:
                tleap_solvate.write('remove model model.%s\n' % water)
            tleap_solvate.write('\n')
        tleap_solvate.write('desc model\n')
        tleap_solvate.write('savepdb model full_pre.pdb\n')
        tleap_solvate.write('quit')
        tleap_solvate.close()
        p = run_with_log(tleap + ' -s -f tleap_solvate_pre.in > tleap_solvate_pre.log')

        # Retrieve residue number for lipids
        # because tleap separates them into different residues

        run_with_log('pdb4amber -i build.pdb -o build_amber.pdb -y')

        renum_data = pd.read_csv('build_amber_renum.txt', sep='\s+',
                                 header=None, names=['old_resname',
                                                     'old_chain',
                                                     'old_resid',
                                                     'new_resname', 'new_resid'])
        revised_resids = []
        resid_counter = 1
        prev_resid = 0
        for i, row in renum_data.iterrows():
            if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                revised_resids.append(resid_counter)
                resid_counter += 1
            else:
                revised_resids.append(resid_counter - 1)
            prev_resid = row['old_resid']

        renum_data['revised_resid'] = revised_resids

        try:
            u = mda.Universe('full_pre.pdb')
        except ValueError('could not convert'):
            raise ValueError('The system is toooo big! '
                             'tleap write incorrect PDB when '
                             'residue exceed 100,000.'
                             'I am not sure how to fix it yet.')

        u_orig = mda.Universe('equil-reference.pdb')

        u.dimensions[0] = u_orig.dimensions[0]
        u.dimensions[1] = u_orig.dimensions[1]
        box_xy = [u.dimensions[0], u.dimensions[1]]
        membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
        # get memb boundries
        membrane_region_z_max = membrane_region.select_atoms('type P').positions[:, 2].max() - 10
        membrane_region_z_min = membrane_region.select_atoms('type P').positions[:, 2].min() + 10
        # water that is within the membrane
        water = u.select_atoms(
            f'byres (resname WAT and prop z > {membrane_region_z_min} and prop z < {membrane_region_z_max})')

        water_around_prot = u.select_atoms('byres (resname WAT and around 5 protein)')

        final_system = u.atoms - water
        final_system = final_system | water_around_prot

        # get WAT that is out of the box
        outside_wat = final_system.select_atoms(
            f'byres (resname WAT and ((prop x > {box_xy[0] / 2}) or (prop x < -{box_xy[0] / 2}) or (prop y > {box_xy[1] / 2}) or (prop y < -{box_xy[1] / 2})))')
        final_system = final_system - outside_wat

        logger.debug(f'Final system: {final_system.n_atoms} atoms')

        # set correct residue number
        revised_resids = np.array(revised_resids)
        total_residues = final_system.residues.n_residues
        final_resids = np.zeros(total_residues, dtype=int)
        final_resids[:len(revised_resids)] = revised_resids
        next_resnum = revised_resids[-1] + 1
        final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
        final_system.residues.resids = final_resids

        final_system_prot = final_system.select_atoms('protein')
        final_system_dum = final_system.select_atoms('resname DUM')
        final_system_others = final_system - final_system_prot - final_system_dum

        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum_data.query(f'old_resid == @resid_str').old_chain.values[0]

        dum_lines = []
        for chain_name in np.unique(final_system.select_atoms('protein').atoms.chainIDs):
            temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

            prot_segment = final_system.select_atoms(f'chainID {chain_name}')

            prot_segment.write(temp_pdb.name)
            temp_pdb.close()

            with open(temp_pdb.name, 'r') as f:
                # store lines start with ATOM
                dum_lines += [line for line in f.readlines() if line.startswith('ATOM')]
            dum_lines.append('TER\n')

        with open('solvate_pre_prot.pdb', 'w') as f:
            f.writelines(dum_lines)

    #    final_system_prot.write('solvate_pre_prot.pdb')

        dum_lines = []
        for residue in final_system_dum.residues:
            # create a temp pdb file in /tmp/
            # write the residue to the temp file
            temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

            residue.atoms.write(temp_pdb.name)
            temp_pdb.close()
            # store atom lines into dum_lines
            with open(temp_pdb.name, 'r') as f:
                # store lines start with ATOM
                dum_lines += [line for line in f.readlines() if line.startswith('ATOM')]
            dum_lines.append('TER\n')

        with open('solvate_pre_dum.pdb', 'w') as f:
            f.writelines(dum_lines)

        non_prot_lines = []

        prev_resid = final_system_others.residues.resids[0]
        for residue in final_system_others.residues:
            if residue.resid != prev_resid:
                non_prot_lines.append('TER\n')
            # create a temp pdb file in /tmp/
            # write the residue to the temp file
            temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

            residue.atoms.write(temp_pdb.name)
            temp_pdb.close()
            # store atom lines into non_prot_lines
            with open(temp_pdb.name, 'r') as f:
                # store lines start with ATOM
                non_prot_lines += [line for line in f.readlines() if line.startswith('ATOM')]
            prev_resid = residue.resid

        with open('solvate_pre_others.pdb', 'w') as f:
            f.writelines(non_prot_lines)

        box_final = final_system_prot.dimensions
        logger.debug(f'Final box dimensions: {box_final[:3]}')
        # Write the final tleap file with the correct system size and removed water molecules
        shutil.copy('tleap.in', 'tleap_solvate.in')
        tleap_solvate = open('tleap_solvate.in', 'a')
        tleap_solvate.write('# Load the necessary parameters\n')
        for i in range(0, len(other_mol)):
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
            tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i].upper(), other_mol[i].lower()))
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
        if comp == 'x':
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
        if comp == 'x':
            tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr.upper(), molr.lower()))
        tleap_solvate.write('# Load the water and jc ion parameters\n')
        if water_model.lower() != 'tip3pf':
            tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
        else:
            tleap_solvate.write('source leaprc.water.fb3\n\n')

        tleap_solvate.write('dum = loadpdb solvate_pre_dum.pdb\n\n')
        tleap_solvate.write('prot = loadpdb solvate_pre_prot.pdb\n\n')
        tleap_solvate.write('others = loadpdb solvate_pre_others.pdb\n\n')
        tleap_solvate.write('model = combine {dum prot others}\n\n')

        if (neut == 'no'):
            tleap_solvate.write('# Add ions for neutralization/ionization\n')
            tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[0], num_cat))
            tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[1], num_ani))
        elif (neut == 'yes'):
            tleap_solvate.write('# Add ions for neutralization/ionization\n')
            if neu_cat != 0:
                tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[0], neu_cat))
            if neu_ani != 0:
                tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[1], neu_ani))
        tleap_solvate.write('\n')
        tleap_solvate.write('set model box {%.2f %.2f %.2f}\n' % (box_final[0], box_final[1], box_final[2]))
        tleap_solvate.write('desc model\n')
        tleap_solvate.write('savepdb model full.pdb\n')
        tleap_solvate.write('saveamberparm model full.prmtop full.inpcrd\n')
        tleap_solvate.write('quit')
        tleap_solvate.close()
        p = run_with_log(tleap + ' -s -f tleap_solvate.in > tleap_solvate.log')

        f = open('tleap_solvate.log', 'r')
        for line in f:
            if "Could not open file" in line:
                logger.error('Error!!!')
                logger.error(line)
                sys.exit(1)
            if "WARNING: The unperturbed charge of the unit:" in line:
                logger.warning(line)
                logger.warning('The system is not neutralized properly after solvation')
            if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
                logger.error('Aborted.The ion types specified in the input file could be wrong.')
                logger.error('Please check the tleap_solvate.log file, and the ion types specified in the input file.\n')
                sys.exit(1)
        f.close()

        # Apply hydrogen mass repartitioning
        shutil.copy(f'../{amber_files_path}/parmed-hmr.in', './')
        run_with_log('parmed -O -n -i parmed-hmr.in > parmed-hmr.log')

    def _restraints(self):
        pose = self.pose_name
        rest = self.sim_config.rest
        bb_start = self.sim_config.bb_start
        bb_end = self.sim_config.bb_end
        stage = self.stage
        mol = self.mol
        molr = self.mol
        comp = self.comp
        bb_equil = self.sim_config.bb_equil
        sdr_dist = self.sim_config.sdr_dist
        dec_method = self.sim_config.dec_method
        other_mol = self.other_mol

        release_eq = self.sim_config.release_eq
        logger.debug('Equil release weights:')
        for i in range(0, len(release_eq)):
            weight = release_eq[i]
            logger.debug('%s' % str(weight))
            setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                             molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
            shutil.copy('./'+pose+'/disang.rest', './'+pose+'/disang%02d.rest' % int(i))
        shutil.copy('./'+pose+'/disang%02d.rest' % int(0), './'+pose+'/disang.rest')

    def _sim_files(self):
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = len(self.sim_config.release_eq)
        pose = self.pose_name
        comp = self.comp
        win = self.win
        stage = self.stage
        eq_steps1 = self.sim_config.eq_steps1
        eq_steps2 = self.sim_config.eq_steps2
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol

        setup.sim_files(hmr, temperature, mol,
                        num_sim, pose, comp, win,
                        stage, eq_steps1, eq_steps2, rng,
                        lipid_sim=lipid_mol)
