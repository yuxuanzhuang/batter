from abc import ABC, abstractmethod
from loguru import logger

import os
import shutil
import pandas as pd
import numpy as np
import MDAnalysis as mda
from contextlib import contextmanager

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.data import build_files as build_files_orig
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
        yield
        os.chdir(cwd)
    

class EquilibrationBuilder(SystemBuilder):

    def _build_complex(self):
        sdr_dist = 0
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
        if len(mol_u.residues) > 1:
            raise ValueError(f'The ligand {self.pose_name} has more than one residue: '
                             f'{mol_u.residues}')
        self.mol = mol_u.residues[0].resname
        mol = self.mol
        if mol in other_mol:
            raise ValueError(f'The ligand {mol}'
                             f'cannot be in the other_mol list: '
                             f'{other_mol}')
        # copy pose
        shutil.copy(f'{self.pose_name}.pdb', f'{mol.lower()}.pdb')
        shutil.copy(f'../ff/ligand.mol2', f'{mol.lower()}.mol2')
        shutil.copy(f'../ff/ligand.frcmod', f'{mol.lower()}.frcmod')


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
        
        p1_resid = h1_entry['new_resid'].values[0]
        p2_resid = h2_entry['new_resid'].values[0]
        p3_resid = h3_entry['new_resid'].values[0]
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

        if any(mol[:3] != mol for mol in lipid_mol):
            logger.warning(
                'The residue names of the lipids are four-letter names.'
                'They were truncated to three-letter names'
                'for compatibility with AMBER.')
            lipid_mol = [mol[:3] for mol in lipid_mol]
        # Convert CHARMM lipid into lipid21
        run_with_log(f'{charmmlipid2amber} -i lipids.pdb -o lipids_amber.pdb')
        u = mda.Universe('lipids_amber.pdb')
        lipid_resnames = set([resname for resname in u.residues.resnames])
        old_lipid_mol = list(lipid_mol)
        lipid_mol = list(lipid_resnames)
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
                            .replace('STAGE', 'equil')
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
        other_mol = self.sim_config.other_mol
        lipid_mol = self.sim_config.lipid_mol

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
            logger.warning('WARNING: Missing residues in the receptor protein sequence. Unless the protein is engineered this is not recommended,')
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
        os.chdir('..')

    def _create_box(self):
        pass
        self.system.add_ligand()
        self.system.solvate()
        self.system.add_ions()
        self.system.add_lipids()

    def _restraints(self):
        pass
        self.system.add_restraints()

    def _sim_files(self):
        pass
        self.system.create_simulation_files()