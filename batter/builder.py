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
import warnings
from typing import Union

from batter.input_process import SimulationConfig, get_configure_from_file
from batter.data import build_files as build_files_orig
from batter.data import amber_files as amber_files_orig
from batter.data import run_files as run_files_orig
from batter.bat_lib import setup, scripts

from batter.utils import (
    run_with_log,
    antechamber,
    tleap,
    cpptraj,
    parmchk2,
    charmmlipid2amber,
    obabel,
    vmd,
    log_info,
    COMPONENTS_LAMBDA_DICT,
    COMPONENTS_FOLDER_DICT,
    COMPONENTS_DICT,
    builder_fail_report,
)


class SystemBuilder(ABC):
    """
    The base class for all system builders.

    The process to build a system involves the following steps:
    1. `_build_complex`: Build the complex.
    2. `_create_box`: Create the box.
    3. `_restraints`: Add restraints.
    4. `_sim_files`: Create simulation files, e.g. input files for AMBER.

    """
    stage = None

    def __init__(self,
                 system: 'batter.System',
                 pose: str,
                 sim_config: SimulationConfig,
                 working_dir: str,
                 ):
        """
        The base class for all system builders.

        Parameters
        ----------
        system : batter.System
            The system to build.
        pose : str
            The name of the pose
        sim_config : batter.input_process.SimulationConfig
            The simulation configuration.
        working_dir : str
            The working directory.
        """
        self.system = system
        self.pose = pose
        self.sim_config = sim_config
        self.other_mol = self.sim_config.other_mol
        self.lipid_mol = self.sim_config.lipid_mol

        self.working_dir = working_dir

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    @property
    def build_file_folder(self):
        return f'{self.comp}_build_files'
    
    @property
    def run_files_folder(self):
        return f'{self.comp}_run_files'
    
    @property
    def amber_files_folder(self):
        return f'{self.comp}_amber_files'

    @builder_fail_report
    def build(self):
    
        with self._change_dir(self.working_dir):
            logger.debug(f'Building {self.pose}...')

            with self._change_dir(self.pose):
                os.makedirs(self.comp_folder, exist_ok=True)
                with self._change_dir(self.comp_folder):
                    if self.win == -1:
                        if os.path.exists(self.build_file_folder):
                            shutil.rmtree(self.build_file_folder, ignore_errors=True)
                        if os.path.exists('exchange_files'):
                            shutil.rmtree('exchange_files', ignore_errors=True)
                        os.makedirs(self.build_file_folder, exist_ok=True)
                        with self._change_dir(self.build_file_folder):
                            logger.debug(f'Creating build_files in  {os.getcwd()}')
                            anchor_found = self._build_complex()
                            if not anchor_found:
                                logger.warning(f'Could not find the ligand anchors for {self.pose}.')
                                return None

                        self.mol = mda.Universe(f'{self.build_file_folder}/{self.pose}.pdb').residues[0].resname
                        self._create_amber_files()
                        if self.win == -1:
                            self._create_run_files()
                    self.mol = mda.Universe(f'{self.build_file_folder}/{self.pose}.pdb').residues[0].resname 
                    os.makedirs(self.window_folder, exist_ok=True)
                    with self._change_dir(self.window_folder):
                        if self.win == -1:
                            self._create_simulation_dir()
                        else:
                            self._copy_simulation_dir()
                        if self.win == -1:
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
        It should return True if the anchor atoms are found,
        False otherwise.
        """
        raise NotImplementedError()

    @log_info
    def _create_amber_files(self):
        dt = self.sim_config.dt
        ntpr = self.sim_config.ntpr
        ntwr = self.sim_config.ntwr
        ntwe = self.sim_config.ntwe
        ntwx = self.sim_config.ntwx
        cut = self.sim_config.cut
        gamma_ln = self.sim_config.gamma_ln
        barostat = self.sim_config.barostat
        if barostat == '2':
            logger.warning('WARNING: Switch to Berendsen barostat')
            barostat = '1'
        
        receptor_ff = self.sim_config.receptor_ff
        ligand_ff = self.sim_config.ligand_ff
        lipid_ff = self.sim_config.lipid_ff
        lipid_mol = self.sim_config.lipid_mol
        if lipid_mol:
            p_coupling = self.p_coupling if hasattr(self, 'p_coupling') else '3'
            c_surften = self.c_surften if hasattr(self, 'c_surften') else '3'
        else:
            p_coupling = '1'
            c_surften = '0'
        
        # if in production, with NVT ensemble (ntp=0)
        if self.win != -1:
            p_coupling = '0'

        if os.path.exists(self.amber_files_folder):
            shutil.rmtree(self.amber_files_folder, ignore_errors=True)

        shutil.copytree(
                amber_files_orig,
                self.amber_files_folder,
                dirs_exist_ok=True)

        for dname, dirs, files in os.walk(self.amber_files_folder):
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

    @abstractmethod
    def _create_run_files(self):
        """
        Create run files.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_simulation_dir(self):
        """
        Create the simulation directory.
        This is only done for the first window.
        """
        raise NotImplementedError()

    @log_info
    def _copy_simulation_dir(self):
        """
        Copy the simulation directory from the first window.
        In reality, only symlink necessary files to reduce the disk usage.
        """
        source_dir = f'../{self.comp}-1'
        files_to_symlink = ['full.prmtop', 'full.inpcrd', 'full.pdb',
                            'vac.pdb', 'vac_ligand.pdb',
                            'vac.prmtop', 'vac_ligand.prmtop',
                            f'fe-{self.mol}.pdb', f'{self.mol}.mol2']
        if self.sim_config.hmr == 'yes':
            files_to_symlink.append('full.hmr.prmtop')

        # Create symlinks for PDB and PRMTOP files
        for file_basename in files_to_symlink:
            for file_path in glob.glob(f'{source_dir}/{file_basename}'):
                file_name = os.path.basename(file_path)
                target_path = os.path.join('.', file_name)

                # Remove existing files or symlinks
                if os.path.exists(target_path) or os.path.islink(target_path):
                    os.remove(target_path)
                try:
                    os.symlink(file_path, target_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to create symlink for {file_path}: {e}")
        
        files_to_copy = ['disang.rest', 'cv.in']
        for file_basename in files_to_copy:
            for file_path in glob.glob(f'{source_dir}/{file_basename}'):
                file_name = os.path.basename(file_path)
                target_path = os.path.join('.', file_name)
                #shutil.copy(file_path, target_path)
                os.system(f'cp {file_path} {target_path}')

    @log_info
    def _create_box(self):
        """
        Create the box.
        It involves
        1. Add ligand (that differs for different systems)
        2. Solvate the system.
        3. Add ions.
        4. For membrane systems, add lipids.
        """
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        mol = self.mol
        comp = self.comp
        if comp == 'x':
            molr = self.molr
            poser = self.poser
        else:
            molr = mol
            poser = self.pose
        buffer_x = self.sim_config.buffer_x
        buffer_y = self.sim_config.buffer_y
        buffer_z = self.sim_config.buffer_z
        water_model = self.sim_config.water_model
        num_waters = self.sim_config.num_waters
        ion_def = self.sim_config.ion_def
        ion_conc = self.sim_config.ion_conc
        neut = self.sim_config.neut
        dec_method = self.sim_config.dec_method

        # if building a lipid system
        # use x, y box dimensions from the lipid system
        if lipid_mol:
            buffer_x = 0
            buffer_y = 0

        # copy all the files from the ff directory
        for file in glob.glob('../../ff/*'):
            if file.endswith('.in') or file.endswith('.pdb'):
                continue
            #shutil.copy(file, '.')
            os.system(f'cp {file} .')


        # Copy tleap files that are used for restraint generation and analysis
        #shutil.copy(f'{self.amber_files_folder}/tleap.in.amber16', 'tleap_vac.in')
        os.system(f'cp {self.amber_files_folder}/tleap.in.amber16 tleap_vac.in')
        #shutil.copy(f'{self.amber_files_folder}/tleap.in.amber16', 'tleap_vac_ligand.in')
        os.system(f'cp {self.amber_files_folder}/tleap.in.amber16 tleap_vac_ligand.in')
        #shutil.copy(f'{self.amber_files_folder}/tleap.in.amber16', 'tleap.in')
        os.system(f'cp {self.amber_files_folder}/tleap.in.amber16 tleap.in')

        # Append tleap file for vacuum
        with open('tleap_vac.in', 'a') as tleap_vac:
            tleap_vac.write('# Load the necessary parameters\n')
            for mol in other_mol:
                tleap_vac.write(f'loadamberparams {mol.lower()}.frcmod\n')
                tleap_vac.write(f'{mol} = loadmol2 {mol.lower()}.mol2\n')
            tleap_vac.write(f'loadamberparams {mol.lower()}.frcmod\n')
            tleap_vac.write(f'{mol} = loadmol2 {mol.lower()}.mol2\n\n')
            if comp == 'x':
                tleap_vac.write(f'loadamberparams {molr.lower()}.frcmod\n')
                tleap_vac.write(f'{molr} = loadmol2 {molr.lower()}.mol2\n\n')
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
        tleap_vac_ligand.write('%s = loadmol2 %s.mol2\n\n' % (mol, mol.lower()))
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
        
        # only one charged ligand present
        if comp == 's' or comp == 'o' or comp == 'z':
            charge_neut = neu_cat - neu_ani - 1*lig_cat + 1*lig_ani
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
        #shutil.copy('tleap.in', 'tleap_solvate_pre.in')
        os.system(f'cp tleap.in tleap_solvate_pre.in')
        tleap_solvate = open('tleap_solvate_pre.in', 'a')
        tleap_solvate.write('# Load the necessary parameters\n')
        for i in range(0, len(other_mol)):
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
            tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i], other_mol[i].lower()))
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol, mol.lower()))
        if comp == 'x':
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
        if comp == 'x':
            tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr, molr.lower()))
        tleap_solvate.write('# Load the water and jc ion parameters\n')
        if water_model.lower() != 'tip3pf':
            tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
        else:
            tleap_solvate.write('source leaprc.water.fb3\n\n')
        tleap_solvate.write('model = loadpdb build.pdb\n\n')
        tleap_solvate.write('# Create water box with chosen model\n')
        tleap_solvate.write('solvatebox model ' + water_box +
                            ' {' + str(buffer_x) + ' ' + str(buffer_y) + ' ' + str(buff) + '} 1\n\n')
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

        # get # of water molecules inside build.pdb
        # to be used for the next iteration
        num_waters = 0
        with open('build.pdb') as myfile:
            for line in myfile:
                if 'WAT' in line:
                    num_waters += 1

        # Retrieve residue number for lipids
        # because tleap separates them into different residues

        run_with_log('pdb4amber -i build.pdb -o build_amber.pdb -y')

        renum_data = pd.read_csv('build_amber_renum.txt', sep=r'\s+',
                                 header=None, names=['old_resname',
                                                     'old_chain',
                                                     'old_resid',
                                                     'new_resname', 'new_resid'])
        # convert all histidine to HIS
        renum_data['old_resname'] = renum_data['old_resname'].replace(
            ['HIS', 'HIE', 'HIP', 'HID'], 'HIS')
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
        
        # reduce the box size on the z axis by 3 angstrom
        # to account for the void space at the boundaries
        u.dimensions[2] = u.dimensions[2] - 3
        u.atoms.positions[:, 2] = u.atoms.positions[:, 2] - 3

        box_xy = [u.dimensions[0], u.dimensions[1]]
        membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
        # get memb boundries
        membrane_region_z_max = membrane_region.select_atoms('type P').positions[:, 2].max() - 10
        membrane_region_z_min = membrane_region.select_atoms('type P').positions[:, 2].min() + 10
        # water that is within the membrane
        water = u.select_atoms(
            f'byres (resname WAT and prop z > {membrane_region_z_min} and prop z < {membrane_region_z_max})')

        #water_around_prot = u.select_atoms('byres (resname WAT and around 5 protein)')
        
        water_around_prot = u.select_atoms('resname WAT').residues[:num_waters].atoms

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
        # filter out the water that is not around protein for ionization
        water_outside_prot = final_system.select_atoms(
            f'byres (resname WAT and not (around 6 protein))')
        final_system_others = final_system_others - water_outside_prot

        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            resid_resname = residue.resname
            if resid_resname in ['HIS', 'HIE', 'HIP', 'HID']:
                # rename it to HIS
                resid_resname = 'HIS'
            residue.atoms.chainIDs = renum_data.query(
                    f'old_resid == @resid_str').query(
                        f'old_resname == @resid_resname').old_chain.values[0]

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
        
        outside_wat_lines = []

        prev_resid = water_outside_prot.residues.resids[0]
        for residue in water_outside_prot.residues:
            if residue.resid != prev_resid:
                outside_wat_lines.append('TER\n')
            # create a temp pdb file in /tmp/
            # write the residue to the temp file
            temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

            residue.atoms.write(temp_pdb.name)
            temp_pdb.close()
            # store atom lines into outside_wat_lines
            with open(temp_pdb.name, 'r') as f:
                # store lines start with ATOM
                outside_wat_lines += [line for line in f.readlines() if line.startswith('ATOM')]
            prev_resid = residue.resid

        with open('solvate_pre_outside_wat.pdb', 'w') as f:
            f.writelines(outside_wat_lines)

        box_final = final_system_prot.dimensions
        
        logger.debug(f'Final box dimensions: {box_final[:3]}')
        # Write the final tleap file with the correct system size and removed water molecules
        #shutil.copy('tleap.in', 'tleap_solvate.in')
        os.system(f'cp tleap.in tleap_solvate.in')
        tleap_solvate = open('tleap_solvate.in', 'a')
        tleap_solvate.write('# Load the necessary parameters\n')
        for i in range(0, len(other_mol)):
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
            tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i], other_mol[i].lower()))
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol, mol.lower()))
        if comp == 'x':
            tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
        if comp == 'x':
            tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr, molr.lower()))
        tleap_solvate.write('# Load the water and jc ion parameters\n')
        if water_model.lower() != 'tip3pf':
            tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
        else:
            tleap_solvate.write('source leaprc.water.fb3\n\n')

        tleap_solvate.write('dum = loadpdb solvate_pre_dum.pdb\n\n')
        tleap_solvate.write('prot = loadpdb solvate_pre_prot.pdb\n\n')
        tleap_solvate.write('others = loadpdb solvate_pre_others.pdb\n\n')
        tleap_solvate.write('outside_wat = loadpdb solvate_pre_outside_wat.pdb\n\n')

        if (neut == 'no'):
            tleap_solvate.write('# Add ions for neutralization/ionization\n')
            tleap_solvate.write('addionsrand outside_wat %s %d\n' % (ion_def[0], num_cat))
            tleap_solvate.write('addionsrand outside_wat %s %d\n' % (ion_def[1], num_ani))
        elif (neut == 'yes'):
            tleap_solvate.write('# Add ions for neutralization/ionization\n')
            if neu_cat != 0:
                tleap_solvate.write('addionsrand outside_wat %s %d\n' % (ion_def[0], neu_cat))
            if neu_ani != 0:
                tleap_solvate.write('addionsrand outside_wat %s %d\n' % (ion_def[1], neu_ani))

        tleap_solvate.write('model = combine {dum prot others outside_wat}\n\n')

        tleap_solvate.write('\n')
        tleap_solvate.write('set model box {%.6f %.6f %.6f}\n' % (box_final[0], box_final[1], box_final[2]))
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

        u = mda.Universe('full.pdb')
        u_vac = mda.Universe('vac.pdb')
        # regenerate full.pdb resid indices
        renum_txt = f'../{self.build_file_folder}/protein_renum.txt'
        if not os.path.exists(renum_txt):
            renum_txt = f'{self.build_file_folder}/protein_renum.txt'

        renum_data = pd.read_csv(
            renum_txt,
            sep=r'\s+',
            header=None,
            names=['old_resname', 'old_chain', 'old_resid',
                    'new_resname', 'new_resid'])

        u.select_atoms('protein').residues.resids = renum_data['old_resid'].values
        u_vac.select_atoms('protein').residues.resids = renum_data['old_resid'].values
        
        # regenerate segments
        seg_txt = 'build_amber_renum.txt'

        segment_renum_data = pd.read_csv(
            seg_txt,
            sep=r'\s+',
            header=None,
            names=['old_resname', 'old_chain', 'old_resid',
                'new_resname', 'new_resid'])
        
        chain_list = renum_data.old_chain.values
        #for res, chain in zip(u.residues[:len(chain_list)], chain_list):
        #    res.atoms.chainIDs = chain  # set chainID for each residue
        #u.residues[len(chain_list):].atoms.chainIDs = 'Z'  # set chainID for extra residues

        u_chain_segments = {}
        for chain in chain_list:
            u_chain_segments[chain] = u.add_Segment(segid=chain)

        for res, chain in zip(u.residues[:len(chain_list)], chain_list):
            res.segment = u_chain_segments[chain]  # assign each residue to its segment

        u.atoms.write('full.pdb')
        u_vac.atoms.write('vac.pdb')

        # Apply hydrogen mass repartitioning
        #shutil.copy(f'{self.amber_files_folder}/parmed-hmr.in', './')
        os.system(f'cp {self.amber_files_folder}/parmed-hmr.in ./')
        run_with_log('parmed -O -n -i parmed-hmr.in > parmed-hmr.log')

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


class EquilibrationBuilder(SystemBuilder):
    stage = 'equil'
    win = -1
    window_folder = '.'
    comp = 'q'
    comp_folder = '.'
    sdr_dist = 0
    dec_method = ''

    @property
    def build_file_folder(self):
        return 'build_files'
    
    @property
    def amber_files_folder(self):
        return 'amber_files'
    
    @property
    def run_files_folder(self):
        return 'run_files'

    @log_info
    def _build_complex(self):
        # only one window for equilibration

        # no ligand copy for equilibration
        sdr_dist = self.sdr_dist
        
        H1 = self.sim_config.H1
        H2 = self.sim_config.H2
        H3 = self.sim_config.H3
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis

        shutil.copytree(build_files_orig, '.', dirs_exist_ok=True)

        all_pose_folder = self.system.poses_folder
        system_name = self.system.system_name

        #shutil.copy(f'{all_pose_folder}/reference.pdb',
        #            f'reference.pdb')
        os.system(f'cp {all_pose_folder}/reference.pdb reference.pdb')
        #shutil.copy(f'{all_pose_folder}/{system_name}_docked.pdb',
        #            f'rec_file.pdb')
        os.system(f'cp {all_pose_folder}/{system_name}_docked.pdb rec_file.pdb')
        #shutil.copy(f'{all_pose_folder}/{self.pose}.pdb',
        #            f'.')
        os.system(f'cp {all_pose_folder}/{self.pose}.pdb .')

        other_mol = self.sim_config.other_mol
        lipid_mol = self.sim_config.lipid_mol
        solv_shell = self.sim_config.solv_shell
        mol_u = mda.Universe(f'{self.pose}.pdb')
        if len(set(mol_u.residues.resnames)) > 1:
            raise ValueError(f'The ligand {self.pose} has more than one residue: '
                             f'{mol_u.atoms.resnames}')
        mol = mol_u.residues[0].resname
        if mol in other_mol:
            raise ValueError(f'The ligand {mol}'
                             f'cannot be in the other_mol list: '
                             f'{other_mol}')
        # rename pose atom name
        #shutil.copy(f'../../ff/{mol.lower()}.mol2', '.')
        os.system(f'cp ../../ff/{mol.lower()}.mol2 .')

        ante_mol = mda.Universe(f'{mol.lower()}.mol2')
        mol_u.atoms.names = ante_mol.atoms.names
        mol_u.atoms.residues.resnames = mol
        mol_u.atoms.write(f'{mol.lower()}.pdb')

        # rename pose param
        # shutil.copy(f'../../ff/ligand.frcmod', f'../../ff/{mol.lower()}.frcmod')
        # shutil.copy(f'../../ff/ligand.mol2', f'../../ff/{mol.lower()}.mol2')

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
                                   .replace('MMM', f"\'{mol}\'"))
        run_with_log(f'{vmd} -dispdev text -e split.tcl', error_match='syntax error')

        #shutil.copy('./protein.pdb', './protein_vmd.pdb')
        os.system(f'cp protein.pdb protein_vmd.pdb')
        run_with_log('pdb4amber -i protein_vmd.pdb -o protein.pdb -y')
        renum_txt = 'protein_renum.txt'

        renum_data = pd.read_csv(
            renum_txt,
            sep=r'\s+',
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
            logger.debug('Receptor is not set as chain A; '
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

        logger.debug(f'Receptor anchors: P1: {P1}; P2: {P2}; P3: {P3}')
        logger.debug(f'Note this is not the same residue number from the receptor file '
                    'because the residues were renumbered from 1.')
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        protein_anchor_file = 'protein_anchors.txt'
        with open(protein_anchor_file, 'w') as f:
            f.write(f'{P1}\n')
            f.write(f'{P2}\n')
            f.write(f'{P3}\n')

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
            logger.debug(
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
                    fout.write(line.replace('MMM', f"\'{mol}\'").replace('mmm', mol.lower())
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
        run_with_log(f'{vmd} -dispdev text -e nochain.tcl')
        run_with_log('./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc')
        run_with_log(f'{vmd} -dispdev text -e measure-fit.tcl')

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
                sep=r'\s+',
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

        run_with_log(f'{vmd} -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            logger.warning('\n')
            logger.warning(f'WARNING: Could not find the ligand first anchor L1 for {self.pose}')
            logger.warning('The ligand is most likely not in the defined binding site in these systems.')
            return False

        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                logger.warning('\n')
                logger.warning(f'WARNING: Could not find the ligand L2 or L3 anchors for {self.pose}')
                logger.warning('Try reducing the min_adis parameter in the input file.')
                return False

        os.rename('./anchors.txt',
                  f'anchors-{self.pose}.txt')
        return True

    @log_info
    def _create_run_files(self):
        pass

    @log_info
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
        pose = self.pose
        other_mol = self.other_mol
        lipid_mol = self.lipid_mol
        ion_mol = ['Na+', 'K+', 'Cl-']

        first_res = self.first_res
        recep_resid_num = self.recep_resid_num
        P1 = self.P1
        P2 = self.P2
        P3 = self.P3

        # Copy a few files
        #shutil.copy(f'{self.build_file_folder}/equil-{mol.lower()}.pdb', './')
        os.system(f'cp {self.build_file_folder}/equil-{mol.lower()}.pdb ./')

        # Use equil-reference.pdb to retrieve the box size
        #shutil.copy(f'{self.build_file_folder}/equil-{mol.lower()}.pdb', './equil-reference.pdb')
        os.system(f'cp {self.build_file_folder}/equil-{mol.lower()}.pdb ./equil-reference.pdb')
        #shutil.copy(f'{self.build_file_folder}/{mol.lower()}-noh.pdb', f'./{mol.lower()}.pdb')
        os.system(f'cp {self.build_file_folder}/{mol.lower()}-noh.pdb ./{mol.lower()}.pdb')
        #shutil.copy(f'{self.build_file_folder}/anchors-{pose}.txt', './anchors.txt')
        os.system(f'cp {self.build_file_folder}/anchors-{pose}.txt ./anchors.txt')

        # Read coordinates for dummy atoms
        for i in range(1, 2):
            #shutil.copy(f'{self.build_file_folder}/dum{i}.pdb', './')
            os.system(f'cp {self.build_file_folder}/dum{i}.pdb ./')
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
                if molecule not in {mol, 'DUM', 'WAT'} and molecule not in other_mol and molecule not in lipid_mol and molecule not in ion_mol:
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
                elif (molecule == 'WAT') or (molecule in other_mol) or (molecule in ion_mol):
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

        logger.debug(f'Ligand anchors: L1: {L1}; L2: {L2}; L3: {L3}')

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

    @log_info
    def _restraints(self):
        # TODO: remove it
        os.chdir('..')
        pose = self.pose
        rest = self.sim_config.rest
        bb_start = self.sim_config.bb_start
        bb_end = self.sim_config.bb_end
        stage = self.stage
        mol = self.mol
        comp = self.comp
        molr = self.mol

        bb_equil = self.sim_config.bb_equil
        sdr_dist = self.sim_config.sdr_dist
        dec_method = self.dec_method

        other_mol = self.other_mol

        release_eq = self.sim_config.release_eq
        logger.debug('Equil release weights:')
        for i in range(0, len(release_eq)):
            weight = release_eq[i]
            logger.debug('%s' % str(weight))
            setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                             molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
            #shutil.copy('./'+pose+'/disang.rest', './'+pose+'/disang%02d.rest' % int(i))
            os.system(f'cp ./{pose}/disang.rest ./{pose}/disang{i:02d}.rest')
        #shutil.copy('./'+pose+'/disang%02d.rest' % int(0), './'+pose+'/disang.rest')
        os.system(f'cp ./{pose}/disang00.rest ./{pose}/disang.rest')
        os.chdir(pose)

    @log_info
    def _sim_files(self):
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = len(self.sim_config.release_eq)
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.eq_steps1
        steps2 = self.sim_config.eq_steps2
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        
        # Find anchors
        with open('disang.rest', 'r') as f:
            data = f.readline().split()
            L1 = data[6].strip()
            L2 = data[7].strip()
            L3 = data[8].strip()

        # Get number of atoms in vacuum
        with open('./vac.pdb') as myfile:
            data = myfile.readlines()
            vac_atoms = data[-3][6:11].strip()

        # Create minimization and NPT equilibration files for big box and small ligand box
        with open(f"{self.amber_files_folder}/mini.in", "rt") as fin:
            with open("./mini.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_lig_name_', mol))
        with open(f"{self.amber_files_folder}/eqnvt.in", "rt") as fin:
            with open("./eqnvt.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
        with open(f"{self.amber_files_folder}/eqnpt0.in", "rt") as fin:
            with open("./eqnpt0.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
        with open(f"{self.amber_files_folder}/eqnpt.in", "rt") as fin:
            with open("./eqnpt.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))

        # Create gradual release files for equilibrium
        for i in range(0, num_sim):
            with open(f'{self.amber_files_folder}/mdin-equil', "rt") as fin:
                with open(f"./mdin-{i:02d}", "wt") as fout:
                    # when no restraint is applied
                    # run longer
                    if self.sim_config.release_eq[i] == 0:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                '_num-atoms_', str(vac_atoms)).replace(
                            '_lig_name_', mol).replace('_num-steps_', str(steps2)).replace('disang_file', f'disang{i:02d}'))
                    else:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                '_num-atoms_', str(vac_atoms)).replace(
                            '_lig_name_', mol).replace('_num-steps_', str(steps1)).replace('disang_file', f'disang{i:02d}'))
                            
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-equil.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('RANGE', str(rng)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace(
                        'STAGE', stage).replace(
                            'POSE', pose).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))
    
    def _find_anchor(self):
        """
        Probably find anchor in equil and fe
        builders can be combined
        """
        # TODO
        raise NotImplementedError('Not implemented yet')


class FreeEnergyBuilder(SystemBuilder):
    stage = 'fe'
    def __init__(self,
                 win: Union[int, str],
                 component: str,
                 system: "batter.System",
                 pose: str,
                 sim_config: SimulationConfig,
                 working_dir: str,
                 molr: str,
                 poser: str,):
        self.win = win
        self.comp = component
        self.molr = molr
        self.poser = poser

        super().__init__(system, pose, sim_config, working_dir)

        dec_method = self.sim_config.dec_method
        if component == 'n':
            dec_method = 'sdr'

        if component in ['a', 'l', 't', 'm', 'c', 'r']:
            dec_method = 'dd'

        if component == 'x':
            dec_method = 'exchange'
            
        self.dec_method = dec_method

        os.makedirs(f"{self.working_dir}/{pose}/{COMPONENTS_FOLDER_DICT[component]}", exist_ok=True)
        self.comp_folder = f"{COMPONENTS_FOLDER_DICT[component]}"
        self.window_folder = f"{self.comp}{self.win:02d}"

        self.lipid_mol = self.sim_config.lipid_mol
        if self.lipid_mol:
            # This will not effect SDR/DD
            # because semi-isotropic barostat is not supported
            # with TI simulations
            # self.p_coupling = '3'
            # self.c_surften = '3'

            # For all systems
            # We will use NVT ensemble during TI simulations
            # self.p_coupling = '0'
            # self.c_surften = '0'
            
            # For all systems
            # We will use NPT isotropic p-coup ensemble during TI simulations
            self.p_coupling = '1'
            self.c_surften = '0'
        else:
            # self.p_coupling = '1'
            # self.c_surften = '0'
            self.p_coupling = '1'
            self.c_surften = '0'
        
    @log_info
    def _build_complex(self):
        """
        Copying files from equilibration
        """
        pose = self.pose
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        
        # sim config values
        solv_shell = self.sim_config.solv_shell
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis
        sdr_dist = self.sim_config.sdr_dist

        shutil.copytree(build_files_orig, '.', dirs_exist_ok=True)

        #shutil.copy(f'../../../../equil/{pose}/build_files/{self.pose}.pdb', './')
        os.system(f'cp ../../../../equil/{pose}/build_files/{self.pose}.pdb ./')
        # Get last state from equilibrium simulations
        #shutil.copy(f'../../../../equil/{pose}/representative.rst7', './')
        os.system(f'cp ../../../../equil/{pose}/representative.rst7 ./')
        #shutil.copy(f'../../../../equil/{pose}/representative.pdb', './aligned-nc.pdb')
        os.system(f'cp ../../../../equil/{pose}/representative.pdb ./aligned-nc.pdb')
        #shutil.copy(f'../../../../equil/{pose}/build_amber_renum.txt', './')
        os.system(f'cp ../../../../equil/{pose}/build_amber_renum.txt ./')
        os.system(f'cp ../../../../equil/{pose}/build_files/protein_renum.txt ./')
        if not os.path.exists('protein_renum.txt'):
            raise FileNotFoundError(f'protein_renum.txt not found in {os.getcwd()}')


        for file in glob.glob(f'../../../../equil/{pose}/full*.prmtop'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob(f'../../../../equil/{pose}/vac*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        
        mol = mda.Universe(f'{self.pose}.pdb').residues[0].resname
        self.mol = mol

        run_with_log(f'{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb')
        renum_data = pd.read_csv('build_amber_renum.txt', sep=r'\s+',
                header=None, names=['old_resname',
                                    'old_chain',
                                    'old_resid',
                                    'new_resname', 'new_resid'])

        u = mda.Universe('rec_file.pdb')
        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum_data.query('old_resid == @resid_str').old_chain.values[0]

        if lipid_mol:
            # fix lipid resids
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
            revised_resids = np.array(revised_resids)
            total_residues = u.atoms.residues.n_residues
            final_resids = np.zeros(total_residues, dtype=int)
            final_resids[:len(revised_resids)] = revised_resids
            next_resnum = revised_resids[-1] + 1
            final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
            u.atoms.residues.resids = final_resids

        u.atoms.write('rec_file.pdb')

        # Used for retrieving the box size
        #shutil.copy('rec_file.pdb', 'equil-reference.pdb')
        os.system('cp rec_file.pdb equil-reference.pdb')

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
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', mol.lower())
                    .replace('MMM', f"\'{mol}\'"))
        run_with_log(f'{vmd} -dispdev text -e split.tcl')

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                     'protein.pdb',
                    f'{mol.lower()}.pdb',
                     'lipids.pdb',
                     'others.pdb',
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

        # Read protein anchors and size from equilibrium
        with open(f'../../../../equil/{pose}/equil-{mol.lower()}.pdb', 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('MMM', f"\'{mol}\'")
                        .replace('mmm', mol.lower())
                        .replace('NN', p1_atom)
                        .replace('P1A', p1_vmd)
                        .replace('FIRST', '2')
                        .replace('LAST', str(rec_res))
                        .replace('STAGE', 'fe')
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

        # Align to reference (equilibrium) structure using VMD's measure fit
        # For FE, to avoid membrane rotation inside the box
        # due to alignment, we just use ues the input structure as the reference
        run_with_log(f'{vmd} -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # fix lipid resids
        if lipid_mol:
            u = mda.Universe('aligned_amber.pdb')
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
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
            # set correct residue number
            revised_resids = np.array(revised_resids)
            u.atoms.residues.resids = final_resids[:len(revised_resids)]

            u.atoms.write('aligned_amber.pdb')
        
        run_with_log(f'{vmd} -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+pose+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+pose+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % mol.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % mol.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])
        return True

    @log_info
    def _create_run_files(self):
        hmr = self.sim_config.hmr

        if os.path.exists(self.run_files_folder):
            shutil.rmtree(self.run_files_folder, ignore_errors=True)
        shutil.copytree(run_files_orig, self.run_files_folder, dirs_exist_ok=True)
        if hmr == 'no':
            replacement = 'full.prmtop'
            for dname, dirs, files in os.walk(self.run_files_folder):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.hmr.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)
        elif hmr == 'yes':
            replacement = 'full.hmr.prmtop'
            for dname, dirs, files in os.walk(self.run_files_folder):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)

    @log_info
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
        mol = self.mol
        molr = self.molr
        poser = self.poser
        resname_lig = mol
        other_mol = self.other_mol
        lipid_mol = self.lipid_mol
        ion_mol = ['Na+', 'K+', 'Cl-']
        comp = self.comp
        sdr_dist = self.sim_config.sdr_dist

        dec_method = self.dec_method

        if os.path.exists(self.amber_files_folder) or os.path.islink(self.amber_files_folder):
            os.remove(self.amber_files_folder)

        os.symlink(f'../{self.amber_files_folder}', self.amber_files_folder)

        for file in glob.glob(f'../{self.build_file_folder}/vac_ligand*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        #shutil.copy(f'../{self.build_file_folder}/{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './build-ini.pdb')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./build-ini.pdb')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/anchors-{self.pose}.txt', './')
        os.system(f'cp ../{self.build_file_folder}/anchors-{self.pose}.txt ./')
        #shutil.copy(f'../{self.build_file_folder}/equil-reference.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/equil-reference.pdb ./')

        for file in glob.glob('../../../ff/*.mol2'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/*.frcmod'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/{mol.lower()}.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/dum.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')

        # Get TER statements
        ter_atom = []
        with open(f'../{self.build_file_folder}/rec_file.pdb') as oldfile, open('rec_file-clean.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'WAT' in line:
                    newfile.write(line)
        run_with_log('pdb4amber -i rec_file-clean.pdb -o rec_amber.pdb -y')
        with open('./rec_amber.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'TER'):
                ter_atom.append(int(lines[i][6:11].strip()))
                
        # Read coordinates for dummy atoms
        if dec_method == 'sdr' or dec_method == 'exchange':
            for i in range(1, 3):
                #shutil.copy(f'../{self.build_file_folder}/dum'+str(i)+'.pdb', './')
                os.system(f'cp ../{self.build_file_folder}/dum'+str(i)+'.pdb ./')
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
        else:
            for i in range(1, 2):
                #shutil.copy(f'../{self.build_file_folder}/dum'+str(i)+'.pdb', './')
                os.system(f'cp ../{self.build_file_folder}/dum'+str(i)+'.pdb ./')
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
        with open('build-ini.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Count atoms of the system
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                molecule = lines[i][17:21].strip() 
                if (molecule != mol) and (molecule != 'DUM') and (molecule != 'WAT') and (molecule not in other_mol) and (molecule not in lipid_mol) and (molecule not in ion_mol):
                    recep_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(molecule)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1
                elif molecule == mol:
                    lig_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(molecule)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1
                elif (molecule == 'WAT') or (molecule in other_mol) or (molecule in ion_mol):
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1
                elif molecule in lipid_mol:
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resid_list = [resid if resid < 10000 else (resid % 9999) + 1 for resid in resid_list]

        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
        lig_resid = recep_last + dum_atom
        oth_tmp = 'None'

        # Get coordinates from reference ligand
        if comp == 'x':
            #shutil.copy('../exchange_files/%s.pdb' % molr.lower(), './')
            os.system(f'cp ../exchange_files/{molr.lower()}.pdb ./')
            #shutil.copy('../exchange_files/anchors-'+poser+'.txt', './')
            os.system(f'cp ../exchange_files/anchors-{poser}.txt ./')
            #shutil.copy('../exchange_files/vac_ligand.pdb', './vac_reference.pdb')
            os.system(f'cp ../exchange_files/vac_ligand.pdb ./vac_reference.pdb')
            #shutil.copy('../exchange_files/vac_ligand.prmtop', './vac_reference.prmtop')
            os.system(f'cp ../exchange_files/vac_ligand.prmtop ./vac_reference.prmtop')
            #shutil.copy('../exchange_files/vac_ligand.inpcrd', './vac_reference.inpcrd')
            os.system(f'cp ../exchange_files/vac_ligand.inpcrd ./vac_reference.inpcrd')
            #shutil.copy('../exchange_files/fe-%s.pdb' % molr.lower(), './build-ref.pdb')
            os.system(f'cp ../exchange_files/fe-{molr.lower()}.pdb ./build-ref.pdb')

            ref_lig_coords = []
            ref_lig_atomlist = []
            ref_lig_rsnmlist = []
            ref_lig_rsidlist = []
            ref_lig_chainlist = []
            ref_lig_atom = 0
            ref_resid_lig = 0
            resname_lig = molr

            # Read coordinates from reference system
            with open('build-ref.pdb') as f_in:
                lines = (line.rstrip() for line in f_in)
                lines = list(line for line in lines if line)  # Non-blank lines in a list

            # Count atoms of the system
            for i in range(0, len(lines)):
                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                    if lines[i][17:20].strip() == molr:
                        ref_lig_coords.append((float(lines[i][30:38].strip()), float(
                            lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                        ref_lig_atomlist.append(lines[i][12:16].strip())
                        ref_lig_rsnmlist.append(lines[i][17:20].strip())
                        ref_lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                        ref_lig_chainlist.append(lines[i][21].strip())
                        ref_lig_atom += 1

        # Write the new pdb file

        build_file = open('build.pdb', 'w')

        # Positions for the dummy atoms
        for i in range(0, dum_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        chain_tmp = 'None'
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
            j = i + 2 - dum_atom
            if j in ter_atom:
                build_file.write('TER\n')

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):
            if comp == 'n':
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]+sdr_dist)))
                build_file.write('%6.2f%6.2f\n' % (0, 0))
            elif comp != 'r':
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                build_file.write('%6.2f%6.2f\n' % (0, 0))

        if comp != 'r':
            build_file.write('TER\n')

        # Extra guests for decoupling

        build_file = open('build.pdb', 'a')
        if (comp == 'e'):
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+1)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

            if dec_method == 'sdr' or dec_method == 'exchange':
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+2)))
                    build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                        lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+3)))
                    build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                        lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
        if (comp == 'v' or comp == 'o' or comp == 'z') and (dec_method == 'sdr' or dec_method == 'exchange'):
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid + 1)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                    lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        # Other ligands for relative calculations
        if (comp == 'x'):
            for i in range(0, ref_lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, ref_lig_atomlist[i], molr, ref_lig_chainlist[i], float(lig_resid + 1)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(ref_lig_coords[i][0]), float(
                    ref_lig_coords[i][1]), float(ref_lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            for i in range(0, ref_lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, ref_lig_atomlist[i], molr, ref_lig_chainlist[i], float(lig_resid + 2)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(ref_lig_coords[i][0]), float(
                    ref_lig_coords[i][1]), float(ref_lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+3)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                    lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        # Positions of the other atoms
        for i in range(0, oth_atom):
            if oth_rsidlist[i] != oth_tmp:
                build_file.write('TER\n')
            oth_tmp = oth_rsidlist[i]
            oth_tmp = oth_tmp if oth_tmp < 10000 else (oth_tmp % 9999) + 1
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, oth_atomlist[i], oth_rsnmlist[i], oth_chainlist[i], oth_tmp))
            build_file.write('%8.3f%8.3f%8.3f' %
                             (float(oth_coords[i][0]), float(oth_coords[i][1]), float(oth_coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')
        build_file.write('END\n')
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
        
        if (comp == 'f' or comp == 'w' or comp == 'c'):
            # Create system with one or two ligands
            build_file = open('build.pdb', 'w')
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            if comp == 'f':
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, chain_list[i], float(lig_resid + 1)))
                    build_file.write('%8.3f%8.3f%8.3f' %
                                     (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
            build_file.write('END\n')
            build_file.close()
            #shutil.copy('./build.pdb', './%s.pdb' % mol.lower())
            os.system(f'cp ./build.pdb ./{mol.lower()}.pdb')
            tleap_vac = open('tleap_vac.in', 'w')
            tleap_vac.write('source leaprc.'+ligand_ff+'\n\n')
            tleap_vac.write('# Load the ligand parameters\n')
            tleap_vac.write('loadamberparams %s.frcmod\n' % (mol.lower()))
            tleap_vac.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
            tleap_vac.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
            tleap_vac.write('check model\n')
            tleap_vac.write('savepdb model vac.pdb\n')
            tleap_vac.write('saveamberparm model vac.prmtop vac.inpcrd\n')
            tleap_vac.write('quit\n\n')
            tleap_vac.close()

            p = run_with_log(tleap + ' -s -f tleap_vac.in > tleap_vac.log')
        
    @log_info
    def _restraints(self):
        # TODO: Refactor this method
        # This is just a hack to avoid the restraints for lambda windows
        # when win is not 0
        if self.win != -1 and COMPONENTS_LAMBDA_DICT[self.comp] == 'lambdas':
            return
        pose = self.pose
        rest = self.sim_config.rest
        bb_start = self.sim_config.bb_start
        bb_end = self.sim_config.bb_end
        stage = self.stage
        mol = self.mol
        molr = self.molr
        comp = self.comp
        bb_equil = self.sim_config.bb_equil
        sdr_dist = self.sim_config.sdr_dist
        dec_method = self.sim_config.dec_method
        other_mol = self.other_mol
        lambdas_comp = self.sim_config.dict()[COMPONENTS_LAMBDA_DICT[self.comp]]
        weight = lambdas_comp[self.win]

        rst = []
        atm_num = []
        mlines = []
        hvy_h = []
        hvy_g = []
        hvy_g2 = []
        msk = []
        pdb_file = ('vac.pdb')
        ligand_pdb_file = ('vac_ligand.pdb')
        reflig_pdb_file = ('vac_reference.pdb')

        # Restraint identifiers
        recep_tr = '#Rec_TR'
        recep_c = '#Rec_C'
        recep_d = '#Rec_D'
        lign_tr = '#Lig_TR'
        lign_c = '#Lig_C'
        lign_d = '#Lig_D'

        # Find anchors
        with open(stage+'-%s.pdb' % mol.lower(), 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            p1_res = P1.split('@')[0][1:]
            p2_res = P2.split('@')[0][1:]
            p3_res = P3.split('@')[0][1:]
            p1_atom = P1.split('@')[1]
            p2_atom = P2.split('@')[1]
            p3_atom = P3.split('@')[1]
            L1 = data[5].strip()
            L2 = data[6].strip()
            L3 = data[7].strip()
            l1_atom = L1.split('@')[1]
            l2_atom = L2.split('@')[1]
            l3_atom = L3.split('@')[1]
            lig_res = L1.split('@')[0][1:]
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get backbone atoms and adjust anchors
        if (comp != 'c' and comp != 'r' and comp != 'f' and comp != 'w'):

            # Get protein backbone atoms
            with open('./vac.pdb') as f_in:
                lines = (line.rstrip() for line in f_in)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                for i in range(0, len(lines)):
                    if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                        if int(lines[i][22:26].strip()) >= 2 and int(lines[i][22:26].strip()) < int(lig_res):
                            data = lines[i][12:16].strip()
                            if data == 'CA' or data == 'N' or data == 'C' or data == 'O':
                                hvy_h.append(lines[i][6:11].strip())

            if dec_method == 'sdr' or dec_method == 'exchange':
                if (comp == 'e' or comp == 'v' or comp == 'n' or comp == 'x' or comp == 'o' or comp == 'z'):

                    rec_res = int(recep_last) + 2
                    lig_res = str((int(lig_res) + 1))
                    L1 = ':'+lig_res+'@'+l1_atom
                    L2 = ':'+lig_res+'@'+l2_atom
                    L3 = ':'+lig_res+'@'+l3_atom
                    hvy_h = []
                    hvy_g = []

                    # Adjust anchors

                    p1_resid = str(int(p1_res) + 1)
                    p2_resid = str(int(p2_res) + 1)
                    p3_resid = str(int(p3_res) + 1)

                    P1 = ":"+p1_resid+"@"+p1_atom
                    P2 = ":"+p2_resid+"@"+p2_atom
                    P3 = ":"+p3_resid+"@"+p3_atom

                    # Get receptor heavy atoms
                    with open('./vac.pdb') as f_in:
                        lines = (line.rstrip() for line in f_in)
                        lines = list(line for line in lines if line)  # Non-blank lines in a list
                        for i in range(0, len(lines)):
                            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                if int(lines[i][22:26].strip()) >= 3 and int(lines[i][22:26].strip()) <= rec_res:
                                    data = lines[i][12:16].strip()
                                    if data == 'CA' or data == 'N' or data == 'C' or data == 'O':
                                        hvy_h.append(lines[i][6:11].strip())

                    # Get bulk ligand heavy atoms
                    with open('./vac.pdb') as f_in:
                        lines = (line.rstrip() for line in f_in)
                        lines = list(line for line in lines if line)  # Non-blank lines in a list
                        if comp == 'x':
                            for i in range(0, len(lines)):
                                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                    if lines[i][22:26].strip() == str(int(lig_res) + 3):
                                        data = lines[i][12:16].strip()
                                        if data[0] != 'H':
                                            hvy_g.append(lines[i][6:11].strip())
                            for i in range(0, len(lines)):
                                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                    if lines[i][22:26].strip() == str(int(lig_res) + 1):
                                        data = lines[i][12:16].strip()
                                        if data[0] != 'H':
                                            hvy_g2.append(lines[i][6:11].strip())
                        if comp == 'e':
                            for i in range(0, len(lines)):
                                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                    if lines[i][22:26].strip() == str(int(lig_res) + 2):
                                        data = lines[i][12:16].strip()
                                        if data[0] != 'H':
                                            hvy_g.append(lines[i][6:11].strip())
                        if comp == 'v' or comp == 'o' or comp == 'z':
                            for i in range(0, len(lines)):
                                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                    if lines[i][22:26].strip() == str(int(lig_res) + 1):
                                        data = lines[i][12:16].strip()
                                        if data[0] != 'H':
                                            hvy_g.append(lines[i][6:11].strip())
                        if comp == 'n':
                            for i in range(0, len(lines)):
                                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                                    if lines[i][22:26].strip() == str(int(lig_res)):
                                        data = lines[i][12:16].strip()
                                        if data[0] != 'H':
                                            hvy_g.append(lines[i][6:11].strip())

        # Adjust anchors for ligand only
        if (comp == 'c' or comp == 'w' or comp == 'f'):
            L1 = L1.replace(':'+lig_res, ':1')
            L2 = L2.replace(':'+lig_res, ':1')
            L3 = L3.replace(':'+lig_res, ':1')

        # Get a relation between atom number and masks
        atm_num = scripts.num_to_mask(pdb_file)
        ligand_atm_num = scripts.num_to_mask(ligand_pdb_file)

        # Get number of ligand atoms
        with open('./vac_ligand.pdb') as myfile:
            data = myfile.readlines()
            vac_atoms = int(data[-3][6:11].strip())

        # Get number of reference ligand atoms
        if comp == 'x':
            with open('./vac_reference.pdb') as myfile:
                data = myfile.readlines()
                ref_atoms = int(data[-3][6:11].strip())

        # Define anchor atom distance restraints on the protein

        rst.append(''+P1+' '+P2+'')
        rst.append(''+P2+' '+P3+'')
        rst.append(''+P3+' '+P1+'')

        # Define protein dihedral restraints in the given range
        nd = 0
        for i in range(0, len(bb_start)):
            beg = bb_start[i] - int(first_res) + 2
            end = bb_end[i] - int(first_res) + 2
            if dec_method == 'sdr' or dec_method == 'exchange':
                if (comp == 'e' or comp == 'v' or comp == 'n' or comp == 'x' or comp == 'o' or comp == 'z'):
                    beg = bb_start[i] - int(first_res) + 3
                    end = bb_end[i] - int(first_res) + 3
            for i in range(beg, end):
                j = i+1
                psi1 = ':'+str(i)+'@N'
                psi2 = ':'+str(i)+'@CA'
                psi3 = ':'+str(i)+'@C'
                psi4 = ':'+str(j)+'@N'
                psit = '%s %s %s %s' % (psi1, psi2, psi3, psi4)
                rst.append(psit)
                nd += 1
                phi1 = ':'+str(i)+'@C'
                phi2 = ':'+str(j)+'@N'
                phi3 = ':'+str(j)+'@CA'
                phi4 = ':'+str(j)+'@C'
                phit = '%s %s %s %s' % (phi1, phi2, phi3, phi4)
                rst.append(phit)
                nd += 1

        # Define translational/rotational and anchor atom distance restraints on the ligand

        rst.append(''+P1+' '+L1+'')
        rst.append(''+P2+' '+P1+' '+L1+'')
        rst.append(''+P3+' '+P2+' '+P1+' '+L1+'')
        rst.append(''+P1+' '+L1+' '+L2+'')
        rst.append(''+P2+' '+P1+' '+L1+' '+L2+'')
        rst.append(''+P1+' '+L1+' '+L2+' '+L3+'')

        # New restraints for ligand only
        if (comp == 'c' or comp == 'w' or comp == 'f'):
            rst = []

        # Get ligand dihedral restraints from ligand parameter/pdb file

        spool = 0
        with open('./vac_ligand.prmtop') as fin:
            lines = (line.rstrip() for line in fin)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
            for line in lines:
                if 'FLAG DIHEDRALS_WITHOUT_HYDROGEN' in line:
                    spool = 1
                elif 'FLAG EXCLUDED_ATOMS_LIST' in line:
                    spool = 0
                if spool != 0 and (len(line.split()) > 3):
                    mlines.append(line)

        for i in range(0, len(mlines)):
            data = mlines[i].split()
            if int(data[3]) > 0:
                anum = []
                for j in range(0, len(data)):
                    anum.append(abs(int(data[j])//3)+1)
                msk.append('%s %s %s %s' % (
                    ligand_atm_num[anum[0]], ligand_atm_num[anum[1]], ligand_atm_num[anum[2]], ligand_atm_num[anum[3]]))

        for i in range(0, len(mlines)):
            data = mlines[i].split()
            if len(data) > 7:
                if int(data[8]) > 0:
                    anum = []
                    for j in range(0, len(data)):
                        anum.append(abs(int(data[j])//3)+1)
                    msk.append('%s %s %s %s' % (
                        ligand_atm_num[anum[5]], ligand_atm_num[anum[6]], ligand_atm_num[anum[7]], ligand_atm_num[anum[8]]))

        excl = msk[:]
        ind = 0
        mat = []
        for i in range(0, len(excl)):
            data = excl[i].split()
            for j in range(0, len(excl)):
                if j == i:
                    break
                data2 = excl[j].split()
                if (data[1] == data2[1] and data[2] == data2[2]) or (data[1] == data2[2] and data[2] == data2[1]):
                    ind = 0
                    for k in range(0, len(mat)):
                        if mat[k] == j:
                            ind = 1
                    if ind == 0:
                        mat.append(j)

        for i in range(0, len(mat)):
            msk[mat[i]] = ''

        if (comp != 'c' and comp != 'w' and comp != 'f'):
            msk = list(filter(None, msk))
            msk = [m.replace(':1', ':'+lig_res) for m in msk]
        else:
            msk = list(filter(None, msk))

        # Remove dihedral restraints on sp carbons to avoid crashes
        sp_carb = []
        with open('./'+mol.lower()+'.mol2') as fin:
            lines = (line.rstrip() for line in fin)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
            for line in lines:
                data = line.split()
                if len(data) > 6:
                    if data[5] == 'cg' or data[5] == 'c1':
                        sp_carb.append(data[1])
        for i in range(0, len(msk)):
            rem_dih = 0
            data = msk[i].split()
            for j in range(0, len(sp_carb)):
                atom_name1 = data[1].split('@')[1]
                atom_name2 = data[2].split('@')[1]
                if atom_name1 == sp_carb[j] or atom_name2 == sp_carb[j]:
                    rem_dih = 1
                    break
            if rem_dih == 0:
                rst.append(msk[i])

        # New restraints for protein only
        if (comp == 'r'):
            rst = []
            rst.append(''+P1+' '+P2+'')
            rst.append(''+P2+' '+P3+'')
            rst.append(''+P3+' '+P1+'')
            nd = 0
            for i in range(beg, end):
                j = i+1
                psi1 = ':'+str(i)+'@N'
                psi2 = ':'+str(i)+'@CA'
                psi3 = ':'+str(i)+'@C'
                psi4 = ':'+str(j)+'@N'
                psit = '%s %s %s %s' % (psi1, psi2, psi3, psi4)
                rst.append(psit)
                nd += 1
                phi1 = ':'+str(i)+'@C'
                phi2 = ':'+str(j)+'@N'
                phi3 = ':'+str(j)+'@CA'
                phi4 = ':'+str(j)+'@C'
                phit = '%s %s %s %s' % (phi1, phi2, phi3, phi4)
                rst.append(phit)
                nd += 1

        # Get initial restraint values for references

        assign_file = open('assign.in', 'w')
        assign_file.write('%s  %s  %s  %s  %s  %s  %s\n' % ('# Anchor atoms', P1, P2, P3, L1, L2, L3))
        assign_file.write('parm full.hmr.prmtop\n')
        assign_file.write('trajin full.inpcrd\n')
        for i in range(0, len(rst)):
            arr = rst[i].split()
            if len(arr) == 2:
                assign_file.write('%s %s %s' % ('distance r'+str(i), rst[i], 'noimage out assign.dat\n'))
            if len(arr) == 3:
                assign_file.write('%s %s %s' % ('angle r'+str(i), rst[i], 'out assign.dat\n'))
            if len(arr) == 4:
                assign_file.write('%s %s %s' % ('dihedral r'+str(i), rst[i], 'out assign.dat\n'))

        assign_file.close()
        run_with_log(cpptraj + ' -i assign.in > assign.log')

        # Assign reference values for restraints
        with open('./assign.dat') as fin:
            lines = (line.rstrip() for line in fin)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
            vals = lines[1].split()
            vals.append(vals.pop(0))
            del vals[-1]

        # If chosen, apply initial reference for the protein backbone restraints
        if (stage == 'fe' and comp != 'c' and comp != 'w' and comp != 'f'):
            if (bb_equil == 'yes'):
                #shutil.copy('../../../../equil/'+pose+'/assign.dat', './assign-eq.dat')
                os.system(f'cp ../../../../equil/{pose}/assign.dat ./assign-eq.dat')
            else:
                #shutil.copy('./assign.dat', './assign-eq.dat')
                os.system(f'cp ./assign.dat ./assign-eq.dat')
            with open('./assign-eq.dat') as fin:
                lines = (line.rstrip() for line in fin)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                valse = lines[1].split()
                valse.append(valse.pop(0))
                del valse[-1]

        if comp == 'l' or comp == 'c':
            rdhf = rest[0]
            rdsf = rest[1]
            ldf = 0
            laf = 0
            ldhf = weight*rest[4]/100
            rcom = rest[5]
        elif comp == 'a' or comp == 'r':
            rdhf = weight*rest[0]/100
            rdsf = weight*rest[1]/100
            ldf = 0
            laf = 0
            ldhf = 0
            rcom = rest[5]
        elif comp == 't':
            rdhf = rest[0]
            rdsf = rest[1]
            ldf = weight*rest[2]/100
            laf = weight*rest[3]/100
            ldhf = rest[4]
            rcom = rest[5]
        elif comp == 'm':
            rdhf = weight*rest[0]/100
            rdsf = weight*rest[1]/100
            ldf = weight*rest[2]/100
            laf = weight*rest[3]/100
            ldhf = weight*rest[4]/100
            rcom = rest[5]
        elif comp == 'n':
            rdhf = weight*rest[0]/100
            rdsf = weight*rest[1]/100
            ldf = 0
            laf = 0
            ldhf = weight*rest[4]/100
            rcom = rest[5]
            lcom = rest[6]
        elif comp == 'v' or comp == 'e' or comp == 'w' or comp == 'f' or comp == 'x' or comp == 'o' or comp == 'z':
            rdhf = rest[0]
            rdsf = rest[1]
            ldf = rest[2]
            laf = rest[3]
            ldhf = rest[4]
            rcom = rest[5]
            lcom = rest[6]

        # Write AMBER restraint file for the full system
        if (comp != 'c' and comp != 'r' and comp != 'w' and comp != 'f'):
            disang_file = open('disang.rest', 'w')
            disang_file.write('%s  %s  %s  %s  %s  %s  %s  %s  %s \n' % ('# Anchor atoms', P1,
                            P2, P3, L1, L2, L3, 'stage = '+stage, 'weight = '+str(weight)))

            for i in range(0, len(rst)):
                data = rst[i].split()
                # Protein conformation (P1-P3 distance restraints)
                if i < 3:
                    if (stage != 'equil'):
                        if len(data) == 2:
                            nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(0.0), float(valse[i]), float(valse[i]), float(999.0), rdsf, rdsf, recep_c))
                    else:
                        if len(data) == 2:
                            nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(0.0), float(vals[i]), float(vals[i]), float(999.0), rdsf, rdsf, recep_c))
                # Protein conformation (backbone restraints)
                elif i >= 3 and i < 3+nd:
                    if (stage != 'equil'):
                        if len(data) == 4:
                            nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                                ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(valse[i]) - 180, float(valse[i]), float(valse[i]), float(valse[i]) + 180, rdhf, rdhf, recep_d))
                    else:
                        if len(data) == 4:
                            nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                                ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, rdhf, rdhf, recep_d))
                # Ligand translational/rotational restraints
                elif i >= 3+nd and i < 9+nd and comp != 'a':
                    if len(data) == 2:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldf, ldf, lign_tr))
                    elif len(data) == 3:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(0.0), float(vals[i]), float(vals[i]), float(180.0), laf, laf, lign_tr))
                    elif len(data) == 4:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, laf, laf, lign_tr))
                    if comp == 'e':
                        if i == (3+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldf, ldf, lign_tr))
                        if i == (4+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])
                                                                        )+','+str(atm_num.index(data[2])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(0.0), float(vals[i]), float(vals[i]), float(180.0), laf, laf, lign_tr))
                        if i == (5+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+',' + \
                                str(atm_num.index(data[2]))+','+str(atm_num.index(data[3])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, laf, laf, lign_tr))
                        if i == (6+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]) +
                                                                        vac_atoms)+','+str(atm_num.index(data[2])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(0.0), float(vals[i]), float(vals[i]), float(180.0), laf, laf, lign_tr))
                        if i == (7+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+',' + \
                                str(atm_num.index(data[2])+vac_atoms)+','+str(atm_num.index(data[3])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, laf, laf, lign_tr))
                        if i == (8+nd):
                            nums2 = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])+vac_atoms)+',' + \
                                str(atm_num.index(data[2])+vac_atoms)+','+str(atm_num.index(data[3])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, laf, laf, lign_tr))
                # Ligand conformation (non-hydrogen dihedrals)
                elif i >= 9+nd and comp != 'a':
                    if len(data) == 4:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                        if (comp == 'v' or comp == 'o' or comp == 'z') and (dec_method == 'sdr' or dec_method == 'exchange'):
                            nums2 = str(atm_num.index(data[0])+vac_atoms)+','+str(atm_num.index(data[1])+vac_atoms)+','+str(
                                atm_num.index(data[2])+vac_atoms)+','+str(atm_num.index(data[3])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                        if comp == 'x' and dec_method == 'exchange':
                            nums2 = str(atm_num.index(data[0])+vac_atoms+2*ref_atoms)+','+str(atm_num.index(data[1])+vac_atoms+2*ref_atoms)+','+str(
                                atm_num.index(data[2])+vac_atoms+2*ref_atoms)+','+str(atm_num.index(data[3])+vac_atoms+2*ref_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                        if comp == 'e':
                            nums2 = str(atm_num.index(data[0])+vac_atoms)+','+str(atm_num.index(data[1])+vac_atoms)+','+str(
                                atm_num.index(data[2])+vac_atoms)+','+str(atm_num.index(data[3])+vac_atoms)+','
                            disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                            disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                            if (dec_method == 'sdr' or dec_method == 'exchange'):
                                nums3 = str(atm_num.index(data[0])+2*vac_atoms)+','+str(atm_num.index(data[1])+2*vac_atoms)+','+str(
                                    atm_num.index(data[2])+2*vac_atoms)+','+str(atm_num.index(data[3])+2*vac_atoms)+','
                                disang_file.write('%s %-23s ' % ('&rst iat=', nums3))
                                disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                    float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                                nums4 = str(atm_num.index(data[0])+3*vac_atoms)+','+str(atm_num.index(data[1])+3*vac_atoms)+','+str(
                                    atm_num.index(data[2])+3*vac_atoms)+','+str(atm_num.index(data[3])+3*vac_atoms)+','
                                disang_file.write('%s %-23s ' % ('&rst iat=', nums4))
                                disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                                    float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))

            # COM restraints
            # TODO: why rcom leads to a crash?
            rcom = 0
            cv_file = open('cv.in', 'w')
            cv_file.write('cv_file \n')
            if False:
                cv_file.write('&colvar \n')
                cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
                cv_file.write(' cv_ni = %s, cv_i = 1,0,' % str(len(hvy_h)+2))
                for i in range(0, len(hvy_h)):
                    cv_file.write(hvy_h[i])
                    cv_file.write(',')
                cv_file.write('\n')
                cv_file.write(' anchor_position = %10.4f, %10.4f, %10.4f, %10.4f \n' %
                            (float(0.0), float(0.0), float(0.0), float(999.0)))
                cv_file.write(' anchor_strength = %10.4f, %10.4f, \n' % (rcom, rcom))
                cv_file.write('/ \n')
            if dec_method == 'sdr' or dec_method == 'exchange':
                if comp == 'e' or comp == 'v' or comp == 'n' or comp == 'x' or comp == 'o' or comp == 'z':
                    cv_file.write('&colvar \n')
                    cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
                    cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(hvy_g)+2))
                    for i in range(0, len(hvy_g)):
                        cv_file.write(hvy_g[i])
                        cv_file.write(',')
                    cv_file.write('\n')
                    cv_file.write(' anchor_position = %10.4f, %10.4f, %10.4f, %10.4f \n' %
                                (float(0.0), float(0.0), float(0.0), float(999.0)))
                    cv_file.write(' anchor_strength = %10.4f, %10.4f, \n' % (lcom, lcom))
                    cv_file.write('/ \n')
                if comp == 'x':
                    cv_file.write('&colvar \n')
                    cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
                    cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(hvy_g2)+2))
                    for i in range(0, len(hvy_g2)):
                        cv_file.write(hvy_g2[i])
                        cv_file.write(',')
                    cv_file.write('\n')
                    cv_file.write(' anchor_position = %10.4f, %10.4f, %10.4f, %10.4f \n' %
                                (float(0.0), float(0.0), float(0.0), float(999.0)))
                    cv_file.write(' anchor_strength = %10.4f, %10.4f, \n' % (lcom, lcom))
                    cv_file.write('/ \n')
            cv_file.close()

            # Analysis of simulations

            if (comp != 'l' and comp != 'a' and comp != 'm' and comp != 'n'):
                restraints_file = open('restraints.in', 'w')
                restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                    '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
                restraints_file.write('noexitonerror\n')
                restraints_file.write('parm vac.prmtop\n')
                # TODO: this is a hack to just read in all the potential trajectories
                # Do this properly when running the analysis
                for i in range(2, 11):
                    restraints_file.write('trajin md%02.0f.nc\n' % i)
         #            for i in range(1, 11):
         #               restraints_file.write('trajin mdin-%02.0f.nc\n' % i)
                for i in range(3+nd, 9+nd):
                    arr = rst[i].split()
                    if len(arr) == 2:
                        restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                    if len(arr) == 3:
                        restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                    if len(arr) == 4:
                        restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
            elif (comp == 'a'):
                restraints_file = open('restraints.in', 'w')
                restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                    '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
                restraints_file.write('noexitonerror\n')
                restraints_file.write('parm vac.prmtop\n')
                for i in range(2, 11):
                    restraints_file.write('trajin md%02.0f.nc\n' % i)
                for i in range(0, 3+nd):
                    arr = rst[i].split()
                    if len(arr) == 2:
                        restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                    if len(arr) == 3:
                        restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                    if len(arr) == 4:
                        restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
            elif (comp == 'l'):
                restraints_file = open('restraints.in', 'w')
                restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                    '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
                restraints_file.write('noexitonerror\n')
                restraints_file.write('parm vac.prmtop\n')
                for i in range(2, 11):
                    restraints_file.write('trajin md%02.0f.nc\n' % i)
                for i in range(9+nd, len(rst)):
                    arr = rst[i].split()
                    if len(arr) == 2:
                        restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                    if len(arr) == 3:
                        restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                    if len(arr) == 4:
                        restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
            elif (comp == 'm' or comp == 'n'):
                restraints_file = open('restraints.in', 'w')
                restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                    '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
                restraints_file.write('noexitonerror\n')
                restraints_file.write('parm vac.prmtop\n')
                for i in range(2, 11):
                    restraints_file.write('trajin md%02.0f.nc\n' % i)
                for i in range(0, len(rst)):
                    arr = rst[i].split()
                    if len(arr) == 2:
                        restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                    if len(arr) == 3:
                        restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                    if len(arr) == 4:
                        restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
        elif comp == 'f':
            while '' in rst:
                rst.remove('')
        # Write restraint file for ligand system
            disang_file = open('disang.rest', 'w')
            disang_file.write('%s  %s  %s  %s  %s  %s  %s  %s  %s \n' % ('# Anchor atoms', P1,
                            P2, P3, L1, L2, L3, 'stage = '+stage, 'weight = '+str(weight)))
            for i in range(0, len(rst)):
                data = rst[i].split()
                # Ligand conformational restraints
                if len(data) == 2:
                    nums = str(ligand_atm_num.index(data[0]))+','+str(ligand_atm_num.index(data[1]))+','
                    nums2 = str(ligand_atm_num.index(data[0])+vac_atoms)+',' + \
                        str(ligand_atm_num.index(data[1])+vac_atoms)+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldsf, ldsf, lign_c))
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldsf, ldsf, lign_c))
                elif len(data) == 4:
                    nums = str(ligand_atm_num.index(data[0]))+','+str(ligand_atm_num.index(data[1])) + \
                        ','+str(ligand_atm_num.index(data[2]))+','+str(ligand_atm_num.index(data[3]))+','
                    nums2 = str(ligand_atm_num.index(data[0])+vac_atoms)+','+str(ligand_atm_num.index(data[1])+vac_atoms)+','+str(
                        ligand_atm_num.index(data[2])+vac_atoms)+','+str(ligand_atm_num.index(data[3])+vac_atoms)+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
            # Analysis of simulations
            restraints_file = open('restraints.in', 'w')
            restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
            restraints_file.write('noexitonerror\n')
            restraints_file.write('parm vac.prmtop\n')
            for i in range(2, 11):
                restraints_file.write('trajin md%02.0f.nc\n' % i)
            for i in range(0, len(rst)):
                arr = rst[i].split()
                if len(arr) == 2:
                    restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                if len(arr) == 3:
                    restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                if len(arr) == 4:
                    restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
        elif comp == 'c' or comp == 'w':
            while '' in rst:
                rst.remove('')
            # Write restraint file for ligand system
            disang_file = open('disang.rest', 'w')
            disang_file.write('%s  %s  %s  %s  %s  %s  %s  %s  %s \n' % ('# Anchor atoms', P1,
                            P2, P3, L1, L2, L3, 'stage = '+stage, 'weight = '+str(weight)))
            for i in range(0, len(rst)):
                data = rst[i].split()
                # Ligand conformational restraints
                if len(data) == 2:
                    nums = str(ligand_atm_num.index(data[0]))+','+str(ligand_atm_num.index(data[1]))+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldsf, ldsf, lign_c))
                elif len(data) == 4:
                    nums = str(ligand_atm_num.index(data[0]))+','+str(ligand_atm_num.index(data[1])) + \
                        ','+str(ligand_atm_num.index(data[2]))+','+str(ligand_atm_num.index(data[3]))+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
            # Analysis of simulations
            restraints_file = open('restraints.in', 'w')
            restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
            restraints_file.write('noexitonerror\n')
            restraints_file.write('parm vac.prmtop\n')
            for i in range(2, 11):
                restraints_file.write('trajin md%02.0f.nc\n' % i)
            for i in range(0, len(rst)):
                arr = rst[i].split()
                if len(arr) == 2:
                    restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                if len(arr) == 3:
                    restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                if len(arr) == 4:
                    restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))
        elif comp == 'r':
            while '' in rst:
                rst.remove('')
            # Write restraint file for protein system
            disang_file = open('disang.rest', 'w')
            disang_file.write('%s  %s  %s  %s  %s  %s  \n' % ('# Anchor atoms', P1,
                            P2, P3, 'stage = '+stage, 'weight = '+str(weight)))
            for i in range(0, len(rst)):
                data = rst[i].split()
                # Protein conformational restraints
                if len(data) == 2:
                    nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(0.0), float(valse[i]), float(valse[i]), float(999.0), rdsf, rdsf, recep_c))
                if len(data) == 4:
                    nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+',' + \
                        str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                    disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                    disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                        float(valse[i]) - 180, float(valse[i]), float(valse[i]), float(valse[i]) + 180, rdhf, rdhf, recep_d))
            # Analysis of simulations
            restraints_file = open('restraints.in', 'w')
            restraints_file.write('%s  %s  %s  %s  %s  %s  %s  %s  \n' % (
                '# Anchor atoms', P1, P2, P3, L1, L2, L3, 'stage = '+stage))
            restraints_file.write('noexitonerror\n')
            restraints_file.write('parm vac.prmtop\n')
            for i in range(2, 11):
                restraints_file.write('trajin md%02.0f.nc\n' % i)
            for i in range(0, len(rst)):
                arr = rst[i].split()
                if len(arr) == 2:
                    restraints_file.write('%s %s %s' % ('distance d'+str(i), rst[i], 'noimage out restraints.dat\n'))
                if len(arr) == 3:
                    restraints_file.write('%s %s %s' % ('angle a'+str(i), rst[i], 'out restraints.dat\n'))
                if len(arr) == 4:
                    restraints_file.write('%s %s %s' % ('dihedral a'+str(i), rst[i], 'out restraints.dat\n'))

        if comp != 'x':
            disang_file.write('\n')
        disang_file.close()

        # Write additional restraints for reference ligand
        if comp == 'x':

            rst = []
            mlines = []
            msk = []

            # Get a relation between atom number and masks for reference ligand
            ligand_atm_num = scripts.num_to_mask(reflig_pdb_file)

            # Find reference ligand anchors
            with open('../../exchange_files/fe-%s.pdb' % molr.lower(), 'r') as f:
                data = f.readline().split()
                P1 = data[2].strip()
                P2 = data[3].strip()
                P3 = data[4].strip()
                p1_res = P1.split('@')[0][1:]
                p2_res = P2.split('@')[0][1:]
                p3_res = P3.split('@')[0][1:]
                p1_atom = P1.split('@')[1]
                p2_atom = P2.split('@')[1]
                p3_atom = P3.split('@')[1]
                L1 = data[5].strip()
                L2 = data[6].strip()
                L3 = data[7].strip()
                l1_atom = L1.split('@')[1]
                l2_atom = L2.split('@')[1]
                l3_atom = L3.split('@')[1]
                lig_res = L1.split('@')[0][1:]

            # Reference ligand TR restraints
            rst.append(''+P1+' '+L1+'')
            rst.append(''+P2+' '+P1+' '+L1+'')
            rst.append(''+P3+' '+P2+' '+P1+' '+L1+'')
            rst.append(''+P1+' '+L1+' '+L2+'')
            rst.append(''+P2+' '+P1+' '+L1+' '+L2+'')
            rst.append(''+P1+' '+L1+' '+L2+' '+L3+'')

            # Get ligand dihedral restraints from ligand parameter/pdb file

            spool = 0
            with open('./vac_reference.prmtop') as fin:
                lines = (line.rstrip() for line in fin)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                for line in lines:
                    if 'FLAG DIHEDRALS_WITHOUT_HYDROGEN' in line:
                        spool = 1
                    elif 'FLAG EXCLUDED_ATOMS_LIST' in line:
                        spool = 0
                    if spool != 0 and (len(line.split()) > 3):
                        mlines.append(line)

            for i in range(0, len(mlines)):
                data = mlines[i].split()
                if int(data[3]) > 0:
                    anum = []
                    for j in range(0, len(data)):
                        anum.append(abs(int(data[j])//3)+1)
                    msk.append('%s %s %s %s' % (
                        ligand_atm_num[anum[0]], ligand_atm_num[anum[1]], ligand_atm_num[anum[2]], ligand_atm_num[anum[3]]))

            for i in range(0, len(mlines)):
                data = mlines[i].split()
                if len(data) > 7:
                    if int(data[8]) > 0:
                        anum = []
                        for j in range(0, len(data)):
                            anum.append(abs(int(data[j])//3)+1)
                        msk.append('%s %s %s %s' % (
                            ligand_atm_num[anum[5]], ligand_atm_num[anum[6]], ligand_atm_num[anum[7]], ligand_atm_num[anum[8]]))

            excl = msk[:]
            ind = 0
            mat = []
            for i in range(0, len(excl)):
                data = excl[i].split()
                for j in range(0, len(excl)):
                    if j == i:
                        break
                    data2 = excl[j].split()
                    if (data[1] == data2[1] and data[2] == data2[2]) or (data[1] == data2[2] and data[2] == data2[1]):
                        ind = 0
                        for k in range(0, len(mat)):
                            if mat[k] == j:
                                ind = 1
                        if ind == 0:
                            mat.append(j)

            for i in range(0, len(mat)):
                msk[mat[i]] = ''

            msk = list(filter(None, msk))
            msk2 = [m.replace(':1', ':'+str(lig_res)) for m in msk]

            # Remove dihedral sp carbons to avoid crashes and write reference rst array
            sp_carb = []
            with open('./'+molr.lower()+'.mol2') as fin:
                lines = (line.rstrip() for line in fin)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                for line in lines:
                    data = line.split()
                    if len(data) > 6:
                        if data[5] == 'cg' or data[5] == 'c1':
                            sp_carb.append(data[1])
            for i in range(0, len(msk2)):
                rem_dih = 0
                data = msk2[i].split()
                for j in range(0, len(sp_carb)):
                    atom_name1 = data[1].split('@')[1]
                    atom_name2 = data[2].split('@')[1]
                    if atom_name1 == sp_carb[j] or atom_name2 == sp_carb[j]:
                        rem_dih = 1
                        break
                if rem_dih == 0:
                    rst.append(msk2[i])

            while '' in rst:
                rst.remove('')

            # Get initial restraint values for references

            #shutil.copy('../../exchange_files/rec_file.pdb', './')
            os.system('cp ../../exchange_files/rec_file.pdb ./')
            #shutil.copy('../../exchange_files/full.hmr.prmtop', './full-ref.hmr.prmtop')
            os.system('cp ../../exchange_files/full.hmr.prmtop ./full-ref.hmr.prmtop')
            assign_file = open('assign2.in', 'w')
            assign_file.write('%s  %s  %s  %s  %s  %s  %s\n' % ('# Anchor atoms', P1, P2, P3, L1, L2, L3))
            assign_file.write('parm full-ref.hmr.prmtop\n')
            assign_file.write('trajin rec_file.pdb\n')
            for i in range(0, len(rst)):
                arr = rst[i].split()
                if len(arr) == 2:
                    assign_file.write('%s %s %s' % ('distance r'+str(i), rst[i], 'noimage out assign2.dat\n'))
                if len(arr) == 3:
                    assign_file.write('%s %s %s' % ('angle r'+str(i), rst[i], 'out assign2.dat\n'))
                if len(arr) == 4:
                    assign_file.write('%s %s %s' % ('dihedral r'+str(i), rst[i], 'out assign2.dat\n'))

            assign_file.close()
            run_with_log(cpptraj + ' -i assign2.in > assign2.log')

            # Assign reference values for restraints
            with open('./assign2.dat') as fin:
                lines = (line.rstrip() for line in fin)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                vals = lines[1].split()
                vals.append(vals.pop(0))
                del vals[-1]

            # Define restraints with the simulation file numbering

            rst = []

            # Adjust numbering
            ref_res = str((int(lig_res) + 3))
            L1 = ':'+ref_res+'@'+l1_atom
            L2 = ':'+ref_res+'@'+l2_atom
            L3 = ':'+ref_res+'@'+l3_atom
            p1_resid = str(int(p1_res) + 1)
            p2_resid = str(int(p2_res) + 1)
            p3_resid = str(int(p3_res) + 1)
            P1 = ":"+p1_resid+"@"+p1_atom
            P2 = ":"+p2_resid+"@"+p2_atom
            P3 = ":"+p3_resid+"@"+p3_atom

            # Reference ligand TR restraints
            rst.append(''+P1+' '+L1+'')
            rst.append(''+P2+' '+P1+' '+L1+'')
            rst.append(''+P3+' '+P2+' '+P1+' '+L1+'')
            rst.append(''+P1+' '+L1+' '+L2+'')
            rst.append(''+P2+' '+P1+' '+L1+' '+L2+'')
            rst.append(''+P1+' '+L1+' '+L2+' '+L3+'')

            ref_res = int(lig_res) + 2
            msk = [m.replace(':1', ':'+str(ref_res)) for m in msk]

            # Remove dihedral sp carbons to avoid crashes and write the ligand dihedrals
            sp_carb = []
            with open('./'+molr.lower()+'.mol2') as fin:
                lines = (line.rstrip() for line in fin)
                lines = list(line for line in lines if line)  # Non-blank lines in a list
                for line in lines:
                    data = line.split()
                    if len(data) > 6:
                        if data[5] == 'cg' or data[5] == 'c1':
                            sp_carb.append(data[1])
            for i in range(0, len(msk)):
                rem_dih = 0
                data = msk[i].split()
                for j in range(0, len(sp_carb)):
                    atom_name1 = data[1].split('@')[1]
                    atom_name2 = data[2].split('@')[1]
                    if atom_name1 == sp_carb[j] or atom_name2 == sp_carb[j]:
                        rem_dih = 1
                        break
                if rem_dih == 0:
                    rst.append(msk[i])

            while '' in rst:
                rst.remove('')

            disang_file = open('disang.rest', 'a')
            for i in range(0, len(rst)):
                data = rst[i].split()
                # Ligand translational/rotational restraints
                if i < 6:
                    if len(data) == 2:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(0.0), float(vals[i]), float(vals[i]), float(999.0), ldf, ldf, lign_tr))
                    elif len(data) == 3:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(0.0), float(vals[i]), float(vals[i]), float(180.0), laf, laf, lign_tr))
                    elif len(data) == 4:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, laf, laf, lign_tr))
                # Ligand conformational restraints
                else:
                    if len(data) == 4:
                        nums = str(atm_num.index(data[0]))+','+str(atm_num.index(data[1])) + \
                            ','+str(atm_num.index(data[2]))+','+str(atm_num.index(data[3]))+','
                        nums2 = str(atm_num.index(data[0])+ref_atoms)+','+str(atm_num.index(data[1])+ref_atoms) + \
                            ','+str(atm_num.index(data[2])+ref_atoms)+','+str(atm_num.index(data[3])+ref_atoms)+','
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))
                        disang_file.write('%s %-23s ' % ('&rst iat=', nums2))
                        disang_file.write('r1= %10.4f, r2= %10.4f, r3= %10.4f, r4= %10.4f, rk2= %11.7f, rk3= %11.7f, &end %s \n' % (
                            float(vals[i]) - 180, float(vals[i]), float(vals[i]), float(vals[i]) + 180, ldhf, ldhf, lign_d))

            disang_file.write('\n')
            disang_file.close()

    @log_info
    def _sim_files(self):

        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        ntwx = self.sim_config.ntwx
        lipid_mol = self.lipid_mol

        # Find anchors
        with open('disang.rest', 'r') as f:
            data = f.readline().split()
            L1 = data[6].strip()
            L2 = data[7].strip()
            L3 = data[8].strip()

        # Get number of atoms in vacuum
        with open('./vac.pdb') as myfile:
            data = myfile.readlines()
            vac_atoms = data[-3][6:11].strip()

        # Create minimization and NPT equilibration files for big box and small ligand box
        if comp != 'c' and comp != 'r' and comp != 'n':
            with open(f"../{self.amber_files_folder}/mini.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_L1_', L1).replace('_L2_', L2).replace('_L3_', L3).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0-fe.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-fe.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
        elif (comp == 'r' or comp == 'c'):
            with open(f"../{self.amber_files_folder}/mini-lig.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        if not 'restraint' in line and not 'ntr = 1' in line:
                            fout.write(line)
            with open(f"../{self.amber_files_folder}/therm1-lig.in", "rt") as fin:
                with open("./therm1.in", "wt") as fout:
                    for line in fin:
                        if not 'restraint' in line and not 'ntr = 1' in line:
                            fout.write(line)
            with open(f"../{self.amber_files_folder}/therm2-lig.in", "rt") as fin:
                with open("./therm2.in", "wt") as fout:
                    for line in fin:
                        if not 'restraint' in line and not 'ntr = 1' in line:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0-lig.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-lig.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        if not 'restraint' in line and not 'ntr = 1' in line:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
        else:  # n component
            with open(f"../{self.amber_files_folder}/mini-sim.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_L1_', L1).replace('_L2_', L2).replace('_L3_', L3).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0-sim.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-sim.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            '_lig_name_', mol))

        if (comp != 'c' and comp != 'r' and comp != 'n'):
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-rest', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                '_num-atoms_', str(vac_atoms)).replace('_num-steps_', n_steps_run).replace('disang_file', 'disang'))
                mdin = open("./mdin-%02d" % int(i), "a")
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')
        elif (comp == 'r' or comp == 'c'):
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-lig', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                        '_num-atoms_', str(vac_atoms)).replace('_num-steps_', n_steps_run).replace('disang_file', 'disang'))

        else:  # n
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-sim', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                            '_num-atoms_', str(vac_atoms)).replace('_num-steps_', n_steps_run).replace('disang_file', 'disang'))
                mdin = open("./mdin-%02d" % int(i), "a")
                mdin.write('  infe = 0,\n')
                mdin.write(' /\n')
                mdin.write(' &pmd \n')
                mdin.write(' output_file = \'cmass.txt\' \n')
                mdin.write(' output_freq = %02d \n' % int(ntwx))
                mdin.write(' cv_file = \'cv.in\' \n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')

        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace(
                                    'POSE', '%s%02d' % (comp, int(win))).replace(
                                        'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))


class SDRFreeEnergyBuilder(FreeEnergyBuilder):
    def _sim_files(self):
        
        dec_method = self.dec_method
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        ntwx = self.sim_config.ntwx
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]


        # Read 'disang.rest' and extract L1, L2, L3
        with open('disang.rest', 'r') as f:
            data = f.readline().split()
            L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read 'vac.pdb' once
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get number of atoms in vacuum (third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Get the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():  # Compare residue name
                last_lig = line[22:26].strip()  # Extract residue number

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        if comp == 'v':
            # Create simulation files for vdw decoupling
            if (dec_method == 'sdr'):
                # Simulation files for simultaneous decoupling
                with open('./vac.pdb') as myfile:
                    data = myfile.readlines()
                    mk2 = int(last_lig)
                    mk1 = int(mk2 - 1)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-lj', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                    '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write(f'  mbar_states = {len(lambdas)}\n')
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt0-lj.in", "rt") as fin:
                    with open("./eqnpt0.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/eqnpt-lj.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/heat-lj.in", "rt") as fin:
                    with open("./heat.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/mini-lj", "rt") as fin:
                    with open("./mini.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))

            # Simulation files for double decoupling
            elif (dec_method == 'dd'):
                with open('./vac.pdb') as myfile:
                    data = myfile.readlines()
                    mk1 = int(last_lig)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-lj-dd', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                    '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write('  mbar_states = %02d\n' % len(lambdas))
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt-lj-dd.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/heat-lj-dd.in", "rt") as fin:
                    with open("./heat.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))

            # Create running scripts for local and server
            with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
                with open("./run_failures.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line)
            with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
                with open("./run-local.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('FERANGE', str(num_sim)))
            with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
                with open("./SLURMM-run", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))

        if (comp == 'e'):
            # Create simulation files for charge decoupling
            if (dec_method == 'sdr') or (dec_method == 'exchange'):
                # Simulation files for simultaneous decoupling
                with open('./vac.pdb') as myfile:
                    data = myfile.readlines()
                    mk4 = int(last_lig)
                    mk3 = int(mk4 - 1)
                    mk2 = int(mk4 - 2)
                    mk1 = int(mk4 - 3)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-ch', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace('_num-steps_', n_steps_run).replace(
                                        'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write('  mbar_states = %02d\n' % len(lambdas))
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt0-ch.in", "rt") as fin:
                    with open("./eqnpt0.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                                'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/eqnpt-ch.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                                'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/heat-ch.in", "rt") as fin:
                    with open("./heat.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                                'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/mini-ch", "rt") as fin:
                    with open("./mini.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                                'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', mol))

            elif (dec_method == 'dd'):
                with open('./vac.pdb') as myfile:
                    # Simulation files for double decoupling
                    data = myfile.readlines()
                    mk2 = int(last_lig)
                    mk1 = int(mk2 - 1)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-ch-dd', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                    '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write('  mbar_states = %02d\n' % len(lambdas))
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt-ch-dd.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/heat-ch-dd.in", "rt") as fin:
                    with open("./heat.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))

            # Create running scripts for local and server
            with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
                with open("./run_failures.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line)
            with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
                with open("./run-local.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('FERANGE', str(num_sim)))
            with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
                with open("./SLURMM-run", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))

        if (comp == 'f'):
            mk1 = '1'
            mk2 = '2'
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-ch-dd', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/heat-ch-lig.in", "rt") as fin:
                with open("./heat.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' %
                                float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-ch-lig.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' %
                                float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))

            # Create running scripts for local and server
            with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
                with open("./run_failures.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line)
            with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
                with open("./run-local.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('FERANGE', str(num_sim)))
            with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
                with open("./SLURMM-run", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))

        if (comp == 'w'):
            for i in range(0, num_sim+1):
                mk1 = '1'
                with open(f'../{self.amber_files_folder}/mdin-lj-dd', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/heat-lj-lig.in", "rt") as fin:
                with open("./heat.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-lj-lig.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))

            # Create running scripts for local and server
            with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
                with open("./run_failures.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line)
            with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
                with open("./run-local.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('FERANGE', str(num_sim)))
            with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
                with open("./SLURMM-run", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))

        if (comp == 'o' or comp == 'z'):
            # Create simulation files for elec+vdw decoupling
            if (dec_method == 'sdr'):
                # Simulation files for simultaneous decoupling
                with open('./vac.pdb') as myfile:
                    data = myfile.readlines()
                    mk2 = int(last_lig)
                    mk1 = int(mk2 - 1)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-uno', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                    '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write('  mbar_states = %02d\n' % len(lambdas))
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt0-uno.in", "rt") as fin:
                    with open("./eqnpt0.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/eqnpt-uno.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/mini-uno", "rt") as fin:
                    with open("./mini.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                            '_lig_name_', mol))

            # Simulation files for double decoupling
            elif (dec_method == 'dd'):
                raise ValueError("Double decoupling not implemented for elec+vdw decoupling")
                with open('./vac.pdb') as myfile:
                    data = myfile.readlines()
                    mk1 = int(last_lig)
                for i in range(0, num_sim+1):
                    with open(f'../{self.amber_files_folder}/mdin-lj-dd', "rt") as fin:
                        with open("./mdin-%02d" % int(i), "wt") as fout:
                            n_steps_run = str(steps1) if i == 0 else str(steps2)
                            for line in fin:
                                if i == 0:
                                    if 'ntx = 5' in line:
                                        line = 'ntx = 1, \n'
                                    elif 'irest' in line:
                                        line = 'irest = 0, \n'
                                    elif 'dt = ' in line:
                                        line = 'dt = 0.001, \n'
                                    elif 'restraintmask' in line:
                                        restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                        if restraint_mask == '':
                                            line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                        else:
                                            line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                                fout.write(line.replace('_temperature_', str(temperature)).replace(
                                        '_num-atoms_', str(vac_atoms)).replace('_num-steps_', n_steps_run).replace('disang_file', 'disang'))
                    mdin = open("./mdin-%02d" % int(i), 'a')
                    mdin.write('  mbar_states = %02d\n' % len(lambdas))
                    mdin.write('  mbar_lambda = ')
                    for i in range(0, len(lambdas)):
                        mdin.write(' %6.5f,' % (lambdas[i]))
                    mdin.write('\n')
                    mdin.write('  infe = 1,\n')
                    mdin.write(' /\n')
                    mdin.write(' &pmd \n')
                    mdin.write(' output_file = \'cmass.txt\' \n')
                    mdin.write(' output_freq = %02d \n' % int(ntwx))
                    mdin.write(' cv_file = \'cv.in\' \n')
                    mdin.write(' /\n')
                    mdin.write(' &wt type = \'END\' , /\n')
                    mdin.write('DISANG=disang.rest\n')
                    mdin.write('LISTOUT=POUT\n')

                with open(f"../{self.amber_files_folder}/eqnpt-lj-dd.in", "rt") as fin:
                    with open("./eqnpt.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))
                with open(f"../{self.amber_files_folder}/heat-lj-dd.in", "rt") as fin:
                    with open("./heat.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace(
                            '_lig_name_', mol))

            # Create running scripts for local and server
            with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
                with open("./run_failures.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line)
            with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
                with open("./run-local.bash", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('FERANGE', str(num_sim)))
            with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
                with open("./SLURMM-run", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))


class EXFreeEnergyBuilder(SDRFreeEnergyBuilder):
    @log_info
    def _build_complex(self):
        """
        Copying files from equilibration
        """

        anchor_found = super()._build_complex()

        if anchor_found == True:
            if not os.path.exists('../exchange_files'):
                shutil.copytree(f'../{self.build_file_folder}',
                                '../exchange_files')
            with self._change_dir('../exchange_files'):
                self._build_exchange_files()
            
            return True
        else:
            return anchor_found

    def _build_exchange_files(self):
        mol = self.mol
        molr = self.molr
        pose = self.pose
        poser = self.poser
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        
        # sim config values
        solv_shell = self.sim_config.solv_shell
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis
        sdr_dist = self.sim_config.sdr_dist

        # Build reference ligand from last state of equilibrium simulations
        
        os.system(f'cp ../../../../equil{poser}/representative.rst7 ./')
        os.system(f'cp ../../../../equil/{poser}/full.pdb ./')
        os.system(f'cp ../../../../equil/{poser}/representative.pdb ./aligned-nc.pdb')
        for file in glob.glob(f'../../../../equil/{poser.lower()}/full*.prmtop'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob(f'../../../../equil/{poser.lower()}/vac*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        run_with_log(f'{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb')

        # restore resid index
        
        #shutil.copy('rec_file.pdb', 'equil-reference.pdb')
        os.system('cp rec_file.pdb equil-reference.pdb')

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
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', molr.lower())
                    .replace('MMM', f"\'{molr.lower()}\'"))
        run_with_log(f'{vmd} -dispdev text -e split.tcl')

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                    'protein.pdb',
                    '%s.pdb' % molr.lower(),
                    'lipids.pdb',
                    'others.pdb',
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

        # Read protein anchors and size from equilibrium
        with open('../../../../equil/'+poser+'/equil-%s.pdb' % molr.lower(), 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('MMM', f"\'{molr}\'")
                    .replace('mmm', molr.lower())
                    .replace('NN', p1_atom)
                    .replace('P1A', p1_vmd)
                    .replace('FIRST', '2')
                    .replace('LAST', str(rec_res))
                    .replace('STAGE', 'fe')
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

        # Align to reference (equilibrium) structure using VMD's measure fit
        run_with_log(f'{vmd} -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        u = mda.Universe('aligned_amber.pdb')
        # Fix lipid
        if lipid_mol:
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
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

        run_with_log(f'{vmd} -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+poser+'.txt')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+poser+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+poser+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % molr.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % molr.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])

    @log_info
    def _sim_files(self):
        # Find anchors
        
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        molr = self.molr
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]
        ntwx = self.sim_config.ntwx

        with open('disang.rest', 'r') as f:
            data = f.readline().split()
            L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read and parse 'vac.pdb' once to reduce repeated file reads
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get the number of atoms in vacuum (from the third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Find the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():
                last_lig = line[22:26].strip()

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        # Calculate simulation parameters
        mk4 = int(last_lig)
        mk1, mk2, mk3 = mk4 - 3, mk4 - 2, mk4 - 1


        for i in range(0, num_sim+1):
            with open(f'../{self.amber_files_folder}/mdin-ex', "rt") as fin:
                with open("./mdin-%02d" % int(i), "wt") as fout:
                    n_steps_run = str(steps1) if i == 0 else str(steps2)
                    for line in fin:
                        if i == 0:
                            if 'ntx = 5' in line:
                                line = 'ntx = 1, \n'
                            elif 'irest' in line:
                                line = 'irest = 0, \n'
                            elif 'dt = ' in line:
                                line = 'dt = 0.001, \n'
                            elif 'restraintmask' in line:
                                restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                if restraint_mask == '':
                                    line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                else:
                                    line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                        fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                            '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)))
            mdin = open("./mdin-%02d" % int(i), 'a')
            mdin.write('  mbar_states = %02d\n' % len(lambdas))
            mdin.write('  mbar_lambda = ')
            for i in range(0, len(lambdas)):
                mdin.write(' %6.5f,' % (lambdas[i]))
            mdin.write('\n')
            mdin.write('  infe = 1,\n')
            mdin.write(' /\n')
            mdin.write(' &pmd \n')
            mdin.write(' output_file = \'cmass.txt\' \n')
            mdin.write(' output_freq = %02d \n' % int(ntwx))
            mdin.write(' cv_file = \'cv.in\' \n')
            mdin.write(' /\n')
            mdin.write(' &wt type = \'END\' , /\n')
            mdin.write('DISANG=disang.rest\n')
            mdin.write('LISTOUT=POUT\n')

        with open(f"../{self.amber_files_folder}/mini-ex", "rt") as fin:
                    with open("./mini.in", "wt") as fout:
                        for line in fin:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', f'{mol},{molr}'))
        with open(f"../{self.amber_files_folder}/eqnpt0-ex.in", "rt") as fin:
            with open("./eqnpt0.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace(
                        'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', f'{mol},{molr}'))
        with open(f"../{self.amber_files_folder}/eqnpt-ex.in", "rt") as fin:
            with open("./eqnpt.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                        'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', f'{mol},{molr}'))
        with open(f"../{self.amber_files_folder}/heat-ex.in", "rt") as fin:
            with open("./heat.in", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('_temperature_', str(temperature)).replace('lbd_val', '%6.5f' % float(weight)).replace(
                        'mk1', str(mk1)).replace('mk2', str(mk2)).replace('mk3', str(mk3)).replace('mk4', str(mk4)).replace(
                            '_lig_name_', f'{mol},{molr}'))


        # Create running scripts for local and server
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                                'SYSTEMNAME', self.system.system_name).replace(
                                    'PARTITIONNAME', self.system.partition))


class RESTFreeEnergyBuilder(FreeEnergyBuilder):
    """
    Builder for restrain free energy calculations system
    """

class UNOFreeEnergyBuilder(SDRFreeEnergyBuilder):
    """
    Builder for vdw + elec single decoupling free energy calculations system
    """
    @log_info
    def _sim_files(self):
        
        dec_method = self.dec_method
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        ntwx = self.sim_config.ntwx
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]

        # Read 'disang.rest' and extract L1, L2, L3
        #with open('disang.rest', 'r') as f:
        #    data = f.readline().split()
        #    L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read 'vac.pdb' once
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get number of atoms in vacuum (third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Get the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():  # Compare residue name
                last_lig = line[22:26].strip()  # Extract residue number

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        # Create simulation files for elec+vdw decoupling
        if (dec_method == 'sdr'):
            # Simulation files for simultaneous decoupling
            with open('./vac.pdb') as myfile:
                data = myfile.readlines()
                mk2 = int(last_lig)
                mk1 = int(mk2 - 1)

            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-uno', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write('  infe = 1,\n')
                mdin.write(' /\n')
                mdin.write(' &pmd \n')
                mdin.write(' output_file = \'cmass.txt\' \n')
                mdin.write(' output_freq = %02d \n' % int(ntwx))
                mdin.write(' cv_file = \'cv.in\' \n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/mini.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))

        # Create running scripts for local and server
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                            'SYSTEMNAME', self.system.system_name).replace(
                                'PARTITIONNAME', self.system.partition))


class UNORESTFreeEnergyBuilder(UNOFreeEnergyBuilder):
    """
    Builder for vdw + elec + restraint single decoupling free energy calculations system
    """
    @log_info
    def _sim_files(self):
        
        dec_method = self.dec_method
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        ntwx = self.sim_config.ntwx
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]

        # Read 'disang.rest' and extract L1, L2, L3
        #with open('disang.rest', 'r') as f:
        #    data = f.readline().split()
        #    L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read 'vac.pdb' once
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get number of atoms in vacuum (third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Get the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():  # Compare residue name
                last_lig = line[22:26].strip()  # Extract residue number

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        # Create simulation files for elec+vdw decoupling
        if (dec_method == 'sdr'):
            # Simulation files for simultaneous decoupling
            with open('./vac.pdb') as myfile:
                data = myfile.readlines()
                mk2 = int(last_lig)
                mk1 = int(mk2 - 1)

            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-unorest', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write('  infe = 1,\n')
                mdin.write(' /\n')
                mdin.write(' &pmd \n')
                mdin.write(' output_file = \'cmass.txt\' \n')
                mdin.write(' output_freq = %02d \n' % int(ntwx))
                mdin.write(' cv_file = \'cv.in\' \n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                mdin.write('DISANG=disang.rest\n')
                mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/mini.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))

        # Create running scripts for local and server
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                            'SYSTEMNAME', self.system.system_name).replace(
                                'PARTITIONNAME', self.system.partition))

        # add lambda.sch to the folder
        with open("./lambda.sch", "wt") as fout:
            fout.write('TypeRestBA, smooth_step2, symmetric, 1.0, 0.0\n')


class UNOFreeEnergyFBBuilder(UNOFreeEnergyBuilder):
    """
    Builder for vdw + elec single decoupling free energy calculations system
    + flat-bottom COM restraints
    """
    @log_info
    def _build_complex(self):
        """
        Copying files from equilibration
        """
        pose = self.pose
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        
        # sim config values
        solv_shell = self.sim_config.solv_shell
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis
        sdr_dist = self.sim_config.sdr_dist

        shutil.copytree(build_files_orig, '.', dirs_exist_ok=True)

        #shutil.copy(f'../../../../equil/{pose}/build_files/{self.pose}.pdb', './')
        os.system(f'cp ../../../../equil/{pose}/build_files/{self.pose}.pdb ./')
        # Get last state from equilibrium simulations
        #shutil.copy(f'../../../../equil/{pose}/representative.rst7', './')
        os.system(f'cp ../../../../equil/{pose}representative.rst7 ./')
        #shutil.copy(f'../../../../equil/{pose}/representative.pdb', './aligned-nc.pdb')
        os.system(f'cp ../../../../equil/{pose}/representative.pdb ./aligned-nc.pdb')
        #shutil.copy(f'../../../../equil/{pose}/build_amber_renum.txt', './')
        os.system(f'cp ../../../../equil/{pose}/build_amber_renum.txt ./')
        os.system(f'cp ../../../../equil/{pose}/build_files/protein_renum.txt ./')
        if not os.path.exists('protein_renum.txt'):
            raise FileNotFoundError(f'protein_renum.txt not found in {os.getcwd()}')
        for file in glob.glob(f'../../../../equil/{pose}/full*.prmtop'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob(f'../../../../equil/{pose}/vac*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        
        mol = mda.Universe(f'{self.pose}.pdb').residues[0].resname
        self.mol = mol

        run_with_log(f'{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb')
        renum_data = pd.read_csv('build_amber_renum.txt', sep=r'\s+',
                header=None, names=['old_resname',
                                    'old_chain',
                                    'old_resid',
                                    'new_resname', 'new_resid'])
        u = mda.Universe('rec_file.pdb')

        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum_data.query(f'old_resid == @resid_str').old_chain.values[0]

        if lipid_mol:
            # fix lipid resids
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
            revised_resids = np.array(revised_resids)
            total_residues = u.atoms.residues.n_residues
            final_resids = np.zeros(total_residues, dtype=int)
            final_resids[:len(revised_resids)] = revised_resids
            next_resnum = revised_resids[-1] + 1
            final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
            u.atoms.residues.resids = final_resids

        u.atoms.write('rec_file.pdb')

        # Used for retrieving the box size
        #shutil.copy('rec_file.pdb', 'equil-reference.pdb')
        os.system('cp rec_file.pdb equil-reference.pdb')

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
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', mol.lower())
                    .replace('MMM', f"\'{mol}\'"))
        run_with_log(f'{vmd} -dispdev text -e split.tcl')

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                     'protein.pdb',
                    f'{mol.lower()}.pdb',
                     'lipids.pdb',
                     'others.pdb',
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

        # Read protein anchors and size from equilibrium
        with open(f'../../../../equil/{pose}/equil-{mol.lower()}.pdb', 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('MMM', f"\'{mol}\'")
                        .replace('mmm', mol.lower())
                        .replace('NN', p1_atom)
                        .replace('P1A', p1_vmd)
                        .replace('FIRST', '2')
                        .replace('LAST', str(rec_res))
                        .replace('STAGE', 'fe')
                        .replace('XDIS', '%4.2f' % l1_x)
                        .replace('YDIS', '%4.2f' % l1_y)
                        .replace('ZDIS', '%4.2f' % l1_z)
                        .replace('RANG', '%4.2f' % l1_range)
                        .replace('DMAX', '%4.2f' % max_adis)
                        .replace('DMIN', '%4.2f' % min_adis)
                        .replace('SDRD', '%4.2f' % sdr_dist)
                        .replace('LIGSITE', '1')
                        .replace('OTHRS', str(other_mol_vmd))
                        .replace('LIPIDS', str(lipid_mol_vmd))
                        )

        # Align to reference (equilibrium) structure using VMD's measure fit
        run_with_log(f'{vmd} -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # fix lipid resids
        if lipid_mol:
            u = mda.Universe('aligned_amber.pdb')
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
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
            # set correct residue number
            revised_resids = np.array(revised_resids)
            u.atoms.residues.resids = final_resids[:len(revised_resids)]

            u.atoms.write('aligned_amber.pdb')
        
        run_with_log(f'{vmd} -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+pose+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+pose+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % mol.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % mol.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])
        return True

    @log_info
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
        mol = self.mol
        molr = self.molr
        poser = self.poser
        resname_lig = mol
        other_mol = self.other_mol
        lipid_mol = self.lipid_mol
        ion_mol = ['Na+', 'K+', 'Cl-']
        comp = self.comp
        sdr_dist = self.sim_config.sdr_dist

        dec_method = self.dec_method

        if os.path.exists(self.amber_files_folder) or os.path.islink(self.amber_files_folder):
            os.remove(self.amber_files_folder)

        os.symlink(f'../{self.amber_files_folder}', self.amber_files_folder)

        for file in glob.glob(f'../{self.build_file_folder}/vac_ligand*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        #shutil.copy(f'../{self.build_file_folder}/{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './build-ini.pdb')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./build-ini.pdb')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/anchors-{self.pose}.txt', './')
        os.system(f'cp ../{self.build_file_folder}/anchors-{self.pose}.txt ./')
        #shutil.copy(f'../{self.build_file_folder}/equil-reference.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/equil-reference.pdb ./')

        for file in glob.glob('../../../ff/*.mol2'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/*.frcmod'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/{mol.lower()}.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/dum.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')

        # Get TER statements
        ter_atom = []
        with open(f'../{self.build_file_folder}/rec_file.pdb') as oldfile, open('rec_file-clean.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'WAT' in line:
                    newfile.write(line)
        run_with_log('pdb4amber -i rec_file-clean.pdb -o rec_amber.pdb -y')
        with open('./rec_amber.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'TER'):
                ter_atom.append(int(lines[i][6:11].strip()))
                
        for i in range(1, 4):
            #shutil.copy(f'../{self.build_file_folder}/dum'+str(i)+'.pdb', './')
            os.system(f'cp ../{self.build_file_folder}/dum'+str(i)+'.pdb ./')
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
        with open('build-ini.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Count atoms of the system
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                molecule = lines[i][17:21].strip() 
                if (molecule != mol) and (molecule != 'DUM') and (molecule != 'WAT') and (molecule not in other_mol) and (molecule not in lipid_mol) and (molecule not in ion_mol):
                    recep_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(molecule)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1
                elif molecule == mol:
                    lig_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(molecule)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1
                elif (molecule == 'WAT') or (molecule in other_mol) or (molecule in ion_mol):
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1
                elif molecule in lipid_mol:
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resid_list = [resid if resid < 10000 else (resid % 9999) + 1 for resid in resid_list]

        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
        lig_resid = recep_last + dum_atom
        oth_tmp = 'None'

        # Write the new pdb file

        build_file = open('build.pdb', 'w')

        # Positions for the dummy atoms
        for i in range(0, dum_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        chain_tmp = 'None'
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
            j = i + 2 - dum_atom
            if j in ter_atom:
                build_file.write('TER\n')

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):

            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')


        for i in range(0, lig_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid + 1)))
            build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

            build_file.write('%6.2f%6.2f\n' % (0, 0))
        build_file.write('TER\n')


        # Positions of the other atoms
        for i in range(0, oth_atom):
            if oth_rsidlist[i] != oth_tmp:
                build_file.write('TER\n')
            oth_tmp = oth_rsidlist[i]
            oth_tmp = oth_tmp if oth_tmp < 10000 else (oth_tmp % 9999) + 1
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, oth_atomlist[i], oth_rsnmlist[i], oth_chainlist[i], oth_tmp))
            build_file.write('%8.3f%8.3f%8.3f' %
                             (float(oth_coords[i][0]), float(oth_coords[i][1]), float(oth_coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')
        build_file.write('END\n')
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
        
    @log_info
    def _restraints(self):
        # TODO: Refactor this method
        # This is just a hack to avoid the restraints for lambda windows
        # when win is not 0
        if self.win != 0 and COMPONENTS_LAMBDA_DICT[self.comp] == 'lambdas':
            return
        pose = self.pose
        rest = self.sim_config.rest
        lcom = rest[6]

        bb_start = self.sim_config.bb_start
        bb_end = self.sim_config.bb_end
        stage = self.stage
        mol = self.mol
        molr = self.molr
        comp = self.comp
        bb_equil = self.sim_config.bb_equil
        sdr_dist = self.sim_config.sdr_dist
        dec_method = self.sim_config.dec_method
        other_mol = self.other_mol
        lambdas_comp = self.sim_config.dict()[COMPONENTS_LAMBDA_DICT[self.comp]]
        weight = lambdas_comp[self.win]

        pdb_file = 'vac.pdb'
        u = mda.Universe(pdb_file)
        ligand_residues = u.select_atoms(f'resname {mol}').residues
        bsite_ag = u.select_atoms(f'name CA and byres protein and around 6 resname {mol}')

        # COM restraints
        cv_file = open('cv.in', 'w')
        cv_file.write('cv_file \n')
        # ignore protein COM restraints
        #cv_file.write('&colvar \n')
        #cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        #cv_file.write(' cv_ni = %s, cv_i = 1,0,' % str(len(hvy_h)+2))
        #for i in range(0, len(hvy_h)):
        #    cv_file.write(hvy_h[i])
        #    cv_file.write(',')
        #cv_file.write('\n')
        #cv_file.write(' anchor_position = %10.4f, %10.4f, %10.4f, %10.4f \n' %
        #            (float(0.0), float(0.0), float(0.0), float(999.0)))
        #cv_file.write(' anchor_strength = %10.4f, %10.4f, \n' % (rcom, rcom))
        #cv_file.write('/ \n')

        # Ligand solvent COM restraints
        lig_solv = ligand_residues[1].atoms
        index_amber = lig_solv.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 0.000, 999.0 \n')
        cv_file.write(f' anchor_strength = {lcom:10.4f}, {lcom:10.4f}, \n')
        cv_file.write('/ \n')

        # Ligand binding site COM restraints
        lig_bs = ligand_residues[0].atoms
        index_amber = lig_bs.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 3,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 0.000, 999.0 \n')
        cv_file.write(' anchor_strength = 50, 50, \n')
        cv_file.write('/ \n')

        index_amber = bsite_ag.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 3,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 3.000, 999.0 \n')
        cv_file.write(f' anchor_strength = {lcom:10.4f}, {lcom:10.4f}, \n')
        cv_file.write('/ \n')
        cv_file.close()

    @log_info
    def _sim_files(self):
        
        dec_method = self.dec_method
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        ntwx = self.sim_config.ntwx
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]

        # Read 'disang.rest' and extract L1, L2, L3
        #with open('disang.rest', 'r') as f:
        #    data = f.readline().split()
        #    L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read 'vac.pdb' once
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get number of atoms in vacuum (third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Get the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():  # Compare residue name
                last_lig = line[22:26].strip()  # Extract residue number

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        # Create simulation files for elec+vdw decoupling
        if (dec_method == 'sdr'):
            # Simulation files for simultaneous decoupling
            with open('./vac.pdb') as myfile:
                data = myfile.readlines()
                mk2 = int(last_lig)
                mk1 = int(mk2 - 1)
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-uno', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write('  infe = 1,\n')
                mdin.write(' /\n')
                mdin.write(' &pmd \n')
                mdin.write(' output_file = \'cmass.txt\' \n')
                mdin.write(' output_freq = %02d \n' % int(ntwx))
                mdin.write(' cv_file = \'cv.in\' \n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                #mdin.write('DISANG=disang.rest\n')
                #mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/eqnpt0-uno.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                        '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt-uno.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                        '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/mini-uno", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_temperature_', str(temperature)).replace(
                            'lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)).replace(
                        '_lig_name_', mol))

        # Create running scripts for local and server
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                            'SYSTEMNAME', self.system.system_name).replace(
                                'PARTITIONNAME', self.system.partition))


class ACESEquilibrationBuilder(FreeEnergyBuilder):
    """
    Builder for running ACES with self-transformation
    """
    @log_info
    def _build_complex(self):
        """
        Copying files from equilibration
        """
        pose = self.pose
        lipid_mol = self.lipid_mol
        other_mol = self.other_mol
        
        # sim config values
        solv_shell = self.sim_config.solv_shell
        l1_x = self.sim_config.l1_x
        l1_y = self.sim_config.l1_y
        l1_z = self.sim_config.l1_z
        l1_range = self.sim_config.l1_range
        max_adis = self.sim_config.max_adis
        min_adis = self.sim_config.min_adis
        sdr_dist = 0

        shutil.copytree(build_files_orig, '.', dirs_exist_ok=True)

        #shutil.copy(f'../../../equil/{pose}/{self.build_file_folder}/{self.pose}.pdb', './')
        os.system(f'cp ../../../equil/{pose}/{self.build_file_folder}/{self.pose}.pdb ./')
        # Get last state from equilibrium simulations
        #shutil.copy(f'../../../equil/{pose}/representative.rst7', './')
        os.system(f'cp ../../../equil/{pose}/representative.rst7 ./')
        #shutil.copy(f'../../../equil/{pose}/representative.pdb', './aligned-nc.pdb')
        os.system(f'cp ../../../equil/{pose}/representative.pdb ./aligned-nc.pdb')
        #shutil.copy(f'../../../equil/{pose}/build_amber_renum.txt', './')
        os.system(f'cp ../../../equil/{pose}/build_amber_renum.txt ./')
        os.system(f'cp ../../../equil/{pose}/build_files/protein_renum.txt ./')
        if not os.path.exists('protein_renum.txt'):
            raise FileNotFoundError(f'protein_renum.txt not found in {os.getcwd()}')


        # Lustre has a problem with copy
        # https://confluence.ecmwf.int/display/UDOC/HPC2020%3A+Python+known+issues
        for file in glob.glob(f'../../../equil/{pose}/full*.prmtop'):
            run_with_log(f'cp {file} .')
            #base_name = os.path.basename(file)
            #os.copy(file, f'./{base_name}')
            #os.symlink(file, f'./{base_name}')
        for file in glob.glob(f'../../../equil/{pose}/vac*'):
            run_with_log(f'cp {file} .')

            #base_name = os.path.basename(file)
            #shutil.copyfile(file, f'./{base_name}')
            #os.symlink(file, f'./{base_name}')
        
        mol = mda.Universe(f'{self.pose}.pdb').residues[0].resname
        self.mol = mol

        run_with_log(f'{cpptraj} -p full.prmtop -y representative.rst7 -x rec_file.pdb')
        renum_data = pd.read_csv('build_amber_renum.txt', sep=r'\s+',
                header=None, names=['old_resname',
                                    'old_chain',
                                    'old_resid',
                                    'new_resname', 'new_resid'])
        u = mda.Universe('rec_file.pdb')

        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum_data.query(f'old_resid == @resid_str').old_chain.values[0]

        if lipid_mol:
            # fix lipid resids
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
            revised_resids = np.array(revised_resids)
            total_residues = u.atoms.residues.n_residues
            final_resids = np.zeros(total_residues, dtype=int)
            final_resids[:len(revised_resids)] = revised_resids
            next_resnum = revised_resids[-1] + 1
            final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
            u.atoms.residues.resids = final_resids

        u.atoms.write('rec_file.pdb')

        # Used for retrieving the box size
        #shutil.copy('rec_file.pdb', 'equil-reference.pdb')
        os.system(f'cp rec_file.pdb equil-reference.pdb')

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
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', mol.lower())
                    .replace('MMM', f"\'{mol}\'"))
        run_with_log(f'{vmd} -dispdev text -e split.tcl')

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                     'protein.pdb',
                    f'{mol.lower()}.pdb',
                     'lipids.pdb',
                     'others.pdb',
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

        # Read protein anchors and size from equilibrium
        with open(f'../../../equil/{pose}/equil-{mol.lower()}.pdb', 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('MMM', f"\'{mol}\'")
                        .replace('mmm', mol.lower())
                        .replace('NN', p1_atom)
                        .replace('P1A', p1_vmd)
                        .replace('FIRST', '2')
                        .replace('LAST', str(rec_res))
                        .replace('STAGE', 'fe')
                        .replace('XDIS', '%4.2f' % l1_x)
                        .replace('YDIS', '%4.2f' % l1_y)
                        .replace('ZDIS', '%4.2f' % l1_z)
                        .replace('RANG', '%4.2f' % l1_range)
                        .replace('DMAX', '%4.2f' % max_adis)
                        .replace('DMIN', '%4.2f' % min_adis)
                        .replace('SDRD', '%4.2f' % sdr_dist)
                        .replace('LIGSITE', '1')
                        .replace('OTHRS', str(other_mol_vmd))
                        .replace('LIPIDS', str(lipid_mol_vmd))
                        )

        # Align to reference (equilibrium) structure using VMD's measure fit
        run_with_log(f'{vmd} -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # fix lipid resids
        if lipid_mol:
            u = mda.Universe('aligned_amber.pdb')
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep=r'\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
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
            # set correct residue number
            revised_resids = np.array(revised_resids)
            u.atoms.residues.resids = final_resids[:len(revised_resids)]

            u.atoms.write('aligned_amber.pdb')
        
        run_with_log(f'{vmd} -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+pose+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+pose+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % mol.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % mol.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])
        return True

    @log_info
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
        mol = self.mol
        molr = self.molr
        poser = self.poser
        resname_lig = mol
        other_mol = self.other_mol
        lipid_mol = self.lipid_mol
        ion_mol = ['Na+', 'K+', 'Cl-']
        comp = self.comp
        sdr_dist = 0

        dec_method = self.dec_method

        if os.path.exists(self.amber_files_folder) or os.path.islink(self.amber_files_folder):
            os.remove(self.amber_files_folder)

        os.symlink(f'../{self.amber_files_folder}', self.amber_files_folder)

        for file in glob.glob(f'../{self.build_file_folder}/vac_ligand*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        #shutil.copy(f'../{self.build_file_folder}/{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './build-ini.pdb')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./build-ini.pdb')
        #shutil.copy(f'../{self.build_file_folder}/fe-{mol.lower()}.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/fe-{mol.lower()}.pdb ./')
        #shutil.copy(f'../{self.build_file_folder}/anchors-{self.pose}.txt', './')
        os.system(f'cp ../{self.build_file_folder}/anchors-{self.pose}.txt ./')
        #shutil.copy(f'../{self.build_file_folder}/equil-reference.pdb', './')
        os.system(f'cp ../{self.build_file_folder}/equil-reference.pdb ./')

        for file in glob.glob('../../../ff/*.mol2'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/*.frcmod'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/{mol.lower()}.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')
        for file in glob.glob('../../../ff/dum.*'):
            #shutil.copy(file, './')
            os.system(f'cp {file} ./')

        # Get TER statements
        ter_atom = []
        with open(f'../{self.build_file_folder}/rec_file.pdb') as oldfile, open('rec_file-clean.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'WAT' in line:
                    newfile.write(line)
        run_with_log('pdb4amber -i rec_file-clean.pdb -o rec_amber.pdb -y')
        with open('./rec_amber.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'TER'):
                ter_atom.append(int(lines[i][6:11].strip()))
                
        # dum1: protein COM
        # dum3: ligand BS COM
        for i in [1, 3]:
            #shutil.copy(f'../{self.build_file_folder}/dum'+str(i)+'.pdb', './')
            os.system(f'cp ../{self.build_file_folder}/dum{i}.pdb ./')
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
        with open('build-ini.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Count atoms of the system
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                molecule = lines[i][17:21].strip() 
                if (molecule != mol) and (molecule != 'DUM') and (molecule != 'WAT') and (molecule not in other_mol) and (molecule not in lipid_mol) and (molecule not in ion_mol):
                    recep_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(molecule)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1
                elif molecule == mol:
                    lig_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(molecule)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1
                elif (molecule == 'WAT') or (molecule in other_mol) or (molecule in ion_mol):
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1
                elif molecule in lipid_mol:
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resid_list = [resid if resid < 10000 else (resid % 9999) + 1 for resid in resid_list]

        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
        lig_resid = recep_last + dum_atom
        oth_tmp = 'None'

        # Write the new pdb file

        build_file = open('build.pdb', 'w')

        # Positions for the dummy atoms
        for i in range(0, dum_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        chain_tmp = 'None'
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
            j = i + 2 - dum_atom
            if j in ter_atom:
                build_file.write('TER\n')

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):

            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')

        for i in range(0, lig_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid + 1)))
            build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                lig_coords[i][1]), float(lig_coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))
        build_file.write('TER\n')

        # Positions of the other atoms
        for i in range(0, oth_atom):
            if oth_rsidlist[i] != oth_tmp:
                build_file.write('TER\n')
            oth_tmp = oth_rsidlist[i]
            oth_tmp = oth_tmp if oth_tmp < 10000 else (oth_tmp % 9999) + 1
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, oth_atomlist[i], oth_rsnmlist[i], oth_chainlist[i], oth_tmp))
            build_file.write('%8.3f%8.3f%8.3f' %
                             (float(oth_coords[i][0]), float(oth_coords[i][1]), float(oth_coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')
        build_file.write('END\n')
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
        
    @log_info
    def _restraints(self):
        # TODO: Refactor this method
        # This is just a hack to avoid the restraints for lambda windows
        # when win is not 0
        if self.win != 0 and COMPONENTS_LAMBDA_DICT[self.comp] == 'lambdas':
            return
        pose = self.pose
        rest = self.sim_config.rest
        lcom = rest[6]

        bb_start = self.sim_config.bb_start
        bb_end = self.sim_config.bb_end
        stage = self.stage
        mol = self.mol
        molr = self.molr
        comp = self.comp
        bb_equil = self.sim_config.bb_equil
        sdr_dist = 0
        dec_method = self.sim_config.dec_method
        other_mol = self.other_mol
        lambdas_comp = self.sim_config.dict()[COMPONENTS_LAMBDA_DICT[self.comp]]
        weight = lambdas_comp[self.win]

        pdb_file = 'vac.pdb'
        u = mda.Universe(pdb_file)
        ligand_residues = u.select_atoms(f'resname {mol}').residues
        bsite_ag = u.select_atoms(f'name CA and byres protein and around 6 resname {mol}')

        # COM restraints
        cv_file = open('cv.in', 'w')
        cv_file.write('cv_file \n')
        # ignore protein COM restraints
        #cv_file.write('&colvar \n')
        #cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        #cv_file.write(' cv_ni = %s, cv_i = 1,0,' % str(len(hvy_h)+2))
        #for i in range(0, len(hvy_h)):
        #    cv_file.write(hvy_h[i])
        #    cv_file.write(',')
        #cv_file.write('\n')
        #cv_file.write(' anchor_position = %10.4f, %10.4f, %10.4f, %10.4f \n' %
        #            (float(0.0), float(0.0), float(0.0), float(999.0)))
        #cv_file.write(' anchor_strength = %10.4f, %10.4f, \n' % (rcom, rcom))
        #cv_file.write('/ \n')

        # Ligand COM restraints
        lig_solv = ligand_residues[1].atoms
        index_amber = lig_solv.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 3.000, 999.0 \n')
        cv_file.write(f' anchor_strength = {lcom:10.4f}, {lcom:10.4f}, \n')
        cv_file.write('/ \n')

        # Ligand COM restraints 2
        lig_bs = ligand_residues[0].atoms
        index_amber = lig_bs.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 3.000, 999.0 \n')
        cv_file.write(f' anchor_strength = {lcom:10.4f}, {lcom:10.4f}, \n')
        cv_file.write('/ \n')

        # Ligand binding site COM restraints
        index_amber = bsite_ag.indices + 1
        cv_file.write('&colvar \n')
        cv_file.write(' cv_type = \'COM_DISTANCE\' \n')
        cv_file.write(' cv_ni = %s, cv_i = 2,0,' % str(len(index_amber)+2))
        for i in range(0, len(index_amber)):
            cv_file.write(str(index_amber[i]))
            cv_file.write(',')
        cv_file.write('\n')
        cv_file.write(' anchor_position = 0.000, 0.000, 0.000, 999.0 \n')
        cv_file.write(' anchor_strength = 50, 50, \n')
        cv_file.write('/ \n')
        cv_file.close()

    @log_info
    def _sim_files(self):
        
        dec_method = self.dec_method
        hmr = self.sim_config.hmr
        temperature = self.sim_config.temperature
        mol = self.mol
        num_sim = self.sim_config.num_fe_range
        pose = self.pose
        comp = self.comp
        win = self.win
        stage = self.stage
        steps1 = self.sim_config.dic_steps1[comp]
        steps2 = self.sim_config.dic_steps2[comp]
        rng = self.sim_config.rng
        lipid_mol = self.lipid_mol
        ntwx = self.sim_config.ntwx
        lambdas = self.system.component_windows_dict[comp]
        weight = lambdas[self.win if self.win != -1 else 0]

        # Read 'disang.rest' and extract L1, L2, L3
        #with open('disang.rest', 'r') as f:
        #    data = f.readline().split()
        #    L1, L2, L3 = data[6].strip(), data[7].strip(), data[8].strip()

        # Read 'vac.pdb' once
        with open('./vac.pdb') as f:
            lines = f.readlines()

        # Get number of atoms in vacuum (third-to-last line)
        vac_atoms = lines[-3][6:11].strip()

        # Get the last ligand residue number
        last_lig = None
        for line in lines:
            if line[17:20].strip().lower() == mol.lower():  # Compare residue name
                last_lig = line[22:26].strip()  # Extract residue number

        if last_lig is None:
            raise ValueError(f"No ligand residue matching '{mol}' found in vac.pdb")

        # Create simulation files for elec+vdw decoupling
        if (dec_method == 'sdr'):
            # Simulation files for simultaneous decoupling
            with open('./vac.pdb') as myfile:
                data = myfile.readlines()
                mk2 = int(last_lig)
                mk1 = int(mk2 - 1)
            for i in range(0, num_sim+1):
                with open(f'../{self.amber_files_folder}/mdin-uno', "rt") as fin:
                    with open("./mdin-%02d" % int(i), "wt") as fout:
                        n_steps_run = str(steps1) if i == 0 else str(steps2)
                        for line in fin:
                            if i == 0:
                                if 'ntx = 5' in line:
                                    line = 'ntx = 1, \n'
                                elif 'irest' in line:
                                    line = 'irest = 0, \n'
                                elif 'dt = ' in line:
                                    line = 'dt = 0.001, \n'
                                elif 'restraintmask' in line:
                                    restraint_mask = line.split('=')[1].strip().replace("'", "").rstrip(',')
                                    if restraint_mask == '':
                                        line = f"restraintmask = '(@CA | :{mol}) & !@H=' \n"
                                    else:
                                        line = f"restraintmask = '(@CA | :{mol} | {restraint_mask}) & !@H=' \n"
                            fout.write(line.replace('_temperature_', str(temperature)).replace('_num-atoms_', str(vac_atoms)).replace(
                                '_num-steps_', n_steps_run).replace('lbd_val', '%6.5f' % float(weight)).replace('mk1', str(mk1)).replace('mk2', str(mk2)))
                mdin = open("./mdin-%02d" % int(i), 'a')
                mdin.write('  mbar_states = %02d\n' % len(lambdas))
                mdin.write('  mbar_lambda = ')
                for i in range(0, len(lambdas)):
                    mdin.write(' %6.5f,' % (lambdas[i]))
                mdin.write('\n')
                mdin.write('  infe = 1,\n')
                mdin.write(' /\n')
                mdin.write(' &pmd \n')
                mdin.write(' output_file = \'cmass.txt\' \n')
                mdin.write(' output_freq = %02d \n' % int(ntwx))
                mdin.write(' cv_file = \'cv.in\' \n')
                mdin.write(' /\n')
                mdin.write(' &wt type = \'END\' , /\n')
                #mdin.write('DISANG=disang.rest\n')
                #mdin.write('LISTOUT=POUT\n')

            with open(f"../{self.amber_files_folder}/mini.in", "rt") as fin:
                with open("./mini.in", "wt") as fout:
                    for line in fin:
                        fout.write(line.replace('_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt0.in", "rt") as fin:
                with open("./eqnpt0.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))
            with open(f"../{self.amber_files_folder}/eqnpt.in", "rt") as fin:
                with open("./eqnpt.in", "wt") as fout:
                    for line in fin:
                        if 'infe' in line:
                            fout.write('  infe = 1,\n')
                        if 'mcwat' in line:
                            fout.write('  mcwat = 0,\n')
                        else:
                            fout.write(line.replace('_temperature_', str(temperature)).replace(
                                    '_lig_name_', mol))

        # Create running scripts for local and server
        with open(f'../{self.run_files_folder}/run_failures.bash', "rt") as fin:
            with open("./run_failures.bash", "wt") as fout:
                for line in fin:
                    fout.write(line)
        with open(f'../{self.run_files_folder}/run-local.bash', "rt") as fin:
            with open("./run-local.bash", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('FERANGE', str(num_sim)))
        with open(f'../{self.run_files_folder}/SLURMM-Am', "rt") as fin:
            with open("./SLURMM-run", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('STAGE', pose).replace('POSE', '%s%02d' % (comp, int(win))).replace(
                            'SYSTEMNAME', self.system.system_name).replace(
                                'PARTITIONNAME', self.system.partition))


class BuilderFactory:
    @staticmethod
    def get_builder(stage,
                    system,
                    pose,
                    sim_config,
                    working_dir,
                    win=0,
                    component='q',
                    molr=None,
                    poser=None,
    ):
        if stage == 'equil':
            return EquilibrationBuilder(
                system=system,
                pose=pose,
                sim_config=sim_config,
                working_dir=working_dir,
            )
        
        match component:
            case 'n' | 'm':
                return RESTFreeEnergyBuilder(
                system=system,
                pose=pose,
                sim_config=sim_config,
                working_dir=working_dir,
                win=win,
                component=component,
                molr=molr,
                poser=poser,
            )
            case 'e' | 'v':
                return SDRFreeEnergyBuilder(
                system=system,
                pose=pose,
                sim_config=sim_config,
                working_dir=working_dir,
                win=win,
                component=component,
                molr=molr,
                poser=poser,
            )
            case 'x':
                return EXFreeEnergyBuilder(
                    system=system,
                    pose=pose,
                    sim_config=sim_config,
                    working_dir=working_dir,
                    win=win,
                    component=component,
                    molr=molr,
                    poser=poser,
                )
            case 'o':
                return UNOFreeEnergyBuilder(
                    system=system,
                    pose=pose,
                    sim_config=sim_config,
                    working_dir=working_dir,
                    win=win,
                    component=component,
                    molr=molr,
                    poser=poser,
                )
            case 'z':
                return UNORESTFreeEnergyBuilder(
                    system=system,
                    pose=pose,
                    sim_config=sim_config,
                    working_dir=working_dir,
                    win=win,
                    component=component,
                    molr=molr,
                    poser=poser,
                )
            case 's':
                return ACESEquilibrationBuilder(
                    system=system,
                    pose=pose,
                    sim_config=sim_config,
                    working_dir=working_dir,
                    win=win,
                    component=component,
                    molr=molr,
                    poser=poser,
                )
            case _:
                raise ValueError(f"Invalid component: {component} for now")