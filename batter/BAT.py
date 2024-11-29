import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
import numpy as np
#from batter.utils.utils import run_with_log, antechamber, tleap, cpptraj
#from batter.batter import System
from batter.input_process import get_configure_from_file
from batter.lib import build, setup, analysis, scripts
from batter.data import run_files, openmm_files
import MDAnalysis as mda
# ignore UserWarning from MDAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import pickle

from loguru import logger
# set logging level to INFO
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{level}</green> | {message}",
    level="INFO"
)

logger.info('Starting the setup of the free energy calculations')
logger.info('This script has been adapted to prepare for membrane protein system')

# Read arguments that define input file and stage
if len(sys.argv) < 5:
    scripts.help_message()
    sys.exit(0)
for i in [1, 3]:
    if '-i' == sys.argv[i].lower():
        input_file = sys.argv[i + 1]
    elif '-s' == sys.argv[i].lower():
        stage = sys.argv[i + 1]
    else:
        scripts.help_message()
        sys.exit(1)

# Read input file
sim_config = get_configure_from_file(input_file)
logger.info('Reading input file')
for field, value in sim_config.model_dump().items():
    logger.info(f"{field}: {value}")
    globals()[field] = value

logger.info('-'*50)

with open('sim_config.pkl', 'wb') as f:
    pickle.dump(sim_config, f)

aa1_poses = []
aa2_poses = []
# Adjust components and windows for OpenMM

if software == 'openmm' and stage == 'fe':
    components_inp = list(components)
    logger.info(f'Components: {components_inp}')
    if sdr_dist == 0:
        dec_method_inp = dec_method
        components = ['t', 'c']
    elif dec_method != 'exchange':
        dec_method_inp = dec_method
        logger.info(f'Decoupling method: {dec_method_inp}')
        dec_method = 'sdr'
        components = ['t', 'c', 'n', 'v']
    else:
        dec_method_inp = dec_method
        logger.info(f'Decoupling method: {dec_method_inp}')
        components = ['t', 'c', 'n', 'v', 'x']
    attach_rest_inp = list(attach_rest)
    logger.info(f'Attach rest: {attach_rest_inp}')
    attach_rest = [100.0]
    lambdas_inp = list(lambdas)
    logger.info(f'Lambdas: {lambdas_inp}')
    lambdas = [0.0]
    dt = str(float(dt)*1000)
    logger.info(f'Timestep: {dt} fs')
    cut = str(float(cut)/10)
    logger.info(f'Cutoff: {cut} nm')

    # Convert equil output file
    os.chdir('equil')
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        rng = len(release_eq) - 1
        if os.path.exists(pose):
            os.chdir(pose)
            convert_file = open('convert.in', 'w')
            convert_file.write('parm full.prmtop\n')
            convert_file.write('trajin md%02d.dcd\n' % rng)
            convert_file.write('trajout md%02d.rst7 onlyframes 10\n' % rng)
            convert_file.close()
            run_with_log(cpptraj + ' -i convert.in > convert.log')
            os.chdir('../')
    os.chdir('../')

if stage == 'equil':
    logger.info('Equilibration stage')
    comp = 'q'
    win = 0
    # Create equilibrium systems for all poses listed in the input file
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        poser = poses_def[0]
        mol = mols[i]
        molr = mols[0]
        rng = len(release_eq) - 1
        if not os.path.exists('./all-poses/'+pose+'.pdb'):
            continue
        logger.info('Setting up '+str(poses_def[i]))
        # Get number of simulations
        num_sim = len(release_eq)


        # Create aligned initial complex
        anch = build.build_equil(
                        pose, celp_st, mol,
                        H1, H2, H3,
                        calc_type, l1_x, l1_y, l1_z,
                        l1_range,
                        min_adis, max_adis,
                        ligand_ff, ligand_ph,
                        retain_lig_prot, ligand_charge,
                        other_mol, solv_shell,
                        lipid_mol, lipid_ff)
        if anch == 'anch1':
            aa1_poses.append(pose)
            os.chdir('../')
            continue
        if anch == 'anch2':
            aa2_poses.append(pose)
            os.chdir('../')
            continue
    
        # Solvate system with ions
        build.create_box(comp, hmr, pose, mol, molr,
                         num_waters, water_model, ion_def,
                         neut, buffer_x, buffer_y, buffer_z,
                         stage, ntpr, ntwr, ntwe, ntwx, cut,
                         gamma_ln, barostat, receptor_ff,
                         ligand_ff, dt, dec_method,
                         other_mol, solv_shell,
                         lipid_mol, lipid_ff)
        # Apply restraints and prepare simulation files
        logger.debug('Equil release weights:')
        for i in range(0, len(release_eq)):
            weight = release_eq[i]
            logger.debug('%s' % str(weight))
            setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                             molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
            shutil.copy('./'+pose+'/disang.rest', './'+pose+'/disang%02d.rest' % int(i))
        shutil.copy('./'+pose+'/disang%02d.rest' % int(0), './'+pose+'/disang.rest')
        setup.sim_files(hmr, temperature, mol,
                        num_sim, pose, comp, win,
                        stage, eq_steps1, eq_steps2, rng,
                        lipid_sim=lipid_mol)
        os.chdir('../')
    if len(aa1_poses) != 0:
        logger.warning('\n')
        logger.warning('WARNING: Could not find the ligand first anchor L1 for', aa1_poses)
        logger.warning('The ligand is most likely not in the defined binding site in these systems.')
    if len(aa2_poses) != 0:
        logger.warning('\n')
        logger.warning('WARNING: Could not find the ligand L2 or L3 anchors for', aa2_poses)
        logger.warning('Try reducing the min_adis parameter in the input file.')

    logger.info('Equilibration systems have been created for all poses listed in the input file.')

elif stage == 'fe':
    logger.info('Start setting simulations in free energy stage')
    # Create systems for all poses after preparation
    num_sim = apr_sim
    # Create and move to free energy directory
    if not os.path.exists('fe'):
        os.makedirs('fe')
    os.chdir('fe')
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        poser = poses_def[0]
        mol = mols[i]
        molr = mols[0]
        fwin = len(release_eq) - 1
        if not os.path.exists('../equil/'+pose):
            continue
        logger.info('Setting up '+str(poses_def[i]))
        # Create and move to pose directory
        if not os.path.exists(pose):
            os.makedirs(pose)
        os.chdir(pose)
        # Generate folder and restraints for all components and windows
        for j in range(0, len(components)):
            comp = components[j]
            # Ligand conformational release in a small box
            if (comp == 'c'):
                if not os.path.exists('rest'):
                    os.makedirs('rest')
                os.chdir('rest')
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        anch = build.build_dec(
                                        fwin, hmr, mol, pose,
                                        molr, poser, comp, win,
                                        water_model, ntpr, ntwr,
                                        ntwe, ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt,
                                        sdr_dist, dec_method,
                                        l1_x, l1_y, l1_z,
                                        l1_range, min_adis, max_adis,
                                        ion_def, other_mol, solv_shell,
                                        # set lipid_mol to empty list
                                        # because it's a ligand box
                                        lipid_mol=[], lipid_ff=lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        build.ligand_box(mol, lig_buffer, water_model, neut, ion_def, comp, ligand_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol, num_sim, pose,
                                        comp, win, stage, c_steps1, c_steps2, rng,
                                        lipid_sim=False)
                    else:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        build.build_dec(fwin, hmr, mol, pose, molr, poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt, sdr_dist, dec_method, l1_x, l1_y, l1_z, l1_range, min_adis, max_adis,
                                        ion_def, other_mol, solv_shell,
                                        lipid_mol=[], lipid_ff=lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol, num_sim, pose,
                                        comp, win, stage, c_steps1, c_steps2, rng,
                                        lipid_sim=False)
                if anch != 'all':
                    break
                os.chdir('../')
            # Receptor conformational release in a separate box
            elif (comp == 'r' or comp == 'n'):
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists('rest'):
                    os.makedirs('rest')
                os.chdir('rest')
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        anch = build.build_dec(
                                    fwin, hmr, mol, pose,
                                    molr, poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                    receptor_ff, ligand_ff, dt,
                                    sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                    l1_range, min_adis, max_adis, ion_def,
                                    other_mol, solv_shell,
                                    lipid_mol, lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        build.create_box(comp, hmr, pose,
                                         mol, molr, num_waters,
                                         water_model, ion_def, neut,
                                         buffer_x, buffer_y, buffer_z, stage,
                                         ntpr, ntwr, ntwe, ntwx,
                                         cut, gamma_ln, barostat,
                                         receptor_ff, ligand_ff, dt,
                                         dec_method, other_mol, solv_shell,
                                         lipid_mol, lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol, num_sim,
                                        pose, comp, win, stage, steps1, steps2, rng,
                                        lipid_sim=lipid_mol)
                    else:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        build.build_dec(fwin, hmr, mol, pose,
                                        molr, poser, comp, win,
                                        water_model, ntpr, ntwr, ntwe,
                                        ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt,
                                        sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                        l1_range, min_adis, max_adis, ion_def,
                                        other_mol, solv_shell,
                                        lipid_mol, lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol,
                                        num_sim, pose, comp, win,
                                        stage, steps1, steps2, rng,
                                        lipid_sim=lipid_mol)
                if anch != 'all':
                    break
                os.chdir('../')
            # Simultaneous/double decoupling/exchange
            elif (comp == 'v' or comp == 'e'):
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if dec_method == 'dd':
                    if not os.path.exists(dec_method):
                        os.makedirs(dec_method)
                    os.chdir(dec_method)
                elif dec_method == 'sdr' or dec_method == 'exchange':
                    if not os.path.exists('sdr'):
                        os.makedirs('sdr')
                    os.chdir('sdr')
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    logger.info('window: %s%02d lambda: %s' % (comp, int(win), str(weight)))
                    if int(win) == 0:
                        anch = build.build_dec(fwin, hmr, mol, pose,
                                               molr, poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                               receptor_ff, ligand_ff, dt,
                                               sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                               l1_range, min_adis, max_adis,
                                               ion_def, other_mol, solv_shell,
                                               lipid_mol, lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        build.create_box(comp, hmr, pose, mol,
                                         molr, num_waters, water_model, ion_def, neut, buffer_x, buffer_y, buffer_z, stage,
                                         ntpr, ntwr, ntwe, ntwx, cut,
                                         gamma_ln, barostat, receptor_ff,
                                         ligand_ff, dt, dec_method,
                                         other_mol,solv_shell,
                                         lipid_mol, lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                    else:
                        build.build_dec(fwin, hmr, mol, pose, molr, poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff,
                                        dt, sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                        l1_range, min_adis, max_adis, ion_def,
                                        other_mol, solv_shell,
                                        lipid_mol, lipid_ff)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                if anch != 'all':
                    break
                os.chdir('../')
            # Bulk systems for dd
            elif (comp == 'f' or comp == 'w'):
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists('dd'):
                    os.makedirs('dd')
                os.chdir('dd')
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    if int(win) == 0:
                        logger.info('window: %s%02d lambda: %s' % (comp, int(win), str(weight)))
                        anch = build.build_dec(fwin, hmr, mol, pose, molr,
                                               poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                               receptor_ff, ligand_ff, dt,
                                               sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                               l1_range, min_adis, max_adis, ion_def,
                                               other_mol, solv_shell,
                                               lipid_mol, lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        build.ligand_box(mol, lig_buffer, water_model,
                                         neut, ion_def, comp, ligand_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                    else:
                        logger.info('window: %s%02d lambda: %s' % (comp, int(win), str(weight)))
                        build.build_dec(fwin, hmr, mol, pose, molr, poser,
                                        comp, win, water_model, ntpr, ntwr, ntwe,
                                        ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt, sdr_dist,
                                        dec_method, l1_x, l1_y, l1_z, l1_range,
                                        min_adis, max_adis, ion_def,
                                        other_mol, solv_shell,
                                        lipid_mol, lipid_ff)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                if anch != 'all':
                    break
                os.chdir('../')
            elif (comp == 'x'):
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists('sdr'):
                    os.makedirs('sdr')
                os.chdir('sdr')
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    logger.info('window: %s%02d lambda: %s' % (comp, int(win), str(weight)))
                    if int(win) == 0:
                        anch = build.build_dec(fwin, hmr, mol, pose, molr,
                                               poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                               receptor_ff, ligand_ff, dt,
                                               sdr_dist, dec_method,
                                               l1_x, l1_y, l1_z, l1_range, min_adis,
                                               max_adis, ion_def,
                                               other_mol, solv_shell,
                                               lipid_mol, lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        build.create_box(comp, hmr, pose, mol,
                                         molr, num_waters, water_model,
                                         ion_def, neut, buffer_x, buffer_y, buffer_z, stage,
                                         ntpr, ntwr, ntwe, ntwx, cut,
                                         gamma_ln, barostat, receptor_ff,
                                         ligand_ff, dt, dec_method,
                                         other_mol, solv_shell,
                                         lipid_mol, lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                    else:
                        build.build_dec(fwin, hmr, mol, pose, molr,
                                        poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt,
                                        sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                        l1_range, min_adis, max_adis, ion_def,
                                        other_mol, solv_shell,
                                        lipid_mol, lipid_ff)
                        setup.dec_files(temperature, mol, num_sim, pose, comp, win, stage,
                                        steps1, steps2, weight, lambdas, dec_method, ntwx)
                if anch != 'all':
                    break

                os.chdir('../')
            # Attachments in the bound system
            else:
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists('rest'):
                    os.makedirs('rest')
                os.chdir('rest')
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if win == 0:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        anch = build.build_dec(fwin, hmr, mol, pose,
                                               molr, poser, comp, win,
                                               water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                               receptor_ff, ligand_ff, dt,
                                               sdr_dist, dec_method,
                                               l1_x, l1_y, l1_z, l1_range,
                                               min_adis, max_adis, ion_def,
                                               other_mol, solv_shell,
                                               lipid_mol, lipid_ff)
                        if anch == 'anch1':
                            aa1_poses.append(pose)
                            break
                        if anch == 'anch2':
                            aa2_poses.append(pose)
                            break
                        if anch != 'altm':
                            build.create_box(comp, hmr, pose, mol, molr, num_waters, water_model, ion_def, neut, buffer_x, buffer_y, buffer_z, stage,
                                             ntpr, ntwr, ntwe, ntwx, cut,
                                             gamma_ln, barostat,
                                             receptor_ff, ligand_ff,
                                             dt, dec_method, other_mol, solv_shell,
                                             lipid_mol, lipid_ff)
                        logger.info('Creating restraints for attaching...')
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol,
                                        num_sim, pose, comp, win,
                                        stage, steps1, steps2, rng,
                                        lipid_sim=lipid_mol)
                    else:
                        logger.info('window: %s%02d weight: %s' % (comp, int(win), str(weight)))
                        build.build_dec(fwin, hmr, mol, pose, molr,
                                        poser, comp, win, water_model, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat,
                                        receptor_ff, ligand_ff, dt,
                                        sdr_dist, dec_method, l1_x, l1_y, l1_z,
                                        l1_range, min_adis, max_adis, ion_def,
                                        other_mol, solv_shell,
                                        lipid_mol, lipid_ff)
                        setup.restraints(pose, rest, bb_start, bb_end, weight, stage, mol,
                                         molr, comp, bb_equil, sdr_dist, dec_method, other_mol)
                        setup.sim_files(hmr, temperature, mol,
                                        num_sim, pose, comp, win,
                                        stage, steps1, steps2, rng,
                                        lipid_sim=lipid_mol)
                if anch == 'anch1' or anch == 'anch2':
                    break
                os.chdir('../')
        os.chdir('../')
    if len(aa1_poses) != 0:
        logger.warning('\n')
        logger.warning('WARNING: Could not find the ligand first anchor L1 for', aa1_poses)
        logger.warning('The ligand most likely left the binding site during equilibration of these systems.')
        for i in aa1_poses:
            shutil.rmtree('./'+i+'')
    if len(aa2_poses) != 0:
        logger.warning('\n')
        logger.warning('WARNING: Could not find the ligand L2 or L3 anchors for', aa2_poses)
        logger.warning('Try reducing the min_adis parameter in the input file.')
        for i in aa2_poses:
            shutil.rmtree('./'+i+'')
    logger.info('Free energy systems have been created for all poses listed in the input file.')
    logger.info('now cd fe/pose0')
    logger.info(f'cp {run_files}/run-express.bash')
    logger.info('and bash run-express.bash')
elif stage == 'analysis':
    logger.info('Analysis stage')
    # Free energy analysis for OpenMM
    if software == 'openmm':
        for i in range(0, len(poses_def)):
            pose = poses_def[i]
            analysis.fe_openmm(components, temperature, pose, dec_method, rest, attach_rest, lambdas,
                               dic_itera1, dic_itera2, itera_steps, dt, dlambda, dec_int, weights, blocks, ti_points)
            os.chdir('../../')
    else:
        # Free energy analysis for AMBER20
        for i in range(0, len(poses_def)):
            pose = poses_def[i]
            analysis.fe_values(blocks, components, temperature, pose, attach_rest, lambdas,
                               weights, dec_int, dec_method, rest, dic_steps1, dic_steps2, dt)
            os.chdir('../../')


# Convert equilibration folders to openmm

if software == 'openmm' and stage == 'equil':
    logger.info('Converting equilibration folders to OpenMM')

    # Adjust a few variables
    cut = str(float(cut)/10)
    dt = str(float(dt)*1000)

    os.chdir('equil')
    for i in range(0, len(poses_def)):
        mol = mols[i]
        pose = poses_def[i]
        rng = len(release_eq) - 1
        if os.path.exists(pose):
            logger.info(pose)
            os.rename(pose, pose+'-amber')
            os.mkdir(pose)
            os.chdir(pose)
            shutil.copy('../'+pose+'-amber/equil-%s.pdb' % mol.lower(), './')
            shutil.copy('../'+pose+'-amber/cv.in', './')
            shutil.copy('../'+pose+'-amber/assign.dat', './')
            for file in glob.glob('../'+pose+'-amber/vac*'):
                shutil.copy(file, './')
            for file in glob.glob('../'+pose+'-amber/full*'):
                shutil.copy(file, './')
            for file in glob.glob('../'+pose+'-amber/disang*'):
                shutil.copy(file, './')
            for file in glob.glob('../'+pose+'-amber/build*'):
                shutil.copy(file, './')
            for file in glob.glob('../'+pose+'-amber/tleap_solvate*'):
                shutil.copy(file, './')
            fin = open(f'{run_files}/local-equil-op.bash', "rt")
            data = fin.read()
            data = data.replace('RANGE', '%02d' % rng)
            fin.close()
            fin = open('run-local.bash', "wt")
            fin.write(data)
            fin.close()
            fin = open(f'{run_files}/PBS-Op', "rt")
            data = fin.read()
            data = data.replace('STAGE', stage).replace('POSE', pose)
            fin.close()
            fin = open('PBS-run', "wt")
            fin.write(data)
            fin.close()
            fin = open(f'{run_files}/SLURMM-Op', "rt")
            data = fin.read()
            data = data.replace('STAGE', stage).replace('POSE', pose)
            fin.close()
            fin = open('SLURMM-run', "wt")
            fin.write(data)
            fin.close()
            for j in range(0, len(release_eq)):
                fin = open(f'{openmm_files}/equil.py', "rt")
                data = fin.read()
                data = data.replace('LIG', mol.upper()).replace('TMPRT', str(temperature)).replace(
                    'TSTP', str(dt)).replace('GAMMA_LN', str(gamma_ln)).replace('STG', '%02d' % j).replace('CTF', cut)
                if hmr == 'yes':
                    data = data.replace('PRMFL', 'full.hmr.prmtop')
                else:
                    data = data.replace('PRMFL', 'full.prmtop')
                if j == rng:
                    data = data.replace('TOTST', str(eq_steps2))
                else:
                    data = data.replace('TOTST', str(eq_steps1))
                fin.close()
                fin = open('equil-%02d.py' % j, "wt")
                fin.write(data)
                fin.close()
            os.chdir('../')
            shutil.rmtree('./'+pose+'-amber')

if software == 'openmm' and stage == 'fe':

    # Redefine input arrays

    components = list(components_inp)
    attach_rest = list(attach_rest_inp)
    dec_method = dec_method_inp
    lambdas = list(lambdas_inp)
    lambdas_rest = []
    for i in attach_rest:
        lbd_rst = float(i)/float(100)
        lambdas_rest.append(lbd_rst)
    Input = lambdas_rest
    lambdas_rest = ['{:.5f}'.format(elem) for elem in Input]

    # Start script

    logger.info('')
    logger.info('#############################')
    logger.info('## OpenMM patch for BAT.py ##')
    logger.info('#############################')
    logger.info('')
    logger.info('Components: ', components)
    logger.info('')
    logger.info('Decoupling lambdas: ', lambdas)
    logger.info('')
    logger.info('Restraint lambdas: ', lambdas_rest)
    logger.info('')
    logger.info('Integration Method: ', dec_int.upper())
    logger.info('')

    # Generate folder and restraints for all components and windows
    for i in range(0, len(poses_def)):
        mol = mols[i]
        molr = mols[0]
        if not os.path.exists(poses_def[i]):
            continue
        os.chdir(poses_def[i])
        for j in range(0, len(components)):
            comp = components[j]
            if comp == 'a' or comp == 'l' or comp == 't' or comp == 'r' or comp == 'c' or comp == 'm' or comp == 'n':
                if not os.path.exists('rest'):
                    os.makedirs('rest')
                os.chdir('rest')
                if not os.path.exists(comp+'-comp'):
                    os.makedirs(comp+'-comp')
                os.chdir(comp+'-comp')
                itera1 = dic_itera1[comp]
                itera2 = dic_itera2[comp]
                shutil.copy(f'{run_files}/local-rest-op.bash', './run-local.bash')
                fin = open(f'{run_files}/PBS-Op', "rt")
                data = fin.read()
                data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                fin.close()
                fin = open('PBS-run', "wt")
                fin.write(data)
                fin.close()
                fin = open(f'{run_files}/SLURMM-Op', "rt")
                data = fin.read()
                data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                fin.close()
                fin = open('SLURMM-run', "wt")
                fin.write(data)
                fin.close()
                fin = open(f'{openmm_files}/rest.py', "rt")
                data = fin.read()
                data = data.replace('LAMBDAS', '[%s]' % ' , '.join(map(str, lambdas_rest))).replace('LIG', mol.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(itera_steps)).replace(
                    'PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                if hmr == 'yes':
                    data = data.replace('PRMFL', 'full.hmr.prmtop')
                else:
                    data = data.replace('PRMFL', 'full.prmtop')
                fin.close()
                fin = open('rest.py', "wt")
                fin.write(data)
                fin.close()
                if comp == 'c':
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/disang.rest', './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/full*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/vac*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/build*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/tleap_solvate*'):
                        shutil.copy(file, './')
                elif comp == 'n':
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/disang.rest', './')
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/cv.in', './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/full*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/vac*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/build*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/n00/tleap_solvate*'):
                        shutil.copy(file, './')
                else:
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/disang.rest', './')
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/cv.in', './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/full*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/vac*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/build*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/tleap_solvate*'):
                        shutil.copy(file, './')
                os.chdir('../../')
            elif comp == 'e' or comp == 'v' or comp == 'w' or comp == 'f':
                if dec_method == 'sdr' or dec_method == 'exchange':
                    if not os.path.exists('sdr'):
                        os.makedirs('sdr')
                    os.chdir('sdr')
                    if dec_int == 'mbar':
                        if not os.path.exists(comp+'-comp'):
                            os.makedirs(comp+'-comp')
                        os.chdir(comp+'-comp')
                        itera1 = dic_itera1[comp]
                        itera2 = dic_itera2[comp]
                        shutil.copy(f'{run_files}/local-sdr-op.bash', './run-local.bash')
                        fin = open(f'{run_files}/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('PBS-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{run_files}/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('SLURMM-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{openmm_files}/sdr.py', "rt")
                        data = fin.read()
                        data = data.replace('LAMBDAS', '[%s]' % ' , '.join(map(str, lambdas))).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                            itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                        if hmr == 'yes':
                            data = data.replace('PRMFL', 'full.hmr.prmtop')
                        else:
                            data = data.replace('PRMFL', 'full.prmtop')
                        fin.close()
                        fin = open('sdr.py', "wt")
                        fin.write(data)
                        fin.close()
                        shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/disang.rest', './')
                        shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/cv.in', './')
                        for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/full*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/vac*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/tleap_solvate*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/build*'):
                            shutil.copy(file, './')
                        os.chdir('../')
                    elif dec_int == 'ti':
                        if not os.path.exists(comp+'-comp'):
                            os.makedirs(comp+'-comp')
                        os.chdir(comp+'-comp')
                        itera1 = int(dic_itera1[comp]*itera_steps)
                        itera2 = int(dic_itera2[comp]/2)
                        for k in range(0, len(lambdas)):
                            if not os.path.exists('%s%02d' % (comp, int(k))):
                                os.makedirs('%s%02d' % (comp, int(k)))
                            os.chdir('%s%02d' % (comp, int(k)))
                            shutil.copy(f'{run_files}/local-sdr-op-ti.bash', './run-local.bash')
                            fin = open(f'{run_files}/SLURMM-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{run_files}/PBS-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{openmm_files}/equil-sdr.py', "rt")
                            data = fin.read()
                            data = data.replace('LBD0', '%8.6f' % lambdas[k]).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                                itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut)
                            if hmr == 'yes':
                                data = data.replace('PRMFL', 'full.hmr.prmtop')
                            else:
                                data = data.replace('PRMFL', 'full.prmtop')
                            fin.close()
                            fin = open('equil-sdr.py', "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{openmm_files}/sdr-ti.py', "rt")
                            data = fin.read()
                            # "Split" initial lambda into two close windows
                            lambda1 = float(lambdas[k] - dlambda/2)
                            lambda2 = float(lambdas[k] + dlambda/2)
                            data = data.replace('LBD1', '%8.6f' % lambda1).replace('LBD2', '%8.6f' % lambda2).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                                itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                            if hmr == 'yes':
                                data = data.replace('PRMFL', 'full.hmr.prmtop')
                            else:
                                data = data.replace('PRMFL', 'full.prmtop')
                            fin.close()
                            fin = open('sdr-ti.py', "wt")
                            fin.write(data)
                            fin.close()
                            shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/disang.rest', './')
                            shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/cv.in', './')
                            for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/full*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/vac*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/tleap_solvate*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/v00/build*'):
                                shutil.copy(file, './')
                            os.chdir('../')
                        os.chdir('../')
                elif dec_method == 'dd':
                    if not os.path.exists('dd'):
                        os.makedirs('dd')
                    os.chdir('dd')
                    if dec_int == 'mbar':
                        if not os.path.exists(comp+'-comp'):
                            os.makedirs(comp+'-comp')
                        os.chdir(comp+'-comp')
                        itera1 = dic_itera1[comp]
                        itera2 = dic_itera2[comp]
                        if not os.path.exists('../run_files'):
                            shutil.copytree(f'{run_files}', '../run_files')
                        shutil.copy(f'{run_files}/local-dd-op.bash', './run-local.bash')
                        fin = open(f'{run_files}/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('PBS-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{run_files}/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('SLURMM-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{openmm_files}/dd.py', "rt")
                        data = fin.read()
                        data = data.replace('LAMBDAS', '[%s]' % ' , '.join(map(str, lambdas))).replace('LIG', mol.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(itera_steps)).replace(
                            'PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                        if hmr == 'yes':
                            data = data.replace('PRMFL', 'full.hmr.prmtop')
                        else:
                            data = data.replace('PRMFL', 'full.prmtop')
                        fin.close()
                        fin = open('dd.py', "wt")
                        fin.write(data)
                        fin.close()
                        if comp == 'f' or comp == 'w':
                            shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/disang.rest', './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/full*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/vac*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/tleap_solvate*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/c00/build*'):
                                shutil.copy(file, './')
                        else:
                            shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/disang.rest', './')
                            shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/cv.in', './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/full*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/vac*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/tleap_solvate*'):
                                shutil.copy(file, './')
                            for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/rest/t00/build*'):
                                shutil.copy(file, './')
                        os.chdir('../')
                    elif dec_int == 'ti':
                        if not os.path.exists(comp+'-comp'):
                            os.makedirs(comp+'-comp')
                        os.chdir(comp+'-comp')
                        itera1 = int(dic_itera1[comp]*itera_steps)
                        itera2 = int(dic_itera2[comp]/2)
                        for k in range(0, len(lambdas)):
                            if not os.path.exists('%s%02d' % (comp, int(k))):
                                os.makedirs('%s%02d' % (comp, int(k)))
                            os.chdir('%s%02d' % (comp, int(k)))
                            shutil.copy(f'{run_files}/local-dd-op-ti.bash', './run-local.bash')
                            fin = open(f'{run_files}/SLURMM-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{run_files}/PBS-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{openmm_files}/equil-dd.py', "rt")
                            data = fin.read()
                            data = data.replace('LBD0', '%8.6f' % lambdas[k]).replace('LIG', mol.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(itera_steps)).replace(
                                'PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut)
                            if hmr == 'yes':
                                data = data.replace('PRMFL', 'full.hmr.prmtop')
                            else:
                                data = data.replace('PRMFL', 'full.prmtop')
                            fin.close()
                            fin = open('equil-dd.py', "wt")
                            fin.write(data)
                            fin.close()
                            fin = open(f'{openmm_files}/dd-ti.py', "rt")
                            data = fin.read()
                            # "Split" initial lambda into two close windows
                            lambda1 = float(lambdas[k] - dlambda/2)
                            lambda2 = float(lambdas[k] + dlambda/2)
                            data = data.replace('LBD1', '%8.6f' % lambda1).replace('LBD2', '%8.6f' % lambda2).replace('LIG', mol.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(itera_steps)).replace(
                                'PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                            if hmr == 'yes':
                                data = data.replace('PRMFL', 'full.hmr.prmtop')
                            else:
                                data = data.replace('PRMFL', 'full.prmtop')
                            fin.close()
                            fin = open('dd-ti.py', "wt")
                            fin.write(data)
                            fin.close()
                            if comp == 'f' or comp == 'w':
                                shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/rest/c00/disang.rest', './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/c00/full*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/c00/vac*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/c00/tleap_solvate*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/c00/build*'):
                                    shutil.copy(file, './')
                            else:
                                shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/disang.rest', './')
                                shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/cv.in', './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/full*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/vac*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/tleap_solvate*'):
                                    shutil.copy(file, './')
                                for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/rest/t00/build*'):
                                    shutil.copy(file, './')
                            os.chdir('../')
                        os.chdir('../')
                os.chdir('../')
            elif comp == 'x':
                if not os.path.exists('sdr'):
                    os.makedirs('sdr')
                os.chdir('sdr')
                if dec_int == 'mbar':
                    if not os.path.exists(comp+'-comp'):
                        os.makedirs(comp+'-comp')
                    os.chdir(comp+'-comp')
                    itera1 = dic_itera1[comp]
                    itera2 = dic_itera2[comp]
                    shutil.copy(f'{run_files}/local-sdr-op.bash', './run-local.bash')
                    fin = open(f'{run_files}/PBS-Op', "rt")
                    data = fin.read()
                    data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                    fin.close()
                    fin = open('PBS-run', "wt")
                    fin.write(data)
                    fin.close()
                    fin = open(f'{run_files}/SLURMM-Op', "rt")
                    data = fin.read()
                    data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                    fin.close()
                    fin = open('SLURMM-run', "wt")
                    fin.write(data)
                    fin.close()
                    fin = open(f'{openmm_files}/sdr.py', "rt")
                    data = fin.read()
                    data = data.replace('LAMBDAS', '[%s]' % ' , '.join(map(str, lambdas))).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                        itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                    if hmr == 'yes':
                        data = data.replace('PRMFL', 'full.hmr.prmtop')
                    else:
                        data = data.replace('PRMFL', 'full.prmtop')
                    fin.close()
                    fin = open('sdr.py', "wt")
                    fin.write(data)
                    fin.close()
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/disang.rest', './')
                    shutil.copy('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/cv.in', './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/full*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/vac*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/tleap_solvate*'):
                        shutil.copy(file, './')
                    for file in glob.glob('../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/build*'):
                        shutil.copy(file, './')
                    os.chdir('../')
                elif dec_int == 'ti':
                    if not os.path.exists(comp+'-comp'):
                        os.makedirs(comp+'-comp')
                    os.chdir(comp+'-comp')
                    itera1 = int(dic_itera1[comp]*itera_steps)
                    itera2 = int(dic_itera2[comp]/2)
                    for k in range(0, len(lambdas)):
                        if not os.path.exists('%s%02d' % (comp, int(k))):
                            os.makedirs('%s%02d' % (comp, int(k)))
                        os.chdir('%s%02d' % (comp, int(k)))
                        shutil.copy(f'{run_files}/local-sdr-op-ti.bash', './run-local.bash')
                        fin = open(f'{run_files}/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                        fin.close()
                        fin = open("SLURMM-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{run_files}/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                        fin.close()
                        fin = open("PBS-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{openmm_files}/equil-sdr.py', "rt")
                        data = fin.read()
                        data = data.replace('LBD0', '%8.6f' % lambdas[k]).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                            itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut)
                        if hmr == 'yes':
                            data = data.replace('PRMFL', 'full.hmr.prmtop')
                        else:
                            data = data.replace('PRMFL', 'full.prmtop')
                        fin.close()
                        fin = open('equil-sdr.py', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open(f'{openmm_files}/sdr-ti.py', "rt")
                        data = fin.read()
                        # "Split" initial lambda into two close windows
                        lambda1 = float(lambdas[k] - dlambda/2)
                        lambda2 = float(lambdas[k] + dlambda/2)
                        data = data.replace('LBD1', '%8.6f' % lambda1).replace('LBD2', '%8.6f' % lambda2).replace('LIG', mol.upper()).replace('LREF', molr.upper()).replace('TMPRT', str(temperature)).replace('TSTP', str(dt)).replace('SPITR', str(
                            itera_steps)).replace('PRIT', str(itera2)).replace('EQIT', str(itera1)).replace('ITCH', str(itcheck)).replace('GAMMA_LN', str(gamma_ln)).replace('CMPN', str(comp)).replace('CTF', cut).replace('BLCKS', str(blocks))
                        if hmr == 'yes':
                            data = data.replace('PRMFL', 'full.hmr.prmtop')
                        else:
                            data = data.replace('PRMFL', 'full.prmtop')
                        fin.close()
                        fin = open('sdr-ti.py', "wt")
                        fin.write(data)
                        fin.close()
                        shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/disang.rest', './')
                        shutil.copy('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/cv.in', './')
                        for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/full*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/vac*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/tleap_solvate*'):
                            shutil.copy(file, './')
                        for file in glob.glob('../../../../../'+stage+'/'+poses_def[i]+'/sdr/x00/build*'):
                            shutil.copy(file, './')
                        os.chdir('../')
                    os.chdir('../')
                os.chdir('../')

        # Clean up amber windows
        dirpath = os.path.join('rest', 't00')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('rest', 'amber_files')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('rest', 'c00')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('rest', 'n00')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('sdr', 'v00')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('sdr', 'amber_files')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
#    dirpath = os.path.join('sdr', 'x00')
#    if os.path.exists(dirpath) and os.path.isdir(dirpath):
#      shutil.rmtree(dirpath)
        os.chdir('../')
