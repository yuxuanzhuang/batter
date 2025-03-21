#!/usr/bin/env python2
import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
from lib import build
from lib import scripts
from lib import setup
from lib import analysis
import numpy as np
from lib.utils import run_with_log, antechamber, tleap, cpptraj
import MDAnalysis as mda
# ignore UserWarning from MDAnalysis
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

from loguru import logger
# set logging level to INFO
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{level}</green> | {message}",
    level="INFO"
)

ion_def = []
poses_list = []
ligand_list = []
poses_def = []
release_eq = []
attach_rest = []
lambdas = []
weights = []
components = []
aa1_poses = []
aa2_poses = []
other_mol = []
bb_start = []
bb_end = []
mols = []
celp_st = []
lipid_mol = []

# Defaults

a_steps1 = 0
a_steps2 = 0
l_steps1 = 0
l_steps2 = 0
t_steps1 = 0
t_steps2 = 0
m_steps1 = 0
m_steps2 = 0
n_steps1 = 0
n_steps2 = 0
c_steps1 = 0
c_steps2 = 0
r_steps1 = 0
r_steps2 = 0
e_steps1 = 0
e_steps2 = 0
v_steps1 = 0
v_steps2 = 0
f_steps1 = 0
f_steps2 = 0
w_steps1 = 0
w_steps2 = 0
x_steps1 = 0
x_steps2 = 0

a_itera1 = 0
a_itera2 = 0
l_itera1 = 0
l_itera2 = 0
t_itera1 = 0
t_itera2 = 0
m_itera1 = 0
m_itera2 = 0
n_itera1 = 0
n_itera2 = 0
c_itera1 = 0
c_itera2 = 0
r_itera1 = 0
r_itera2 = 0
e_itera1 = 0
e_itera2 = 0
v_itera1 = 0
v_itera2 = 0
f_itera1 = 0
f_itera2 = 0
w_itera1 = 0
w_itera2 = 0
x_itera1 = 0
x_itera2 = 0

sdr_dist = 0
rng = 0
rec_dihcf_force = 0
buffer_z = 0
num_waters = 0
ion_conc = 0.0
retain_lig_prot = 'no'
ligand_ph = 7.0
ligand_charge = 'nd'
software = 'amber'
solv_shell = 0.0
dlambda = 0.001

ntpr = '1000'
ntwr = '10000'
ntwe = '0'
ntwx = '2500'
cut = '9.0'
barostat = '2'
ti_points = 0

logger.info('Starting the setup of the free energy calculations')
logger.info('This script has been adapted to prepare for membrane protein system')
logger.info('Now reading the input file')

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

# Open input file
with open(input_file) as f_in:
    # Remove spaces and tabs
    lines = (line.strip(' \t\n\r') for line in f_in)
    lines = list(line for line in lines if line)  # Non-blank lines in a list

for i in range(0, len(lines)):
    # split line using the equal sign, and remove text after #
    if not lines[i][0] == '#':
        lines[i] = lines[i].split('#')[0].split('=')

# Read parameters from input file
for i in range(0, len(lines)):
    if not lines[i][0] == '#':
        key = lines[i][0].strip().lower()
        value = lines[i][1].strip()
        match key:
            case 'temperature':
                temperature = scripts.check_input('float', value, input_file, key)
            # Wildcard matching for keys ending with '_steps'
            case key if (key.startswith(('eq', 'a', 'l', 't',
                                         'm', 'n', 'c', 'r',
                                         'e', 'v', 'f', 'w',
                                         'x')) and
                         (key.endswith('_steps1') or key.endswith('_steps2'))):
                globals()[key] = scripts.check_input('int', value, input_file, key)
            # OpenMM specific keys:
            case key if (key.startswith(('a', 'l', 't',
                                         'm', 'n', 'c', 'r',
                                         'e', 'v', 'f', 'w',
                                         'x')) and
                         (key.endswith('_itera1') or key.endswith('_itera2'))):
                globals()[key] = scripts.check_input('int', value, input_file, key)
            case 'itera_steps':
                itera_steps = scripts.check_input('int', value, input_file, key)
            case 'itcheck':
                itcheck = value
            case 'ti_points':
                ti_points = scripts.check_input('int', value, input_file, key)
            case 'poses_list' | 'ligand_list' | 'other_mol':
                newline = value.strip('\'\"-,.:;#()][').split(',')
                for item in newline:
                    if key == 'poses_list':
                        poses_list.append(scripts.check_input('int', item, input_file, key))
                    elif key == 'ligand_list':
                        ligand_list.append(item.strip())
                    elif key == 'other_mol':
                        other_mol.append(item.strip())
            # Example for handling specific strings:
            case 'calc_type':
                if value.lower() in {'dock', 'rank', 'crystal'}:
                    calc_type = value.lower()
                else:
                    logger.error('Please choose dock, rank, or crystal for the calculation type')
                    sys.exit(1)
            case 'retain_lig_prot':
                retain_lig_prot = value.lower()
            case 'celpp_receptor':
                newline = value.strip('\'\"-,.:;#()][').split(',')
                for item in newline:
                    celp_st.append(item)
            case 'p1' | 'p2' | 'p3':
                if key == 'p1':
                    H1 = value
                elif key == 'p2':
                    H2 = value
                elif key == 'p3':
                    H3 = value
            case 'ligand_name':
                newline = value.strip('\'\"-,.:;#()][').split(',')
                for item in newline:
                    mols.append(item)
            case 'fe_type':
                fe_type = value.lower()
                if fe_type not in {'rest', 'dd', 'sdr', 'sdr-rest', 'express', 'dd-rest', 'relative', 'custom'}:
                    logger.error('Free energy type not recognized. Please choose a valid option.')
                    sys.exit(1)
            case 'dec_int':
                dec_int = value.lower()
                if dec_int not in {'mbar', 'ti'}:
                    logger.error('Decoupling integration method not recognized. Please choose ti or mbar.')
                    sys.exit(1)
            case 'dec_method':
                dec_method = value.lower()
                if dec_method not in {'dd', 'sdr', 'exchange'}:
                    logger.error('Decoupling method not recognized. Please choose dd, sdr, or exchange.')
                    sys.exit(1)
            case 'blocks':
                blocks = scripts.check_input('int', value, input_file, key)
            case 'hmr':
                hmr = value.lower()
                if hmr not in {'yes', 'no'}:
                    logger.error('Wrong input! Please use yes or no for hydrogen mass repartitioning.')
                    sys.exit(1)
            # Handle multi-choice options:
            case 'water_model':
                if value.lower() in {'tip3p', 'tip4pew', 'spce', 'opc', 'tip3pf'}:
                    water_model = value.upper()
                else:
                    logger.error('Water model not supported. Please choose TIP3P, TIP4PEW, SPCE, OPC or TIP3PF')
                    sys.exit(1)
            case 'num_waters':
                num_waters = scripts.check_input('int', value, input_file, key)
            case 'neutralize_only':
                neut = value.lower()
                if neut not in {'yes', 'no'}:
                    logger.error('Wrong input! Please use yes or no to indicate whether neutralization only or extra ions will be added.')
                    sys.exit(1)
            case 'cation':
                cation = value
            case 'anion':
                anion = value
            case 'ion_conc':
                ion_conc = scripts.check_input('float', value, input_file, key)
            case 'buffer_x' | 'buffer_y' | 'buffer_z':
                globals()[key] = scripts.check_input('float', value, input_file, key)
            case 'lig_buffer':
                lig_buffer = scripts.check_input('float', value, input_file, key)
            case 'rec_dihcf_force' | 'rec_discf_force' | 'lig_distance_force' | \
                 'lig_angle_force' | 'lig_dihcf_force' | 'rec_com_force' | 'lig_com_force':
                globals()[key] = scripts.check_input('float', value, input_file, key)
            case 'sdr_dist' | 'l1_x' | 'l1_y' | 'l1_z' | 'l1_range' | \
                 'min_adis' | 'max_adis' | 'solv_shell' | 'dlambda':
                globals()[key] = scripts.check_input('float', value, input_file, key)
            case 'rec_bb':
                rec_bb = value.lower()
                if rec_bb not in {'yes', 'no'}:
                    logger.error('Wrong input! Please use yes or no to indicate whether protein backbone restraints will be used.')
                    sys.exit(1)
            case 'bb_start' | 'bb_end':
                newline = value.strip('\'\"-,.:;#()][').split(',')
                for item in newline:
                    globals()[key].append(scripts.check_input('int', item, input_file, key))
            case 'bb_equil':
                bb_equil = value.lower() if value.lower() == 'yes' else 'no'
            case 'release_eq' | 'attach_rest' | 'lambdas':
                strip_line = value.strip('\'\"-,.:;#()][').split()
                for item in strip_line:
                    globals()[key].append(scripts.check_input('float', item, input_file, key))
            case 'components':
                strip_line = value.strip('\'\"-,.:;#()][').split()
                for item in strip_line:
                    components.append(item)
            case 'ntpr' | 'ntwr' | 'ntwe' | 'ntwx' | 'cut' | 'gamma_ln' | 'barostat' | 'receptor_ff':
                globals()[key] = value
            case 'ligand_ff':
                ligand_ff = value.lower()
                if ligand_ff not in {'gaff', 'gaff2'}:
                    logger.error('Wrong input! Available options for ligand force-field are gaff and gaff2.')
                    sys.exit(1)
            case 'ligand_ph' | 'ligand_charge':
                globals()[key] = scripts.check_input('float', value, input_file, key)
            case 'dt':
                dt = value
            case 'software':
                software = value.lower()
                if software not in {'openmm', 'amber'}:
                    logger.error('Simulation software not recognized. Please choose openmm or amber.')
                    sys.exit(1)
            # New lipid options
            case 'lipid_mol':
                newline = value.strip('\'\"-,.:;#()][').split(',')
                for item in newline:
                    lipid_mol.append(item.strip())
            case 'lipid_ff':
                lipid_ff = value.lower()
                if lipid_ff not in {'lipid21'}:
                    logger.error('Wrong input! Available options for lipid force-field are lipid21.')
                    sys.exit(1)
            case _:
                logger.error(f"Unrecognized key: {key}")
                sys.exit(1)

if len(bb_start) != len(bb_end):
    logger.error('Wrong input! Please use arrays of the same size for bb_start and bb_end.')
    sys.exit(1)

if num_waters == 0 and buffer_z == 0:
    logger.error('Wrong input! Please choose either a number of water molecules or a z buffer value.')
    sys.exit(1)

if num_waters != 0 and buffer_z != 0:
    logger.error('Wrong input! Please choose either a number of water molecules or a z buffer value.')
    sys.exit(1)

if buffer_x <= solv_shell or buffer_y <= solv_shell:
    logger.error('Wrong input! Solvation buffers cannot be smaller than the solv_shell variable.')
    sys.exit(1)

if buffer_z != 0 and buffer_z <= solv_shell:
    logger.error('Wrong input! Solvation buffers cannot be smaller than the solv_shell variable.')
    sys.exit(1)

if other_mol == ['']:
    other_mol = []

if lipid_mol == ['']:
    lipid_mol = []

if lipid_mol:
        # convert back to lipid
    charmm_amber_lipid_df = pd.read_csv('build_files/charmmlipid2amber.csv', header=1, sep=',')

    logger.info(f'Converting lipid input: {lipid_mol}')
    lipid_mol.extend(charmm_amber_lipid_df.query('residue in @lipid_mol')['replace'].apply(lambda x: x.split()[1]).unique().tolist())
    logger.info(f'Amber lipids: {lipid_mol}')

# Number of simulations, 1 equilibrium and 1 production
apr_sim = 2

# Define free energy components
if fe_type == 'custom':
    try:
        dec_method
    except NameError:
        logger.error('Wrong input! Please choose a decoupling method (dd, sdr or exchange) when using the custom option.')
        sys.exit(1)
elif fe_type == 'rest':
    components = ['c', 'a', 'l', 't', 'r']
    dec_method = 'dd'
elif fe_type == 'sdr':
    components = ['e', 'v']
    dec_method = 'sdr'
elif fe_type == 'dd':
    components = ['e', 'v', 'f', 'w']
    dec_method = 'dd'
elif fe_type == 'sdr-rest':
    components = ['c', 'a', 'l', 't', 'r', 'e', 'v']
    dec_method = 'sdr'
elif fe_type == 'express':
    components = ['m', 'n', 'e', 'v']
    dec_method = 'sdr'
elif fe_type == 'dd-rest':
    components = ['c', 'a', 'l', 't', 'r', 'e', 'v', 'f', 'w']
    dec_method = 'dd'
elif fe_type == 'relative':
    components = ['x', 'e', 'n', 'm']
    dec_method = 'exchange'

if (dec_method == 'sdr' or dec_method == 'exchange') and sdr_dist == 0:
    logger.error('Wrong input! Please choose a positive value for the sdr_dist variable when performing sdr or exchange.')
    sys.exit(1)

for i in components:
    if i == 'n' and sdr_dist == 0:
        logger.error('Wrong input! Please choose a positive value for the sdr_dist variable when using the n component.')
        sys.exit(1)

# Do not apply protein backbone restraints
if rec_bb == 'no':
    bb_start = [1]
    bb_end = [0]
    bb_equil = 'no'


# Create poses definitions
if calc_type == 'dock':
    celp_st = celp_st[0]
    for i in range(0, len(poses_list)):
        poses_def.append('pose'+str(poses_list[i]))
elif calc_type == 'rank':
    celp_st = celp_st[0]
    for i in range(0, len(ligand_list)):
        poses_def.append(ligand_list[i])
elif calc_type == 'crystal':
    for i in range(0, len(celp_st)):
        poses_def.append(celp_st[i])

# Obtain all ligand names
if calc_type != 'crystal':
    mols = []
    for i in range(0, len(poses_def)):
        with open('./all-poses/%s.pdb' % poses_def[i].lower()) as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
            for j in range(0, len(lines)):
                if (lines[j][0:6].strip() == 'ATOM') or (lines[j][0:6].strip() == 'HETATM'):
                    lig_name = (lines[j][17:20].strip())
                    mols.append(lig_name)
                    break

logger.info(f'Receptor/complex structures: {celp_st}')
logger.info(f'Ligand names: {mols}')
logger.info(f'Cobinders names: {other_mol}')
logger.info(f'Lipid names: {lipid_mol}')

for i in range(0, len(mols)):
    if mols[i] in other_mol:
        logger.error('Same residue name ('+mols[i]+') found in ligand name and cobinders, please change one of them')
        sys.exit(1)


# Create restraint definitions
rest = [rec_dihcf_force, rec_discf_force, lig_distance_force,
        lig_angle_force, lig_dihcf_force, rec_com_force, lig_com_force]

# Create ion definitions
ion_def = [cation, anion, ion_conc]

# Define number of steps for all stages (amber)
dic_steps1 = {}
dic_steps2 = {}
dic_steps1['a'] = a_steps1
dic_steps2['a'] = a_steps2
dic_steps1['l'] = l_steps1
dic_steps2['l'] = l_steps2
dic_steps1['t'] = t_steps1
dic_steps2['t'] = t_steps2
dic_steps1['m'] = m_steps1
dic_steps2['m'] = m_steps2
dic_steps1['n'] = n_steps1
dic_steps2['n'] = n_steps2
dic_steps1['c'] = c_steps1
dic_steps2['c'] = c_steps2
dic_steps1['r'] = r_steps1
dic_steps2['r'] = r_steps2
dic_steps1['v'] = v_steps1
dic_steps2['v'] = v_steps2
dic_steps1['e'] = e_steps1
dic_steps2['e'] = e_steps2
dic_steps1['w'] = w_steps1
dic_steps2['w'] = w_steps2
dic_steps1['f'] = f_steps1
dic_steps2['f'] = f_steps2
dic_steps1['x'] = x_steps1
dic_steps2['x'] = x_steps2

# Define number of steps for all stages (openmm)
dic_itera1 = {}
dic_itera2 = {}
dic_itera1['a'] = a_itera1
dic_itera2['a'] = a_itera2
dic_itera1['l'] = l_itera1
dic_itera2['l'] = l_itera2
dic_itera1['t'] = t_itera1
dic_itera2['t'] = t_itera2
dic_itera1['m'] = m_itera1
dic_itera2['m'] = m_itera2
dic_itera1['n'] = n_itera1
dic_itera2['n'] = n_itera2
dic_itera1['c'] = c_itera1
dic_itera2['c'] = c_itera2
dic_itera1['r'] = r_itera1
dic_itera2['r'] = r_itera2
dic_itera1['v'] = v_itera1
dic_itera2['v'] = v_itera2
dic_itera1['e'] = e_itera1
dic_itera2['e'] = e_itera2
dic_itera1['w'] = w_itera1
dic_itera2['w'] = w_itera2
dic_itera1['f'] = f_itera1
dic_itera2['f'] = f_itera2
dic_itera1['x'] = x_itera1
dic_itera2['x'] = x_itera2

# Obtain Gaussian Quadrature lambdas and weights

if dec_int == 'ti':
    if ti_points != 0:
        lambdas = []
        weights = []
        x, y = np.polynomial.legendre.leggauss(ti_points)
        # Adjust Gaussian lambdas
        for i in range(0, len(x)):
            lambdas.append(float((x[i]+1)/2))
        # Adjust Gaussian weights
        for i in range(0, len(y)):
            weights.append(float(y[i]/2))
    else:
        logger.error('Wrong input! Please choose a positive integer for the ti_points variable when using the TI-GQ method')
        sys.exit(1)
    logger.info(f'lambda values: {lambdas}')
    logger.info(f'Gaussian weights: {weights}')
elif dec_int == 'mbar':
    if lambdas == []:
        logger.error('Wrong input! Please choose a set of lambda values when using the MBAR method')
        sys.exit(1)
    if ti_points != 0:
        logger.error('Wrong input! Do not define the ti_points variable when applying the MBAR method, instead choose a set of lambda values')
        sys.exit(1)
    logger.info(f'lambda values: {lambdas}')


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
        logger.info('Creating box...')
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
                        logger.info('Creating box for ligand only...')
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
                        logger.info('Creating box for protein/simultaneous release...')
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
                        logger.info('Creating box for ligand decoupling in bulk...')
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
                            logger.info('Creating box for attaching restraints...')
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
    logger.info('cp ../../run_files/run-express.bash')
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
            fin = open('../../run_files/local-equil-op.bash', "rt")
            data = fin.read()
            data = data.replace('RANGE', '%02d' % rng)
            fin.close()
            fin = open('run-local.bash', "wt")
            fin.write(data)
            fin.close()
            fin = open('../../run_files/PBS-Op', "rt")
            data = fin.read()
            data = data.replace('STAGE', stage).replace('POSE', pose)
            fin.close()
            fin = open('PBS-run', "wt")
            fin.write(data)
            fin.close()
            fin = open('../../run_files/SLURMM-Op', "rt")
            data = fin.read()
            data = data.replace('STAGE', stage).replace('POSE', pose)
            fin.close()
            fin = open('SLURMM-run', "wt")
            fin.write(data)
            fin.close()
            for j in range(0, len(release_eq)):
                fin = open('../../lib/equil.py', "rt")
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
                shutil.copy('../../../../run_files/local-rest-op.bash', './run-local.bash')
                fin = open('../../../../run_files/PBS-Op', "rt")
                data = fin.read()
                data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                fin.close()
                fin = open('PBS-run', "wt")
                fin.write(data)
                fin.close()
                fin = open('../../../../run_files/SLURMM-Op', "rt")
                data = fin.read()
                data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                fin.close()
                fin = open('SLURMM-run', "wt")
                fin.write(data)
                fin.close()
                fin = open('../../../../lib/rest.py', "rt")
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
                        shutil.copy('../../../../run_files/local-sdr-op.bash', './run-local.bash')
                        fin = open('../../../../run_files/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('PBS-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../run_files/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('SLURMM-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../lib/sdr.py', "rt")
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
                            shutil.copy('../../../../../run_files/local-sdr-op-ti.bash', './run-local.bash')
                            fin = open('../../../../../run_files/SLURMM-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open('../../../../../run_files/PBS-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open('../../../../../lib/equil-sdr.py', "rt")
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
                            fin = open('../../../../../lib/sdr-ti.py', "rt")
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
                            shutil.copytree('../../../../run_files', '../run_files')
                        shutil.copy('../../../../run_files/local-dd-op.bash', './run-local.bash')
                        fin = open('../../../../run_files/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('PBS-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../run_files/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                        fin.close()
                        fin = open('SLURMM-run', "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../lib/dd.py', "rt")
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
                            shutil.copy('../../../../../run_files/local-dd-op-ti.bash', './run-local.bash')
                            fin = open('../../../../../run_files/SLURMM-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open('../../../../../run_files/PBS-Op', "rt")
                            data = fin.read()
                            data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open('../../../../../lib/equil-dd.py', "rt")
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
                            fin = open('../../../../../lib/dd-ti.py', "rt")
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
                    shutil.copy('../../../../run_files/local-sdr-op.bash', './run-local.bash')
                    fin = open('../../../../run_files/PBS-Op', "rt")
                    data = fin.read()
                    data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                    fin.close()
                    fin = open('PBS-run', "wt")
                    fin.write(data)
                    fin.close()
                    fin = open('../../../../run_files/SLURMM-Op', "rt")
                    data = fin.read()
                    data = data.replace('POSE', comp).replace('STAGE', poses_def[i])
                    fin.close()
                    fin = open('SLURMM-run', "wt")
                    fin.write(data)
                    fin.close()
                    fin = open('../../../../lib/sdr.py', "rt")
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
                        shutil.copy('../../../../../run_files/local-sdr-op-ti.bash', './run-local.bash')
                        fin = open('../../../../../run_files/SLURMM-Op', "rt")
                        data = fin.read()
                        data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                        fin.close()
                        fin = open("SLURMM-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../../run_files/PBS-Op', "rt")
                        data = fin.read()
                        data = data.replace('STAGE', poses_def[i]).replace('POSE', '%s%02d' % (comp, int(k)))
                        fin.close()
                        fin = open("PBS-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open('../../../../../lib/equil-sdr.py', "rt")
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
                        fin = open('../../../../../lib/sdr-ti.py', "rt")
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
