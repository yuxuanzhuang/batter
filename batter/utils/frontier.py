import click
from loguru import logger
import pickle
import pandas as pd
import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
import numpy as np
import json
# from batter.utils.utils import run_with_log, antechamber, tleap, cpptraj
# from batter.batter import System
from batter.data import frontier_files


def generate_frontier_files(version=24, dec_method='sdr'):
    """
    Generate the frontier files for the system
    to run them in a bundle.

    # Example simulations 
    """
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
            'therm1.in', 'therm2.in',
            'eqnpt0.in',
            'eqnpt.in_00',
            'eqnpt.in_01', 'eqnpt.in_02',
            'eqnpt.in_03', 'eqnpt.in_04',
            'mdin.in', 'mdin.in.extend'
        ],
        'sdr': [
            'mini.in',
            'heat.in_00',
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
                                    line = f"cv_file = '{sim_folder_name}/cv.in'\n"
                                if 'output_file' in line:
                                    line = f"output_file = '{sim_folder_name}/cmass.txt'\n"
                                if 'disang' in line:
                                    line = f"DISANG={sim_folder_name}/disang.rest\n"
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
                                        line = '  ntp = 0,\n'
                                    if 'csurften' in line:
                                        line = '\n'
                                outfile.write(line)

                        f.write(f'# {component} {i} {stage}\n')
                        if stage == 'mdin.in':
                            f.write(f'-O -i {sim_folder_name}/mdin.in_frontier -p {sim_folder_name}/full.hmr.prmtop -c {sim_folder_name}/eqnpt.in_04.rst7 '
                                    f'-o {sim_folder_name}/mdin-00.out -r {sim_folder_name}/mdin-00.rst7 -x {sim_folder_name}/mdin-00.nc '
                                    f'-ref {sim_folder_name}/full.inpcrd -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-00.log '
                                    f'-e {sim_folder_name}/mdin-00.mden\n')
                        elif stage == 'mdin.in.extend':
                            f.write(f'-O -i {sim_folder_name}/mdin.in_frontier -p {sim_folder_name}/full.hmr.prmtop -c {sim_folder_name}/mdin-CURRNUM.rst7 '
                                    f'-o {sim_folder_name}/mdin-NEXTNUM.out -r {sim_folder_name}/mdin-NEXTNUM.rst7 -x {sim_folder_name}/mdin-NEXTNUM.nc '
                                    f'-ref {sim_folder_name}/full.inpcrd -inf {sim_folder_name}/mdinfo -l {sim_folder_name}/mdin-NEXTNUM.log '
                                    f'-e {sim_folder_name}/mdin-NEXTNUM.mden\n')
                        else:
                            f.write(
                                f'-O -i {sim_folder_name}/{stage.split("_")[0]}_frontier -p {prmtop} -c {stage_previous.replace("REPXXX", f"{i:02d}")} '
                                f'-o {sim_folder_name}/{stage}.out -r {sim_folder_name}/{stage}.rst7 -x {sim_folder_name}/{stage}.nc '
                                f'-ref {inpcrd} -inf {sim_folder_name}/{stage}.mdinfo -l {sim_folder_name}/{stage}.log '
                                f'-e {sim_folder_name}/{stage}.mden\n'
                            )
                        if stage == 'mdin.in':
                            all_replicates[component].append(f'{sim_folder_name}')
                    stage_previous = f'{sim_folder_temp}REPXXX/{stage}.rst7'
        logger.debug(f'all_replicates: {all_replicates}')
        return all_replicates

    def write_sbatch_file(pose, components):
        for component in components:
            folder = os.getcwd()
            folder = '_'.join(folder.split(os.sep)[-4:])
            # write the sbatch file for equilibration
            file_temp = f'{frontier_files}/fep_run.sbatch'
            lines = open(file_temp).readlines()
            lines.append(f'\n\n\n')
            lines.append(f'# {pose} {component}\n')

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
                else:
                    lines.append(
                        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_SPFP.MPI -ng {n_sims} -groupfile {g_name}\n'
                    )
            lines = [line
                     .replace('NUM_NODES', str(n_nodes))
                     .replace('FEP_SIM_XXX', f'{folder}_{component}_{pose}') for line in lines]
            with open(sbatch_file, 'w') as f:
                f.writelines(lines)
             

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
                '# Get the latest mdin-xx.rst7 file in the current directory\n',
                'latest_file=$(ls pose0/sdr/e00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)\n\n',
                '# Check if any mdin-xx.rst7 files exist\n',
                'if [[ -z "$latest_file" ]]; then\n',
                '  echo "No old production files found in the current directory."\n',
                '  echo "Run sbatch fep_md.sbatch."\n',
                '  exit 1\n',
                'fi\n\n',
                '# Extract the latest number (xx) and calculate the next number\n',
                'latest_num=$(echo "$latest_file" | grep -oP \'(?<=-)\d{2}(?=\\.rst7)\')\n',
                'next_num=$(printf "%02d" $((10#$latest_num + 1)))\n\n',
                '# Replace REPNUM in the groupfile with the current number\n',
                '# sed "s/CURRNUM/$latest_num/g" mdin_extend.groupfile > temp_mdin.groupfile\n',
                '# Replace NEXTNUM in the groupfile with the next number\n',
                '# sed "s/NEXTNUM/$next_num/g" temp_mdin.groupfile > current_mdin.groupfile\n\n',
                '# Run the production simulation\n',
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
                    f'srun -N {n_nodes} -n {n_sims} pmemd.hip_SPFP.MPI -ng {n_sims} -groupfile {groupfile_name_prod} &\n',
                    f'sleep 0.5\n'
                        ]
                    )
                with open(sbatch_extend_file, 'a') as f:
                    f.writelines(
                        [ 
                        f'sed "s/CURRNUM/$latest_num/g" {pose}/groupfiles/{comp}_mdin.in.extend.groupfile > {pose}/groupfiles/{comp}_temp_mdin.groupfile\n',
                        f'sed "s/NEXTNUM/$next_num/g" {pose}/groupfiles/{comp}_temp_mdin.groupfile > {pose}/groupfiles/{comp}_current_mdin.groupfile\n',
                        f'# {pose} {comp}\n',
                        f'srun -N {n_nodes} -n {n_sims} pmemd.hip_SPFP.MPI -ng {n_sims} -groupfile {pose}/groupfiles/{comp}_current_mdin.groupfile &\n',
                        f'sleep 0.5\n\n'
                        ]
                    )
        # append wait
        with open(sbatch_file, 'a') as f:
            f.write('wait\n')

        with open(sbatch_extend_file, 'a') as f:
            f.write('wait\n')

    all_replicates = []

    for pose in poses_def:
        all_replicates.append(write_2_pose(pose, components))
        write_sbatch_file(pose, components)
        logger.debug(f'Generated groupfiles for {pose}')
    logger.debug(all_replicates)
    write_production_sbatch(all_replicates)
    # copy env.amber.24
    env_amber_file = f'{frontier_files}/env.amber.{version}'
    shutil.copy(env_amber_file, 'fe/env.amber')
    logger.info('Generated groupfiles for all poses')

