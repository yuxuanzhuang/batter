import click
import os
import glob
import subprocess
from batter import MABFESystem
import numpy as np
from loguru import logger
            
def check_stage(pose, comp, n_windows, fe_folder):
    sim_type = 'rest' if comp in ['m', 'n'] else 'sdr'
    # check equilibration of FE has finished
    # mini.rst7
    mini_file = f'{fe_folder}/{pose}/{sim_type}/{comp}-1/mini.out'
    if not os.path.exists(mini_file):
        logger.debug(f'{mini_file} does not exist')
        return 'eq_mini'
    # eqnpt_pre.rst7
    eq_file = f'{fe_folder}/{pose}/{sim_type}/{comp}-1/eqnpt_pre.rst7'
    if not os.path.exists(eq_file):
        logger.debug(f'{eq_file} does not exist')
        return 'eqnpt_pre'
    # eqnpt00.rst7, eqnpt01.rst7, eqnpt02.rst7, eqnpt03.rst7, eqnpt04.rst7
    for eq_stage in range(5):
        eq_file = f'{fe_folder}/{pose}/{sim_type}/{comp}-1/eqnpt{eq_stage:02d}.rst7'
        if not os.path.exists(eq_file):
            logger.debug(f'{eq_file} does not exist')
            return f'eqnpt{eq_stage:02d}'

    for window in range(n_windows):
        folder_2_check = f'{fe_folder}/{pose}/{sim_type}/{comp}{window:02d}'
        if not os.path.exists(folder_2_check):
            logger.debug(f'{folder_2_check} does not exist')
            return 'no_folder'
        # use mini.in.out instead of mini.in.rst7
        min_rst7 = f'{folder_2_check}/mini.in.out'
        if not os.path.exists(min_rst7):
            return 'min'
        mdin_files = glob.glob(f'{folder_2_check}/mdin-*.rst7')
        if not mdin_files:
            return '-1'
        mdin_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        last_mdin = mdin_files[-1]
        size = os.path.getsize(last_mdin)
        if size == 0:
            logger.warning(f'{last_mdin} is empty, use second last')
            if len(mdin_files) == 1:
                return '-1'
            last_mdin = mdin_files[-2]
            if os.path.getsize(last_mdin) == 0:
                raise ValueError(f'Second last {last_mdin} is also empty, panicking...')
        return int(last_mdin.split('-')[-1].split('.')[0])

@click.command(help='Run the simulations in batch in Frontier.')
@click.option('--folders', '-f',
              multiple=True,
              help='Folders of the simulations to gather simulations from.',
              required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--resubmit',
              is_flag=True, help='Whether to submit and resubmit the script until all queried jobs finished.')
@click.option('--remd', is_flag=True, help='Whether run it as remd.')
@click.option('--nrestarts', '-n', default=10, help='Number of extensions to run for each simulation.')
@click.option('--window_json', '-w', default=None,
              help='JSON file with the windows inforation e.g. lambda values and rest force constant to run.',
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option('--overwrite', is_flag=True, help='Whether to overwrite the existing prepared frontier files.')

def run_in_batch(
        folders,
        resubmit,
        remd,
        nrestarts,
        window_json=None,
        overwrite=False,
        ):

    total_num_nodes = 0
    total_num_jobs = 0
    run_lines = []
    sim_to_run = False
    cwd = os.getcwd()
    len_md = nrestarts

    for folder in folders:
        system = MABFESystem(folder)
        if window_json is not None:
            system.load_window_json(window_json)
            overwrite = True
        if not os.path.exists(f'{system.fe_folder}/pose0/groupfiles') or overwrite:
            logger.warning(f'generating run files...')
            system.generate_frontier_files(remd=remd)
        run_lines.append(f'# {folder}')
        run_lines.append(f'cd {system.fe_folder}')
        for pose in system.sim_config.poses_def:
            for comp_ind, comp in enumerate(system.sim_config.components):
                # check the status of the component
                windows = system.component_windows_dict[comp]
                n_windows = len(windows)
                n_nodes = int(np.ceil(n_windows / 8))
                last_rst7 = check_stage(pose, comp, n_windows, fe_folder=system.fe_folder)
                if last_rst7 == 'no_folder':
                    logger.warning(f'{pose} {comp} {last_rst7}')
                    continue
                else:
                    sim_to_run = True
                
                if last_rst7 == 'eq_mini':
                    # only run eq for the first component
                    # as the rest will be run in the same job
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows * 8} pmemd.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_mini.in.groupfile || echo "Error in {pose}/{comp} eq_mini" &'
                    logger.info(f'{pose} {comp} eq_mini')
                    run_lines.append(f'# {pose} {comp} eq_mini')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt_pre':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows * 8} pmemd.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt0.in.groupfile || echo "Error in {pose}/{comp} eqnpt_pre" &'
                    logger.info(f'{pose} {comp} eqnpt_pre')
                    run_lines.append(f'# {pose} {comp} eqnpt_pre')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt00':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt.in_00.groupfile || echo "Error in {pose}/{comp} eqnpt00" &'
                    logger.info(f'{pose} {comp} eqnpt00')
                    run_lines.append(f'# {pose} {comp} eqnpt00')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt01':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt.in_01.groupfile || echo "Error in {pose}/{comp} eqnpt01" &'
                    logger.info(f'{pose} {comp} eqnpt01')
                    run_lines.append(f'# {pose} {comp} eqnpt01')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt02':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt.in_02.groupfile || echo "Error in {pose}/{comp} eqnpt02" &'
                    logger.info(f'{pose} {comp} eqnpt02')
                    run_lines.append(f'# {pose} {comp} eqnpt02')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt03':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt.in_03.groupfile || echo "Error in {pose}/{comp} eqnpt03" &'
                    logger.info(f'{pose} {comp} eqnpt03')
                    run_lines.append(f'# {pose} {comp} eqnpt03')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'eqnpt04':
                    if comp_ind != 0:
                        continue
                    n_windows = len(system.sim_config.components)
                    n_nodes = int(np.ceil(n_windows / 8))
                    run_line = f'srun -N {np.ceil(n_nodes):.0f} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/fe_eq_eqnpt.in_04.groupfile || echo "Error in {pose}/{comp} eqnpt04" &'
                    logger.info(f'{pose} {comp} eqnpt04')
                    run_lines.append(f'# {pose} {comp} eqnpt04')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == 'min':
                    run_line = f'srun -N {n_nodes} -n {n_windows * 8} pmemd.MPI -ng {n_windows} -groupfile {pose}/groupfiles/{comp}_mini.in.groupfile  || echo "Error in {pose}/{comp} min" &'
                    logger.info(f'{pose} {comp} min')
                    run_lines.append(f'# {pose} {comp} min')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                elif last_rst7 == '-1':
                    if remd and comp in ['e', 'v', 'x', 'o', 's']:
                        run_line = f'srun -N {n_nodes} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -rem 3 -remlog {pose}/rem_{comp}_{last_rst7}.log -groupfile {pose}/groupfiles/{comp}_mdin.in.groupfile || echo "Error in {pose}/{comp} md" &'
                    else:
                        run_line = f'srun -N {n_nodes} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/{comp}_mdin.in.groupfile || echo "Error in {pose}/{comp} md" &'
                    logger.info(f'{pose} {comp} md start')
                    run_lines.append(f'# {pose} {comp} md start')
                    run_lines.append(run_line)
                    run_lines.append(f'sleep 0.3\n')
                    run_lines.append(f'\n')
                else:
                    if remd and comp in ['e', 'v', 'x', 'o', 's']:
                        run_line = f'srun -N {n_nodes} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -rem 3 -remlog {pose}/rem_{comp}_{last_rst7}.log -groupfile {pose}/groupfiles/{comp}_current_mdin.groupfile || echo "Error in {pose}/{comp} md" &'
                    else:
                        run_line = f'srun -N {n_nodes} -n {n_windows} pmemd.hip_DPFP.MPI -ng {n_windows} -groupfile {pose}/groupfiles/{comp}_current_mdin.groupfile || echo "Error in {pose}/{comp} md" &'
                    if last_rst7 <= len_md:
                        next_rst7 = last_rst7 + 1
                        logger.info(f'{pose} {comp} md {last_rst7}')
                        run_lines.append(f'# {pose} {comp} md {last_rst7}')
                        run_lines.append(
                                f'latest_file=$(ls {pose}/*/{comp}00/mdin-??.rst7 2>/dev/null | sort | tail -n 1)')
                        # check the size latest_file is not 0
                        # Check if the latest file exists and is not empty
                        run_lines.append(f'if [ ! -s "$latest_file" ]; then')
                        run_lines.append(f'    echo "Last file for {pose}/{comp}: $latest_file is empty"')

                        # Attempt to use the second-to-last file if the latest is empty
                        run_lines.append(
                            f'    latest_file=$(ls {pose}/*/{comp}00/mdin-??.rst7 2>/dev/null | sort | tail -n 2 | head -n 1)')
                        run_lines.append(f'    echo "Using second last file for {pose}/{comp}: $latest_file"')
                        run_lines.append(f'fi')
                        run_lines.append(f'echo "Last file for {pose}/{comp}: $latest_file"')
                        run_lines.append(f'latest_num=$(echo "$latest_file" | grep -oP "(?<=-)[0-9]{{2}}(?=\.rst7)")')
                        run_lines.append(f'next_num=$(printf "%02d" $((10#$latest_num + 1)))')
                        run_lines.append(f'echo "Next number: $next_num"')
                        run_lines.append(f'sed "s/CURRNUM/${{latest_num}}/g" {pose}/groupfiles/{comp}_mdin.in.extend.groupfile > {pose}/groupfiles/{comp}_temp_mdin.groupfile')
                        run_lines.append(f'sed "s/NEXTNUM/${{next_num}}/g" {pose}/groupfiles/{comp}_temp_mdin.groupfile > {pose}/groupfiles/{comp}_current_mdin.groupfile')
                        run_lines.append(run_line)
                        run_lines.append(f'sleep 0.3\n')
                        run_lines.append(f'\n')
                    else:
                        logger.info(f'{pose} {comp} md {last_rst7} finished')
                total_num_nodes += n_nodes
                total_num_jobs += n_windows
        
        run_lines.append(f'cd {cwd}')
        run_lines.append(f'\n')

    total_num_nodes = int(np.ceil(total_num_nodes))
    total_num_jobs = int(total_num_jobs)

    logger.info(f'Total number of nodes: {total_num_nodes}')
    logger.info(f'Total number of jobs: {total_num_jobs}')
    if not sim_to_run:
        logger.info('No jobs to run')
        return

    headers = [
        '#!/usr/bin/env bash',
        '#SBATCH -A BIP152',
        f'#SBATCH -J {cwd}',
        '#SBATCH -o run.out',
        '#SBATCH -t 00:50:00',
        '#SBATCH -p batch',
        f'#SBATCH -N {total_num_nodes}',
        '#SBATCH -S 0',
        '#SBATCH --open-mode=append',
        '#SBATCH --dependency=singleton',
        '#SBATCH --export=ALL',
        'source ~/env.amber > /dev/null 2>&1',
        'echo $AMBERHOME',
        'if [ -z "${AMBERHOME}" ]; then echo "AMBERHOME is not set" && exit 0; fi',
        'export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
        'echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"',
    ]

    with open('run_in_batch.sbatch', 'w') as f:
        f.write('\n'.join(headers))
        f.write('\n\n\n')
        f.write('\n'.join(run_lines))
        f.write('\n')
        if resubmit:
            f.write('sleep 200\n')

            command = f'batter run-in-batch'
            for folder in folders:
                command += f' -f {folder}'
            command += ' --resubmit'
            if remd:
                command += ' --remd'
            f.write(f'{command}\n')
        f.write('wait\n')

    if resubmit:
        result = subprocess.run(['sbatch', 'run_in_batch.sbatch'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)