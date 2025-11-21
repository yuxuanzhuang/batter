#!/bin/bash

# # AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_MPI_EXEC=${PMEMD_MPI_EXEC:-pmemd.cuda.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}

# Define constants for filenames
PRMTOP="full.hmr.prmtop"
log_file="run.log"
INPCRD="full.inpcrd"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

if [[ -f FAILED ]]; then
    rm FAILED
fi

source check_run.bash

if [[ $only_eq -eq 1 ]]; then
    # Minimization
    # if mini_eq is found use mini_eq.in
    if [[ -f mini_eq.in ]]; then
        echo "Using mini_eq.in for minimization."
    else
        echo "mini_eq.in not found, using mini.in instead."
        cp mini.in mini_eq.in
    fi
    $PMEMD_DPFP_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
    check_sim_failure "Minimization" "$log_file"

    if ! check_min_energy "mini.out" -1000; then
        echo "Minimization not passed with cuda; try CPU"
        rm -f "$log_file"
        rm -f mini.rst7 mini.nc mini.out
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            $MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_MPI_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        else
            $PMEMD_CPU_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        fi
        check_sim_failure "Minimization" "$log_file"

        if ! check_min_energy "mini.out" -1000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out
            exit 1
        fi
    fi

    if [[ $overwrite -eq 0 && -s eqnpt04.rst7 ]]; then
        echo "Skipping equilibration steps."
    else
        # Equilibration with protein and lipid restrained
        # this is to equilibrate the density of water
        # Note we are not using the GPU version here
        # because for large box size change, an error will be raised.
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            $MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 >> "$log_file" 2>&1
        else
            $PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 >> "$log_file" 2>&1
        fi
        check_sim_failure "Pre equilibration" "$log_file"

        # Equilibration with protein restrained
        $PMEMD_EXEC -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> "$log_file" 2>&1
        check_sim_failure "Equilibration stage 0" "$log_file"
        for step in {1..4}; do
            prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
            curr=$(printf "eqnpt%02d" $step)
            $PMEMD_EXEC -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> "$log_file" 2>&1
            check_sim_failure "Equilibration stage $step" "$log_file"
        done
    fi

    # run minimization for each windows at this stage
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" $i)
        if [[ -s $win_folder/mini.rst7 ]]; then
            echo "Skipping minimization for window $i, already exists."
        else
            echo "Running minimization for window $i"
            cd $win_folder
    $PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> "$log_file" 2>&1
            check_sim_failure "Minimization for window $i" "$log_file"
            if ! check_min_energy "mini.in.out" -1000; then
                echo "Minimization not passed with cuda; try CPU"
                rm -f "$log_file"
                rm -f mini.in.rst7 mini.in.nc mini.in.out
                if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
                    $MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_MPI_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> "$log_file" 2>&1
                else
                    $PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> "$log_file" 2>&1
                fi
                check_sim_failure "Minimization for window $i" "$log_file"
                if ! check_min_energy "mini.in.out" -1000; then
                    echo "Minimization with CPU also failed for window $i, exiting."
                    rm -f mini.in.rst7 mini.in.nc mini.in.out
                    exit 1
                fi
            fi
            cd ../COMPONENT-1
        fi
    done

    cpptraj -p $PRMTOP -y eqnpt04.rst7 -x eq_output.pdb >> "$log_file" 2>&1

    echo "Only equilibration requested and finished."
    if [[ -s eq_output.pdb ]]; then
        echo "EQ_FINISHED" > EQ_FINISHED
        echo "Job completed at $(date)"
    fi
    exit 0
fi

if [[ $overwrite -eq 0 && -s mdin-01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    # Initial MD production run
    $PMEMD_EXEC -O -i mdin-00 -p $PRMTOP -c mini.in.rst7 -o mdin-00.out -r mdin-00.rst7 -x mdin-00.nc -ref mini.in.rst7 >> "$log_file" 2>&1
    check_sim_failure "MD stage 0" "$log_file"
fi

i=1
while [ $i -le FERANGE ]; do
    j=$((i - 1))
    k=$((i + 1))
    x=$(printf "%02d" $i)
    y=$(printf "%02d" $j)
    z=$(printf "%02d" $k)
    # x is the current step, y is the previous step, z is the next step
    if [[ $overwrite -eq 0 && -s mdin-$z.rst7 ]]; then
        echo "Skipping md$x steps."
    else
        $PMEMD_EXEC -O -i mdin-$x -p $PRMTOP -c mdin-$y.rst7 -o mdin-$x.out -r mdin-$x.rst7 -x mdin-$x.nc -ref mini.in.rst7 >> $log_file 2>&1
        check_sim_failure "MD stage $i" "$log_file"
    fi
    i=$((i + 1))
done

cpptraj -p $PRMTOP -y mdin-$x.rst7 -x output.pdb >> "$log_file" 2>&1

# check output.pdb exists
# to catch cases where the simulation did not run to completion
if [[ -s output.pdb ]]; then
    echo "FINISHED" > FINISHED
    exit 0
fi
