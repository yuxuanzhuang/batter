#!/bin/bash

# Define constants for filenames
PRMTOP="full.hmr.prmtop"
log_file="run.log"
INPCRD=${INPCRD:-full.inpcrd}
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

if [[ -f FAILED ]]; then
    rm FAILED
fi

source run_failures.bash

if [[ $overwrite -eq 0 && -s mini.rst7 ]]; then
    echo "Skipping minimization steps."
else
    # Minimization
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
    else
        pmemd -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
    fi
    check_sim_failure "Minimization" "$log_file"
fi

if [[ $only_eq -eq 1 ]]; then
    if [[ $overwrite -eq 0 && -s md00.rst7 ]]; then
        echo "Skipping equilibration steps."
    else
        # Equilibration with protein and lipid restrained
        # this is to equilibrate the density of water
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 >> "$log_file" 2>&1
        else
            pmemd -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 >> "$log_file" 2>&1
        fi
        check_sim_failure "Pre equilibration" "$log_file"

        # Equilibration with COM restrained
        pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> "$log_file" 2>&1
        check_sim_failure "Equilibration stage 0" "$log_file"
        for step in {1..4}; do
            prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
            curr=$(printf "eqnpt%02d" $step)
            pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> "$log_file" 2>&1
            check_sim_failure "Equilibration stage $step" "$log_file"
        done

        cpptraj -p $PRMTOP -y eqnpt04.rst7 -x eq_output.pdb >> "$log_file" 2>&1
    fi
    echo "Only equilibration requested and finished."
    if [[ -s eq_output.pdb ]]; then
        echo "EQ_FINISHED" > EQ_FINISHED
        echo "Job completed at $(date)"
    fi
    exit 0
fi

if [[ $overwrite -eq 0 && -s md01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    # Initial MD production run
    pmemd.cuda -O -i mdin-00 -p $PRMTOP -c mini.rst7 -o mdin-00.out -r mdin-00.rst7 -x mdin-00.nc -ref mini.rst7 >> "$log_file" 2>&1
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
        pmemd.cuda -O -i mdin-$x -p $PRMTOP -c mdin-$y.rst7 -o mdin-$x.out -r mdin-$x.rst7 -x mdin-$x.nc -ref mini.rst7 >> $log_file 2>&1
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