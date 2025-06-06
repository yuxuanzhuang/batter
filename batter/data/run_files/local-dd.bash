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

check_sim_failure() {
    local stage=$1
    
    if grep -q "Terminated Abnormally" "$log_file"; then
        echo "$stage Simulation failed."
        cat $log_file
        exit 1
    elif grep -q "command not found" "$log_file"; then
        echo "$stage Simulation failed"
        cat $log_file
        exit 1
    elif grep -q "illegal memory" "$log_file"; then
        echo "$stage Simulation failed."
        cat $log_file
        exit 1
    elif grep -q "Error" "$log_file"; then
        echo "$stage Simulation failed."
        cat $log_file
        exit 1
    else
        echo "$stage complete at $(date)"
    fi
}


if [[ $overwrite -eq 0 && -f eqnpt_pre.rst7 ]]; then
    echo "Skipping minimization steps."
else
    # Minimization
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD > "$log_file" 2>&1
    else
        pmemd -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD > "$log_file" 2>&1
    fi
    check_sim_failure "Minimization"
fi

if [[ $only_eq -eq 1 ]]; then
    if [[ $overwrite -eq 0 && -f md00.rst7 ]]; then
        echo "Skipping equilibration steps."
    else
        # Equilibration with protein and lipid restrained
        # this is to equilibrate the density of water
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 > "$log_file" 2>&1
        else
            pmemd -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini.rst7 > "$log_file" 2>&1
        fi
        check_sim_failure "Pre equilibration"

        # Equilibration with COM restrained
        pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 > "$log_file" 2>&1
        check_sim_failure "Equilibration stage 0"
        for step in {1..4}; do
            prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
            curr=$(printf "eqnpt%02d" $step)
            pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev > "$log_file" 2>&1
            check_sim_failure "Equilibration stage $step"
        done
    fi
    echo "Only equilibration requested."
    exit 0
fi

if [[ $overwrite -eq 0 && -f md01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    # Initial MD production run
    pmemd.cuda -O -i mdin-00 -p $PRMTOP -c mini.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref mini.rst7 > "$log_file" 2>&1
    check_sim_failure "MD stage 0"
fi

if [[ $overwrite -eq 0 && -f md02.rst7 ]]; then
    echo "Skipping md01 steps."
else
    pmemd.cuda -O -i mdin-01 -p $PRMTOP -c md00.rst7 -o md-01.out -r md01.rst7 -x md01.nc -ref mini.rst7 > "$log_file" 2>&1
    check_sim_failure "MD stage 1"
fi
if [[ $overwrite -eq 0 && -f md03.rst7 ]]; then
    echo "Skipping md02 steps."
else
    pmemd.cuda -O -i mdin-02 -p $PRMTOP -c md01.rst7 -o md-02.out -r md02.rst7 -x md02.nc -ref mini.rst7 > "$log_file" 2>&1
    check_sim_failure "MD stage 2"
fi
if [[ $overwrite -eq 0 && -f md04.rst7 ]]; then
    echo "Skipping md03 steps."
else
    pmemd.cuda -O -i mdin-03 -p $PRMTOP -c md02.rst7 -o md-03.out -r md03.rst7 -x md03.nc -ref mini.rst7 > "$log_file" 2>&1
    check_sim_failure "MD stage 3"
fi
if [[ $overwrite -eq 0 && -f output.pdb ]]; then
    echo "Skipping md04 steps."
else
    pmemd.cuda -O -i mdin-04 -p $PRMTOP -c md03.rst7 -o md-04.out -r md04.rst7 -x md04.nc -ref mini.rst7 > "$log_file" 2>&1
    check_sim_failure "MD stage 4"
fi
cpptraj -p $PRMTOP -y md04.rst7 -x output.pdb > "$log_file" 2>&1