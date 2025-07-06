#!/bin/bash

# Constants
PRMTOP="full.hmr.prmtop"
INPCRD="full.inpcrd"
log_file="run.log"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

# remove FAILED file if it exists
if [[ -f FAILED ]]; then
    rm FAILED
fi

source check_run.bash

if [[ $overwrite -eq 0 && -s mini.rst7 ]]; then
    echo "Skipping EM steps." 
else
    pmemd.cuda_DPFP -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
    check_sim_failure "Minimization" "$log_file"
    if ! check_min_energy "mini.out" -10000; then
        echo "Minimization not passed with cuda; trying CPU"
        rm -f "$log_file"
        rm -f mini.rst7 mini.nc mini.out
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        else
            pmemd -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        fi
        check_sim_failure "Minimization" "$log_file"
        if ! check_min_energy "mini.out" -10000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out
            exit 1
        fi
    fi

fi
if [[ $overwrite -eq 0 && -s md00.rst7 ]]; then
    echo "Skipping equilibration steps."
else
    # Equilbration with gradually-incrase lambda of the ligand
    # this can fix issues e.g. ligand entaglement https://pubs.acs.org/doi/10.1021/ct501111d
    pmemd.cuda -O -i eqnvt.in -p $PRMTOP -c mini.rst7 -o eqnvt.out -r eqnvt.rst7 -x eqnvt.nc -ref $INPCRD >> "$log_file" 2>&1
    check_sim_failure "NVT" "$log_file"
    # check ligand entaglement
    python check_penetration.py

    # Equilibration with protein and ligand restrained
    # this is to equilibrate the density of water
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> "$log_file" 2>&1
    else
        pmemd -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> "$log_file" 2>&1
    fi
    check_sim_failure "Pre equilibration" "$log_file"

    # Equilibration with C-alpha restrained
    pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> "$log_file" 2>&1
    check_sim_failure "Equilibration stage 0" "$log_file"
    for step in {1..4}; do
        prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
        curr=$(printf "eqnpt%02d" $step)
        pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> "$log_file" 2>&1
        check_sim_failure "Equilibration stage $step" "$log_file"
    done
fi
if [[ $only_eq -eq 1 ]]; then
    echo "Only equilibration requested."
    exit 0
fi

if [[ $overwrite -eq 0 && -s md01.rst7 ]]; then
    echo "Skipping md00 steps."
else
# Initial MD run
pmemd.cuda -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref eqnpt04.rst7 >> $log_file 2>&1
check_sim_failure "MD stage 0" "$log_file"
fi

i=1
while [ $i -le RANGE ]; do
    j=$((i - 1))
    k=$((i + 1))
    x=$(printf "%02d" $i)
    y=$(printf "%02d" $j)
    z=$(printf "%02d" $k)
    # x is the current step, y is the previous step, z is the next step
    if [[ $overwrite -eq 0 && -s md$z.rst7 ]]; then
        echo "Skipping md$x steps."
    else
        pmemd.cuda -O -i mdin-$x -p $PRMTOP -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref eqnpt04.rst7 >> $log_file 2>&1
        check_sim_failure "MD stage $i" "$log_file"
    fi
    i=$((i + 1))
done

cpptraj -p $PRMTOP -y md$x.rst7 -x output.pdb >> "$log_file" 2>&1

# check output.pdb exists
# to catch cases where the simulation did not run to completion
if [[ -s output.pdb ]]; then
    echo "EQFINISHED" > FINISHED
    exit 0
fi
