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
    else
        echo "$stage complete at $(date)"
    fi
}

if [[ $overwrite -eq 0 && -f eqnpt_pre.rst7 ]]; then
    echo "Skipping EM steps." 
else
    # Minimization
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        mpirun --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE pmemd.MPI -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD -nt $SLURM_JOB_CPUS_PER_NODE > "$log_file" 2>&1
    else
        pmemd -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD > "$log_file" 2>&1
    fi
    check_sim_failure "Minimization"
fi

if [[ $overwrite -eq 0 && -f md00.rst7 ]]; then
    echo "Skipping equilibration steps."
else
    # Heating steps
    # pmemd.cuda -O -i therm1.in -p $PRMTOP -c mini.rst7 -o therm1.out -r therm1.rst7 -x therm1.nc -ref $INPCRD > "$log_file" 2>&1
    # check_sim_failure "Heating 1"
    # pmemd.cuda -O -i therm2.in -p $PRMTOP -c therm1.rst7 -o therm2.out -r therm2.rst7 -x therm2.nc -ref therm1.rst7 > "$log_file" 2>&1
    # check_sim_failure "Heating 2"

    # Equilibration with protein restrained
    # pmemd.cuda -O -i eqnpt0.in -p $PRMTOP -c therm2.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x traj_pre.nc -ref therm2.rst7 > "$log_file" 2>&1
    pmemd.cuda -O -i eqnpt0.in -p $PRMTOP -c mini.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x traj_pre.nc -ref mini.rst7 > "$log_file" 2>&1
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
if [[ $only_eq -eq 1 ]]; then
    echo "Only equilibration requested."
    exit 0
fi

if [[ $overwrite -eq 0 && -f md01.rst7 ]]; then
    echo "Skipping md00 steps."
else
# Initial MD run
pmemd.cuda -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref eqnpt04.rst7 > $log_file 2>&1
check_sim_failure "MD stage 0"
fi

i=1
while [ $i -le RANGE ]; do
    j=$((i - 1))
    k=$((i + 1))
    x=$(printf "%02d" $i)
    y=$(printf "%02d" $j)
    z=$(printf "%02d" $k)
    # x is the current step, y is the previous step, z is the next step
    if [[ $overwrite -eq 0 && -f md$z.rst7 ]]; then
        echo "Skipping md$x steps."
    else
        pmemd.cuda -O -i mdin-$x -p $PRMTOP -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref eqnpt04.rst7 > $log_file 2>&1
        check_sim_failure "MD stage $i"
    fi
    i=$((i + 1))
done

cpptraj -p $PRMTOP -y md$x.rst7 -x output.pdb > "$log_file" 2>&1