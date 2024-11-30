#!/bin/bash

# Define constants for filenames
PRMTOP="full.hmr.prmtop"
INPCRD="full.inpcrd"
log_file="run.log"

check_sim_failure() {
    local stage=$1
    
    if grep -q "Terminated Abnormally" "$log_file"; then
        echo "$stage Simulation failed."
        exit 1
    elif grep -q "command not found" "$log_file"; then
        echo "$stage Simulation failed; simulaiton command not found."
        exit 1
    else
        echo "$stage complete."
    fi
}

# Minimization
pmemd -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "Minimization"

# Heating steps
pmemd.cuda -O -i therm1.in -p $PRMTOP -c mini.rst7 -o therm1.out -r therm1.rst7 -x therm1.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "Heating 1"
pmemd.cuda -O -i therm2.in -p $PRMTOP -c therm1.rst7 -o therm2.out -r therm2.rst7 -x therm2.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "Heating 2"

# Equilibration with protein restrained
pmemd.cuda -O -i eqnpt0.in -p $PRMTOP -c therm2.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x traj_pre.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "Pre Equilibration"

# Equilibration with COM restrained
pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "Equilibration stage 0"
for step in {1..4}; do
    prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
    curr=$(printf "eqnpt%02d" $step)
    pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $INPCRD > "$log_file" 2>&1
    check_sim_failure "Equilibration stage $step"
done

# Initial MD production run
pmemd.cuda -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref $INPCRD > "$log_file" 2>&1
check_sim_failure "MD stage 0"

# Additional MD production runs
for i in {1..2}; do
    prev=$(printf "md%02d.rst7" $((i - 1)))
    curr=$(printf "md%02d" $i)

    pmemd.cuda -O -i mdin-${curr: -2} -p $PRMTOP -c $prev -o md-${curr: -2}.out -r $curr.rst7 -x md${curr: -2}.nc -ref $INPCRD > "$log_file" 2>&1

    check_sim_failure "MD stage $i"
done