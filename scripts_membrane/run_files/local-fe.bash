#!/bin/bash

# Define constants for filenames
PRMTOP="full.hmr.prmtop"
INPCRD="full.inpcrd"

# Minimization
pmemd.cuda -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD
echo "Minimization complete."

# Heating steps
pmemd.cuda -O -i therm1.in -p $PRMTOP -c mini.rst7 -o therm1.out -r therm1.rst7 -x therm1.nc -ref $INPCRD
pmemd.cuda -O -i therm2.in -p $PRMTOP -c therm1.rst7 -o therm2.out -r therm2.rst7 -x therm2.nc -ref $INPCRD
echo "Heating complete."

# Equilibration with protein restrained
pmemd.cuda -O -i eqnpt0.in -p $PRMTOP -c therm2.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x traj_pre.nc -ref $INPCRD

# Equilibration with COM restrained
pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref $INPCRD
for step in {1..4}; do
    prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
    curr=$(printf "eqnpt%02d" $step)
    pmemd.cuda -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $INPCRD
done
echo "Pre-equilibration complete."

# Initial MD production run
pmemd.cuda -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref $INPCRD

# Additional MD production runs
for i in {1..2}; do
    prev=$(printf "md%02d.rst7" $((i - 1)))
    curr=$(printf "md%02d" $i)

    echo "Running: pmemd.cuda -O -i mdin-$curr -p $PRMTOP -c $prev -o md-$curr.out -r $curr.rst7 -x md$curr.nc -ref $INPCRD"
    pmemd.cuda -O -i mdin-${curr: -2} -p $PRMTOP -c $prev -o md-${curr: -2}.out -r $curr.rst7 -x md${curr: -2}.nc -ref $INPCRD

    echo "Completed # $i production stage out of 2 stages"
done