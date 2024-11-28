#!/bin/bash

# Constants
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

# Initial MD run
pmemd.cuda -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md00.nc -ref $INPCRD
echo "Initial Equilibration run complete."

i=1
while [ $i -le RANGE ]; do
    j=$((i - 1))
    x=$(printf "%02d" $i)
    y=$(printf "%02d" $j)

    echo "Running: pmemd.cuda -O -i mdin-$x -p $PRMTOP -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref $INPCRD"
    pmemd.cuda -O -i mdin-$x -p $PRMTOP -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref $INPCRD

    echo "Completed # $i equilibration stage out of RANGE stages"
    let i=$i+1
done