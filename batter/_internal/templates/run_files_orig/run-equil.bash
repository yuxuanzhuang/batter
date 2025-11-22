#!/bin/bash

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}

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

if [[ $overwrite -eq 0 && -s md00.rst7 ]]; then
    echo "Skipping EM steps." 
else
    $PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
    check_sim_failure "Minimization" "$log_file" mini.rst7
    if ! check_min_energy "mini.out" -1000; then
        echo "Minimization not passed with cuda; trying CPU"
        rm -f "$log_file"
        rm -f mini.rst7 mini.nc mini.out
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            $MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        else
            $PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> "$log_file" 2>&1
        fi
        check_sim_failure "Minimization" "$log_file" mini.rst7
        if ! check_min_energy "mini.out" -1000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out
            exit 1
        fi
    fi

fi
if [[ $overwrite -eq 0 && -s md00.rst7 ]]; then
    echo "Skipping equilibration steps."
else
    python check_penetration.py mini.rst7

    # if RING_PENETRATION file is found
    if [[ -f RING_PENETRATION ]]; then
        echo "Ligand ring penetration detected previously; using longer equilibration."

        # Equilbration with gradually-incrase lambda of the ligand
        # this can fix issues e.g. ligand entaglement https://pubs.acs.org/doi/10.1021/ct501111d
    $PMEMD_DPFP_EXEC -O -i eqnvt.in -p $PRMTOP -c mini.rst7 -o eqnvt.out -r eqnvt.rst7 -x eqnvt.nc -ref $INPCRD >> "$log_file" 2>&1
        check_sim_failure "NVT" "$log_file" eqnvt.rst7
        python check_penetration.py eqnvt.rst7
        if [[ -f RING_PENETRATION ]]; then
            echo "Ligand ring penetration still detected after NVT; exiting."
            exit 1
        fi
    else
        # no NVT needed
        cp mini.rst7 eqnvt.rst7
    fi

    # Equilibration with protein and ligand restrained
    # this is to equilibrate the density of water
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        $MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_CPU_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> "$log_file" 2>&1
    else
        $PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> "$log_file" 2>&1
    fi
    check_sim_failure "Pre equilibration" "$log_file" eqnpt_pre.rst7

    # Equilibration with C-alpha restrained
    $PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> "$log_file" 2>&1
    check_sim_failure "Equilibration stage 0" "$log_file"
    for step in {1..4}; do
        prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
        curr=$(printf "eqnpt%02d" $step)
        $PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> "$log_file" 2>&1
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
$PMEMD_DPFP_EXEC -O -i mdin-00 -p $PRMTOP -c eqnpt04.rst7 -o md-00.out -r md00.rst7 -x md-00.nc -ref eqnpt04.rst7 >> $log_file 2>&1
check_sim_failure "MD stage 0" "$log_file" md00.rst7
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
        $PMEMD_DPFP_EXEC -O -i mdin-$x -p $PRMTOP -c md$y.rst7 -o md-$x.out -r md$x.rst7 -x md-$x.nc -ref eqnpt04.rst7 >> $log_file 2>&1
        check_sim_failure "MD stage $i" "$log_file" md$x.rst7
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
