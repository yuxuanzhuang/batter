#!/bin/bash

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}

# Define constants for filenames
PRMTOP="full.hmr.prmtop"
log_file="run.log"
INPCRD="full.inpcrd"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}
retry=${RETRY_COUNT:-0}

# Echo commands before executing them so the full invocation is visible
print_and_run() {
    echo "$@"
    eval "$@"
}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

if [[ -f FAILED ]]; then
    rm FAILED
fi

source check_run.bash

if [[ $only_eq -eq 1 ]]; then
    # no eq needed, just copy the INPCRD to mini.in.rst7
    cp $INPCRD mini.rst7
    check_sim_failure "Minimization" "$log_file" mini.rst7

    # run minimization for each windows at this stage
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" $i)
        if [[ -s $win_folder/mini.rst7 ]]; then
            echo "Skipping minimization for window $i, already exists."
        else
            echo "Running minimization for window $i"
            cd $win_folder
            cp ../COMPONENT-1/mini.rst7 mini.in.rst7
            cd ../COMPONENT-1
        fi
    done

    print_and_run "cpptraj -p $PRMTOP -y mini.rst7 -x eq_output.pdb >> \"$log_file\" 2>&1"

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
    print_and_run "$PMEMD_EXEC -O -i mdin-00 -p $PRMTOP -c mini.in.rst7 -o mdin-00.out -r mdin-00.rst7 -x mdin-00.nc -ref mini.in.rst7 -AllowSmallBox >> \"$log_file\" 2>&1"
    check_sim_failure "MD stage 0" "$log_file" mdin-00.rst7
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
        print_and_run "$PMEMD_EXEC -O -i mdin-$x -p $PRMTOP -c mdin-$y.rst7 -o mdin-$x.out -r mdin-$x.rst7 -x mdin-$x.nc -ref mini.in.rst7 -AllowSmallBox >> \"$log_file\" 2>&1"
        check_sim_failure "MD stage $i" "$log_file" mdin-$x.rst7 mdin-$y.rst7 $retry
    fi
    i=$((i + 1))
done

print_and_run "cpptraj -p $PRMTOP -y mdin-$x.rst7 -x output.pdb >> \"$log_file\" 2>&1"

# check output.pdb exists
# to catch cases where the simulation did not run to completion
if [[ -s output.pdb ]]; then
    echo "FINISHED" > FINISHED
    exit 0
fi
