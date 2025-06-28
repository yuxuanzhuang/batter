#!/bin/bash

# Define constants for filenames
# pose
PRMTOP="full.hmr.prmtop"
POSEFOLDER=${POSEFOLDER:-pose0}
CFOLDER=${CFOLDER:-sdr}
COMP=${COMP:-z}
log_file="${POSEFOLDER}/${CFOLDER}/${COMP}_run.log"
INPCRD=${INPCRD:-full.inpcrd}
overwrite=${OVERWRITE:-0}

source check_run.bash

if [[ -f ${POSEFOLDER}/${CFOLDER}/${COMP}_FINISHED ]]; then
    echo "POSEFOLDER is complete."
    exit 0
fi

if [[ -f ${POSEFOLDER}/${CFOLDER}/${COMP}_FAILED ]]; then
    rm ${POSEFOLDER}/${CFOLDER}/${COMP}_FAILED
fi

# Check if initial MD is already done
if [[ -s ${POSEFOLDER}/${CFOLDER}/${COMP}00/mdin-01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    # Initial MD production run
    mpirun -np NWINDOWS --oversubscribe pmemd.cuda.MPI -ng NWINDOWS -rem 3 -groupfile ${POSEFOLDER}/groupfiles/${COMP}_mdin.in.groupfile >> "$log_file" 2>&1
    check_sim_failure "md00" "$log_file"

fi

i=1
while [ $i -le FERANGE ]; do
    j=$((i - 1))
    k=$((i + 1))
    x=$(printf "%02d" $i)
    y=$(printf "%02d" $j)
    z=$(printf "%02d" $k)

    # x is the current step, y is the previous step, z is the next step
    if [[ $overwrite -eq 0 && -s ${POSEFOLDER}/${CFOLDER}/${COMP00}/mdin-${z}.rst7 ]]; then
        echo "Skipping md${x} steps."
    else
        mpirun -np NWINDOWS --oversubscribe pmemd.cuda.MPI -ng NWINDOWS -rem 3 -groupfile ${POSEFOLDER}/groupfiles/${COMP}_mdin.in.stage${$x}.groupfile >> "$log_file" 2>&1
        check_sim_failure "md${x}" "$log_file"
    fi
    i=$((i + 1))
done

if [[ -s ${POSEFOLDER}/${CFOLDER}/${COMP}00/mdin-${$x}.rst7 ]]; then
    echo "FINISHED" > ${POSEFOLDER}/${CFOLDER}/${COMP}_FINISHED
    exit 0
fi