#!/bin/bash

# Define constants for filenames
# pose
PRMTOP="full.hmr.prmtop"
PFOLDER=${PFOLDER:-pose0}
CFOLDER=${CFOLDER:-sdr}
COMP=${COMP:-z}
NWINDOWS=${NWINDOWS:-24}
log_file="${PFOLDER}/${CFOLDER}/${COMP}_run.log"
INPCRD=${INPCRD:-full.inpcrd}
overwrite=${OVERWRITE:-0}
REMD=${REMD:-0}

source batch_run/check_run.bash

if [[ -f ${PFOLDER}/${CFOLDER}/${COMP}_FINISHED ]]; then
    echo "${PFOLDER} ${COMP} is complete."
    exit 0
fi

if [[ -f ${PFOLDER}/${CFOLDER}/${COMP}_FAILED ]]; then
    rm ${PFOLDER}/${CFOLDER}/${COMP}_FAILED
fi

# Check if initial MD is already done
if [[ -s ${PFOLDER}/${CFOLDER}/${COMP}00/mdin-01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    # if remd add -rem 3
    if [[ $REMD -eq 1 ]]; then
        REMD_FLAG="-rem 3 -remlog ${PFOLDER}/${CFOLDER}/rem_${COMP}_0.log"
    else
        REMD_FLAG=""
    fi
    # Initial MD production run
    mpirun -np ${NWINDOWS} --oversubscribe pmemd.cuda.MPI -ng ${NWINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/groupfiles/${COMP}_mdin.in.groupfile >> "$log_file" 2>&1
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
    if [[ $overwrite -eq 0 && -s ${PFOLDER}/${CFOLDER}/${COMP}00/mdin-${z}.rst7 ]]; then
        echo "Skipping md${x} steps."
    else
        # if remd add -rem 3
        if [[ $REMD -eq 1 ]]; then
            REMD_FLAG="-rem 3 -remlog ${PFOLDER}/${CFOLDER}/rem_${COMP}_${x}.log"
        else
            REMD_FLAG=""
        fi
        mpirun -np ${NWINDOWS} --oversubscribe pmemd.cuda.MPI -ng ${NWINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/groupfiles/${COMP}_mdin.in.stage${x}.groupfile >> "$log_file" 2>&1
        check_sim_failure "md${x}" "$log_file"
    fi
    i=$((i + 1))
done

if [[ -s ${PFOLDER}/${CFOLDER}/${COMP}00/mdin-${x}.rst7 ]]; then
    echo "FINISHED" > ${PFOLDER}/${CFOLDER}/${COMP}_FINISHED
    exit 0
fi