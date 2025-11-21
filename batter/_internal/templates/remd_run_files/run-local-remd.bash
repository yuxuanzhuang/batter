#!/bin/bash

PRMTOP="full.hmr.prmtop"
COMP="COMPONENT"
NWINDOWS=NWINDOWS
FERANGE=FERANGE
PFOLDER="."
REMD=1
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_MPI_EXEC=${PMEMD_MPI_EXEC:-pmemd.cuda.MPI}
MPI_EXEC=${MPI_EXEC:-mpirun}
overwrite=${OVERWRITE:-0}
log_file="${PFOLDER}/${COMP}_run.remd.log"

if [[ -f ${PFOLDER}/${COMP}_FINISHED ]]; then
    echo "${COMP} is complete."
    exit 0
fi

if [[ -f ${PFOLDER}/${COMP}_FAILED ]]; then
    rm -f ${PFOLDER}/${COMP}_FAILED
fi

if [[ -s ${PFOLDER}/${COMP}00/mdin-01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${COMP}_0.log"
    $MPI_EXEC -np ${NWINDOWS} --oversubscribe ${PMEMD_MPI_EXEC} -ng ${NWINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/groupfiles/${COMP}_mdin.in.remd.groupfile >> "$log_file" 2>&1
    if [[ -f check_run.bash ]]; then source check_run.bash; check_sim_failure "md00" "$log_file"; fi
fi

i=1
while [ $i -le ${FERANGE} ]; do
    x=$(printf "%02d" $i)
    z=$(printf "%02d" $((i + 1)))
    if [[ $overwrite -eq 0 && -s ${PFOLDER}/${COMP}00/mdin-${z}.rst7 ]]; then
        echo "Skipping md${x} steps."
    else
        REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${COMP}_${x}.log"
        $MPI_EXEC -np ${NWINDOWS} --oversubscribe ${PMEMD_MPI_EXEC} -ng ${NWINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/groupfiles/${COMP}_mdin.in.stage${x}.remd.groupfile >> "$log_file" 2>&1
        if [[ -f check_run.bash ]]; then source check_run.bash; check_sim_failure "md${x}" "$log_file"; fi
    fi
    i=$((i + 1))
done

final_stage=$(printf "%02d" $FERANGE)
if [[ -s ${PFOLDER}/${COMP}00/mdin-${final_stage}.rst7 ]]; then
    echo "FINISHED" > ${PFOLDER}/${COMP}_FINISHED
    exit 0
fi
