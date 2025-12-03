#!/bin/bash

# # AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_MPI_EXEC=${PMEMD_MPI_EXEC:-pmemd.cuda.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}

PRMTOP="full.hmr.prmtop"
N_WINDOWS=NWINDOWS
FE_RANGE=FERANGE
PFOLDER="."
REMD=1
overwrite=${OVERWRITE:-0}
COMP=${COMP:-$(basename "$PWD")}
log_file="${PFOLDER}/run.log"
retry=${RETRY_COUNT:-0}

if [[ -f ${PFOLDER}/FINISHED ]]; then
    echo "REMD is complete."
    exit 0
fi

if [[ -f ${PFOLDER}/FAILED ]]; then
    rm -f ${PFOLDER}/FAILED
fi

if [[ -s ${PFOLDER}/${COMP}00/mdin-01.rst7 ]]; then
    echo "Skipping md00 steps."
else
    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_0.log"
    $MPI_EXEC -np ${N_WINDOWS} --oversubscribe ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/remd/mdin.in.remd.groupfile >> "$log_file" 2>&1
fi

i=1
while [ $i -le ${FE_RANGE} ]; do
    x=$(printf "%02d" $i)
    z=$(printf "%02d" $((i + 1)))
    if [[ $overwrite -eq 0 && -s ${PFOLDER}/${COMP}00/mdin-${z}.rst7 ]]; then
        echo "Skipping md${x} steps."
    else
        REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${x}.log"
        $MPI_EXEC -np ${N_WINDOWS} --oversubscribe ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/remd/mdin.in.stage${x}.remd.groupfile >> "$log_file" 2>&1
    fi
    i=$((i + 1))
done

final_stage=$(printf "%02d" $FE_RANGE)
if [[ -s ${PFOLDER}/${COMP}00/mdin-${final_stage}.rst7 ]]; then
    echo "FINISHED" > ${PFOLDER}/FINISHED
    exit 0
fi
