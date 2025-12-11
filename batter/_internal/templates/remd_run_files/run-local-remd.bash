#!/bin/bash

# # AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_MPI_EXEC=${PMEMD_MPI_EXEC:-pmemd.cuda.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}
MPI_FLAGS=${MPI_FLAGS:-}

PRMTOP="full.hmr.prmtop"
N_WINDOWS=NWINDOWS
PFOLDER="."
REMD=1
overwrite=${OVERWRITE:-0}
COMP=${COMP:-$(basename "$PWD")}
log_file="${PFOLDER}/run.log"
retry=${RETRY_COUNT:-0}

# Echo commands before executing them so the full invocation is visible
print_and_run() {
    echo "$@"
    eval "$@"
}

# Build an MPI launch prefix that works for mpirun or srun.
if [[ -z "${MPI_FLAGS}" ]]; then
    mpi_base=$(echo "${MPI_EXEC}" | awk '{print $1}')
    mpi_base=${mpi_base##*/}
    if [[ "${mpi_base}" == srun* ]]; then
        MPI_FLAGS="-n ${N_WINDOWS}"
    else
        MPI_FLAGS="-np ${N_WINDOWS} --oversubscribe"
    fi
fi
MPI_LAUNCH="${MPI_EXEC} ${MPI_FLAGS}"

if [[ -f ${PFOLDER}/FINISHED ]]; then
    echo "REMD is complete."
    exit 0
fi

if [[ -f ${PFOLDER}/FAILED ]]; then
    rm -f ${PFOLDER}/FAILED
fi

if [[ -s ${PFOLDER}/${COMP}00/mdin-00.rst7 ]]; then
    echo "Skipping md00 steps."
else
    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_0.log"
    print_and_run "$MPI_LAUNCH ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${PFOLDER}/remd/mdin.in.remd.groupfile >> \"$log_file\" 2>&1"
fi

if [[ -s ${PFOLDER}/${COMP}00/mdin-00.rst7 ]]; then
    echo "FINISHED" > ${PFOLDER}/FINISHED
    exit 0
fi
