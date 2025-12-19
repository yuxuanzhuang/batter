#!/bin/bash

# # AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_MPI_EXEC=${PMEMD_MPI_EXEC:-pmemd.cuda.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}
MPI_FLAGS=${MPI_FLAGS:-}
CPPTRAJ_EXEC=${CPPTRAJ_EXEC:-cpptraj}

PRMTOP="full.hmr.prmtop"
N_WINDOWS=NWINDOWS
PFOLDER="."
REMD=1
overwrite=${OVERWRITE:-0}
COMP=${COMP:-$(basename "$PWD")}
log_file="${PFOLDER}/run.log"
retry=${RETRY_COUNT:-0}

source check_run.bash

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

# Determine progress from the first window
WIN0=$(printf "%s%02d" "${COMP}" 0)
tmpl0="${PFOLDER}/${WIN0}/mdin-remd-template"
if [[ ! -f "$tmpl0" ]]; then
    echo "[ERROR] Missing mdin-remd-template in ${WIN0}; cannot continue."
    exit 1
fi

total_steps=$(parse_total_steps "$tmpl0")
chunk_steps=$(parse_nstlim "$tmpl0")
last_idx=$(highest_index_for_pattern "${PFOLDER}/${WIN0}/mdin-*.rst7")
if [[ $last_idx -lt 0 ]]; then
    current_steps=0
else
    current_steps=$(( (last_idx + 1) * chunk_steps ))
fi

while [[ $current_steps -lt $total_steps ]]; do
    remaining=$((total_steps - current_steps))
    run_steps=$chunk_steps
    if [[ $remaining -lt $chunk_steps ]]; then
        run_steps=$remaining
    fi
    seg_idx=$((current_steps / chunk_steps))
    first_run=$([[ $current_steps -eq 0 ]] && echo 1 || echo 0)

    # Build per-window mdin and groupfile for this segment
    groupfile="${PFOLDER}/remd/mdin.in.remd.current"
    : > "$groupfile"
    for ((i = 0; i < N_WINDOWS; i++)); do
        win=$(printf "%s%02d" "${COMP}" "$i")
        tmpl="${PFOLDER}/${win}/mdin-remd-template"
        [[ -f "$tmpl" ]] || {
            echo "[ERROR] Missing template $tmpl" >&2
            exit 1
        }
        current_mdin="${PFOLDER}/${win}/mdin-remd-current"
        write_mdin_current "$tmpl" "$run_steps" "$first_run" > "$current_mdin"

        prev_rst="mini.in.rst7"
        if [[ $seg_idx -gt 0 ]]; then
            prev_rst=$(printf "mdin-%02d.rst7" $((seg_idx - 1)))
        fi
        out_tag=$(printf "mdin-%02d" "$seg_idx")
        echo "-O -i ${win}/mdin-remd-current -p ${PRMTOP} -c ${win}/${prev_rst} -o ${win}/${out_tag}.out -r ${win}/${out_tag}.rst7 -x ${win}/${out_tag}.nc -ref ${win}/mini.in.rst7 -inf ${win}/mdinfo -l ${win}/${out_tag}.log -e ${win}/${out_tag}.mden" >> "$groupfile"
    done

    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${seg_idx}.log"
    print_and_run "$MPI_LAUNCH ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${groupfile} >> \"$log_file\" 2>&1"

    current_steps=$((current_steps + run_steps))
done

echo "FINISHED" > ${PFOLDER}/FINISHED
