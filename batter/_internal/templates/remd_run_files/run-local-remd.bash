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

if [[ ! -f ./check_run.bash ]]; then
    echo "[ERROR] Missing check_run.bash in ${PFOLDER}; cannot continue."
    exit 1
fi
source ./check_run.bash

# Write a REMD mdin current file:
# - keep nstlim fixed to the REMD interval
# - update numexchg based on remaining steps
# - set irest/ntx according to first_run
write_mdin_remd_current() {
    local tmpl=$1
    local nstlim_value=$2
    local numexchg_value=$3
    local first_run=$4
    if [[ ! -f $tmpl ]]; then
        echo "[ERROR] Missing template $tmpl" >&2
        return 1
    fi
    local text
    text=$(<"$tmpl")
    if [[ $first_run -eq 1 ]]; then
        text=$(echo "$text" | sed -E 's/^[[:space:]]*irest[[:space:]]*=.*/  irest = 0,/' | sed -E 's/^[[:space:]]*ntx[[:space:]]*=.*/  ntx   = 1,/')
    else
        text=$(echo "$text" | sed -E 's/^[[:space:]]*irest[[:space:]]*=.*/  irest = 1,/' | sed -E 's/^[[:space:]]*ntx[[:space:]]*=.*/  ntx   = 5,/')
    fi
    text=$(echo "$text" | sed -E "s/^[[:space:]]*nstlim[[:space:]]*=.*/  nstlim = ${nstlim_value},/")
    if echo "$text" | grep -Eq "^[[:space:]]*numexchg[[:space:]]*="; then
        text=$(echo "$text" | sed -E "s/^[[:space:]]*numexchg[[:space:]]*=.*/  numexchg = ${numexchg_value},/")
    else
        text=$(echo "$text" | awk -v val="$numexchg_value" '
            BEGIN { in_cntrl=0; inserted=0 }
            { line=$0 }
            tolower(line) ~ /^&cntrl/ { in_cntrl=1 }
            if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/ && inserted==0) {
                print "  numexchg = " val ","
                inserted=1
            }
            print line
            if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/) { in_cntrl=0 }
        ')
    fi
    echo "$text"
}

# Determine completed time (ps) from restart and latest md-*.out index for window 0.
remd_progress() {
    local win0=$1
    local pattern=$2
    local idx tps prev_tps
    idx=$(highest_out_index_for_pattern "$pattern")
    tps=$(completed_time_ps_from_rst "${win0}/md-current.rst7")
    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        prev_tps=$(completed_time_ps_from_rst "${win0}/md-previous.rst7")
        if [[ -n $prev_tps && $prev_tps != 0 && $prev_tps != 0.0 ]]; then
            tps="$prev_tps"
        else
            tps=0
        fi
    fi
    if [[ $idx -lt 0 ]]; then
        echo "$tps -1"
        return
    fi
    echo "$tps $idx"
}

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
dt_ps=$(parse_dt_ps "$tmpl0")
total_ps=$(awk -v s="$total_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

read current_ps last_idx < <(remd_progress "${PFOLDER}/${WIN0}" "${PFOLDER}/${WIN0}/md-*.out")
[[ -z $current_ps ]] && current_ps=0

echo "Current completed time (from restart): ${current_ps} ps / ${total_ps} ps (dt=${dt_ps} ps)"

current_steps=$(awk -v t="$current_ps" -v dt="$dt_ps" 'BEGIN{if (dt<=0) {print 0; exit} printf "%d\n", (t/dt)+0.5}')
remaining_steps=$(( total_steps - current_steps ))
if (( remaining_steps < 0 )); then
    remaining_steps=0
fi
remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')
if awk -v tot="$total_ps" -v rem="$remaining_ps" 'BEGIN{exit !(tot>=100 && rem<=100)}'; then
    remaining_steps=0
    current_ps="$total_ps"
fi

if (( remaining_steps > 0 )); then
    run_steps=$remaining_steps
    run_ps=$(awk -v s="$run_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

    # numexchg controls total steps for REMD (steps = nstlim * numexchg)
    run_exchg=$(( (run_steps + chunk_steps - 1) / chunk_steps ))
    (( run_exchg > 0 )) || { echo "[ERROR] Computed run_exchg=0"; exit 1; }

    seg_idx=$((last_idx + 1))
    first_run=$([[ $last_idx -lt 0 ]] && echo 1 || echo 0)

    # Build per-window mdin and groupfile for this segment
    groupfile="${PFOLDER}/remd/mdin.in.remd.groupfile"
    : > "$groupfile"
    win_00=$(printf "%s%02d" "${COMP}" 0)
    for ((i = 0; i < N_WINDOWS; i++)); do
        win=$(printf "%s%02d" "${COMP}" "$i")
        tmpl="${PFOLDER}/${win}/mdin-remd-template"
        [[ -f "$tmpl" ]] || {
            echo "[ERROR] Missing template $tmpl" >&2
            exit 1
        }
        current_mdin="${PFOLDER}/${win}/mdin-remd-current"
        write_mdin_remd_current "$tmpl" "$chunk_steps" "$run_exchg" "$first_run" > "$current_mdin"

        # Determine restart input per window (prefer rolling restarts, else eq.rst7)
        rst_in="eq.rst7"
        if [[ -s "${win}/md-current.rst7" ]]; then
            mv -f "${win}/md-current.rst7" "${win}/md-previous.rst7"
            rst_in="md-previous.rst7"
        elif [[ -s "${win}/md-previous.rst7" ]]; then
            rst_in="md-previous.rst7"
        fi
        if [[ ! -s "${win}/${rst_in}" ]]; then
            echo "[ERROR] Missing restart file ${win}/${rst_in}; cannot continue."
            exit 1
        fi

        out_tag=$(printf "md-%02d" "$seg_idx")
        echo "-O -i ${win}/mdin-remd-current -p ${win_00}/${PRMTOP} -c ${win}/${rst_in} -o ${win}/${out_tag}.out -r ${win}/md-current.rst7 -x ${win}/${out_tag}.nc -ref ${win_00}/eq.rst7 -inf ${win}/mdinfo -l ${win}/${out_tag}.log -e ${win}/${out_tag}.mden" >> "$groupfile"
    done

    # keep a compat copy for older tooling
    cp -f "$groupfile" "${PFOLDER}/remd/mdin.in.remd.current" >/dev/null 2>&1 || true

    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${seg_idx}.log"
    print_and_run "$MPI_LAUNCH ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${groupfile} >> \"$log_file\" 2>&1"
    rc=$?
    echo "[INFO] pmemd step rc=$rc dir=${PFOLDER} at $(date)" | tee -a "$log_file"
    if (( rc != 0 )); then
        echo "[ERROR] pmemd failed in ${PFOLDER}; skipping post-step" | tee -a "$log_file"
        cleanup_failed_md_segment "$COMP" "$seg_idx" "$N_WINDOWS" "$PFOLDER"
        exit $rc
    fi
else
    current_ps="$total_ps"
fi

# if we reach here, REMD step completed successfully
echo "FINISHED" > ${PFOLDER}/FINISHED
echo "[INFO] REMD complete; writing per-window FINISHED markers."
for ((i = 0; i < N_WINDOWS; i++)); do
    win=$(printf "%s%02d" "${COMP}" "$i")
    echo "FINISHED" > "${PFOLDER}/${win}/FINISHED"
    echo "[INFO] ${win}: FINISHED"
done
exit 0
fi
