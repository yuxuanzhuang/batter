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

# Determine completed time (ps) and latest mdin-*.out index for window 0.
remd_progress() {
    local win0=$1
    local pattern=$2
    local idx tps prev_idx prev_out prev_tps mdout
    idx=$(highest_out_index_for_pattern "$pattern")
    if [[ $idx -lt 0 ]]; then
        echo "0 -1"
        return
    fi
    mdout=$(printf "%s/mdin-%02d.out" "$win0" "$idx")
    [[ -f $mdout ]] || mdout=$(printf "%s/mdin-%d.out" "$win0" "$idx")
    if [[ ! -f $mdout ]]; then
        echo "0 -1"
        return
    fi
    tps=$(completed_time_ps_from_out "$mdout")
    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        prev_idx=$((idx - 1))
        if (( prev_idx >= 0 )); then
            prev_out=$(printf "%s/mdin-%02d.out" "$win0" "$prev_idx")
            [[ -f $prev_out ]] || prev_out=$(printf "%s/mdin-%d.out" "$win0" "$prev_idx")
            if [[ -f $prev_out ]]; then
                prev_tps=$(completed_time_ps_from_out "$prev_out")
                if [[ -n $prev_tps && $prev_tps != 0 && $prev_tps != 0.0 ]]; then
                    echo "[WARN] Latest out $mdout has 0 ps; using $prev_out (TIME(PS)=$prev_tps) and removing $mdout" >&2
                    rm -f "$mdout"
                    echo "$prev_tps $prev_idx"
                    return
                fi
            fi
        fi
        echo "0 -1"
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

echo "Current completed time (from OUT): ${current_ps} ps / ${total_ps} ps (dt=${dt_ps} ps)"

if awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur < tot)}'; then
    remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')

    run_ps="$remaining_ps"

    run_steps=$(awk -v ps="$run_ps" -v dt="$dt_ps" 'BEGIN{printf "%d\n", ps/dt}')
    (( run_steps > 0 )) || { echo "[ERROR] Computed run_steps=0 (dt=$dt_ps, run_ps=$run_ps)"; exit 1; }

    # numexchg controls total steps for REMD (steps = nstlim * numexchg)
    run_exchg=$(( (run_steps + chunk_steps - 1) / chunk_steps ))
    (( run_exchg > 0 )) || { echo "[ERROR] Computed run_exchg=0"; exit 1; }

    seg_idx=$((last_idx + 1))
    first_run=$([[ $last_idx -lt 0 ]] && echo 1 || echo 0)

    # Build per-window mdin and groupfile for this segment
    groupfile="${PFOLDER}/remd/mdin.in.remd.groupfile"
    : > "$groupfile"
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
        echo "-O -i ${win}/mdin-remd-current -p ${win}/${PRMTOP} -c ${win}/${rst_in} -o ${win}/${out_tag}.out -r ${win}/md-current.rst7 -x ${win}/${out_tag}.nc -ref ${win}/eq.rst7 -inf ${win}/mdinfo -l ${win}/${out_tag}.log -e ${win}/${out_tag}.mden" >> "$groupfile"
    done

    # keep a compat copy for older tooling
    cp -f "$groupfile" "${PFOLDER}/remd/mdin.in.remd.current" >/dev/null 2>&1 || true

    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${seg_idx}.log"
    print_and_run "$MPI_LAUNCH ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${groupfile} >> \"$log_file\" 2>&1"

    read current_ps last_idx < <(remd_progress "${PFOLDER}/${WIN0}" "${PFOLDER}/${WIN0}/md-*.out")
    [[ -z $current_ps ]] && current_ps=0
fi

if awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur >= tot)}'; then
    echo "FINISHED" > ${PFOLDER}/FINISHED
    exit 0
fi
