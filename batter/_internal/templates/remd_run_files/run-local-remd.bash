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

PRMTOP="full_merged.prmtop"
N_WINDOWS=NWINDOWS
PFOLDER="."
PFOLDER_ABS=$(cd "${PFOLDER}" 2>/dev/null && pwd -P)
REMD=1
overwrite=${OVERWRITE:-0}
COMP=${COMP:-$(basename "$PWD")}
log_file="${PFOLDER}/run.log"
retry=${RETRY_COUNT:-${RETRY:-}}

if [[ ! -f ./check_run.bash ]]; then
    echo "[ERROR] Missing check_run.bash in ${PFOLDER}; cannot continue."
    exit 1
fi
source ./check_run.bash

if ! declare -F production_is_complete >/dev/null 2>&1; then
production_is_complete() {
    local current_ps=$1
    local total_ps=$2
    local dt_ps=${3:-0}

    awk -v cur="$current_ps" -v tot="$total_ps" -v dt="$dt_ps" '
        BEGIN {
            tol = dt * 0.5
            if (tol < 1e-6) {
                tol = 1e-6
            }
            exit !((cur + tol) >= tot)
        }
    '
}
fi

# Write a REMD mdin current file:
# - keep nstlim fixed from mdin-remd-template
# - update numexchg based on remaining steps
# - always continue from restart coordinates and velocities
cap_dumpfreq_for_remd_chunk() {
    local nstlim_value=$1
    local dumpfreq_value

    [[ $nstlim_value =~ ^[0-9]+$ && $nstlim_value -gt 0 ]] || { cat; return; }
    dumpfreq_value=$nstlim_value

    # Keep the center-of-mass print interval inside each exchange block.
    awk -v freq="$dumpfreq_value" '
        BEGIN { IGNORECASE = 1 }
        {
            line = $0
            if (line ~ /DUMPFREQ/ && match(line, /istep1[[:space:]]*=[[:space:]]*[0-9]+/)) {
                token = substr(line, RSTART, RLENGTH)
                value = token
                sub(/.*=/, "", value)
                gsub(/[[:space:]]/, "", value)
                if (value + 0 > freq + 0) {
                    line = substr(line, 1, RSTART - 1) "istep1=" int(freq) substr(line, RSTART + RLENGTH)
                }
            }
            print line
        }
    '
}

write_mdin_remd_current() {
    local tmpl=$1
    local nstlim_value=$2
    local numexchg_value=$3
    local dumpave_file=${5:-}
    if [[ ! -f $tmpl ]]; then
        echo "[ERROR] Missing template $tmpl" >&2
        return 1
    fi
    local text
    text=$(<"$tmpl")
    text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "irest" "1")
    text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "ntx" "5")
    text=$(printf "%s\n" "$text" | cap_dumpfreq_for_remd_chunk "$nstlim_value")
    if echo "$text" | grep -Eq "^[[:space:]]*numexchg[[:space:]]*="; then
        text=$(echo "$text" | sed -E "s/^[[:space:]]*numexchg[[:space:]]*=.*/  numexchg = ${numexchg_value},/")
    else
        text=$(echo "$text" | awk -v val="$numexchg_value" '
            BEGIN { in_cntrl=0; inserted=0 }
            {
                line=$0
                if (tolower(line) ~ /^&cntrl/) { in_cntrl=1 }
                if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/ && inserted==0) {
                    print "  numexchg = " val ","
                    inserted=1
                }
                print line
                if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/) { in_cntrl=0 }
            }
        ')
    fi
    if [[ -n $dumpave_file ]]; then
        text=$(echo "$text" | awk -v dumpave="$dumpave_file" '
            BEGIN{IGNORECASE=1}
            /^[[:space:]]*DUMPAVE[[:space:]]*=/ {
                print "DUMPAVE=" dumpave
                next
            }
            { print }
        ')
    fi
    echo "$text"
}

# Select the newest valid numbered restart within one window.
select_window_restart_name() {
    local win_dir=$1
    local start_ps=$2
    local retry_count=${3:-}

    (
        cd "$win_dir" || exit 1
        select_valid_md_restart "eq.rst7" "$start_ps" "$retry_count" >&2
        printf "%s\n" "$SELECTED_MD_RESTART"
    )
}

# Echo commands before executing them so the full invocation is visible
print_and_run() {
    echo "$@"
    eval "$@"
}

reduce_dt_for_remd_windows() {
    local stage=$1
    local retry_count=${2:-${RETRY_COUNT:-${RETRY:-0}}}
    local dec=${3:-0.001}

    if [[ $retry_count -lt 3 ]]; then
        return 0
    fi

    local i win tmpl
    for ((i = 0; i < N_WINDOWS; i++)); do
        win=$(printf "%s%02d" "${COMP}" "$i")
        tmpl="${PFOLDER}/${win}/mdin-remd-template"
        [[ -f "$tmpl" ]] || continue
        reduce_dt_on_failure "$tmpl" "$dec" "${stage} (${win})" "$retry_count"
    done
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

reset_attempt_failed_archive_marker
archive_existing_log_file "$log_file"

# Determine progress from the first window
WIN0=$(printf "%s%02d" "${COMP}" 0)
tmpl0="${PFOLDER}/${WIN0}/mdin-remd-template"
if [[ ! -f "$tmpl0" ]]; then
    echo "[ERROR] Missing mdin-remd-template in ${WIN0}; cannot continue."
    exit 1
fi

for ((i = 0; i < N_WINDOWS; i++)); do
    win=$(printf "%s%02d" "${COMP}" "$i")
    apply_retry_dt_reduction "${PFOLDER}/${win}/mdin-remd-template" "$retry" 0.001 "REMD startup"
done

total_steps=$(parse_total_steps "$tmpl0")
dt_ps=$(parse_dt_ps "$tmpl0")
target_dt_ps=$(parse_target_dt_ps "$tmpl0")
chunk_steps=$(parse_nstlim "$tmpl0")
total_ps=$(awk -v s="$total_steps" -v dt="$target_dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

production_start_marker="${PFOLDER}/${WIN0}/production-start.ps"
start_ps=$(production_start_ps "$production_start_marker" "${PFOLDER}/${WIN0}/eq.rst7")
restart_name=$(select_window_restart_name "${PFOLDER}/${WIN0}" "$start_ps" "$retry") || exit 1
if [[ ! -s "${PFOLDER}/${WIN0}/${restart_name}" ]]; then
    echo "[ERROR] Missing restart file ${WIN0}/${restart_name}; cannot continue."
    exit 1
fi
restart_ps=$(completed_time_ps_from_rst "${PFOLDER}/${WIN0}/${restart_name}")
[[ -z $restart_ps ]] && restart_ps=0
current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
[[ -z $current_ps ]] && current_ps=0

last_idx=$(latest_md_index "${PFOLDER}/${WIN0}/md-*.out")
[[ $last_idx -lt 0 ]] && last_idx=0
restart_seg_idx=0
if parsed_restart_seg_idx=$(md_segment_index_from_restart "$restart_name" 2>/dev/null); then
    restart_seg_idx=$parsed_restart_seg_idx
fi
(( restart_seg_idx > last_idx )) && last_idx=$restart_seg_idx

echo "Current completed production time: ${current_ps} ps / ${total_ps} ps (restart=${restart_ps} ps, start=${start_ps} ps, dt=${dt_ps} ps)"

remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')
remaining_steps=$(remaining_steps_from_time "$total_ps" "$current_ps" "$dt_ps")
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
    first_run=$([[ $restart_name == "eq.rst7" ]] && echo 1 || echo 0)
    out_tag=$(printf "md-%02d" "$seg_idx")
    rst_out="${out_tag}.rst7"

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
        cmass_file=$(printf "cmass-%02d.txt" "$seg_idx")
        dumpave_file="${win}/${cmass_file}"
        write_mdin_remd_current "$tmpl" "$chunk_steps" "$run_exchg" "$first_run" "$dumpave_file" > "$current_mdin"

        window_start_ps=$(production_start_ps "${PFOLDER}/${win}/production-start.ps" "${PFOLDER}/${win}/eq.rst7")
        rst_in=$(select_window_restart_name "${PFOLDER}/${win}" "$window_start_ps" "$retry") || exit 1
        if [[ ! -s "${win}/${rst_in}" ]]; then
            echo "[ERROR] Missing restart file ${win}/${rst_in}; cannot continue."
            exit 1
        fi
        window_restart_ps=$(completed_time_ps_from_rst "${win}/${rst_in}")
        if ! awk -v got="$window_restart_ps" -v expected="$restart_ps" -v dt="$dt_ps" '
            BEGIN {
                delta = got - expected
                if (delta < 0) delta = -delta
                tol = dt * 0.5
                if (tol < 1e-6) tol = 1e-6
                exit !(delta <= tol)
            }
        '; then
            echo "[ERROR] Restart time mismatch for ${win}/${rst_in}: ${window_restart_ps} ps; expected ${restart_ps} ps."
            exit 1
        fi

        echo "-O -i ${win}/mdin-remd-current -p ${win_00}/${PRMTOP} -c ${win}/${rst_in} -o ${win}/${out_tag}.out -r ${win}/${rst_out} -x ${win}/${out_tag}.nc -ref ${win_00}/eq.rst7 -inf ${win}/mdinfo -l ${win}/${out_tag}.log -e ${win}/${out_tag}.mden" >> "$groupfile"
    done

    # keep a compat copy for older tooling
    cp -f "$groupfile" "${PFOLDER}/remd/mdin.in.remd.current" >/dev/null 2>&1 || true

    REMD_FLAG="-rem 3 -remlog ${PFOLDER}/rem_${seg_idx}.log"
    print_and_run "$MPI_LAUNCH ${PMEMD_MPI_EXEC} -ng ${N_WINDOWS} ${REMD_FLAG} -groupfile ${groupfile} >> \"$log_file\" 2>&1"
    rc=$?
    echo "[INFO] pmemd step rc=$rc dir=${PFOLDER_ABS} at $(date)" | tee -a "$log_file"
    if (( rc != 0 )); then
        echo "[ERROR] pmemd failed in ${PFOLDER_ABS}; skipping post-step" | tee -a "$log_file"
        archive_failed_md_segment "$COMP" "$seg_idx" "$N_WINDOWS" "$PFOLDER" "$retry"
        reduce_dt_for_remd_windows "REMD segment ${seg_idx}" "$retry"
        exit $rc
    fi

    missing_restart=0
    for ((i = 0; i < N_WINDOWS; i++)); do
        win=$(printf "%s%02d" "${COMP}" "$i")
        if [[ ! -s "${PFOLDER}/${win}/${rst_out}" ]]; then
            echo "[ERROR] Missing or empty restart after REMD segment ${seg_idx}: ${win}/${rst_out}" | tee -a "$log_file"
            missing_restart=1
        fi
    done
    if (( missing_restart )); then
        archive_failed_md_segment "$COMP" "$seg_idx" "$N_WINDOWS" "$PFOLDER" "$retry"
        reduce_dt_for_remd_windows "REMD segment ${seg_idx}" "$retry"
        exit 1
    fi
    restart_ps=$(completed_time_ps_from_rst "${PFOLDER}/${WIN0}/${rst_out}")
    [[ -z $restart_ps ]] && restart_ps=0
    current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
    [[ -z $current_ps ]] && current_ps=0
else
    current_ps="$total_ps"
fi

if production_is_complete "$current_ps" "$total_ps" "$dt_ps"; then
    echo "FINISHED" > ${PFOLDER}/FINISHED
    echo "[INFO] REMD complete; writing per-window FINISHED markers."
    for ((i = 0; i < N_WINDOWS; i++)); do
        win=$(printf "%s%02d" "${COMP}" "$i")
        echo "FINISHED" > "${PFOLDER}/${win}/FINISHED"
        echo "[INFO] ${win}: FINISHED"
    done
    exit 0
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
