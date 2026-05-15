move_failed_file_if_present() {
    local src=$1
    local archive_dir=$2

    [[ -n $src && -e $src ]] || return 1
    mv -f "$src" "$archive_dir/"
}

archive_existing_log_file() {
    local log_path=${1:-}
    local log_dir log_name archive_dir timestamp archived_path suffix

    [[ -n $log_path ]] || return 0
    [[ -e "$log_path" ]] || return 0

    log_dir=$(dirname "$log_path")
    log_name=$(basename "$log_path")
    archive_dir="${log_dir}/ARCHIVED_LOGS"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    archived_path="${archive_dir}/${timestamp}_${log_name}"

    mkdir -p "$archive_dir"

    if [[ -e "$archived_path" ]]; then
        suffix=1
        while [[ -e "${archived_path}.${suffix}" ]]; do
            suffix=$((suffix + 1))
        done
        archived_path="${archived_path}.${suffix}"
    fi

    mv -f "$log_path" "$archived_path"
    echo "[INFO] Archived existing log file to ${archived_path}"
}

archive_failed_job_files() {
    local retry_count=${1:-${RETRY_COUNT:-${RETRY:-0}}}
    shift || true

    local timestamp archive_dir moved_any=0 f stem
    timestamp=$(date +"%Y%m%d_%H%M%S")
    archive_dir="WRONG_FAIL/${timestamp}_job_attempt_${retry_count}"
    mkdir -p "$archive_dir"

    for f in "$@"; do
        if move_failed_file_if_present "$f" "$archive_dir"; then
            moved_any=1
        fi

        if [[ -n $f && $f == *.rst7 ]]; then
            stem=${f%.rst7}
            if move_failed_file_if_present "${stem}.out" "$archive_dir"; then
                moved_any=1
            fi
            if move_failed_file_if_present "${stem}.nc" "$archive_dir"; then
                moved_any=1
            fi
            if move_failed_file_if_present "${stem}.log" "$archive_dir"; then
                moved_any=1
            fi
            if move_failed_file_if_present "${stem}.mden" "$archive_dir"; then
                moved_any=1
            fi
            if move_failed_file_if_present "${stem}.mdinfo" "$archive_dir"; then
                moved_any=1
            fi
        fi
    done

    if move_failed_file_if_present "mdinfo" "$archive_dir"; then
        moved_any=1
    fi

    if (( moved_any )); then
        echo "[INFO] Archived failed job files to ${archive_dir}"
    else
        rmdir "$archive_dir" 2>/dev/null || true
        rmdir WRONG_FAIL 2>/dev/null || true
    fi
}

ATTEMPT_FAILED_MARKER=${ATTEMPT_FAILED_MARKER:-ATTEMPT_FAILED}

write_attempt_failed_marker() {
    printf "FAILED\n" > "$ATTEMPT_FAILED_MARKER"
}

consume_prior_failure_marker() {
    local prior_failed=0

    if [[ -f "$ATTEMPT_FAILED_MARKER" ]]; then
        prior_failed=1
        rm -f "$ATTEMPT_FAILED_MARKER"
    fi

    echo "$prior_failed"
}

mark_failed_and_exit() {
    local message=${1:-}
    if [[ -n $message ]]; then
        echo "$message"
    fi
    write_attempt_failed_marker
    exit 1
}

require_nonempty_file_or_attempt_fail() {
    local required_path=$1
    local message=${2:-"[ERROR] Missing required file ${required_path}; aborting."}

    if [[ -n $required_path && -s $required_path ]]; then
        return 0
    fi

    mark_failed_and_exit "$message"
}

remove_empty_file_if_present() {
    local path=$1

    [[ -n $path && -e $path && ! -s $path ]] || return 1

    rm -f "$path"
    echo "[INFO] Removed stale empty file $path"
}

md_out_has_amber_control_data() {
    local path=$1
    [[ -s "$path" ]] || return 1
    grep -Eq 'CONTROL[[:space:]]+DATA[[:space:]]+FOR[[:space:]]+THE[[:space:]]+RUN' "$path"
}

archive_incomplete_md_out_if_present() {
    local path=$1
    local retry_count=${2:-}

    [[ -n $path && -s $path ]] || return 1
    md_out_has_amber_control_data "$path" && return 1

    local stem
    stem=${path%.out}
    retry_count=$(retry_count_for_template "mdin-template" "$retry_count")
    archive_failed_job_files "$retry_count" \
        "$path" \
        "${stem}.nc" \
        "${stem}.log" \
        "${stem}.mden" \
        "${stem}.mdinfo"
    echo "[INFO] Archived incomplete MD output $path before restart."
    return 0
}

cleanup_stale_empty_md_artifacts() {
    local pattern f
    local patterns=(
        "md-*.out"
        "md*.out"
        "md-*.nc"
        "md*.nc"
        "md-*.log"
        "md*.log"
        "md-*.mden"
        "md*.mden"
        "md-*.mdinfo"
        "md*.mdinfo"
        "md-current.rst7"
        "md-previous.rst7"
        "cmass.txt"
    )

    if [[ -n ${ZSH_VERSION-} ]]; then
        setopt local_options null_glob
        for pattern in "${patterns[@]}"; do
            for f in ${~pattern}; do
                remove_empty_file_if_present "$f" || true
            done
        done
        if [[ ! -s md-current.rst7 && ! -s md-previous.rst7 ]]; then
            for f in md-*.out md*.out; do
                archive_incomplete_md_out_if_present "$f" || true
            done
        fi
        return 0
    fi

    local nullglob_was_on=0
    shopt -q nullglob && nullglob_was_on=1
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for f in $pattern; do
            remove_empty_file_if_present "$f" || true
        done
    done
    if [[ ! -s md-current.rst7 && ! -s md-previous.rst7 ]]; then
        for f in md-*.out md*.out; do
            [[ -e "$f" ]] || continue
            archive_incomplete_md_out_if_present "$f" || true
        done
    fi
    if [[ $nullglob_was_on -eq 0 ]]; then
        shopt -u nullglob
    fi
}

should_skip_completed_step() {
    local stage=$1
    local artifact=$2
    local overwrite=${3:-0}
    local prior_failed=${4:-0}
    local rerun_after_failure=${5:-0}

    if [[ $overwrite -ne 0 ]]; then
        return 1
    fi

    if [[ -z $artifact || ! -s $artifact ]]; then
        return 1
    fi

    if [[ $prior_failed -eq 1 && $rerun_after_failure -eq 1 ]]; then
        echo "[INFO] Prior failure marker found; rerunning ${stage} despite existing artifact ${artifact}."
        return 1
    fi

    echo "[INFO] Skipping ${stage}; found existing artifact ${artifact}."
    return 0
}

check_sim_failure() {
    local stage=$1
    local log_file=$2
    local rst_file=$3
    local rst_file_prev=${4:-}
    local retry_count=${5:-${RETRY_COUNT:-${RETRY:-}}}
    local extra_files=()
    if (( $# > 5 )); then
        extra_files=("${@:6}")
    fi
    retry_count=$(retry_count_for_template "mdin-template" "$retry_count")

    cleanup_outputs() {
        if ((${#extra_files[@]})); then
            archive_failed_job_files "$retry_count" "$log_file" "$rst_file" "${extra_files[@]}"
        else
            archive_failed_job_files "$retry_count" "$log_file" "$rst_file"
        fi
    }

    # If log doesn't exist yet, don't treat as failure here
    [[ -f "$log_file" ]] || return 0

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|segmentation fault|MPI_ABORT|FATAL|cudaGetDeviceCount|Calculation halted" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        tail -n 200 "$log_file" || true
        cleanup_outputs
        if [[ $retry_count -ge 3 ]]; then
            reduce_dt_on_failure "mdin-template" 0.001 "$stage" "$retry_count"
        fi

        # if not the first retry attempt, remove the previous restart file to avoid repeated failure
        if [[ -n "$rst_file_prev" && $retry_count -gt 0 ]]; then
            echo "[INFO] Removing previous restart file $rst_file_prev before retrying."
            rm -f "$rst_file_prev"
        fi
        write_attempt_failed_marker
        exit 1
    fi

    if [[ -n "$rst_file" && (! -f "$rst_file" || ! -s "$rst_file") ]]; then
        echo "[ERROR] $stage simulation failed. Restart file missing or empty: $rst_file"
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_on_failure "mdin-template" 0.001 "$stage" "$retry_count" 1
        fi
        if [[ -n "$rst_file_prev" && $retry_count -gt 0 ]]; then
            echo "[INFO] Removing previous restart file $rst_file_prev before retrying."
            rm -f "$rst_file_prev"
        fi
        write_attempt_failed_marker
        exit 1
    fi

    echo "[INFO] $stage completed successfully at $(date)"
}

check_min_energy() {
    local energy_file="$1"
    local threshold="$2"

    local energy_value source_label

    # 1) Try last EAMBER in the file (most direct)
    energy_value=$(awk '
        $1=="EAMBER" && $2=="=" { v=$3 }
        END { if (v!="") print v }
    ' "$energy_file")
    source_label="EAMBER"

    # 2) Fallback: last ENERGY from the NSTEP table (take the second column of the data line)
    if [[ -z "$energy_value" ]]; then
        energy_value=$(awk '
            /^[[:space:]]*NSTEP[[:space:]]+ENERGY[[:space:]]+RMS[[:space:]]+GMAX/ { in_tbl=1; next }
            in_tbl && NF>=2 {
                # data lines usually start with an integer step
                if ($1 ~ /^[0-9]+$/) v=$2
            }
            END { if (v!="") print v }
        ' "$energy_file")
        source_label="ENERGY"
    fi

    if [[ -z "$energy_value" ]]; then
        echo "Error: Could not find EAMBER or ENERGY in $energy_file"
        return 2
    fi

    # 3) Overflow detection (only look for stars in energy fields)
    # - ENERGY column in NSTEP table: second field becomes ********
    # - EAMBER line: value after '=' becomes ********
    if tail -n 600 "$energy_file" | awk '
        # NSTEP table overflow
        /^[[:space:]]*[0-9]+[[:space:]]+\*{6,}/ { exit 0 }
        # EAMBER overflow
        $1=="EAMBER" && $2=="=" && $3 ~ /\*{6,}/ { exit 0 }
        END { exit 1 }
    '; then
        echo "Error: Overflow detected in ENERGY/EAMBER field in $energy_file"
        return 1
    fi

    # 4) Validate numeric
    if ! [[ "$energy_value" =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "Error: Energy value '$energy_value' is not a valid number"
        return 1
    fi

    # 5) Catch absurd energies (blow-up heuristic)
    if awk -v val="$energy_value" 'BEGIN { exit (val < -1.0e8 || val > 1.0e8) ? 0 : 1 }'; then
        echo "Error: Energy magnitude too large: $energy_value"
        return 1
    fi

    printf "%s: %.4f kcal/mol (threshold: %s)\n" "$source_label" "$energy_value" "$threshold"

    if awk -v val="$energy_value" -v thr="$threshold" 'BEGIN { exit (val < thr ? 0 : 1) }'; then
        echo "Energy is below threshold."
        return 0
    else
        echo "Energy is above threshold."
        return 1
    fi
}

highest_out_index_for_pattern() {
    local pattern=$1
    local require_nonempty=${2:-0}
    local max=-1
    local f n

    if [[ -n ${ZSH_VERSION-} ]]; then
        setopt local_options null_glob
        for f in ${~pattern}; do
            [[ -e "$f" ]] || continue
            if [[ $require_nonempty -eq 1 && ! -s "$f" ]]; then
                continue
            fi
            if [[ $f =~ ([0-9]+)\.out$ ]]; then
                n=${match[1]}
                n=$((10#$n))
                (( n > max )) && max=$n
            fi
        done
    else
        for f in $pattern; do
            [[ -e "$f" ]] || continue
            if [[ $require_nonempty -eq 1 && ! -s "$f" ]]; then
                continue
            fi
            if [[ $f =~ ([0-9]+)\.out$ ]]; then
                n=${BASH_REMATCH[1]}
                n=$((10#$n))
                (( n > max )) && max=$n
            fi
        done
    fi

    echo "$max"
}

latest_md_index() {
    local pattern=${1:-"md-*.out"}
    highest_out_index_for_pattern "$pattern" 1
}

cleanup_failed_md_segment() {
    local comp=$1
    local seg_idx=$2
    local n_windows=$3
    local pfolder=${4:-.}

    if [[ -z $comp || -z $seg_idx || -z $n_windows ]]; then
        echo "[WARN] cleanup_failed_md_segment missing args; skip."
        return
    fi

    local out_tag win
    out_tag=$(printf "md-%02d" "$seg_idx")
    for ((i = 0; i < n_windows; i++)); do
        win=$(printf "%s%02d" "$comp" "$i")
        rm -f "${pfolder}/${win}/${out_tag}.out" \
              "${pfolder}/${win}/${out_tag}.nc" \
              "${pfolder}/${win}/${out_tag}.log" \
              "${pfolder}/${win}/${out_tag}.mden" \
              "${pfolder}/${win}/md-current.rst7"
    done
}

# Report stage based ONLY on which OUT files exist.
# - production: md-*.out present
# - equilibration: eqnpt*.out present
# - minimization: mini*.out present
# - not_started: none of the above
report_progress() {
    local stage="not_started"
    local seg=-1
    local tps=0

    seg=$(latest_md_index "md-*.out")
    [[ $seg -lt 0 ]] && seg=$(latest_md_index "md*.out")
    if [[ $seg -ge 0 ]]; then
        stage="production"
        tps=$(completed_steps "mdin-template" 2>/dev/null || echo 0)
    elif ls eqnpt*.out >/dev/null 2>&1; then
        stage="equilibration"
        seg=$(highest_out_index_for_pattern "eqnpt*.out")
        # try to parse TIME(PS) from the latest eqnpt out
        tps=$(completed_time_ps_from_out "$(printf "eqnpt%02d.out" "$seg")" 2>/dev/null || echo 0)
    elif ls mini*.out >/dev/null 2>&1; then
        stage="minimization"
        seg=$(highest_out_index_for_pattern "mini*.out")
        tps=$(completed_time_ps_from_out "$(printf "mini%02d.out" "$seg")" 2>/dev/null || echo 0)
    fi

    echo "[progress] stage=${stage} last_out_index=${seg} time_ps=${tps}"
}

parse_total_steps() {
    local tmpl=${1:-mdin-template}

    [[ -f $tmpl ]] || { echo "[ERROR] Missing template $tmpl" >&2; return 1; }

    local total
    total=$(
        grep -E '^[!#][[:space:]]*total_steps[[:space:]]*=[[:space:]]*[0-9]+' "$tmpl" \
        | tail -1 \
        | sed -E 's/.*total_steps[[:space:]]*=[[:space:]]*([0-9]+).*/\1/'
    )

    [[ -n $total ]] || { echo "[ERROR] total_steps comment not found in $tmpl" >&2; return 1; }
    printf "%s\n" "$total"
}

parse_nstlim() {
    local tmpl=${1:-mdin-template}
    local nst
    nst=$(grep -E "^[[:space:]]*nstlim[[:space:]]*=" "$tmpl" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
    [[ -n $nst ]] || { echo "[ERROR] Could not parse nstlim from $tmpl" >&2; return 1; }
    echo "$nst"
}

# Parse dt (ps) from template; default 0.001 ps if missing/unparsable.
parse_dt_ps() {
    local tmpl=${1:-mdin-template}
    local dt

    [[ -f $tmpl ]] || { echo 0.001; return; }

    dt=$(
        awk '
        BEGIN{IGNORECASE=1}
        {
            # Match dt = 0.004 or dt=0.004 (allow spaces, commas)
            if (match($0, /^[[:space:]]*dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/)) {
                s = substr($0, RSTART, RLENGTH)
                sub(/.*dt[[:space:]]*=[[:space:]]*/, "", s)
                print s
                exit
            }
        }
        ' "$tmpl"
    )

    [[ -n $dt ]] && echo "$dt" || echo 0.001
}

parse_target_dt_ps() {
    local tmpl=${1:-mdin-template}
    local dt

    [[ -f $tmpl ]] || { echo 0.001; return; }

    dt=$(
        grep -E '^[!#][[:space:]]*target_dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?' "$tmpl" \
        | tail -1 \
        | sed -E 's/.*target_dt[[:space:]]*=[[:space:]]*([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?).*/\1/' \
        | tr 'dD' 'eE'
    )

    [[ -n $dt ]] && echo "$dt" || parse_dt_ps "$tmpl"
}

retry_count_for_template() {
    local tmpl=${1:-mdin-template}
    local explicit=${2:-}
    local dir f value

    if [[ $explicit =~ ^[0-9]+$ ]]; then
        echo "$explicit"
        return
    fi

    dir=$(dirname -- "$tmpl")
    local attempt_files=()
    [[ -n ${JOB_ATTEMPT_FILE:-} ]] && attempt_files+=("$JOB_ATTEMPT_FILE")
    attempt_files+=("job_attempt.txt" "${dir}/job_attempt.txt" "${dir}/../job_attempt.txt")

    for f in "${attempt_files[@]}"; do
        [[ -f "$f" ]] || continue
        value=$(tr -d '[:space:]' < "$f")
        if [[ $value =~ ^[0-9]+$ ]]; then
            echo "$value"
            return
        fi
    done

    if [[ ${RETRY_COUNT:-} =~ ^[0-9]+$ ]]; then
        echo "$RETRY_COUNT"
        return
    fi
    if [[ ${RETRY:-} =~ ^[0-9]+$ ]]; then
        echo "$RETRY"
        return
    fi

    echo 0
}

retry_adjusted_dt_ps() {
    local tmpl=${1:-mdin-template}
    local retry_count=${2:-}
    local dec=${3:-0.001}
    local reduction_start=${4:-3}

    [[ -f "$tmpl" ]] || { echo 0.001; return; }
    retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
    [[ $retry_count =~ ^[0-9]+$ ]] || { parse_dt_ps "$tmpl"; return; }

    local current_dt target_dt reductions desired_dt
    current_dt=$(parse_dt_ps "$tmpl")
    if grep -Eq '^[!#][[:space:]]*target_dt[[:space:]]*=' "$tmpl"; then
        target_dt=$(parse_target_dt_ps "$tmpl")
    else
        target_dt="$current_dt"
    fi

    reductions=$((retry_count - reduction_start))
    if [[ $reductions -le 0 ]]; then
        echo "$current_dt"
        return
    fi

    desired_dt=$(awk -v target="$target_dt" -v dec="$dec" -v n="$reductions" 'BEGIN{printf "%.6f\n", target - dec*n}')
    if awk -v nd="$desired_dt" 'BEGIN{exit !(nd>0)}'; then
        echo "$desired_dt"
    else
        echo "$current_dt"
        return
    fi
}

sync_current_mdin_from_template() {
    local tmpl=${1:-mdin-template}
    local current_mdin=${2:-}
    local retry_count=${3:-}

    [[ -n "$current_mdin" && -f "$current_mdin" ]] || return 0

    local nstlim_value tmp
    if [[ $(basename -- "$current_mdin") == "mdin-remd-current" ]]; then
        rewrite_mdin_dt_file "$current_mdin" "$(parse_dt_ps "$tmpl")"
        return 0
    fi

    nstlim_value=$(parse_nstlim "$current_mdin" 2>/dev/null || parse_nstlim "$tmpl" 2>/dev/null) || return 0
    tmp="${current_mdin}.tmp"
    write_mdin_current "$tmpl" "$nstlim_value" 0 "$current_mdin" "$retry_count" > "$tmp" && mv "$tmp" "$current_mdin"
}

ensure_target_dt_marker() {
    local tmpl=${1:-mdin-template}
    local target_dt=${2:-}

    [[ -f "$tmpl" ]] || return 0
    if grep -Eq '^[!#][[:space:]]*target_dt[[:space:]]*=' "$tmpl"; then
        return 0
    fi

    [[ -n $target_dt ]] || target_dt=$(parse_dt_ps "$tmpl")
    printf "! target_dt=%s\n" "$target_dt" > "${tmpl}.tmp"
    cat "$tmpl" >> "${tmpl}.tmp"
    mv "${tmpl}.tmp" "$tmpl"
}

remaining_steps_from_time() {
    local total_ps=$1
    local current_ps=$2
    local dt_ps=$3

    awk -v tot="$total_ps" -v cur="$current_ps" -v dt="$dt_ps" '
        BEGIN {
            rem = tot - cur
            if (dt <= 0 || rem <= 0) {
                print 0
                exit
            }
            n = rem / dt
            whole = int(n)
            if (n - whole > 1e-9) {
                whole += 1
            }
            print whole
        }
    '
}

apply_retry_dt_reduction() {
    local tmpl=${1:-mdin-template}
    local retry_count=${2:-${RETRY_COUNT:-${RETRY:-}}}
    local dec=${3:-0.001}
    local stage=${4:-"retry startup"}

    [[ -f "$tmpl" ]] || return 0
    retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
    [[ $retry_count =~ ^[0-9]+$ ]] || return 0

    local current_dt desired_dt
    current_dt=$(parse_dt_ps "$tmpl")
    if [[ $retry_count -gt 3 ]]; then
        ensure_target_dt_marker "$tmpl" "$current_dt"
    fi
    desired_dt=$(retry_adjusted_dt_ps "$tmpl" "$retry_count" "$dec" 3)

    if ! awk -v nd="$desired_dt" 'BEGIN{exit !(nd>0)}'; then
        echo "[WARN] dt reduction skipped for $tmpl at ${stage} (retry=${retry_count}, dec=${dec})."
        return 0
    fi
    if ! awk -v current="$current_dt" -v desired="$desired_dt" 'BEGIN{diff=current-desired; if (diff<0) diff=-diff; exit !(diff>1e-9)}'; then
        return 0
    fi

    rewrite_mdin_dt_file "$tmpl" "$desired_dt"

    local current_mdin
    if current_mdin=$(current_mdin_for_template "$tmpl"); then
        sync_current_mdin_from_template "$tmpl" "$current_mdin" "$retry_count"
    fi

    echo "[INFO] Applied retry dt in $tmpl for ${stage} (attempt ${retry_count}): ${current_dt} -> ${desired_dt}"
}

rewrite_mdin_dt_file() {
    local target=$1
    local new_dt=$2

    [[ -f "$target" ]] || return 0
    if ! awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*dt[[:space:]]*=/ {found=1; exit} END{exit !found}' "$target"; then
        return 0
    fi

    awk -v newdt="$new_dt" '
        BEGIN{IGNORECASE=1; done=0}
        {
            if (!done && match($0, /^[[:space:]]*dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/)) {
                sub(/dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/, "dt=" newdt)
                done=1
            }
            print
        }
    ' "$target" > "${target}.tmp" && mv "${target}.tmp" "$target"
}

current_mdin_for_template() {
    local tmpl=${1:-mdin-template}
    local dir base

    dir=$(dirname -- "$tmpl")
    base=$(basename -- "$tmpl")

    case "$base" in
        mdin-template|mdin-batch-template)
            echo "${dir}/mdin-current"
            ;;
        mdin-remd-template)
            echo "${dir}/mdin-remd-current"
            ;;
        *)
            return 1
            ;;
    esac
}

reduce_dt_on_failure() {
    local tmpl=${1:-mdin-template}
    local dec=${2:-0.001}
    local stage=${3:-unknown}
    local retry_count=${4:-}
    local reduction_start=${5:-2}

    [[ -f "$tmpl" ]] || { echo "[WARN] $tmpl not found; skip dt reduction."; return; }
    if ! awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*dt[[:space:]]*=/ {found=1; exit} END{exit !found}' "$tmpl"; then
        echo "[WARN] dt not found in $tmpl; skip dt reduction."
        return
    fi

    retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
    [[ $retry_count =~ ^[0-9]+$ ]] || return

    local dt new_dt
    dt=$(parse_dt_ps "$tmpl")
    ensure_target_dt_marker "$tmpl" "$dt"
    new_dt=$(retry_adjusted_dt_ps "$tmpl" "$retry_count" "$dec" "$reduction_start")
    if ! awk -v nd="$new_dt" 'BEGIN{exit !(nd>0)}'; then
        echo "[WARN] dt reduction skipped (current dt=${dt}, dec=${dec})."
        return
    fi

    if awk -v current="$dt" -v desired="$new_dt" 'BEGIN{diff=current-desired; if (diff<0) diff=-diff; exit !(diff>1e-9)}'; then
        rewrite_mdin_dt_file "$tmpl" "$new_dt"
    fi

    local current_mdin
    if current_mdin=$(current_mdin_for_template "$tmpl"); then
        sync_current_mdin_from_template "$tmpl" "$current_mdin" "$retry_count"
    fi

    # remove old sims if there's any.
    rm -f md-*
    echo "[INFO] Reduced dt in $tmpl after ${stage} failure (attempt ${retry_count}): ${dt} -> ${new_dt}"
}

completed_time_ps_from_out() {
    local out_file=$1
    [[ -f $out_file ]] || { echo 0; return; }

    awk '
      BEGIN{IGNORECASE=1}
      match($0, /TIME\(PS\)[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/) {
        s = substr($0, RSTART, RLENGTH)
        sub(/.*=/, "", s)
        gsub(/[[:space:]]/, "", s)
        last = s
      }
      END { if (last != "") printf "%s\n", last; else print 0 }
    ' "$out_file"
}

completed_time_ps_from_rst() {
    local rst_file=$1
    [[ -f $rst_file ]] || { echo 0; return; }
    command -v ncdump >/dev/null 2>&1 || { echo 0; return; }

    ncdump -v time "$rst_file" 2>/dev/null | awk '
      BEGIN{IGNORECASE=1}
      tolower($1) == "time" && $2 == "=" {
        gsub(/;/, "", $3)
        print $3
        exit
      }
    '
}

completed_steps() {
    local tmpl=${1:-mdin-template}
    local tps prev_tps

    tps=$(completed_time_ps_from_rst "md-current.rst7")

    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        prev_tps=$(completed_time_ps_from_rst "md-previous.rst7")
        if [[ -n $prev_tps && $prev_tps != 0 && $prev_tps != 0.0 ]]; then
            tps="$prev_tps"
        else
            echo 0
            return
        fi
    fi

    if [[ -f $tmpl ]]; then
        local ntwr dt dt_ps
        ntwr=$(
            awk '
              BEGIN{IGNORECASE=1}
              {gsub(/!.*/, "", $0)}                           # strip comments
              {
                # find ntwr=...
                if (match($0, /(^|[^a-z0-9_])ntwr[[:space:]]*=[[:space:]]*[-+]?[0-9]+/)) {
                  s=substr($0, RSTART, RLENGTH)
                  sub(/.*ntwr[[:space:]]*=[[:space:]]*/, "", s)
                  print s
                  exit
                }
              }' "$tmpl"
        )
        dt=$(
            awk '
              BEGIN{IGNORECASE=1}
              {gsub(/!.*/, "", $0)}
              {
                # find dt=... (allow decimals and exponent)
                if (match($0, /(^|[^a-z0-9_])dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?/)) {
                  s=substr($0, RSTART, RLENGTH)
                  sub(/.*dt[[:space:]]*=[[:space:]]*/, "", s)
                  gsub(/[dD]/, "e", s)   # Fortran D exponent -> e
                  print s
                  exit
                }
              }' "$tmpl"
        )

        if [[ -n $ntwr && $ntwr -gt 0 && -n $dt ]]; then
            # compute restart interval in ps: ntwr * dt
            dt_ps=$(awk -v dt="$dt" 'BEGIN{printf "%.10f", dt+0.0}')
            local interval_ps
            interval_ps=$(awk -v n="$ntwr" -v dt="$dt_ps" 'BEGIN{printf "%.10f", n*dt}')

            if awk -v x="$interval_ps" 'BEGIN{exit !(x>0)}'; then
                tps=$(awk -v t="$tps" -v step="$interval_ps" '
                    BEGIN{
                        # Snap-to-grid tolerance:
                        # - absolute floor to handle ps-level floating noise
                        # - plus a tiny relative part proportional to step
                        eps = 1e-6
                        rel = step * 1e-12
                        if (rel > eps) eps = rel

                        # "Floor with tolerance": if t is extremely close to next boundary, snap up.
                        k = int((t + eps) / step)
                        out = k * step

                        # Format & trim trailing zeros
                        s = sprintf("%.10f", out)
                        sub(/\.?0+$/, "", s)
                        print s
                    }')
            fi
        fi
    fi

    echo "$tps"
}


write_mdin_current() {
    local tmpl=${1:-mdin-template}
    local nstlim_value=$2
    local first_run=$3
    local current_mdin=${4:-mdin-current}
    local retry_count=${5:-}

    [[ -f $tmpl ]] || { echo "[ERROR] Missing template $tmpl" >&2; return 1; }

    local text
    text=$(<"$tmpl")

    local template_dt effective_dt
    template_dt=$(parse_dt_ps "$tmpl")
    retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
    effective_dt=$(retry_adjusted_dt_ps "$tmpl" "$retry_count" 0.001 3)

    if awk -v eff="$effective_dt" -v template="$template_dt" 'BEGIN{exit !(eff != template)}'; then
        text=$(echo "$text" | awk -v newdt="$effective_dt" '
            BEGIN{IGNORECASE=1; done=0}
            {
                if (!done && match($0, /^[[:space:]]*dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/)) {
                    sub(/dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/, "dt=" newdt)
                    done=1
                }
                print
            }
        ')
    fi

    text=$(echo "$text" \
        | sed -E 's/^[[:space:]]*irest[[:space:]]*=.*/  irest = 1,/' \
        | sed -E 's/^[[:space:]]*ntx[[:space:]]*=.*/  ntx   = 5,/')

    text=$(echo "$text" | sed -E "s/^[[:space:]]*nstlim[[:space:]]*=.*/  nstlim = ${nstlim_value},/")
    echo "$text"
}
