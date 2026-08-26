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
        record_attempt_failed_archive "$archive_dir"
        echo "[INFO] Archived failed job files to ${archive_dir}"
    else
        rmdir "$archive_dir" 2>/dev/null || true
        rmdir WRONG_FAIL 2>/dev/null || true
        reset_attempt_failed_archive_marker
    fi
}

ATTEMPT_FAILED_MARKER=${ATTEMPT_FAILED_MARKER:-ATTEMPT_FAILED}
ATTEMPT_FAILED_ARCHIVE_MARKER=${ATTEMPT_FAILED_ARCHIVE_MARKER:-ATTEMPT_FAILED_ARCHIVE}

reset_attempt_failed_archive_marker() {
    : > "$ATTEMPT_FAILED_ARCHIVE_MARKER"
}

record_attempt_failed_archive() {
    local archive_dir=$1
    [[ -n $archive_dir ]] || return 0
    printf "%s\n" "$archive_dir" > "$ATTEMPT_FAILED_ARCHIVE_MARKER"
}

append_attempt_failed_archive() {
    local archive_dir=$1
    [[ -n $archive_dir ]] || return 0
    printf "%s\n" "$archive_dir" >> "$ATTEMPT_FAILED_ARCHIVE_MARKER"
}

write_attempt_failed_marker() {
    printf "FAILED\n" > "$ATTEMPT_FAILED_MARKER"
}

consume_prior_failure_marker() {
    local prior_failed=0

    reset_attempt_failed_archive_marker

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
    reset_attempt_failed_archive_marker
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

is_amber_restart_path() {
    case "$1" in
        *.inpcrd|*.rst7|*.restrt) return 0 ;;
        *) return 1 ;;
    esac
}

has_netcdf_magic() {
    local restart_path=$1
    local magic

    magic=$(dd if="$restart_path" bs=1 count=4 2>/dev/null | od -An -tx1 | tr -d ' \n')
    case "$magic" in
        43444601|43444602|43444605|89484446*) return 0 ;;
        *) return 1 ;;
    esac
}

netcdf_restart_validation_status() {
    local restart_path=$1
    local size

    if ! command -v ncdump >/dev/null 2>&1; then
        if has_netcdf_magic "$restart_path"; then
            echo "ok"
            return 0
        fi
        echo "ncdump unavailable for binary restart"
        return 0
    fi

    if ! ncdump -h "$restart_path" >/dev/null 2>&1; then
        echo "invalid NetCDF restart header"
        return 0
    fi

    size=$(wc -c < "$restart_path" | tr -d ' ')
    ncdump -h "$restart_path" 2>/dev/null | awk -v file_size="$size" '
        BEGIN {
            atom = 0
            spatial = 0
            has_coords = 0
            coord_bytes = 0
            vel_bytes = 0
            conventions = 0
        }
        /^[[:space:]]*atom[[:space:]]*=/ {
            atom = $3 + 0
        }
        /^[[:space:]]*spatial[[:space:]]*=/ {
            spatial = $3 + 0
        }
        /^[[:space:]]*(float|double)[[:space:]]+coordinates\(atom,[[:space:]]*spatial\)/ {
            has_coords = 1
            coord_bytes = ($1 == "double" ? 8 : 4)
        }
        /^[[:space:]]*(float|double)[[:space:]]+velocities\(atom,[[:space:]]*spatial\)/ {
            vel_bytes = ($1 == "double" ? 8 : 4)
        }
        /:Conventions[[:space:]]*=[[:space:]]*"AMBERRESTART"/ {
            conventions = 1
        }
        END {
            if (atom <= 0) {
                print "NetCDF restart missing atom dimension"
                exit
            }
            if (spatial <= 0) {
                print "NetCDF restart missing spatial dimension"
                exit
            }
            if (!has_coords) {
                print "NetCDF restart missing coordinates variable"
                exit
            }
            if (!conventions) {
                print "NetCDF file is not marked AMBERRESTART"
                exit
            }

            expected = atom * spatial * (coord_bytes + vel_bytes)
            if (file_size + 0 < expected) {
                printf "NetCDF restart shorter than expected payload (%d < %d bytes)", file_size, expected
                exit
            }
            print "ok"
        }
    '
}

amber_restart_validation_status() {
    local restart_path=$1

    if [[ -z $restart_path || ! -s $restart_path ]]; then
        echo "missing or empty"
        return 0
    fi

    if has_netcdf_magic "$restart_path" \
        && command -v ncdump >/dev/null 2>&1 \
        && ncdump -h "$restart_path" >/dev/null 2>&1; then
        netcdf_restart_validation_status "$restart_path"
        return 0
    fi

    if ! LC_ALL=C grep -Iq . "$restart_path"; then
        if has_netcdf_magic "$restart_path"; then
            netcdf_restart_validation_status "$restart_path"
        else
            echo "binary restart is not readable as NetCDF"
        fi
        return 0
    fi

    awk '
        BEGIN {
            count = 0
            status = ""
        }
        NR == 2 {
            natom_raw = $1
            if (natom_raw !~ /^[0-9]+$/ || natom_raw + 0 <= 0) {
                print "invalid atom count in restart header"
                status = "bad"
                exit
            }
            natom = natom_raw + 0
            coord = natom * 3
            coord_box = coord + 6
            vel = natom * 6
            vel_box = vel + 6
            next
        }
        NR > 2 {
            for (i = 1; i <= NF; i++) {
                if ($i !~ /^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([EeDd][-+]?[0-9]+)?$/) {
                    printf "non-numeric restart payload on line %d", NR
                    status = "bad"
                    exit
                }
                count++
            }
        }
        END {
            if (status != "") {
                exit
            }
            if (NR < 2) {
                print "missing restart header"
                exit
            }
            if (count == coord || count == coord_box || count == vel || count == vel_box) {
                print "ok"
                exit
            }
            printf "expected %d/%d coordinate or %d/%d coordinate+velocity fields, found %d", coord, coord_box, vel, vel_box, count
        }
    ' "$restart_path"
}

amber_restart_is_complete() {
    local restart_path=$1
    local status

    status=$(amber_restart_validation_status "$restart_path")
    [[ $status == "ok" ]]
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
    grep -Eq 'CONTROL[[:space:]]+DATA[[:space:]]+FOR[[:space:]]+THE[[:space:]]+RUN|Amber[[:space:]]+[0-9]+[[:space:]]+PMEMD|File Assignments:|Here is the input file:' "$path"
}

md_out_has_completion_marker() {
    local path=$1
    [[ -s "$path" ]] || return 1
    grep -Eq 'Final Performance Info|Total wall time' "$path"
}

output_file_for_restart_artifact() {
    local artifact=$1

    case "$artifact" in
        *.rst7) printf '%s\n' "${artifact%.rst7}.out" ;;
        *.restrt) printf '%s\n' "${artifact%.restrt}.out" ;;
        *) return 1 ;;
    esac
}

restart_artifact_has_incomplete_output() {
    local artifact=$1
    local out_file

    is_amber_restart_path "$artifact" || return 1
    out_file=$(output_file_for_restart_artifact "$artifact" 2>/dev/null) || return 1
    [[ -s "$out_file" ]] || return 1
    md_out_has_amber_control_data "$out_file" || return 1
    ! md_out_has_completion_marker "$out_file"
}

cmass_file_for_md_stem() {
    local stem=$1
    local n

    if [[ $stem =~ ^md-?([0-9]+)$ ]]; then
        n=${BASH_REMATCH[1]}
        printf "cmass-%02d.txt\n" "$((10#$n))"
    fi
}

md_segment_index_from_stem() {
    local stem=$1

    if [[ $stem =~ ^md-?([0-9]+)$ ]]; then
        printf "%d\n" "$((10#${BASH_REMATCH[1]}))"
        return 0
    fi
    return 1
}

md_segment_index_from_stage() {
    local stage=$1

    if [[ $stage =~ ^MD[[:space:]]+segment[[:space:]]+([0-9]+)$ ]]; then
        printf "%d\n" "$((10#${BASH_REMATCH[1]}))"
        return 0
    fi
    return 1
}

should_archive_previous_md_segment_for_failure() {
    local stage=$1
    local retry_count=${2:-0}
    local seg

    seg=$(md_segment_index_from_stage "$stage") || return 1
    (( seg > 1 )) || return 1
    [[ $retry_count =~ ^[0-9]+$ ]] || retry_count=0

    # Initial failure keeps the prior explicit restart (for example md-02.rst7)
    # so the next attempt can rerun the failed segment. If the same later
    # segment fails again, archive the prior segment too so the next restart is
    # one explicit segment farther back.
    (( retry_count >= 2 ))
}

previous_md_segment_files_for_stage() {
    local stage=$1
    local seg prev stem cmass_file

    seg=$(md_segment_index_from_stage "$stage") || return 0
    (( seg > 1 )) || return 0

    prev=$((seg - 1))
    for stem in "$(printf "md-%02d" "$prev")" "$(printf "md%02d" "$prev")"; do
        printf "%s\n" \
            "${stem}.out" \
            "${stem}.nc" \
            "${stem}.log" \
            "${stem}.mden" \
            "${stem}.mdinfo" \
            "${stem}.rst7"
        cmass_file=$(cmass_file_for_md_stem "$stem")
        [[ -n $cmass_file ]] && printf "%s\n" "$cmass_file"
    done
}

md_segment_index_from_restart() {
    local restart_file=$1
    local stem

    stem=${restart_file%.rst7}
    md_segment_index_from_stem "$stem"
}

md_restart_path_for_index() {
    local idx=$1
    printf "md-%02d.rst7\n" "$idx"
}

latest_md_restart_index() {
    local pattern f idx max=-1
    local patterns=("md-*.rst7" "md[0-9]*.rst7")

    local nullglob_was_on=0
    shopt -q nullglob && nullglob_was_on=1
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for f in $pattern; do
            idx=$(md_segment_index_from_restart "$f") || continue
            (( idx > max )) && max=$idx
        done
    done
    if [[ $nullglob_was_on -eq 0 ]]; then
        shopt -u nullglob
    fi
    echo "$max"
}

latest_md_restart_path() {
    local idx path compact

    idx=$(latest_md_restart_index)
    [[ $idx -ge 0 ]] || return 1
    path=$(printf "md-%02d.rst7" "$idx")
    if [[ -e "$path" ]]; then
        printf "%s\n" "$path"
        return 0
    fi
    compact=$(printf "md%02d.rst7" "$idx")
    if [[ -e "$compact" ]]; then
        printf "%s\n" "$compact"
        return 0
    fi
    return 1
}

cleanup_finished_md_restarts() {
    local pattern f
    local patterns=("md-*.rst7" "md[0-9]*.rst7")

    local nullglob_was_on=0
    shopt -q nullglob && nullglob_was_on=1
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for f in $pattern; do
            rm -f "$f"
        done
    done
    if [[ $nullglob_was_on -eq 0 ]]; then
        shopt -u nullglob
    fi
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
        "${stem}.mdinfo" \
        "$(cmass_file_for_md_stem "$stem")"
    echo "[INFO] Archived incomplete MD output $path before restart."
    return 0
}

archive_suspect_md_restart_if_present() {
    local restart_file=$1
    local out_file=$2
    local retry_count=${3:-}

    [[ -n $restart_file && -s "$restart_file" ]] || return 1
    [[ -n $out_file && -s "$out_file" ]] || return 1
    md_out_has_amber_control_data "$out_file" || return 1
    ! md_out_has_completion_marker "$out_file" || return 1

    local stem
    stem=${out_file%.out}
    retry_count=$(retry_count_for_template "mdin-template" "$retry_count")
    archive_failed_job_files "$retry_count" \
        "$out_file" \
        "${stem}.nc" \
        "${stem}.log" \
        "${stem}.mden" \
        "${stem}.mdinfo" \
        "$(cmass_file_for_md_stem "$stem")" \
        "$restart_file"
    echo "[INFO] Archived incomplete MD segment $out_file and suspect restart $restart_file before resume."
    return 0
}

md_nc_frame_count() {
    local nc_file=$1

    [[ -s "$nc_file" ]] || return 1
    command -v ncdump >/dev/null 2>&1 || return 1

    ncdump -h "$nc_file" 2>/dev/null | awk '
      /frame[[:space:]]*=[[:space:]]*UNLIMITED/ {
        s=$0
        sub(/^.*\(/, "", s)
        sub(/[[:space:]]+currently\).*$/, "", s)
        if (s ~ /^[0-9]+$/) {
          print s
          found=1
          exit
        }
      }
      END { if (!found) exit 1 }
    '
}

md_nc_has_zero_frames() {
    local nc_file=$1
    local frames

    frames=$(md_nc_frame_count "$nc_file" 2>/dev/null) || return 1
    [[ $frames =~ ^[0-9]+$ ]] || return 1
    (( frames == 0 ))
}

archive_zero_frame_md_trajectory_if_present() {
    local nc_file=$1
    local retry_count=${2:-}
    local stem out_file stage seg previous_idx f cleanup_retry_count
    local -a files_to_archive=()
    local -a previous_md_files=()

    [[ -n $nc_file && -s "$nc_file" ]] || return 1
    md_nc_has_zero_frames "$nc_file" || return 1

    stem=${nc_file%.nc}
    out_file="${stem}.out"
    if [[ -s "$out_file" ]] && md_out_has_completion_marker "$out_file"; then
        return 1
    fi

    retry_count=$(retry_count_for_template "mdin-template" "$retry_count")
    files_to_archive=(
        "$out_file" \
        "$nc_file" \
        "${stem}.log" \
        "${stem}.mden" \
        "${stem}.mdinfo" \
        "$(cmass_file_for_md_stem "$stem")" \
        "${stem}.rst7"
    )

    if seg=$(md_segment_index_from_stem "$stem"); then
        stage="MD segment ${seg}"
        cleanup_retry_count=$retry_count
        if [[ $cleanup_retry_count =~ ^[0-9]+$ && $cleanup_retry_count -gt 0 ]]; then
            cleanup_retry_count=$((cleanup_retry_count - 1))
        fi
        if should_archive_previous_md_segment_for_failure "$stage" "$cleanup_retry_count"; then
            while IFS= read -r f; do
                [[ -n $f ]] && previous_md_files+=("$f")
            done < <(previous_md_segment_files_for_stage "$stage")
            if (( ${#previous_md_files[@]} > 0 )); then
                previous_idx=$((seg - 1))
                echo "[INFO] Archiving previous MD segment ${previous_idx} because zero-frame ${nc_file} failed again."
                files_to_archive+=("${previous_md_files[@]}")
            fi
        elif (( seg > 1 )); then
            previous_idx=$((seg - 1))
            echo "[INFO] Keeping previous MD segment ${previous_idx} and $(md_restart_path_for_index "$previous_idx") for retry after zero-frame ${nc_file}."
        fi
    fi

    archive_failed_job_files "$retry_count" "${files_to_archive[@]}"
    echo "[INFO] Archived zero-frame MD trajectory $nc_file before restart."
    return 0
}

cleanup_zero_frame_md_trajectories() {
    local retry_count=${1:-}
    local pattern nc_file seen_files=" "
    local patterns=("md-*.nc" "md[0-9]*.nc")

    if [[ -n ${ZSH_VERSION-} ]]; then
        setopt local_options null_glob
        for pattern in "${patterns[@]}"; do
            for nc_file in ${~pattern}; do
                case "$seen_files" in
                    *" $nc_file "*) continue ;;
                esac
                seen_files="${seen_files}${nc_file} "
                archive_zero_frame_md_trajectory_if_present "$nc_file" "$retry_count" || true
            done
        done
        return 0
    fi

    local nullglob_was_on=0
    shopt -q nullglob && nullglob_was_on=1
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for nc_file in $pattern; do
            case "$seen_files" in
                *" $nc_file "*) continue ;;
            esac
            seen_files="${seen_files}${nc_file} "
            archive_zero_frame_md_trajectory_if_present "$nc_file" "$retry_count" || true
        done
    done
    if [[ $nullglob_was_on -eq 0 ]]; then
        shopt -u nullglob
    fi
}

cleanup_suspect_md_resume_state() {
    local retry_count=${1:-}
    local resume_mode=${2:-strict}
    local latest_idx out_file restart_file

    [[ $resume_mode == strict ]] || return 0

    latest_idx=$(latest_md_index "md-*.out")
    if [[ $latest_idx -lt 0 ]]; then
        latest_idx=$(latest_md_index "md*.out")
    fi
    [[ $latest_idx -ge 0 ]] || return 0

    out_file=$(printf "md-%02d.out" "$latest_idx")
    if [[ ! -e "$out_file" ]]; then
        out_file=$(printf "md%02d.out" "$latest_idx")
    fi

    restart_file=$(printf "md-%02d.rst7" "$latest_idx")
    if [[ ! -e "$restart_file" ]]; then
        restart_file=$(printf "md%02d.rst7" "$latest_idx")
    fi
    archive_suspect_md_restart_if_present "$restart_file" "$out_file" "$retry_count" || true
}

cleanup_stale_empty_md_artifacts() {
    local resume_mode=${1:-strict}
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
        "md-*.rst7"
        "md[0-9]*.rst7"
        "cmass.txt"
        "cmass-*.txt"
    )

    if [[ -n ${ZSH_VERSION-} ]]; then
        setopt local_options null_glob
        for pattern in "${patterns[@]}"; do
            for f in ${~pattern}; do
                remove_empty_file_if_present "$f" || true
            done
        done
        if ! latest_md_restart_path >/dev/null 2>&1; then
            for f in md-*.out md*.out; do
                archive_incomplete_md_out_if_present "$f" || true
            done
        fi
        cleanup_suspect_md_resume_state "" "$resume_mode"
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
    if ! latest_md_restart_path >/dev/null 2>&1; then
        for f in md-*.out md*.out; do
            [[ -e "$f" ]] || continue
            archive_incomplete_md_out_if_present "$f" || true
        done
    fi
    if [[ $nullglob_was_on -eq 0 ]]; then
        shopt -u nullglob
    fi
    cleanup_suspect_md_resume_state "" "$resume_mode"
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

    if is_amber_restart_path "$artifact" && ! amber_restart_is_complete "$artifact"; then
        echo "[INFO] Existing artifact ${artifact} is not a complete Amber restart ($(amber_restart_validation_status "$artifact")); rerunning ${stage}."
        return 1
    fi

    if restart_artifact_has_incomplete_output "$artifact"; then
        echo "[INFO] Existing artifact ${artifact} has incomplete Amber output $(output_file_for_restart_artifact "$artifact"); rerunning ${stage}."
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
    local command_status=${SIM_COMMAND_STATUS:-0}
    local -a extra_files=()
    local extra_file_count=0
    local _seen_numeric_failure_files=""
    if (( $# > 5 )); then
        extra_files=("${@:6}")
        extra_file_count=${#extra_files[@]}
    fi
    retry_count=$(retry_count_for_template "mdin-template" "$retry_count")
    SIM_COMMAND_STATUS=0

    cleanup_outputs() {
        local -a files_to_archive=("$log_file" "$rst_file")
        local -a previous_md_files=()
        local f previous_idx

        if (( extra_file_count > 0 )); then
            files_to_archive+=("${extra_files[@]}")
        fi
        if should_archive_previous_md_segment_for_failure "$stage" "$retry_count"; then
            while IFS= read -r f; do
                [[ -n $f ]] && previous_md_files+=("$f")
            done < <(previous_md_segment_files_for_stage "$stage")
            if (( ${#previous_md_files[@]} > 0 )); then
                previous_idx=$(md_segment_index_from_stage "$stage")
                previous_idx=$((previous_idx - 1))
                echo "[INFO] Archiving previous MD segment ${previous_idx} because ${stage} failed again."
                files_to_archive+=("${previous_md_files[@]}")
            fi
        elif previous_idx=$(md_segment_index_from_stage "$stage"); then
            if (( previous_idx > 1 )); then
                previous_idx=$((previous_idx - 1))
                echo "[INFO] Keeping previous MD segment ${previous_idx} and $(md_restart_path_for_index "$previous_idx") for retry after ${stage} failed."
            fi
        fi

        archive_failed_job_files "$retry_count" "${files_to_archive[@]}"
    }

    remove_previous_restart() {
        if [[ -n "$rst_file_prev" && "$rst_file_prev" != "0" ]]; then
            echo "[INFO] Removing previous restart file $rst_file_prev before retrying."
            rm -f "$rst_file_prev"
        fi
    }

    amber_output_has_numeric_failure() {
        local output_file=$1
        [[ -f "$output_file" && -s "$output_file" ]] || return 1

        # Minimization may briefly print overflows while still recovering to a
        # usable restart. If Amber wrote FINAL RESULTS, judge the final state
        # instead of transient early minimization lines.
        if awk '
            /FINAL RESULTS/ {
                final_start = NR
            }
            {
                lines[NR] = $0
            }
            END {
                start = final_start ? final_start : 1
                for (i = start; i <= NR; i++) {
                    line = tolower(lines[i])
                    if (line ~ /(^|[^[:alpha:]])(nan|infinity|inf)([^[:alpha:]]|$)/) {
                        exit 0
                    }
                }
                exit 1
            }
        ' "$output_file"; then
            return 0
        fi

        if awk '
            /FINAL RESULTS/ {
                final_start = NR
            }
            {
                lines[NR] = $0
            }
            END {
                start = final_start ? final_start : 1
                for (i = start; i <= NR; i++) {
                    if (lines[i] ~ /^[[:space:]]*(Etot|BOND|ANGLE|DIHED|1-4 NB|1-4 EEL|VDWAALS|EAMBER|SC_|lambda =).*\*\*\*\*\*/) {
                        exit 0
                    }
                }
                exit 1
            }
        ' "$output_file"; then
            return 0
        fi

        return 1
    }

    numeric_failure_file() {
        local -a candidates=()
        local f stem seen

        if [[ -n "$rst_file" && "$rst_file" == *.rst7 ]]; then
            stem=${rst_file%.rst7}
            candidates+=("${stem}.out")
        fi
        candidates+=("$log_file")
        if (( extra_file_count > 0 )); then
            for f in "${extra_files[@]}"; do
                case "$f" in
                    *.out|*.log|*.mdinfo|mdinfo) candidates+=("$f") ;;
                esac
            done
        fi

        for f in "${candidates[@]}"; do
            [[ -n "$f" ]] || continue
            seen=" ${_seen_numeric_failure_files:-} "
            [[ "$seen" == *" $f "* ]] && continue
            _seen_numeric_failure_files="${_seen_numeric_failure_files:-} $f"
            if amber_output_has_numeric_failure "$f"; then
                printf '%s\n' "$f"
                return 0
            fi
        done
        return 1
    }

    dt_reduction_template_for_failure() {
        local handoff_input
        if [[ "$rst_file" == eq-handoff-*.rst7 ]]; then
            handoff_input="${rst_file%.rst7}.in"
            if [[ -f "$handoff_input" ]]; then
                printf '%s\n' "$handoff_input"
                return
            fi
        fi
        if [[ "$rst_file" == "eq.rst7" && -f eq.in ]]; then
            printf '%s\n' "eq.in"
            return
        fi
        printf '%s\n' "mdin-template"
    }

    reduce_dt_for_failed_stage() {
        local reduction_start=${1:-2}
        local tmpl
        tmpl=$(dt_reduction_template_for_failure)
        reduce_dt_on_failure "$tmpl" 0.001 "$stage" "$retry_count" "$reduction_start"
    }

    recoverable_gpu_box_grid_restart() {
        case "$stage" in
            "MD segment "*) ;;
            *) return 1 ;;
        esac

        [[ -f "$log_file" ]] || return 1
        grep -Eqi "Periodic box dimensions have changed too much|GPU code does not automatically reorganize grid cells" "$log_file" || return 1
        [[ -n "$rst_file" && -s "$rst_file" ]] || return 1

        if is_amber_restart_path "$rst_file" && ! amber_restart_is_complete "$rst_file"; then
            return 1
        fi
        return 0
    }

    if [[ $command_status =~ ^[0-9]+$ && $command_status -ne 0 ]]; then
        if recoverable_gpu_box_grid_restart; then
            echo "[INFO] $stage hit Amber GPU periodic-box grid-cell restart condition; found usable $rst_file, continuing with next segment."
            return 0
        fi
        echo "[ERROR] $stage simulation failed. Command exited with status $command_status."
        if [[ -f "$log_file" ]]; then
            tail -n 200 "$log_file" || true
        fi
        cleanup_outputs
        if [[ $retry_count -ge 3 ]]; then
            reduce_dt_for_failed_stage 3
        fi
        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    # If log doesn't exist yet, don't treat as failure here
    [[ -f "$log_file" ]] || return 0

    if recoverable_gpu_box_grid_restart; then
        echo "[INFO] $stage hit Amber GPU periodic-box grid-cell restart condition; found usable $rst_file, continuing with next segment."
        return 0
    fi

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|segmentation fault|MPI_ABORT|FATAL|cudaGetDeviceCount|Calculation halted" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        tail -n 200 "$log_file" || true
        cleanup_outputs
        if [[ $retry_count -ge 3 ]]; then
            reduce_dt_for_failed_stage 3
        fi

        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    if [[ -n "$rst_file" && (! -f "$rst_file" || ! -s "$rst_file") ]]; then
        echo "[ERROR] $stage simulation failed. Restart file missing or empty: $rst_file"
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_for_failed_stage 1
        fi
        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    if [[ -n "$rst_file" ]] && is_amber_restart_path "$rst_file" && ! amber_restart_is_complete "$rst_file"; then
        echo "[ERROR] $stage simulation failed. Restart file is incomplete or malformed: $rst_file ($(amber_restart_validation_status "$rst_file"))"
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_for_failed_stage 1
        fi
        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    if [[ -n "$rst_file" ]] && restart_artifact_has_incomplete_output "$rst_file"; then
        local out_file
        out_file=$(output_file_for_restart_artifact "$rst_file")
        echo "[ERROR] $stage simulation failed. Amber output did not reach normal completion marker: $out_file"
        tail -n 200 "$out_file" || true
        cleanup_outputs
        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    local bad_output_file
    if bad_output_file=$(numeric_failure_file); then
        echo "[ERROR] $stage simulation failed. Numeric failure detected in ${bad_output_file}:"
        tail -n 200 "$bad_output_file" || true
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_for_failed_stage 1
        fi
        remove_previous_restart
        write_attempt_failed_marker
        exit 1
    fi

    echo "[INFO] $stage completed successfully at $(date)"
}

run_fe_window_equilibration() {
    local stage=$1
    local initial_restart=$2
    local topology=$3
    local current_restart=$initial_restart
    local input stem
    local -a handoff_inputs=()

    shopt -s nullglob
    handoff_inputs=(eq-handoff-[0-9][0-9].in)
    shopt -u nullglob

    if (( ${#handoff_inputs[@]} == 0 )); then
        if [[ -s eq-handoff.json ]]; then
            echo "[ERROR] Staged EQ inputs were cleaned after a prior successful fe_equil run. Regenerate the FE window inputs before forcing equilibration again."
            write_attempt_failed_marker
            return 1
        fi
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $topology -c $initial_restart -o eq.out -r eq.rst7 -x eq.nc -ref $initial_restart >> \"$log_file\" 2>&1"
        check_sim_failure "$stage" "$log_file" eq.rst7
        return 0
    fi

    rm -f eq-handoff-[0-9][0-9].out eq-handoff-[0-9][0-9].rst7 \
        eq-handoff-[0-9][0-9].nc eq-final.nc eq.nc eq.out
    for input in "${handoff_inputs[@]}"; do
        stem=${input%.in}
        print_and_run "$PMEMD_EXEC -O -i $input -p $topology -c $current_restart -o ${stem}.out -r ${stem}.rst7 -ref $initial_restart >> \"$log_file\" 2>&1"
        check_sim_failure "$stage ($stem)" "$log_file" "${stem}.rst7"
        current_restart="${stem}.rst7"
    done

    print_and_run "$PMEMD_EXEC -O -i eq.in -p $topology -c $current_restart -o eq.out -r eq.rst7 -ref $initial_restart >> \"$log_file\" 2>&1"
    check_sim_failure "$stage (final)" "$log_file" eq.rst7
    SIM_COMMAND_STATUS=0
    return 0
}

cleanup_fe_equilibration_artifacts() {
    local component_prefix=$1
    local n_windows=$2
    local seed_dir=${3:-$PWD}
    local window_dir
    local staged_count=0
    local cleanup_status=0
    local i

    if [[ ${KEEP_FE_EQUIL_ARTIFACTS:-0} == 1 ]]; then
        echo "[INFO] KEEP_FE_EQUIL_ARTIFACTS=1; retaining staged FE equilibration artifacts."
        return 0
    fi
    if [[ ! $n_windows =~ ^[0-9]+$ ]]; then
        echo "[WARN] Cannot clean FE equilibration artifacts: invalid window count '$n_windows'."
        return 1
    fi

    # Preflight every staged window before deleting anything. This keeps all
    # intermediate files available when a component equilibration is partial.
    for ((i=0; i<n_windows; i++)); do
        window_dir=$(printf '%s/../%s%02d' "$seed_dir" "$component_prefix" "$i")
        [[ -s "$window_dir/eq-handoff.json" ]] || continue
        staged_count=$((staged_count + 1))
        if [[ ! -s "$window_dir/eq.rst7" ]]; then
            echo "[WARN] Retaining FE equilibration artifacts because the final restart is missing: $window_dir/eq.rst7"
            return 1
        fi
    done
    if (( staged_count == 0 )); then
        return 0
    fi

    for ((i=0; i<n_windows; i++)); do
        window_dir=$(printf '%s/../%s%02d' "$seed_dir" "$component_prefix" "$i")
        [[ -s "$window_dir/eq-handoff.json" ]] || continue
        if ! rm -f -- \
            "$window_dir"/eq-handoff-*.* \
            "$window_dir"/eq-final.nc \
            "$window_dir"/eq-trajectory-combine.cpptraj \
            "$window_dir"/eq_init.rst7 \
            "$window_dir"/eq.nc \
            "$window_dir"/eq.out \
            "$window_dir"/eq.in \
            "$window_dir"/cmass.txt; then
            echo "[WARN] Could not remove every FE equilibration artifact under $window_dir."
            cleanup_status=1
        fi
    done
    if ! rm -f -- "$seed_dir"/eq.rst7.[0-9]* "$seed_dir"/eq.nc; then
        echo "[WARN] Could not remove every seed-window FE equilibration artifact under $seed_dir."
        cleanup_status=1
    fi

    if (( cleanup_status == 0 )); then
        echo "[INFO] Removed transient FE equilibration artifacts from $staged_count target window(s)."
    fi
    return "$cleanup_status"
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

archive_failed_md_segment() {
    local comp=$1
    local seg_idx=$2
    local n_windows=$3
    local pfolder=${4:-.}
    local retry_count=${5:-${RETRY_COUNT:-${RETRY:-0}}}

    if [[ -z $comp || -z $seg_idx || -z $n_windows ]]; then
        echo "[WARN] archive_failed_md_segment missing args; skip."
        return
    fi

    local out_tag cmass_tag timestamp win window_dir archive_dir src
    local i moved_any
    out_tag=$(printf "md-%02d" "$seg_idx")
    cmass_tag=$(cmass_file_for_md_stem "$out_tag")
    timestamp=$(date +"%Y%m%d_%H%M%S")

    for ((i = 0; i < n_windows; i++)); do
        win=$(printf "%s%02d" "$comp" "$i")
        window_dir="${pfolder}/${win}"
        archive_dir="${window_dir}/WRONG_FAIL/${timestamp}_job_attempt_${retry_count}"
        moved_any=0
        mkdir -p "$archive_dir"

        for src in "${window_dir}/${out_tag}".* \
                   "${window_dir}/${cmass_tag}" \
                   "${window_dir}/mdinfo"; do
            if move_failed_file_if_present "$src" "$archive_dir"; then
                moved_any=1
            fi
        done

        if (( moved_any )); then
            append_attempt_failed_archive "$archive_dir"
            echo "[INFO] Archived failed grouped MD files to ${archive_dir}"
        else
            rmdir "$archive_dir" 2>/dev/null || true
            rmdir "${window_dir}/WRONG_FAIL" 2>/dev/null || true
        fi
    done
}

# Compatibility for component folders generated by older BATTER releases.
cleanup_failed_md_segment() {
    archive_failed_md_segment "$@"
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
        tps=$(production_restart_ps 2>/dev/null || echo 0)
        if [[ -s production-start.ps ]]; then
            tps=$(production_elapsed_ps "$tps" "$(cat production-start.ps)")
        fi
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

scale_steps_for_dt() {
    local steps=$1
    local target_dt=$2
    local current_dt=$3

    awk -v steps="$steps" -v target="$target_dt" -v current="$current_dt" '
        BEGIN {
            if (steps <= 0 || target <= 0 || current <= 0) {
                print steps
                exit
            }
            n = steps * target / current
            whole = int(n)
            if (n - whole > 1e-9) {
                whole += 1
            }
            if (whole < 1) {
                whole = 1
            }
            print whole
        }
    '
}

scaled_nstlim_for_dt() {
    local tmpl=${1:-mdin-template}
    local current_dt=${2:-}
    local target_dt nstlim

    nstlim=$(parse_nstlim "$tmpl") || return 1
    target_dt=$(parse_target_dt_ps "$tmpl")
    [[ -n $current_dt ]] || current_dt=$(parse_dt_ps "$tmpl")

    scale_steps_for_dt "$nstlim" "$target_dt" "$current_dt"
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
    local _dec=${3:-0.001}
    local _reduction_start=${4:-3}

    [[ -f "$tmpl" ]] || { echo 0.001; return; }
    retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
    [[ $retry_count =~ ^[0-9]+$ ]] || { parse_dt_ps "$tmpl"; return; }

    local current_dt target_dt desired_dt
    current_dt=$(parse_dt_ps "$tmpl")
    if grep -Eq '^[!#][[:space:]]*target_dt[[:space:]]*=' "$tmpl"; then
        target_dt=$(parse_target_dt_ps "$tmpl")
    else
        target_dt="$current_dt"
    fi

    desired_dt="$target_dt"
    if [[ $retry_count -ge $_reduction_start ]]; then
        local reduction_steps
        reduction_steps=$((retry_count - _reduction_start + 1))
        if [[ $reduction_steps -gt 3 ]]; then
            reduction_steps=3
        fi
        desired_dt=$(awk -v target="$target_dt" -v dec="$_dec" -v steps="$reduction_steps" 'BEGIN{v=target-steps*dec; if (v<0.001) v=0.001; printf "%.6f\n", v}')
    fi

    awk -v target="$target_dt" -v desired="$desired_dt" -v current="$current_dt" '
        BEGIN {
            if (desired <= 0) {
                printf "%.6f\n", current
            } else if (desired > target) {
                printf "%.6f\n", target
            } else {
                printf "%.6f\n", desired
            }
        }
    '
}

sync_current_mdin_from_template() {
    local tmpl=${1:-mdin-template}
    local current_mdin=${2:-}
    local retry_count=${3:-}
    local effective_dt_override=${4:-}

    [[ -n "$current_mdin" && -f "$current_mdin" ]] || return 0

    local nstlim_value tmp dumpave_file
    if [[ $(basename -- "$current_mdin") == "mdin-remd-current" ]]; then
        rewrite_mdin_dt_file "$current_mdin" "$(parse_dt_ps "$tmpl")"
        return 0
    fi

    nstlim_value=$(parse_nstlim "$current_mdin" 2>/dev/null || parse_nstlim "$tmpl" 2>/dev/null) || return 0
    dumpave_file=$(awk '
        BEGIN{IGNORECASE=1}
        /^[[:space:]]*DUMPAVE[[:space:]]*=/ {
            sub(/^[[:space:]]*DUMPAVE[[:space:]]*=[[:space:]]*/, "")
            print
            exit
        }
    ' "$current_mdin")
    tmp="${current_mdin}.tmp"
    write_mdin_current "$tmpl" "$nstlim_value" 0 "$current_mdin" "$retry_count" "" "$dumpave_file" "$effective_dt_override" > "$tmp" && mv "$tmp" "$current_mdin"
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
            tol = dt * 0.5
            if (tol < 1e-6) {
                tol = 1e-6
            }
            if (rem <= tol) {
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
    if [[ $retry_count -ge 3 ]]; then
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
        sync_current_mdin_from_template "$tmpl" "$current_mdin" "$retry_count" "$desired_dt"
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
    if ! awk -v current="$dt" -v desired="$new_dt" 'BEGIN{diff=current-desired; if (diff<0) diff=-diff; exit !(diff>1e-9)}'; then
        return 0
    fi

    rewrite_mdin_dt_file "$tmpl" "$new_dt"

    local current_mdin
    if current_mdin=$(current_mdin_for_template "$tmpl"); then
        sync_current_mdin_from_template "$tmpl" "$current_mdin" "$retry_count" "$new_dt"
    fi

    # Remove MD output artifacts after a dt reduction, but keep restart backups
    # so retries can step back one segment instead of restarting the window.
    rm -f md-*.out md*.out \
          md-*.nc md*.nc \
          md-*.log md*.log \
          md-*.mden md*.mden \
          md-*.mdinfo md*.mdinfo \
          cmass.txt cmass-*.txt
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
    local tps fallback_tps

    if command -v ncdump >/dev/null 2>&1; then
        tps=$(ncdump -v time "$rst_file" 2>/dev/null | awk '
      BEGIN{IGNORECASE=1}
      tolower($1) == "time" && $2 == "=" {
        gsub(/;/, "", $3)
        print $3
        exit
      }
        ')
        if [[ -n $tps && $tps != 0 && $tps != 0.0 && $tps != 0.000 && $tps != 0.0000 ]]; then
            echo "$tps"
            return
        fi
    fi

    if LC_ALL=C grep -Iq . "$rst_file"; then
        fallback_tps=$(awk '
          BEGIN{
            num="^[-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?$"
          }
          /^time[[:space:]]*=/ {
            s=$0
            sub(/^time[[:space:]]*=[[:space:]]*/, "", s)
            gsub(/[[:space:];]/, "", s)
            gsub(/[dD]/, "e", s)
            if (s ~ num) {
              print s
              exit
            }
          }
          NR == 2 && NF >= 2 {
            s=$2
            gsub(/[dD]/, "e", s)
            if (s ~ num) {
              print s
              exit
            }
          }
        ' "$rst_file")
    fi

    if [[ -n $fallback_tps ]]; then
        echo "$fallback_tps"
    elif [[ -n $tps ]]; then
        echo "$tps"
    else
        echo 0
    fi
}

md_restart_is_valid_for_resume() {
    local restart_file=$1
    local start_ps=${2:-0}
    local restart_ps

    [[ -s "$restart_file" ]] || return 1
    restart_ps=$(completed_time_ps_from_rst "$restart_file")

    awk -v rst="$restart_ps" -v start="$start_ps" '
      BEGIN {
        # Valid production restarts must be parseable and ahead of the initial
        # production restart. Files at or before start_ps cannot represent useful
        # resumed production progress.
        exit !(rst + 0.0 > start + 1.0e-7)
      }
    '
}

archive_invalid_md_restart_if_present() {
    local restart_file=$1
    local latest_out_file=${2:-}
    local retry_count=${3:-}
    local start_ps=${4:-0}
    local restart_ps

    [[ -e "$restart_file" ]] || return 1
    restart_ps=$(completed_time_ps_from_rst "$restart_file")

    local files=("$restart_file")
    if [[ -n $latest_out_file && -e "$latest_out_file" ]]; then
        local stem
        stem=${latest_out_file%.out}
        files+=(
            "$latest_out_file"
            "${stem}.nc"
            "${stem}.log"
            "${stem}.mden"
            "${stem}.mdinfo"
        )
    fi

    archive_failed_job_files "$retry_count" "${files[@]}"
    echo "[WARN] Archived invalid MD restart ${restart_file} before resume (restart=${restart_ps} ps, start=${start_ps} ps)."
    return 0
}

select_valid_md_restart() {
    local initial_restart=$1
    local start_ps=${2:-0}
    local retry_count=${3:-}
    local idx restart_file compact_restart latest_out_file compact_out

    SELECTED_MD_RESTART="$initial_restart"

    idx=$(latest_md_restart_index)
    while [[ $idx =~ ^[0-9]+$ && $idx -ge 1 ]]; do
        restart_file=$(printf "md-%02d.rst7" "$idx")
        compact_restart=$(printf "md%02d.rst7" "$idx")
        if [[ ! -e "$restart_file" && -e "$compact_restart" ]]; then
            restart_file="$compact_restart"
        fi
        if [[ -e "$restart_file" ]]; then
            if md_restart_is_valid_for_resume "$restart_file" "$start_ps"; then
                SELECTED_MD_RESTART="$restart_file"
                return 0
            fi
            latest_out_file=$(printf "md-%02d.out" "$idx")
            compact_out=$(printf "md%02d.out" "$idx")
            if [[ ! -e "$latest_out_file" && -e "$compact_out" ]]; then
                latest_out_file="$compact_out"
            fi
            archive_invalid_md_restart_if_present \
                "$restart_file" "$latest_out_file" "$retry_count" "$start_ps"
        fi
        idx=$((idx - 1))
    done

    return 0
}

completed_steps() {
    local tmpl=${1:-mdin-template}
    local tps

    tps=$(production_restart_ps)
    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        echo 0
        return
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

production_restart_ps() {
    local restart_file=${1:-}
    local tps

    if [[ -z $restart_file ]]; then
        restart_file=$(latest_md_restart_path 2>/dev/null || true)
    fi
    if [[ -z $restart_file ]]; then
        echo 0
        return
    fi

    tps=$(completed_time_ps_from_rst "$restart_file")
    [[ -n $tps ]] || tps=0
    echo "$tps"
}

production_start_ps() {
    local marker=${1:-production-start.ps}
    local initial_rst=${2:-eq.rst7}
    local start_ps

    if [[ -s "$marker" ]]; then
        start_ps=$(tr -d '[:space:]' < "$marker")
        if [[ $start_ps =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
            echo "$start_ps"
            return
        fi
    fi

    start_ps=$(completed_time_ps_from_rst "$initial_rst")
    [[ -n $start_ps ]] || start_ps=0

    mkdir -p "$(dirname -- "$marker")" 2>/dev/null || true
    printf "%s\n" "$start_ps" > "$marker" 2>/dev/null || true
    echo "$start_ps"
}

production_elapsed_ps() {
    local absolute_ps=${1:-0}
    local start_ps=${2:-0}

    awk -v abs="$absolute_ps" -v start="$start_ps" '
      BEGIN {
        elapsed = abs - start
        if (elapsed < 0) {
          elapsed = 0
        }
        s = sprintf("%.10f", elapsed)
        sub(/\.?0+$/, "", s)
        if (s == "") {
          s = "0"
        }
        print s
      }
    '
}

completed_production_ps() {
    local tmpl=${1:-mdin-template}
    local marker=${2:-production-start.ps}
    local initial_rst=${3:-eq.rst7}
    local absolute_ps start_ps

    absolute_ps=$(production_restart_ps)
    [[ -n $absolute_ps ]] || absolute_ps=0
    start_ps=$(production_start_ps "$marker" "$initial_rst")
    production_elapsed_ps "$absolute_ps" "$start_ps"
}

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


mdin_set_cntrl_value() {
    local key=$1
    local value=$2

    awk -v key="$key" -v value="$value" '
        BEGIN {
            in_cntrl = 0
            inserted = 0
            key_pattern = "^[[:space:]]*" tolower(key) "[[:space:]]*="
        }
        {
            line = $0
            lower = tolower(line)
            if (lower ~ /^[[:space:]]*&cntrl/) {
                in_cntrl = 1
            }
            if (lower ~ key_pattern) {
                print "  " key " = " value ","
                inserted = 1
                next
            }
            if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/ && inserted == 0) {
                print "  " key " = " value ","
                inserted = 1
            }
            print line
            if (in_cntrl && line ~ /^[[:space:]]*\/[[:space:]]*$/) {
                in_cntrl = 0
            }
        }
    '
}

mdin_get_cntrl_value() {
    local key=$1

    awk -v key="$key" '
        BEGIN {
            key_pattern = "^[[:space:]]*" tolower(key) "[[:space:]]*="
        }
        {
            line = $0
            lower = tolower(line)
            if (lower ~ key_pattern) {
                sub(/^[^=]*=/, "", line)
                sub(/,.*/, "", line)
                gsub(/[[:space:]]/, "", line)
                print line
                exit
            }
        }
    '
}

mdin_has_cntrl_value() {
    local key=$1

    awk -v key="$key" '
        BEGIN {
            key_pattern = "^[[:space:]]*" tolower(key) "[[:space:]]*="
            found = 0
        }
        {
            if (tolower($0) ~ key_pattern) {
                found = 1
                exit
            }
        }
        END {
            exit !found
        }
    '
}

mdin_cap_cntrl_frequency_to_nstlim() {
    local key=$1
    local nstlim_value=$2
    local input current_value

    [[ $nstlim_value =~ ^[0-9]+$ && $nstlim_value -gt 0 ]] || { cat; return; }

    input=$(cat)
    current_value=$(printf "%s\n" "$input" | mdin_get_cntrl_value "$key")
    if [[ $current_value =~ ^[0-9]+$ && $current_value -gt 0 && $current_value -gt $nstlim_value ]]; then
        printf "%s\n" "$input" | mdin_set_cntrl_value "$key" "$nstlim_value"
    else
        printf "%s\n" "$input"
    fi
}

mdin_cap_dumpfreq_to_nstlim() {
    local nstlim_value=$1

    [[ $nstlim_value =~ ^[0-9]+$ && $nstlim_value -gt 0 ]] || { cat; return; }

    awk -v nstlim="$nstlim_value" '
        BEGIN { IGNORECASE = 1 }
        {
            line = $0
            if (line ~ /DUMPFREQ/ && match(line, /istep1[[:space:]]*=[[:space:]]*[0-9]+/)) {
                token = substr(line, RSTART, RLENGTH)
                value = token
                sub(/.*=/, "", value)
                gsub(/[[:space:]]/, "", value)
                if (value + 0 > nstlim + 0) {
                    line = substr(line, 1, RSTART - 1) "istep1=" int(nstlim) substr(line, RSTART + RLENGTH)
                }
            }
            print line
        }
    '
}

can_skip_short_final_tail() {
    local total_ps=$1
    local current_ps=$2
    local remaining_ps=$3

    awk -v tot="$total_ps" -v cur="$current_ps" -v rem="$remaining_ps" '
        BEGIN {
            if (tot <= 0 || cur <= 0 || rem <= 0) {
                exit 1
            }
            frac = rem / tot
            exit !(tot >= 100 && rem <= 100 && frac <= 0.025)
        }
    '
}

write_mdin_current() {
    local tmpl=${1:-mdin-template}
    local nstlim_value=$2
    local first_run=$3
    local current_mdin=${4:-mdin-current}
    local retry_count=${5:-}
    local initial_time_ps=${6:-}
    local dumpave_file=${7:-}
    local effective_dt_override=${8:-}

    [[ -f $tmpl ]] || { echo "[ERROR] Missing template $tmpl" >&2; return 1; }

    local text freq_key
    text=$(<"$tmpl")

    local template_dt effective_dt
    template_dt=$(parse_dt_ps "$tmpl")
    if [[ $effective_dt_override =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
        effective_dt="$effective_dt_override"
    else
        retry_count=$(retry_count_for_template "$tmpl" "$retry_count")
        effective_dt=$(retry_adjusted_dt_ps "$tmpl" "$retry_count" 0.001 3)
    fi

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

    text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "irest" "1")
    text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "ntx" "5")
    if [[ $first_run == 1 ]]; then
        if [[ $initial_time_ps =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
            text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "t" "$initial_time_ps")
        fi
    fi

    text=$(printf "%s\n" "$text" | mdin_set_cntrl_value "nstlim" "$nstlim_value")
    for freq_key in ntpr ntwr ntwx ntwe; do
        text=$(printf "%s\n" "$text" | mdin_cap_cntrl_frequency_to_nstlim "$freq_key" "$nstlim_value")
    done
    text=$(printf "%s\n" "$text" | mdin_cap_dumpfreq_to_nstlim "$nstlim_value")
    if [[ -n $dumpave_file ]]; then
        text=$(printf "%s\n" "$text" | awk -v dumpave="$dumpave_file" '
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
