check_sim_failure() {
    local stage=$1
    local log_file=$2
    local rst_file=$3
    local rst_file_prev=${4:-}
    local retry_count=${5:-0}
    local extra_files=()
    if (( $# > 5 )); then
        extra_files=("${@:6}")
    fi

    cleanup_outputs() {
        local f
        for f in "${extra_files[@]}"; do
            [[ -n "$f" ]] || continue
            rm -f "$f"
        done
    }

    # If log doesn't exist yet, don't treat as failure here
    [[ -f "$log_file" ]] || return 0

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|segmentation fault|MPI_ABORT|FATAL" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        tail -n 200 "$log_file" || true
        rm -f "$log_file"
        rm -f "$rst_file"
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_on_failure "mdin-template" 0.001 "$stage" "$retry_count"
        fi

        # if not the first retry attempt, remove the previous restart file to avoid repeated failure
        if [[ -n "$rst_file_prev" && $retry_count -gt 0 ]]; then
            echo "[INFO] Removing previous restart file $rst_file_prev before retrying."
            rm -f "$rst_file_prev"
        fi
        exit 1
    fi

    if [[ -n "$rst_file" && (! -f "$rst_file" || ! -s "$rst_file") ]]; then
        echo "[ERROR] $stage simulation failed. Restart file missing or empty: $rst_file"
        rm -f "$log_file"
        rm -f "$rst_file"
        cleanup_outputs
        if [[ $retry_count -ge 2 ]]; then
            reduce_dt_on_failure "mdin-template" 0.001 "$stage" "$retry_count"
        fi
        if [[ -n "$rst_file_prev" && $retry_count -gt 0 ]]; then
            echo "[INFO] Removing previous restart file $rst_file_prev before retrying."
            rm -f "$rst_file_prev"
        fi
        exit 1
    fi

    echo "[INFO] $stage completed successfully at $(date)"
}

check_min_energy() {
    local energy_file=$1
    local threshold=$2

    local energy_value source_label
    energy_value=$(awk '/FINAL RESULTS/,/EAMBER/ { if ($1 == "EAMBER") { print $3; exit } }' "$energy_file")
    source_label="EAMBER energy"

    if [[ -z $energy_value ]]; then
        energy_value=$(awk '
            /FINAL RESULTS/ { in_final=1; next }
            in_final && /^[[:space:]]*NSTEP[[:space:]]+ENERGY[[:space:]]+RMS[[:space:]]+GMAX/ {
                if (getline line) {
                    split(line, fields)
                    if (length(fields) >= 2) found=fields[2]
                }
            }
            END { if (found) print found }
        ' "$energy_file")
        source_label="ENERGY (NSTEP table)"
    fi

    if [[ -z $energy_value ]]; then
        energy_value=$(awk '
            /^[[:space:]]*NSTEP[[:space:]]+ENERGY[[:space:]]+RMS[[:space:]]+GMAX/ {
                if (getline line) {
                    split(line, fields)
                    if (length(fields) >= 2) last=fields[2]
                }
            }
            END { if (last) print last }
        ' "$energy_file")
        source_label="ENERGY (NSTEP table)"
    fi

    if [[ -z $energy_value ]]; then
        echo "Error: Could not find EAMBER or ENERGY in $energy_file"
        return 2
    fi

    # Only check the last energy block for overflow markers.
    if awk '
        /^[[:space:]]*NSTEP[[:space:]]+ENERGY[[:space:]]+RMS[[:space:]]+GMAX/ {block=""; inblock=1}
        inblock {block = block $0 "\n"}
        inblock && NF==0 {inblock=0}
        END {print block}
    ' "$energy_file" | grep -q "********"; then
        echo "Error: Overflow detected in last energy block of $energy_file"
        return 1
    fi

    if ! [[ $energy_value =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "Error: Energy value '$energy_value' is not a valid number"
        return 1
    fi

    # Catch absurd energies that often signal numerical blow-up.
    if awk -v val="$energy_value" 'BEGIN { exit (val < -1.0e8 || val > 1.0e8) ? 0 : 1 }'; then
        echo "Error: Energy magnitude too large: $energy_value"
        return 1
    fi

    local formatted_value
    formatted_value=$(printf "%.4f" "$energy_value")
    echo "$source_label: $formatted_value kcal/mol (threshold: $threshold)"

    if awk -v val="$energy_value" -v thr="$threshold" 'BEGIN { exit (val < thr ? 0 : 1) }'; then
        echo "Energy is below threshold."
        return 0
    else
        echo "Energy is above threshold."
        return 1
    fi
}

# Highest numeric index in files matching pattern, where the index is the digits
# immediately before the ".out" extension.
# Works in bash and zsh; ignores non-matching names; safe when glob matches nothing.
highest_out_index_for_pattern() {
    local pattern=$1
    local max=-1
    local f n

    if [[ -n ${ZSH_VERSION-} ]]; then
        setopt local_options null_glob
        for f in ${~pattern}; do
            [[ -e "$f" ]] || continue
            if [[ $f =~ ([0-9]+)\.out$ ]]; then
                n=${match[1]}
                n=$((10#$n))
                (( n > max )) && max=$n
            fi
        done
    else
        for f in $pattern; do
            [[ -e "$f" ]] || continue
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
    highest_out_index_for_pattern "$pattern"
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

    if ls md-*.out >/dev/null 2>&1 || ls md*.out >/dev/null 2>&1; then
        stage="production"
        seg=$(latest_md_index "md-*.out")
        [[ $seg -lt 0 ]] && seg=$(latest_md_index "md*.out")
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

reduce_dt_on_failure() {
    local tmpl=${1:-mdin-template}
    local dec=${2:-0.001}
    local stage=${3:-unknown}
    local retry_count=${4:-0}

    [[ -f "$tmpl" ]] || { echo "[WARN] $tmpl not found; skip dt reduction."; return; }
    if ! awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*dt[[:space:]]*=/ {found=1; exit} END{exit !found}' "$tmpl"; then
        echo "[WARN] dt not found in $tmpl; skip dt reduction."
        return
    fi

    local dt new_dt
    dt=$(parse_dt_ps "$tmpl")
    new_dt=$(awk -v dt="$dt" -v dec="$dec" 'BEGIN{printf "%.6f\n", dt-dec}')
    if ! awk -v nd="$new_dt" 'BEGIN{exit !(nd>0)}'; then
        echo "[WARN] dt reduction skipped (current dt=${dt}, dec=${dec})."
        return
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
    ' "$tmpl" > "${tmpl}.tmp" && mv "${tmpl}.tmp" "$tmpl"

    # remove old sims if there's any.
    rm md-*
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

    # ---- fallback if latest restart is missing/bad ----
    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        prev_tps=$(completed_time_ps_from_rst "md-previous.rst7")
        if [[ -n $prev_tps && $prev_tps != 0 && $prev_tps != 0.0 ]]; then
            tps="$prev_tps"
        else
            echo 0
            return
        fi
    fi

    # ---- align to restart-write boundary from mdin template ----
    # We interpret "divisible by ntwr" as: floor TIME(PS) to a multiple of (ntwr * dt_ps)
    # because ntwr is in steps, while tps is in picoseconds.
    #
    # If we can't parse ntwr or dt, we just return tps unchanged.
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

        # Default Amber dt is ps (e.g., 0.002). If dt is missing, we can't align.
        if [[ -n $ntwr && $ntwr -gt 0 && -n $dt ]]; then
            # compute restart interval in ps: ntwr * dt
            dt_ps=$(awk -v dt="$dt" 'BEGIN{printf "%.10f", dt+0.0}')
            local interval_ps
            interval_ps=$(awk -v n="$ntwr" -v dt="$dt_ps" 'BEGIN{printf "%.10f", n*dt}')

            # Floor tps to nearest multiple of interval_ps (never increase).
            # Guard against interval_ps == 0 due to parse weirdness.
            if awk -v x="$interval_ps" 'BEGIN{exit !(x>0)}'; then
                tps=$(awk -v t="$tps" -v step="$interval_ps" '
                    BEGIN{
                      # number of full intervals completed:
                      k = int(t/step + 1e-12)
                      out = k*step
                      # print with reasonable precision; trim trailing zeros via %g-like logic
                      printf "%.10f\n", out
                    }' | sed -E 's/([0-9])0+$/\1/; s/\.$//')
            fi
        fi
    fi

    echo "$tps"
}


write_mdin_current() {
    local tmpl=${1:-mdin-template}
    local nstlim_value=$2
    local first_run=$3

    [[ -f $tmpl ]] || { echo "[ERROR] Missing template $tmpl" >&2; return 1; }

    local text
    text=$(<"$tmpl")

    if [[ $first_run -eq 1 ]]; then
        text=$(echo "$text" \
            | sed -E 's/^[[:space:]]*irest[[:space:]]*=.*/  irest = 0,/' \
            | sed -E 's/^[[:space:]]*ntx[[:space:]]*=.*/  ntx   = 1,/')
    else
        text=$(echo "$text" \
            | sed -E 's/^[[:space:]]*irest[[:space:]]*=.*/  irest = 1,/' \
            | sed -E 's/^[[:space:]]*ntx[[:space:]]*=.*/  ntx   = 5,/')
    fi

    text=$(echo "$text" | sed -E "s/^[[:space:]]*nstlim[[:space:]]*=.*/  nstlim = ${nstlim_value},/")
    echo "$text"
}
