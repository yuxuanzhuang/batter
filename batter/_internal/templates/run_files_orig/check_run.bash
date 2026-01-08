check_sim_failure() {
    local stage=$1
    local log_file=$2
    local rst_file=$3
    local rst_file_prev=${4:-}
    local retry_count=${5:-0}

    # If log doesn't exist yet, don't treat as failure here
    [[ -f "$log_file" ]] || return 0

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|segmentation fault|MPI_ABORT|FATAL" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        tail -n 200 "$log_file" || true
        rm -f "$log_file"
        rm -f "$rst_file"

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

    if ! [[ $energy_value =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "Error: Energy value '$energy_value' is not a valid number"
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
            if (match($0, /^[[:space:]]*dt[[:space:]]*=[[:space:]]*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/, m)) {
                print m[1]
                exit
            }
        }
        ' "$tmpl"
    )

    [[ -n $dt ]] && echo "$dt" || echo 0.001
}

completed_time_ps_from_out() {
    local out_file=$1
    [[ -f $out_file ]] || { echo 0; return; }

    awk '
      BEGIN{IGNORECASE=1}
      match($0, /TIME\(PS\)[[:space:]]*=[[:space:]]*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/, m) { last=m[1] }
      END { if (last != "") printf "%s\n", last; else print 0 }
    ' "$out_file"
}

completed_steps() {
    local tmpl=${1:-mdin-template}
    local seg mdout tps prev_seg prev_out prev_tps

    # ---- find latest md output ----
    seg=$(latest_md_index "md-*.out")
    echo "[DEBUG] Latest md segment index: $seg" >&2
    if [[ $seg -lt 0 ]]; then
        echo 0
        return
    fi

    mdout=$(printf "md-%02d.out" "$seg")
    [[ -f $mdout ]] || mdout=$(printf "md%02d.out" "$seg")
    [[ -f $mdout ]] || { echo 0; return; }

    tps=$(completed_time_ps_from_out "$mdout")

    # ---- fallback if latest is bad/truncated ----
    if [[ -z $tps || $tps == 0 || $tps == 0.0 || $tps == 0.000 || $tps == 0.0000 ]]; then
        prev_seg=$((seg - 1))
        if (( prev_seg >= 0 )); then
            prev_out=$(printf "md-%02d.out" "$prev_seg")
            [[ -f $prev_out ]] || prev_out=$(printf "md%02d.out" "$prev_seg")

            if [[ -f $prev_out ]]; then
                prev_tps=$(completed_time_ps_from_out "$prev_out")
                if [[ -n $prev_tps && $prev_tps != 0 && $prev_tps != 0.0 ]]; then
                    echo "[WARN] Latest out $mdout has 0 ps; using $prev_out (TIME(PS)=$prev_tps) and removing $mdout" >&2
                    rm -f "$mdout"
                    tps="$prev_tps"
                else
                    echo 0
                    return
                fi
            else
                echo 0
                return
            fi
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
                if (match($0, /(^|[^a-z0-9_])ntwr[[:space:]]*=[[:space:]]*[-+]?[0-9]+/, m)) {
                  s=substr($0, RSTART, RLENGTH)
                  gsub(/.*ntwr[[:space:]]*=[[:space:]]*/, "", s)
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
                if (match($0, /(^|[^a-z0-9_])dt[[:space:]]*=[[:space:]]*[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?/, m)) {
                  s=substr($0, RSTART, RLENGTH)
                  gsub(/.*dt[[:space:]]*=[[:space:]]*/, "", s)
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