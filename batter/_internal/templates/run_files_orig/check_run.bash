check_sim_failure() {
    local stage=$1
    local log_file=$2
    local rst_file=$3
    local rst_file_prev=${4:-}
    local retry_count=${5:-0}

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|Error|failed|segmentation fault" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        cat "$log_file"
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

    # Prefer EAMBER from the FINAL RESULTS block; fall back to ENERGY from the NSTEP table
    local energy_value source_label
    energy_value=$(awk '/FINAL RESULTS/,/EAMBER/ { if ($1 == "EAMBER") { print $3; exit } }' "$energy_file")
    source_label="EAMBER energy"

    if [[ -z $energy_value ]]; then
        # First, try to grab ENERGY from the NSTEP table that appears after the FINAL RESULTS section
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
        # Fallback: any NSTEP ENERGY table in the file
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

    # Validate numeric (support scientific notation)
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

# Lightweight progress reporting helpers (stage + completed steps)
highest_index_for_pattern() {
    local pattern=$1
    local max=-1
    for f in $pattern; do
        [[ -e "$f" ]] || continue
        # extract the first integer chunk from the filename
        local n
        n=$(echo "$f" | sed -E 's/[^0-9]*([0-9]+).*/\1/')
        [[ $n =~ ^[0-9]+$ ]] || continue
        if [[ $n -gt $max ]]; then
            max=$n
        fi
    done
    echo "$max"
}

report_progress() {
    local stage="not_started"
    local steps=0

    if [[ -f FINISHED ]]; then
        stage="production"
        steps=$(highest_index_for_pattern "mdin-*.rst7")
        [[ $steps -lt 0 ]] && steps="completed"
    elif ls mdin-*.rst7 >/dev/null 2>&1; then
        stage="production"
        steps=$(highest_index_for_pattern "mdin-*.rst7")
        [[ $steps -lt 0 ]] && steps=0
    elif ls eqnpt*.rst7 >/dev/null 2>&1; then
        stage="equilibration"
        steps=$(highest_index_for_pattern "eqnpt*.rst7")
        [[ $steps -lt 0 ]] && steps=0
    elif ls mini*.rst7 >/dev/null 2>&1; then
        stage="minimization"
        steps=$(highest_index_for_pattern "mini*.rst7")
        [[ $steps -lt 0 ]] && steps=0
    fi

    echo "[progress] stage=${stage} steps_completed=${steps}"
}

# --------- Equil/production helper parsers ---------
parse_total_steps() {
    local tmpl=${1:-mdin-template}
    if [[ ! -f $tmpl ]]; then
        echo "[ERROR] Missing template $tmpl" >&2
        return 1
    fi
    local total
    total=$(grep -E "^[!#]\\s*eq_steps\\s*=\\s*[0-9]+" "$tmpl" | tail -1 | sed -E 's/.*eq_steps\\s*=\\s*([0-9]+).*/\1/')
    if [[ -z $total ]]; then
        echo "[ERROR] eq_steps comment not found in $tmpl (expected '! eq_steps=<total_steps>')" >&2
        return 1
    fi
    echo "$total"
}

parse_nstlim() {
    local tmpl=${1:-mdin-template}
    local nst
    nst=$(grep -E "^[[:space:]]*nstlim" "$tmpl" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
    if [[ -z $nst ]]; then
        echo "[ERROR] Could not parse nstlim from $tmpl" >&2
        return 1
    fi
    echo "$nst"
}

latest_md_index() {
    local pattern=${1:-"md*.out"}
    local idx
    idx=$(highest_index_for_pattern "$pattern")
    echo "$idx"
}

completed_steps() {
    local tmpl=${1:-mdin-template}
    local idx
    idx=$(latest_md_index "md*.out")
    if [[ $idx -lt 0 ]]; then
        echo 0
        return
    fi
    local mdout
    mdout=$(printf "md-%02d.out" "$idx")
    if [[ ! -f $mdout ]]; then
        mdout=$(printf "md%02d.out" "$idx")
    fi
    if [[ ! -f $mdout ]]; then
        echo $(( (idx + 1) * $(parse_nstlim "$tmpl") ))
        return
    fi
    local nstep
    nstep=$(grep "NSTEP" "$mdout" | tail -1 | awk '{for(i=1;i<=NF;i++){if($i=="NSTEP"){print $(i+2); exit}}}')
    if [[ -z $nstep ]]; then
        echo $(( (idx + 1) * $(parse_nstlim "$tmpl") ))
    else
        echo "$nstep"
    fi
}

write_mdin_current() {
    local tmpl=${1:-mdin-template}
    local nstlim_value=$2
    local first_run=$3
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
    echo "$text"
}
