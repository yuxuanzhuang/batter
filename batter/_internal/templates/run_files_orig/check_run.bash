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
