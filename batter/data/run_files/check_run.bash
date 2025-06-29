check_sim_failure() {
    local stage=$1
    local log_file=$2

    if grep -Eqi "Terminated Abnormally|command not found|illegal memory|Error|failed" "$log_file"; then
        echo "[ERROR] $stage simulation failed. Detected error in $log_file:"
        cat "$log_file"
        # clean up the log file
        rm -f "$log_file"
        exit 1
    else
        echo "[INFO] $stage completed successfully at $(date)"
    fi
}

check_min_energy() {
    local energy_file=$1
    local threshold=$2

    # Find EAMBER value from FINAL RESULTS block
    local eam_line
    eam_line=$(awk '/FINAL RESULTS/,/EAMBER/ { if ($1 == "EAMBER") print $3 }' "$energy_file")

    if [[ -z $eam_line ]]; then
        echo "Error: Could not find EAMBER energy in $energy_file"
        return 2
    fi

    # Check if eam_line is a valid number
    if ! [[ $eam_line =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error: EAMBER value '$eam_line' is not a valid number"
        return 1
    fi

    local eam_value
    eam_value=$(printf "%.4f" "$eam_line")

    echo "EAMBER energy: $eam_value kcal/mol (threshold: $threshold)"

    if (( $(echo "$eam_value < $threshold" | bc -l) )); then
        echo "Energy is below threshold."
        return 0
    else
        echo "Energy is above threshold."
        return 1
    fi
}