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