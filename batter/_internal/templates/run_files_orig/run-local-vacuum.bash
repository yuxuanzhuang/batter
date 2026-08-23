#!/usr/bin/env bash
set -euo pipefail

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
CPPTRAJ_EXEC=${CPPTRAJ_EXEC:-cpptraj}

# Define constants for filenames
PRMTOP="full_merged.prmtop"
log_file="run.log"
INPCRD="full.inpcrd"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}
retry=${RETRY_COUNT:-${RETRY:-}}

# Echo commands before executing them so the full invocation is visible
print_and_run() {
    echo "$@"
    local errexit_was_on=0
    case $- in
        *e*) errexit_was_on=1 ;;
    esac
    SIM_COMMAND_STATUS=0
    set +e
    eval "$@"
    SIM_COMMAND_STATUS=$?
    if [[ $errexit_was_on -eq 1 ]]; then
        set -e
    else
        set +e
    fi
}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

source check_run.bash

rm -f FAILED

consume_prior_failure_marker >/dev/null

archive_existing_log_file "$log_file"
cleanup_stale_empty_md_artifacts relaxed
cleanup_zero_frame_md_trajectories "$retry"

# ------------------------- only_eq mode -------------------------
if [[ $only_eq -eq 1 ]]; then
    # no equilibration needed here; just seed a restart
    cp "$INPCRD" mini.rst7
    check_sim_failure "Seed restart" "$log_file" mini.rst7

    # propagate restart to each window folder
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" "$i")
        if [[ -s "$win_folder/mini.rst7" ]]; then
            echo "Skipping seed for window $i, already exists."
        else
            echo "Seeding window $i"
            cp "mini.rst7" "$win_folder/eq.rst7"
        fi
    done

    print_and_run "$CPPTRAJ_EXEC -i /dev/stdin >> \"$log_file\" 2>&1 <<'EOF'
parm $PRMTOP
trajin mini.rst7
trajout eq_output.pdb pdb include_ep
run
EOF"

    echo "Only seeding requested and finished."
    if [[ -s eq_output.pdb ]]; then
        echo "EQ_FINISHED" > EQ_FINISHED
        echo "[INFO] EQ_FINISHED marker written."
        echo "Job completed at $(date)"
    fi
    exit 0
fi

# ------------------------- production mode -------------------------
tmpl="mdin-template"
mdin_current="mdin-current"

if [[ ! -f $tmpl ]]; then
    echo "[ERROR] Missing mdin template: $tmpl"
    exit 1
fi

apply_retry_dt_reduction "$tmpl" "$retry" 0.001 "production startup"

dt_ps=$(parse_dt_ps "$tmpl")
target_dt_ps=$(parse_target_dt_ps "$tmpl")
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(scaled_nstlim_for_dt "$tmpl" "$dt_ps")

# Convert target steps -> ps using the original requested dt; rerun steps use current dt.
total_ps=$(awk -v s="$total_steps" -v dt="$target_dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')
chunk_ps=$(awk -v s="$chunk_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

# Progress is production elapsed time, not absolute Amber restart time.
production_start_marker="production-start.ps"
production_initial_rst="mini.in.rst7"
start_ps=$(production_start_ps "$production_start_marker" "$production_initial_rst")
select_valid_md_restart "$production_initial_rst" "$start_ps" "$retry"
rst_in="$SELECTED_MD_RESTART"
require_nonempty_file_or_attempt_fail "$rst_in" "[ERROR] Missing restart file $rst_in; cannot continue."
restart_ps=$(production_restart_ps "$rst_in")
[[ -z $restart_ps ]] && restart_ps=0
current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
[[ -z $current_ps ]] && current_ps=0

echo "Current completed production time: $current_ps ps / $total_ps ps (restart=$restart_ps ps, start=$start_ps ps, dt=$dt_ps ps)"

# Determine current segment index from existing OUT files and the selected restart.
restart_seg_idx=0
if parsed_restart_seg_idx=$(md_segment_index_from_restart "$rst_in" 2>/dev/null); then
    restart_seg_idx=$parsed_restart_seg_idx
fi
seg_idx=$(latest_md_index "md-*.out")
if [[ $seg_idx -lt 0 ]]; then
    seg_idx=0
fi
if (( restart_seg_idx > seg_idx )); then
    seg_idx=$restart_seg_idx
fi

last_rst="$rst_in"

remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')
remaining_steps=$(remaining_steps_from_time "$total_ps" "$current_ps" "$dt_ps")
if can_skip_short_final_tail "$total_ps" "$current_ps" "$remaining_ps"; then
    remaining_steps=0
    current_ps="$total_ps"
fi

if (( remaining_steps > 0 )); then
    run_steps=$remaining_steps
    if (( run_steps > chunk_steps )); then
        run_steps=$chunk_steps
    fi
    run_ps=$(awk -v s="$run_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

    first_run=0
    if [[ "$rst_in" == "$production_initial_rst" ]]; then
        first_run=1
    fi

    out_tag=$(printf "md-%02d" $((seg_idx + 1)))
    rst_out="${out_tag}.rst7"
    cmass_file=$(printf "cmass-%02d.txt" $((seg_idx + 1)))
    echo "[INFO] Running segment $((seg_idx + 1)) -> ${out_tag}.out for ${run_steps} steps (${run_ps} ps); restart_in=$rst_in"

    write_mdin_current "$tmpl" "$run_steps" "$first_run" "$mdin_current" "$retry" "$start_ps" "$cmass_file" > "$mdin_current"

    # Preflight: ensure output directory writable (avoids Fortran OPEN errors)
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r $rst_out -x ${out_tag}.nc -ref $rst_in -AllowSmallBox >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "$rst_out" "" "$retry" "${out_tag}.out" "${out_tag}.nc" "$cmass_file"

    # Update production elapsed time from the explicit segment restart.
    restart_ps=$(production_restart_ps "$rst_out")
    [[ -z $restart_ps ]] && restart_ps=0
    current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed production time: $current_ps ps / $total_ps ps (restart=$restart_ps ps, start=$start_ps ps)"

    rst_in="$rst_out"
    last_rst="$rst_out"
fi

if production_is_complete "$current_ps" "$total_ps" "$dt_ps"; then
    require_nonempty_file_or_attempt_fail "$last_rst" "[ERROR] Production is marked complete but restart $last_rst is missing."
    print_and_run "$CPPTRAJ_EXEC -i /dev/stdin >> \"$log_file\" 2>&1 <<'EOF'
parm $PRMTOP
trajin ${last_rst}
trajout output.pdb pdb include_ep
run
EOF"

    # check output.pdb exists to catch cases where the simulation did not run to completion
    if [[ -s output.pdb ]]; then
        cleanup_finished_md_restarts
        echo "FINISHED" > FINISHED
        echo "[INFO] FINISHED marker written."
        exit 0
    fi

    mark_failed_and_exit "[ERROR] output.pdb not created or empty; marking ATTEMPT_FAILED."
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
