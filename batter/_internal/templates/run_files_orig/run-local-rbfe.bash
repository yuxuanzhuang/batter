#!/bin/bash

# # AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}
MPI_FLAGS=${MPI_FLAGS:-}
CPPTRAJ_EXEC=${CPPTRAJ_EXEC:-cpptraj}

# Define constants for filenames
PRMTOP="full_merged.prmtop"
PRMTOP_MERGED="full_merged.prmtop"
log_file="run.log"
INPCRD="full.inpcrd"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}
skip_window_eq=${SKIP_WINDOW_EQ:-0}
retry=${RETRY_COUNT:-${RETRY:-}}
rerun_eq_steps_after_failure=${RERUN_EQ_STEPS_AFTER_FAILURE:-0}

# if retry is 5 during equilibration-only runs, use PMEMD_DPFP_EXEC instead of PMEMD_EXEC
if [[ $only_eq -eq 1 && $retry =~ ^[0-9]+$ && $retry -eq 5 ]]; then
    PMEMD_EXEC=${PMEMD_DPFP_EXEC}
fi

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

# Build an MPI launch prefix that works for mpirun or srun.
if [[ -z "${MPI_FLAGS}" ]]; then
    mpi_base=$(echo "${MPI_EXEC}" | awk '{print $1}')
    mpi_base=${mpi_base##*/}
    if [[ "${mpi_base}" == srun* ]]; then
        MPI_FLAGS="-n ${SLURM_JOB_CPUS_PER_NODE:-1}"
    else
        MPI_FLAGS="--oversubscribe -np ${SLURM_JOB_CPUS_PER_NODE:-1}"
    fi
fi
MPI_LAUNCH="${MPI_EXEC} ${MPI_FLAGS}"

source check_run.bash

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    report_progress
    exit 0
fi
rm -f FAILED

prior_failed=$(consume_prior_failure_marker)

should_skip_eq_step() {
    should_skip_completed_step "$1" "$2" "$overwrite" "$prior_failed" "$rerun_eq_steps_after_failure"
}

write_noshake_minimization_input() {
    local src=$1
    local dst=$2
    awk '
        /^[[:space:]]*ntf[[:space:]]*=/ { sub(/=[[:space:]]*[0-9]+,/, "= 1,") }
        /^[[:space:]]*ntc[[:space:]]*=/ { sub(/=[[:space:]]*[0-9]+,/, "= 1,") }
        { print }
    ' "$src" > "$dst"
}

minimization_failed_for_noshake_retry() {
    local out_file=$1
    local rst_file=$2
    local status=${SIM_COMMAND_STATUS:-0}

    if [[ $status =~ ^[0-9]+$ && $status -ne 0 ]]; then
        return 0
    fi
    if [[ -f "$log_file" ]] && grep -Eqi "Coordinate resetting cannot be accomplished|try ntc=1|SHAKE|Calculation halted|Terminated Abnormally|FATAL" "$log_file"; then
        return 0
    fi
    if [[ -f "$out_file" ]] && grep -Eqi "Coordinate resetting cannot be accomplished|try ntc=1|SHAKE|Calculation halted|Terminated Abnormally|FATAL" "$out_file"; then
        return 0
    fi
    if [[ ! -s "$rst_file" ]]; then
        return 0
    fi
    if is_amber_restart_path "$rst_file" && ! amber_restart_is_complete "$rst_file"; then
        return 0
    fi
    return 1
}

run_rbfe_seed_minimization_cuda() {
    local mdin=$1
    print_and_run "$PMEMD_DPFP_EXEC -O -i $mdin -p $PRMTOP_MERGED -c $INPCRD -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref $INPCRD >> \"$log_file\" 2>&1"
}

archive_existing_log_file "$log_file"
cleanup_stale_empty_md_artifacts relaxed
cleanup_zero_frame_md_trajectories "$retry"

report_progress

if [[ $only_eq -eq 1 ]]; then
    if ! should_skip_eq_step "RBFE minimization seed" "mini.in.rst7"; then
        mini_input="mini.in"
        noshake_mini_input="mini_noshake.in"
        run_rbfe_seed_minimization_cuda "$mini_input"
        if minimization_failed_for_noshake_retry "mini.in.out" "mini.in.rst7"; then
            echo "[WARN] RBFE minimization with ntc=2 failed; retrying with ntc=1."
            archive_failed_job_files "$retry" "$log_file" mini.in.rst7
            rm -f "$log_file" mini.in.rst7 mini.in.nc mini.in.out
            write_noshake_minimization_input "$mini_input" "$noshake_mini_input"
            mini_input="$noshake_mini_input"
            run_rbfe_seed_minimization_cuda "$mini_input"
        fi
        check_sim_failure "RBFE minimization seed" "$log_file" mini.in.rst7
        if ! check_min_energy "mini.in.out" -1000; then
            echo "[WARN] CUDA RBFE minimization energy did not pass threshold; continuing from mini.in.rst7 without CPU minimization."
        fi
    fi
    # run one long equilbration with dynamically changed lambda value
    seed_eq_ran=0
    if ! should_skip_eq_step "RBFE equilibration seed" "eq.rst7"; then
        require_nonempty_file_or_attempt_fail "mini.in.rst7" "[ERROR] Missing mini.in.rst7; cannot continue to RBFE equilibration seed."
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP_MERGED -c mini.in.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref mini.in.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration for window $i" "$log_file" eq.rst7
        seed_eq_ran=1
    fi

    # lambda values for EACH EQ frame
    lambda_eq_list=(LAMBDA_EQ_LIST)

    # lambda values for EACH window folder index i
    lambda_set_list=(LAMBDA_SET_LIST)

    # 1) Convert eq.nc to per-frame rst7 files: eq.rst7.1, eq.rst7.2, ...
    if [[ $overwrite -ne 0 || $seed_eq_ran -eq 1 || ($prior_failed -eq 1 && $rerun_eq_steps_after_failure -eq 1) || ! -s eq.rst7.1 ]]; then
        rm -f eq.rst7.[0-9]*
        $CPPTRAJ_EXEC -p $PRMTOP_MERGED -i /dev/stdin <<'EOF'
trajin eq.nc
trajout eq.rst7 multi restart
run
EOF
    fi

    # Find closest index in lambda_eq_list to a target lambda
    closest_index() {
    local target="$1"
    # Print "best_index best_lambda best_absdiff"
    awk -v target="$target" '
        BEGIN { best_i=0; best_d=1e99; best_l=0; }
        {
        l=$1
        d = l - target
        if (d < 0) d = -d
        if (d < best_d) { best_d=d; best_i=NR-1; best_l=l }
        }
        END { printf "%d %.10g %.10g\n", best_i, best_l, best_d }
    ' < <(printf "%s\n" "${lambda_eq_list[@]}")
    }

    # 2) For each window, pick closest EQ lambda frame and copy restart
    for ((i=0; i<NWINDOWS; i++)); do
        win_folder=$(printf "../COMPONENT%02d" "$i")
        lambda_win="${lambda_set_list[$i]}"

        read -r best_i best_l best_d < <(closest_index "$lambda_win")

        # cpptraj "multi" numbering starts at 1 => frame file index = best_i + 1
        frame=$((best_i + 1))
        src="eq.rst7.${frame}"
        dst="${win_folder}/eq_init.rst7"

        if [[ ! -f "$src" ]]; then
            echo "ERROR: missing source restart $src (check eq_init.rst7.* generation)" >&2
            exit 1
        fi
        if should_skip_completed_step "Equilibration for window $i" "${win_folder}/eq.rst7" "$overwrite" "$prior_failed" "$rerun_eq_steps_after_failure"; then
            continue
        fi
        mkdir -p "$win_folder"
        cp -f "$src" "$dst"

        printf "window %02d lambda=%s -> closest_eq_lambda=%s (diff=%s) : %s -> %s\n" \
            "$i" "$lambda_win" "$best_l" "$best_d" "$src" "$dst"
        
        cd "$win_folder"
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP_MERGED -c eq_init.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref eq_init.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration for window $i" "$log_file" eq.rst7
        cd ../COMPONENT-1
    done

    print_and_run "$CPPTRAJ_EXEC -i /dev/stdin >> \"$log_file\" 2>&1 <<'EOF'
parm $PRMTOP
trajin eq.rst7
trajout eq_output.pdb pdb include_ep
run
EOF"

    echo "Only equilibration requested and finished."
    if [[ -s eq_output.pdb ]]; then
        echo "EQ_FINISHED" > EQ_FINISHED
        echo "[INFO] EQ_FINISHED marker written."
        echo "Job completed at $(date)"
    fi
    exit 0
fi

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
production_initial_rst="eq.rst7"
start_ps=$(production_start_ps "$production_start_marker" "$production_initial_rst")
select_valid_md_restart "$production_initial_rst" "$start_ps" "$retry"
rst_in="$SELECTED_MD_RESTART"
require_nonempty_file_or_attempt_fail "$rst_in" "[ERROR] Missing restart file $rst_in; cannot continue."
restart_ps=$(production_restart_ps "$rst_in")
[[ -z $restart_ps ]] && restart_ps=0
current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
[[ -z $current_ps ]] && current_ps=0

echo "Current completed production time: ${current_ps} ps / ${total_ps} ps (restart=${restart_ps} ps, start=${start_ps} ps, dt=${dt_ps} ps)"

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
win_00=../COMPONENT00

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

    # Preflight: must be able to write restart output in this directory
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP_MERGED -c $rst_in -o ${out_tag}.out -r $rst_out -x ${out_tag}.nc -ref ${win_00}/eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "$rst_out" "" "$retry" "${out_tag}.out" "${out_tag}.nc" "$cmass_file"

    # Update production elapsed time from the explicit segment restart.
    restart_ps=$(production_restart_ps "$rst_out")
    [[ -z $restart_ps ]] && restart_ps=0
    current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed production time: ${current_ps} ps / ${total_ps} ps (restart=${restart_ps} ps, start=${start_ps} ps)"

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
