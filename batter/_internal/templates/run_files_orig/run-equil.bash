#!/usr/bin/env bash
set -euo pipefail

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}
CPPTRAJ_EXEC=${CPPTRAJ_EXEC:-cpptraj}

BATTER_SOURCE_ROOT="${BATTER_SOURCE_ROOT:-__BATTER_SOURCE_ROOT__}"
if [[ -d "$BATTER_SOURCE_ROOT/batter" ]]; then
    export PYTHONPATH="$BATTER_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

# Constants
PRMTOP="full.hmr.prmtop"
PRMTOP_MERGED="full_merged.prmtop"
INPCRD="full.inpcrd"
log_file="run.log"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}
retry_count=${RETRY_COUNT:-${RETRY:-}}
if [[ -n ${RERUN_EQ_STEPS_AFTER_FAILURE+x} ]]; then
    rerun_eq_steps_after_failure=${RERUN_EQ_STEPS_AFTER_FAILURE}
else
    rerun_eq_steps_after_failure=auto
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
    return 0
}

# ---- load helpers FIRST ----
source check_run.bash

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi
if [[ $rerun_eq_steps_after_failure != 1 ]]; then
    rm -f FAILED
fi

prior_failed=$(consume_prior_failure_marker)

if [[ $rerun_eq_steps_after_failure == 1 ]]; then
    prior_failed=1
elif [[ $rerun_eq_steps_after_failure == auto ]]; then
    rerun_eq_steps_after_failure=0
    if [[ $prior_failed -eq 1 ]]; then
        echo "[INFO] Prior failure marker found; preserving completed equilibration stages."
    elif [[ $retry_count =~ ^[0-9]+$ && $retry_count -gt 1 ]]; then
        echo "[INFO] Retry attempt ${retry_count} detected; preserving completed equilibration stages."
    fi
fi

should_skip_eq_step() {
    should_skip_completed_step "$1" "$2" "$overwrite" "$prior_failed" "$rerun_eq_steps_after_failure"
}

pre_equil_restart_is_complete() {
    [[ -s eqnpt_pre.rst7 ]] || return 1
    if is_amber_restart_path "eqnpt_pre.rst7" && ! amber_restart_is_complete "eqnpt_pre.rst7"; then
        return 1
    fi
    return 0
}

reset_minimization_after_failed_pre_equil() {
    if [[ $prior_failed -ne 1 ]]; then
        return 0
    fi
    if pre_equil_restart_is_complete; then
        return 0
    fi
    if [[ -s mini.rst7 || -s mini2.rst7 || -s eqnvt.rst7 ]]; then
        echo "[INFO] Prior failure occurred before Pre equilibration completed; rerunning minimization instead of reusing mini.rst7/mini2.rst7; rerunning minimization/NVT prep instead of reusing mini.rst7/mini2.rst7/eqnvt.rst7."
        rm -f mini.rst7 mini.out mini.nc mini_noshake.in mini2.rst7 mini2.out eqnvt.rst7 eqnvt.out eqnvt.nc
    fi
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

run_minimization_cuda() {
    local mdin=$1
    local out_file=$2
    local rst_file=$3
    local nc_file=$4
    local coord=$5
    print_and_run "$PMEMD_DPFP_EXEC -O -i $mdin -p $PRMTOP -c $coord -o $out_file -r $rst_file -x $nc_file -ref $INPCRD >> \"$log_file\" 2>&1"
}

run_penetration_check() {
    local rst_path=$1
    shift || true
    local err_file=".penetration_check.err"
    local status=0
    local errexit_was_on=0
    local action="Checking"
    if [[ " $* " == *" --repair "* ]]; then
        action="Checking and repairing"
    fi
    echo "[INFO] ${action} ligand ring penetration in ${rst_path}."
    case $- in
        *e*) errexit_was_on=1 ;;
    esac
    set +e
    python check_penetration.py "$@" "$rst_path" 2>"$err_file"
    status=$?
    if [[ $errexit_was_on -eq 1 ]]; then
        set -e
    fi
    if [[ $status -ne 0 ]]; then
        if grep -Eq "ModuleNotFoundError: No module named '(MDAnalysis|networkx|parmed|batter)'" "$err_file"; then
            cat "$err_file" >&2
            rm -f "$err_file"
            mark_failed_and_exit "[ERROR] Ring penetration check could not run; missing BATTER Python deps (MDAnalysis/networkx/parmed/batter)."
        fi
        cat "$err_file" >&2
        rm -f "$err_file"
        mark_failed_and_exit "[ERROR] Ring penetration check failed for ${rst_path}."
    fi
    rm -f "$err_file"
    if [[ -f RING_PENETRATION ]]; then
        echo "[INFO] Ligand ring penetration marker present for ${rst_path}."
    else
        echo "[INFO] No ligand ring penetration detected in ${rst_path}."
    fi
}

archive_existing_log_file "$log_file"
cleanup_stale_empty_md_artifacts relaxed
cleanup_zero_frame_md_trajectories "$retry_count"

tmpl="mdin-template"
mdin_current="mdin-current"

# sanity check template exists
if [[ ! -f $tmpl ]]; then
    echo "[ERROR] Missing mdin template: $tmpl"
    exit 1
fi

# template-driven MD params
apply_retry_dt_reduction "$tmpl" "$retry_count" 0.001 "production startup"

reset_minimization_after_failed_pre_equil

dt_ps=$(parse_dt_ps "$tmpl")
target_dt_ps=$(parse_target_dt_ps "$tmpl")
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(scaled_nstlim_for_dt "$tmpl" "$dt_ps")
total_ps=$(awk -v s="$total_steps" -v dt="$target_dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')
chunk_ps=$(awk -v s="$chunk_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

# ---------------- Minimization ----------------
mini_input="mini.in"
noshake_mini_input="mini_noshake.in"
if ! should_skip_eq_step "Minimization" "mini.rst7"; then
    run_minimization_cuda "$mini_input" "mini.out" "mini.rst7" "mini.nc" "$INPCRD"
    if minimization_failed_for_noshake_retry "mini.out" "mini.rst7"; then
        echo "[WARN] Minimization with ntf=2, ntc=2 failed; retrying with ntf=1, ntc=1."
        archive_failed_job_files "$retry_count" "$log_file" mini.rst7
        rm -f "$log_file" mini.rst7 mini.nc mini.out
        write_noshake_minimization_input "$mini_input" "$noshake_mini_input"
        mini_input="$noshake_mini_input"
        run_minimization_cuda "$mini_input" "mini.out" "mini.rst7" "mini.nc" "$INPCRD"
    fi
    check_sim_failure "Minimization" "$log_file" mini.rst7

    if ! check_min_energy "mini.out" -1000; then
        echo "[WARN] CUDA minimization energy did not pass threshold; continuing from mini.rst7 without CPU minimization."
    fi
else
    if [[ -f "$noshake_mini_input" ]]; then
        mini_input="$noshake_mini_input"
    fi
fi

if ! should_skip_eq_step "Minimization 2" "mini2.rst7"; then
    require_nonempty_file_or_attempt_fail "mini.rst7" "[ERROR] Missing mini.rst7; cannot continue to Minimization 2."
    echo "[INFO] Skipping CPU Minimization 2; continuing from CUDA minimization restart."
    cp mini.rst7 mini2.rst7
    printf "Skipped CPU Minimization 2; copied mini.rst7 to mini2.rst7.\n" > mini2.out
fi

# ---------------- Equilibration ----------------
if ! should_skip_eq_step "NVT preparation" "eqnvt.rst7"; then
    require_nonempty_file_or_attempt_fail "mini2.rst7" "[ERROR] Missing mini2.rst7; cannot continue to NVT preparation."
    rm -f RING_PENETRATION_REPAIRED
    run_penetration_check "mini2.rst7"

    if [[ -f RING_PENETRATION ]]; then
        echo "Ligand ring penetration detected after minimization; attempting local repair before NVT."
        run_penetration_check "mini2.rst7" --repair
        run_penetration_check "mini2.rst7"
    fi
    if [[ -f RING_PENETRATION ]]; then
        mark_failed_and_exit "Ligand ring penetration still detected after mini2.rst7 repair; exiting."
    fi

    if [[ -f RING_PENETRATION_REPAIRED ]]; then
        echo "Ligand ring penetration repaired before NVT; running NVT equilibration."

        print_and_run "$PMEMD_DPFP_EXEC -O -i eqnvt.in -p $PRMTOP_MERGED -c mini2.rst7 -o eqnvt.out -r eqnvt.rst7 -x eqnvt.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        check_sim_failure "NVT" "$log_file" eqnvt.rst7

        run_penetration_check "eqnvt.rst7"
        if [[ -f RING_PENETRATION ]]; then
            echo "Ligand ring penetration still detected after NVT; attempting local repair."
            run_penetration_check "eqnvt.rst7" --repair
            run_penetration_check "eqnvt.rst7"
        fi
        if [[ -f RING_PENETRATION ]]; then
            mark_failed_and_exit "Ligand ring penetration still detected after NVT repair; exiting."
        fi
    else
        cp mini2.rst7 eqnvt.rst7
    fi
fi

if ! should_skip_eq_step "Pre equilibration" "eqnpt_pre.rst7"; then
    require_nonempty_file_or_attempt_fail "eqnvt.rst7" "[ERROR] Missing eqnvt.rst7; cannot continue to Pre equilibration."
    # Equilibration with protein and ligand restrained (CPU for stability)
    if [[ ${SLURM_JOB_CPUS_PER_NODE:-1} -gt 1 ]]; then
        print_and_run "$MPI_EXEC --oversubscribe -np ${SLURM_JOB_CPUS_PER_NODE:-1} $PMEMD_CPU_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP_MERGED -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    else
        print_and_run "$PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP_MERGED -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    fi
    check_sim_failure "Pre equilibration" "$log_file" eqnpt_pre.rst7
fi

if ! should_skip_eq_step "Equilibration stage 0" "eqnpt00.rst7"; then
    require_nonempty_file_or_attempt_fail "eqnpt_pre.rst7" "[ERROR] Missing eqnpt_pre.rst7; cannot continue to Equilibration stage 0."
    # Equilibration with C-alpha restrained
    print_and_run "$PMEMD_DPFP_EXEC -O -i eqnpt0.in -p $PRMTOP_MERGED -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration stage 0" "$log_file" eqnpt00.rst7
fi

for step in {1..4}; do
    prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
    curr=$(printf "eqnpt%02d" $step)
    if should_skip_eq_step "Equilibration stage $step" "${curr}.rst7"; then
        continue
    fi
    require_nonempty_file_or_attempt_fail "$prev" "[ERROR] Missing ${prev}; cannot continue to Equilibration stage $step."
    print_and_run "$PMEMD_EXEC -O -i eqnpt.in -p $PRMTOP_MERGED -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration stage $step" "$log_file" "${curr}.rst7" "$prev"
done

if ! should_skip_eq_step "Long equilibration" "eqnpt_eq.rst7"; then
    require_nonempty_file_or_attempt_fail "eqnpt04.rst7" "[ERROR] Missing eqnpt04.rst7; cannot continue to Long equilibration."
    print_and_run "$PMEMD_EXEC -O -i eqnpt_eq.in -p $PRMTOP_MERGED -c eqnpt04.rst7 -o eqnpt_eq.out -r eqnpt_eq.rst7 -x eqnpt_eq.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Long equilibration" "$log_file" eqnpt_eq.rst7
fi

if ! should_skip_eq_step "Equilibration disappear" "eqnpt_disappear.rst7"; then
    require_nonempty_file_or_attempt_fail "eqnpt_eq.rst7" "[ERROR] Missing eqnpt_eq.rst7; cannot continue to Equilibration disappear."
    print_and_run "$PMEMD_EXEC -O -i eqnpt_disappear.in -p $PRMTOP_MERGED -c eqnpt_eq.rst7 -o eqnpt_disappear.out -r eqnpt_disappear.rst7 -x eqnpt_disappear.nc -ref eqnpt_eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration disappear" "$log_file" eqnpt_disappear.rst7
fi

if ! should_skip_eq_step "Equilibration appear" "eqnpt_appear.rst7"; then
    require_nonempty_file_or_attempt_fail "eqnpt_disappear.rst7" "[ERROR] Missing eqnpt_disappear.rst7; cannot continue to Equilibration appear."
    print_and_run "$PMEMD_EXEC -O -i eqnpt_appear.in -p $PRMTOP_MERGED -c eqnpt_disappear.rst7 -o eqnpt_appear.out -r eqnpt_appear.rst7 -x eqnpt_appear.nc -ref eqnpt_eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration appear" "$log_file" eqnpt_appear.rst7 0 "eqnpt_appear.rst7" "eqnpt_appear.nc"
fi

if [[ $only_eq -eq 1 ]]; then
    echo "Only equilibration requested."
    exit 0
fi

# ---------------- Production MD (progress = elapsed production time) ----------------

# current progress (ps) from rolling restart, minus the initial production restart time
production_start_marker="production-start.ps"
production_initial_rst="eqnpt_appear.rst7"
start_ps=$(production_start_ps "$production_start_marker" "$production_initial_rst")
select_valid_md_restart "$production_initial_rst" "$start_ps" "$retry_count"
rst_in="$SELECTED_MD_RESTART"
require_nonempty_file_or_attempt_fail "$rst_in" "[ERROR] Missing restart file $rst_in; cannot continue."
restart_ps=$(production_restart_ps)
[[ -z $restart_ps ]] && restart_ps=0
current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
[[ -z $current_ps ]] && current_ps=0

echo "Current completed production time: $current_ps ps / $total_ps ps (restart=$restart_ps ps, start=$start_ps ps, dt=$dt_ps ps)"

last_rst="$rst_in"

# determine current segment index from OUT files (not from time)
seg_idx=$(latest_md_index "md-*.out")
if [[ $seg_idx -lt 0 ]]; then
    seg_idx=0
fi

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
    cmass_file=$(printf "cmass-%02d.txt" $((seg_idx + 1)))
    echo "[INFO] Running segment $((seg_idx + 1)) -> ${out_tag}.out for ${run_steps} steps (${run_ps} ps); restart_in=$rst_in"

    write_mdin_current "$tmpl" "$run_steps" "$first_run" "$mdin_current" "$retry_count" "$start_ps" "$cmass_file" > "$mdin_current"

    # Preflight: ensure directory is writable (avoid Fortran OPEN failures)
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    # archive prior restart if present
    if [[ -f md-current.rst7 ]]; then
        require_nonempty_file_or_attempt_fail "md-current.rst7" "[ERROR] Found md-current.rst7 but empty; aborting."
        mv -f md-current.rst7 md-previous.rst7
        if [[ "$rst_in" == "md-current.rst7" ]]; then
            rst_in="md-previous.rst7"
        fi
    fi

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7" "" "$retry_count" "${out_tag}.out" "${out_tag}.nc" "$cmass_file"

    # Update production elapsed time from the rolling restart.
    restart_ps=$(production_restart_ps)
    [[ -z $restart_ps ]] && restart_ps=0
    current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed production time: $current_ps ps / $total_ps ps (restart=$restart_ps ps, start=$start_ps ps)"

    rst_in="md-current.rst7"
    last_rst="md-current.rst7"
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
        echo "FINISHED" > FINISHED
        echo "[INFO] FINISHED marker written."
        exit 0
    fi

    mark_failed_and_exit "[ERROR] output.pdb not created or empty; marking ATTEMPT_FAILED."
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
