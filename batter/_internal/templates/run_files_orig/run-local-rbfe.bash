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
PRMTOP="full.hmr.prmtop"
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
    eval "$@"
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

prior_failed=$(consume_prior_failure_marker)

should_skip_eq_step() {
    should_skip_completed_step "$1" "$2" "$overwrite" "$prior_failed" "$rerun_eq_steps_after_failure"
}

archive_existing_log_file "$log_file"
cleanup_stale_empty_md_artifacts

report_progress

if [[ $only_eq -eq 1 ]]; then
    if ! should_skip_eq_step "RBFE minimization seed" "mini.in.rst7"; then
        print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        if ! check_min_energy "mini.in.out" -1000; then
            echo "Minimization not passed with cuda; try CPU"
            rm -f "$log_file"
            rm -f mini.in.rst7 mini.in.nc mini.in.out
            if [[ ${SLURM_JOB_CPUS_PER_NODE:-1} -gt 1 ]]; then
                print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            else
                print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            fi
            check_sim_failure "Minimization for window $i" "$log_file" mini.in.rst7
            if ! check_min_energy "mini.in.out" -1000; then
                echo "Minimization with CPU also failed for window $i, exiting."
                rm -f mini.in.rst7 mini.in.nc mini.in.out
                mark_failed_and_exit
            fi
        fi
    fi
    # run one long equilbration with dynamically changed lambda value
    if ! should_skip_eq_step "RBFE equilibration seed" "eq.rst7"; then
        require_nonempty_file_or_attempt_fail "mini.in.rst7" "[ERROR] Missing mini.in.rst7; cannot continue to RBFE equilibration seed."
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP_MERGED -c mini.in.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref mini.in.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration for window $i" "$log_file" eq.rst7
    fi

    # lambda values for EACH EQ frame
    lambda_eq_list=(LAMBDA_EQ_LIST)

    # lambda values for EACH window folder index i
    lambda_set_list=(LAMBDA_SET_LIST)

    # 1) Convert eq.nc to per-frame rst7 files: eq.rst7.1, eq.rst7.2, ...
    if [[ $overwrite -ne 0 || ($prior_failed -eq 1 && $rerun_eq_steps_after_failure -eq 1) || ! -s eq.rst7.1 ]]; then
        $CPPTRAJ_EXEC -p full.prmtop -i /dev/stdin <<'EOF'
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
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP -c eq_init.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref eq_init.rst7 >> \"$log_file\" 2>&1"
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
restart_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
[[ -z $restart_ps ]] && restart_ps=0
current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
[[ -z $current_ps ]] && current_ps=0

echo "Current completed production time: ${current_ps} ps / ${total_ps} ps (restart=${restart_ps} ps, start=${start_ps} ps, dt=${dt_ps} ps)"

# Determine current segment index from existing OUT files
seg_idx=$(latest_md_index "md-*.out")
if [[ $seg_idx -lt 0 ]]; then
    seg_idx=0
fi

# Choose restart input (needed to run; NOT used for progress)
rst_in="eq.rst7"
if [[ -s md-current.rst7 ]]; then
    rst_in="md-current.rst7"
elif [[ -s md-previous.rst7 ]]; then
    rst_in="md-previous.rst7"
fi

require_nonempty_file_or_attempt_fail "$rst_in" "[ERROR] Missing restart file $rst_in; cannot continue."

last_rst="md-current.rst7"
win_00=../COMPONENT00

remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')
remaining_steps=$(remaining_steps_from_time "$total_ps" "$current_ps" "$dt_ps")
if awk -v tot="$total_ps" -v rem="$remaining_ps" 'BEGIN{exit !(tot>=100 && rem<=100)}'; then
    remaining_steps=0
    current_ps="$total_ps"
fi

if (( remaining_steps > 0 )); then
    run_steps=$remaining_steps
    if (( run_steps > chunk_steps )); then
        run_steps=$chunk_steps
    fi
    run_ps=$(awk -v s="$run_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

    # first_run if no md-*.out exists yet
    first_run=0
    if [[ $(latest_md_index "md-*.out") -lt 0 ]]; then
        first_run=1
    fi

    out_tag=$(printf "md-%02d" $((seg_idx + 1)))
    echo "[INFO] Running segment $((seg_idx + 1)) -> ${out_tag}.out for ${run_steps} steps (${run_ps} ps); restart_in=$rst_in"

    write_mdin_current "$tmpl" "$run_steps" "$first_run" "$mdin_current" > "$mdin_current"

    # Preflight: must be able to write restart output in this directory
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    # Rotate md-current restart (avoid Fortran OPEN issues / keep backup)
    if [[ -f md-current.rst7 ]]; then
        require_nonempty_file_or_attempt_fail "md-current.rst7" "[ERROR] md-current.rst7 exists but empty; aborting."
        mv -f md-current.rst7 md-previous.rst7
        if [[ "$rst_in" == "md-current.rst7" ]]; then
            rst_in="md-previous.rst7"
        fi
    fi

    # Run MD: always write restart to md-current.rst7
    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP_MERGED -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref ${win_00}/eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7" "" "$retry" "${out_tag}.out" "${out_tag}.nc"

    # Update production elapsed time from the rolling restart.
    restart_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
    [[ -z $restart_ps ]] && restart_ps=0
    current_ps=$(production_elapsed_ps "$restart_ps" "$start_ps")
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed production time: ${current_ps} ps / ${total_ps} ps (restart=${restart_ps} ps, start=${start_ps} ps)"

    rst_in="md-current.rst7"
    last_rst="md-current.rst7"
fi

if awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur >= tot)}'; then
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
