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
log_file="run.log"
INPCRD="full.inpcrd"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}
skip_window_eq=${SKIP_WINDOW_EQ:-0}
retry=${RETRY_COUNT:-0}

# if retry > 3, use PMEMD_DPFP_EXEC instead of PMEMD_EXEC
if [[ $retry -gt 3 ]]; then
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

if [[ -f FAILED ]]; then
    rm -f FAILED
fi

report_progress

if [[ $only_eq -eq 1 ]]; then
    # Minimization
    # if mini_eq is found use mini_eq.in
    if [[ -f mini_eq.in ]]; then
        echo "Using mini_eq.in for minimization."
    else
        echo "mini_eq.in not found, using mini.in instead."
        cp mini.in mini_eq.in
    fi
    print_and_run "$PMEMD_DPFP_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    check_sim_failure "Minimization" "$log_file" mini.rst7
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini_eq.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    else
        print_and_run "$PMEMD_CPU_EXEC -O -i mini_eq.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    fi
    check_sim_failure "Minimization" "$log_file" mini2.rst7

    if ! check_min_energy "mini2.out" -1000; then
        echo "Minimization not passed with cuda; try CPU"
        rm -f "$log_file"
        rm -f mini.rst7 mini.nc mini.out
        rm -f mini2.rst7 mini2.nc mini2.out
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini_eq.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        else
            print_and_run "$PMEMD_CPU_EXEC -O -i mini_eq.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$PMEMD_CPU_EXEC -O -i mini_eq.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        fi
        check_sim_failure "Minimization" "$log_file" mini.rst7
        check_sim_failure "Minimization" "$log_file" mini2.rst7

        if ! check_min_energy "mini2.out" -1000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out
            rm -f mini2.rst7 mini2.nc mini2.out
            exit 1
        fi
    fi

    if [[ $overwrite -eq 0 && -s eqnpt04.rst7 ]]; then
        echo "Skipping equilibration steps."
    else
        # Equilibration with protein and lipid restrained
        # this is to equilibrate the density of water
        # Note we are not using the GPU version here
        # because for large box size change, an error will be raised.
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP -c mini2.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini2.rst7 >> \"$log_file\" 2>&1"
        else
            print_and_run "$PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP -c mini2.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref mini2.rst7 >> \"$log_file\" 2>&1"
        fi
        check_sim_failure "Pre equilibration" "$log_file" eqnpt_pre.rst7

        # Equilibration with protein restrained
        print_and_run "$PMEMD_EXEC -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration stage 0" "$log_file" eqnpt00.rst7
        for step in {1..4}; do
            prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
            curr=$(printf "eqnpt%02d" $step)
            print_and_run "$PMEMD_EXEC -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> \"$log_file\" 2>&1"
            check_sim_failure "Equilibration stage $step" "$log_file" ${curr}.rst7 $prev $retry
        done
    fi

    # only do it if N_WINDOWS is not 1
    if [[ NWINDOWS -gt 1 ]]; then
        print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Minimization for FEP" "$log_file" mini.in.rst7
        if ! check_min_energy "mini.in.out" -1000; then
            echo "Minimization not passed with cuda; try CPU"
            rm -f "$log_file"
            rm -f mini.in.rst7 mini.in.nc mini.in.out
            if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
                print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
            else
                print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
            fi
            check_sim_failure "Minimization for window $i" "$log_file" mini.in.rst7
            if ! check_min_energy "mini.in.out" -1000; then
                echo "Minimization with CPU also failed for window $i, exiting."
                rm -f mini.in.rst7 mini.in.nc mini.in.out
                exit 1
            fi
        fi
        
        # run one long equilbration with dynamically changed lambda value
        print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP -c mini.in.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref mini.in.rst7 >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration for window $i" "$log_file" eq.rst7

        # lambda values for EACH EQ frame
        lambda_eq_list=(LAMBDA_EQ_LIST)

        # lambda values for EACH window folder index i
        lambda_set_list=(LAMBDA_SET_LIST)

        # 1) Convert eq.nc to per-frame rst7 files: eq.rst7.1, eq.rst7.2, ...
        cpptraj -p full.prmtop -i /dev/stdin <<'EOF'
trajin eq.nc
trajout eq.rst7 multi restart
run
EOF
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
            dst="${win_folder}/eq.rst7"

            if [[ ! -f "$src" ]]; then
                echo "ERROR: missing source restart $src (check eq.rst7.* generation)" >&2
                exit 1
            fi
            mkdir -p "$win_folder"
            cp -f "$src" "$dst"

            printf "window %02d lambda=%s -> closest_eq_lambda=%s (diff=%s) : %s -> %s\n" \
                "$i" "$lambda_win" "$best_l" "$best_d" "$src" "$dst"
        done
    fi

    print_and_run "$CPPTRAJ_EXEC -p $PRMTOP -y eqnpt04.rst7 -x eq_output.pdb >> \"$log_file\" 2>&1"

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

dt_ps=$(parse_dt_ps "$tmpl")
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(parse_nstlim "$tmpl")

# Convert steps -> ps (loop is time-based)
total_ps=$(awk -v s="$total_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')
chunk_ps=$(awk -v s="$chunk_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

# Progress from restart (completed_steps prints a single number to STDOUT)
current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
[[ -z $current_ps ]] && current_ps=0

echo "Current completed time (from restart): ${current_ps} ps / ${total_ps} ps (dt=${dt_ps} ps)"

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

[[ -s "$rst_in" ]] || {
    echo "[ERROR] Missing restart file $rst_in; cannot continue."
    exit 1
}

last_rst="md-current.rst7"

current_steps=$(awk -v t="$current_ps" -v dt="$dt_ps" 'BEGIN{if (dt<=0) {print 0; exit} printf "%d\n", (t/dt)+0.5}')
remaining_steps=$(( total_steps - current_steps ))
if (( remaining_steps < 0 )); then
    remaining_steps=0
fi
remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')
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

    write_mdin_current "$tmpl" "$run_steps" "$first_run" > "$mdin_current"

    # Preflight: must be able to write restart output in this directory
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    # Rotate md-current restart (avoid Fortran OPEN issues / keep backup)
    if [[ -f md-current.rst7 ]]; then
        [[ -s md-current.rst7 ]] || { echo "[ERROR] md-current.rst7 exists but empty; aborting."; exit 1; }
        mv -f md-current.rst7 md-previous.rst7
        if [[ "$rst_in" == "md-current.rst7" ]]; then
            rst_in="md-previous.rst7"
        fi
    fi

    # Run MD: always write restart to md-current.rst7
    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7" "" "$retry" "${out_tag}.out" "${out_tag}.nc"

    # Update progress from restart. completed_steps will:
    # - if latest out has 0 ps, use previous and delete the bad latest.
    current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed time (from restart): ${current_ps} ps / ${total_ps} ps"

    rst_in="md-current.rst7"
    last_rst="md-current.rst7"
fi

if awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur >= tot)}'; then
    print_and_run "$CPPTRAJ_EXEC -p $PRMTOP -y ${last_rst} -x output.pdb >> \"$log_file\" 2>&1"

    if [[ -s output.pdb ]]; then
        echo "FINISHED" > FINISHED
        echo "[INFO] FINISHED marker written."
        exit 0
    fi

    echo "[ERROR] output.pdb not created or empty; marking FAILED."
    exit 1
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
