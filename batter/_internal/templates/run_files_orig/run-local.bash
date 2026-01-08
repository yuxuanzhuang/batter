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
retry=${RETRY_COUNT:-0}

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
    print_and_run "$PMEMD_DPFP_EXEC -O -i mini_eq.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
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

    # run minimization for each windows at this stage
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" $i)
        if [[ -s $win_folder/eq.rst7 ]]; then
            echo "Skipping equilibration for window $i, already exists."
        else
            echo "Running equilibration for window $i"
            cd $win_folder
            print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> \"$log_file\" 2>&1"
            check_sim_failure "Minimization for window $i" "$log_file" mini.in.rst7
            if ! check_min_energy "mini.in.out" -1000; then
                echo "Minimization not passed with cuda; try CPU"
                rm -f "$log_file"
                rm -f mini.in.rst7 mini.in.nc mini.in.out
                if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
                    print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> \"$log_file\" 2>&1"
                else
                    print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/eqnpt04.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/eqnpt04.rst7 >> \"$log_file\" 2>&1"
                fi
                check_sim_failure "Minimization for window $i" "$log_file" mini.in.rst7
                if ! check_min_energy "mini.in.out" -1000; then
                    echo "Minimization with CPU also failed for window $i, exiting."
                    rm -f mini.in.rst7 mini.in.nc mini.in.out
                    exit 1
                fi
            fi
            print_and_run "$PMEMD_EXEC -O -i eq.in -p $PRMTOP -c mini.in.rst7 -o eq.out -r eq.rst7 -x eq.nc -ref mini.in.rst7 >> \"$log_file\" 2>&1"
            check_sim_failure "Equilibration for window $i" "$log_file" eq.rst7
            cd ../COMPONENT-1
        fi
    done

    print_and_run "$CPPTRAJ_EXEC -p $PRMTOP -y eqnpt04.rst7 -x eq_output.pdb >> \"$log_file\" 2>&1"

    echo "Only equilibration requested and finished."
    if [[ -s eq_output.pdb ]]; then
        echo "EQ_FINISHED" > EQ_FINISHED
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

# Progress from OUT only (completed_steps prints a single number to STDOUT)
current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
[[ -z $current_ps ]] && current_ps=0

echo "Current completed time (from OUT): ${current_ps} ps / ${total_ps} ps (dt=${dt_ps} ps)"

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

while awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur < tot)}'; do
    remaining_ps=$(awk -v tot="$total_ps" -v cur="$current_ps" 'BEGIN{printf "%.6f\n", tot-cur}')

    run_ps="$chunk_ps"
    if awk -v rem="$remaining_ps" -v ch="$chunk_ps" 'BEGIN{exit !(rem < ch)}'; then
        run_ps="$remaining_ps"
    fi

    # Convert to steps for nstlim
    run_steps=$(awk -v ps="$run_ps" -v dt="$dt_ps" 'BEGIN{printf "%d\n", ps/dt}')
    (( run_steps > 0 )) || { echo "[ERROR] Computed run_steps=0 (dt=$dt_ps, run_ps=$run_ps)"; exit 1; }

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
    fi

    # Run MD: always write restart to md-current.rst7
    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref eq.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7"

    # Update progress (OUT only). completed_steps will:
    # - if latest out has 0 ps, use previous and delete the bad latest.
    current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed time (from OUT): ${current_ps} ps / ${total_ps} ps"

    # Advance for next loop
    seg_idx=$((seg_idx + 1))
    rst_in="md-current.rst7"
    last_rst="md-current.rst7"
done

print_and_run "$CPPTRAJ_EXEC -p $PRMTOP -y ${last_rst} -x output.pdb >> \"$log_file\" 2>&1"

if [[ -s output.pdb ]]; then
    echo "FINISHED" > FINISHED
    exit 0
fi

echo "[ERROR] output.pdb not created or empty; marking FAILED."
echo "FAILED" > FAILED
exit 1
