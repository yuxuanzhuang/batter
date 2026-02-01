#!/usr/bin/env bash
set -euo pipefail

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}

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

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

if [[ -f FAILED ]]; then
    rm -f FAILED
fi

source check_run.bash

# ------------------------- only_eq mode -------------------------
if [[ $only_eq -eq 1 ]]; then
    # no equilibration needed here; just seed a restart
    cp "$INPCRD" mini.rst7

    # run minimization for each windows at this stage
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" $i)
        if [[ -s $win_folder/eq.rst7 ]]; then
            echo "Skipping equilibration for window $i, already exists."
        else
            echo "Running equilibration for window $i"
            cd $win_folder
            print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/mini.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/mini.rst7 >> \"$log_file\" 2>&1"
            if ! check_min_energy "mini.in.out" -1000; then
                echo "Minimization not passed with cuda; try CPU"
                rm -f "$log_file"
                rm -f mini.in.rst7 mini.in.nc mini.in.out
                if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
                    print_and_run "$MPI_LAUNCH $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/mini.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/mini.rst7 >> \"$log_file\" 2>&1"
                else
                    print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c ../COMPONENT-1/mini.rst7 -o mini.in.out -r mini.in.rst7 -x mini.in.nc -ref ../COMPONENT-1/mini.rst7 >> \"$log_file\" 2>&1"
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

    print_and_run "cpptraj -p $PRMTOP -y mini.rst7 -x eq_output.pdb >> \"$log_file\" 2>&1"

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

dt_ps=$(parse_dt_ps "$tmpl")
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(parse_nstlim "$tmpl")

# Convert steps -> ps for loop control
total_ps=$(awk -v s="$total_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')
chunk_ps=$(awk -v s="$chunk_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

# Progress from restart
current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
[[ -z $current_ps ]] && current_ps=0

echo "Current completed time (from restart): $current_ps ps / $total_ps ps (dt=$dt_ps ps)"

# Determine current segment index from existing OUT files
seg_idx=$(latest_md_index "md-*.out")
if [[ $seg_idx -lt 0 ]]; then
    seg_idx=0
fi

# Choose initial restart input (needed to run, not for progress)
rst_in="eq.rst7"
if [[ -s md-current.rst7 ]]; then
    rst_in="md-current.rst7"
elif [[ -s md-previous.rst7 ]]; then
    rst_in="md-previous.rst7"
fi

if [[ ! -s $rst_in ]]; then
    echo "[ERROR] Missing restart file $rst_in; cannot continue."
    exit 1
fi

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

    # Preflight: ensure output directory writable (avoids Fortran OPEN errors)
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    # Rotate restart outputs
    if [[ -f md-current.rst7 ]]; then
        [[ -s md-current.rst7 ]] || { echo "[ERROR] md-current.rst7 exists but empty; aborting."; exit 1; }
        mv -f md-current.rst7 md-previous.rst7
        if [[ "$rst_in" == "md-current.rst7" ]]; then
            rst_in="md-previous.rst7"
        fi
    fi

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref mini.in.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7" "" "$retry" "${out_tag}.out" "${out_tag}.nc"

    # Update progress from restart
    current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed time (from restart): $current_ps ps / $total_ps ps"

    rst_in="md-current.rst7"
    last_rst="md-current.rst7"
fi

if awk -v cur="$current_ps" -v tot="$total_ps" 'BEGIN{exit !(cur >= tot)}'; then
    print_and_run "cpptraj -p $PRMTOP -y ${last_rst} -x output.pdb >> \"$log_file\" 2>&1"

    # check output.pdb exists to catch cases where the simulation did not run to completion
    if [[ -s output.pdb ]]; then
        echo "FINISHED" > FINISHED
        echo "[INFO] FINISHED marker written."
        exit 0
    fi

    echo "[ERROR] output.pdb not created or empty; marking FAILED."
    echo "FAILED" > FAILED
    exit 1
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
