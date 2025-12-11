#!/bin/bash

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
    rm FAILED
fi

source check_run.bash

if [[ $only_eq -eq 1 ]]; then
    # no eq needed, just copy the INPCRD to mini.in.rst7
    cp $INPCRD mini.rst7
    check_sim_failure "Minimization" "$log_file" mini.rst7

    # run minimization for each windows at this stage
    for i in $(seq 0 $((NWINDOWS - 1))); do
        win_folder=$(printf "../COMPONENT%02d" $i)
        if [[ -s $win_folder/mini.rst7 ]]; then
            echo "Skipping minimization for window $i, already exists."
        else
            echo "Running minimization for window $i"
            cd $win_folder
            cp ../COMPONENT-1/mini.rst7 mini.in.rst7
            cd ../COMPONENT-1
        fi
    done

    print_and_run "cpptraj -p $PRMTOP -y mini.rst7 -x eq_output.pdb >> \"$log_file\" 2>&1"

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

total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(parse_nstlim "$tmpl")
current_steps=$(completed_steps "$tmpl")
echo "Current completed steps: $current_steps / $total_steps"

last_rst="mini.in.rst7"

while [[ $current_steps -lt $total_steps ]]; do
    remaining=$((total_steps - current_steps))
    run_steps=$chunk_steps
    if [[ $remaining -lt $chunk_steps ]]; then
        run_steps=$remaining
    fi

    seg_idx=$(( (current_steps + chunk_steps - 1) / chunk_steps ))

    rst_prev="mini.in.rst7"
    if [[ -s md-current.rst7 ]]; then
        rst_prev="md-current.rst7"
    fi

    if [[ ! -f $rst_prev ]]; then
        echo "[ERROR] Missing restart file $rst_prev; cannot continue."
        exit 1
    fi

    if [[ -f md-current.rst7 ]]; then
        if [[ ! -s md-current.rst7 ]]; then
            echo "[ERROR] Found md-current.rst7 but file is empty; aborting to avoid corrupt restart."
            exit 1
        fi
        mv -f md-current.rst7 md-previous.rst7
        rst_prev="md-previous.rst7"
    fi

    echo "[INFO] Using restart $rst_prev -> md-current.rst7 for segment $((seg_idx + 1))"

    write_mdin_current "$tmpl" "$run_steps" $((current_steps == 0 ? 1 : 0)) > "$mdin_current"

    out_tag=$(printf "md-%02d" "$((seg_idx + 1))")
    rst_out="md-current.rst7"

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_prev -o ${out_tag}.out -r $rst_out -x ${out_tag}.nc -ref mini.in.rst7 -AllowSmallBox >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "$rst_out"

    current_steps=$((current_steps + run_steps))
    last_rst="$rst_out"
done

print_and_run "cpptraj -p $PRMTOP -y ${last_rst} -x output.pdb >> \"$log_file\" 2>&1"

# check output.pdb exists
# to catch cases where the simulation did not run to completion
if [[ -s output.pdb ]]; then
    echo "FINISHED" > FINISHED
    exit 0
fi
