#!/bin/bash

# AMBER Constants
PMEMD_EXEC=${PMEMD_EXEC:-pmemd.cuda}
PMEMD_CPU_MPI_EXEC=${PMEMD_CPU_MPI_EXEC:-pmemd.MPI}
PMEMD_DPFP_EXEC=${PMEMD_DPFP_EXEC:-pmemd.cuda_DPFP}
PMEMD_CPU_EXEC=${PMEMD_CPU_EXEC:-pmemd}
SANDER_EXEC=${SANDER_EXEC:-sander}
MPI_EXEC=${MPI_EXEC:-mpirun}

# Constants
PRMTOP="full.hmr.prmtop"
INPCRD="full.inpcrd"
log_file="run.log"
overwrite=${OVERWRITE:-0}
only_eq=${ONLY_EQ:-0}

# Echo commands before executing them so the full invocation is visible
print_and_run() {
    echo "$@"
    eval "$@"
}

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

# remove FAILED file if it exists
if [[ -f FAILED ]]; then
    rm FAILED
fi

source check_run.bash

tmpl="mdin-template"
mdin_current="mdin-current"

# sanity check template exists
if [[ ! -f $tmpl ]]; then
    echo "[ERROR] Missing mdin template: $tmpl"
    exit 1
fi

# prepare template-driven MD input on the fly
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(parse_nstlim "$tmpl")

if [[ $overwrite -eq 0 && -s eqnpt_appear.rst7 ]]; then
    echo "Skipping EM steps." 
else
    print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    check_sim_failure "Minimization" "$log_file" mini.rst7
    print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    check_sim_failure "Minimization 2" "$log_file" mini2.rst7

    if ! check_min_energy "mini2.out" -1000; then
        echo "Minimization not passed with cuda; trying CPU"
        rm -f "$log_file"
        rm -f mini.rst7 mini.nc mini.out
        rm -f mini2.rst7 mini2.nc mini2.out
        if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
            print_and_run "$MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        else
            print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        fi
        check_sim_failure "Minimization" "$log_file" mini.rst7
        check_sim_failure "Minimization 2" "$log_file" mini2.rst7
        if ! check_min_energy "mini2.out" -1000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out
            rm -f mini2.rst7 mini2.nc mini2.out
            exit 1
        fi
    fi

fi
if [[ $overwrite -eq 0 && -s eqnpt_appear.rst7 ]]; then
    echo "Skipping equilibration steps."
else
    python check_penetration.py mini2.rst7

    # if RING_PENETRATION file is found
    if [[ -f RING_PENETRATION ]]; then
        echo "Ligand ring penetration detected previously; using longer equilibration."

        # Equilbration with gradually-incrase lambda of the ligand
        # this can fix issues e.g. ligand entaglement https://pubs.acs.org/doi/10.1021/ct501111d
        print_and_run "$PMEMD_DPFP_EXEC -O -i eqnvt.in -p $PRMTOP -c mini2.rst7 -o eqnvt.out -r eqnvt.rst7 -x eqnvt.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        check_sim_failure "NVT" "$log_file" eqnvt.rst7
        python check_penetration.py eqnvt.rst7
        if [[ -f RING_PENETRATION ]]; then
            echo "Ligand ring penetration still detected after NVT; exiting."
            exit 1
        fi
    else
        # no NVT needed
        cp mini2.rst7 eqnvt.rst7
    fi

    # Equilibration with protein and ligand restrained
    # this is to equilibrate the density of water
    if [[ $SLURM_JOB_CPUS_PER_NODE -gt 1 ]]; then
        print_and_run "$MPI_EXEC --oversubscribe -np $SLURM_JOB_CPUS_PER_NODE $PMEMD_CPU_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    else
        print_and_run "$PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    fi
    check_sim_failure "Pre equilibration" "$log_file" eqnpt_pre.rst7

    # Equilibration with C-alpha restrained
    print_and_run "$PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration stage 0" "$log_file"
    for step in {1..4}; do
        prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
        curr=$(printf "eqnpt%02d" $step)
        print_and_run "$PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration stage $step" "$log_file"
    done

    # Additional disappear/appear equilibration steps
    print_and_run "$PMEMD_EXEC -O -i eqnpt_disappear.in -p $PRMTOP -c eqnpt04.rst7 -o eqnpt_disappear.out -r eqnpt_disappear.rst7 -x eqnpt_disappear.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration disappear" "$log_file" eqnpt_disappear.rst7

    print_and_run "$PMEMD_EXEC -O -i eqnpt_appear.in -p $PRMTOP -c eqnpt_disappear.rst7 -o eqnpt_appear.out -r eqnpt_appear.rst7 -x eqnpt_appear.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration appear" "$log_file" eqnpt_appear.rst7
fi
if [[ $only_eq -eq 1 ]]; then
    echo "Only equilibration requested."
    exit 0
fi

current_steps=$(completed_steps "$tmpl")
echo "Current completed steps: $current_steps / $total_steps"

last_rst="eqnpt_appear.rst7"

while [[ $current_steps -lt $total_steps ]]; do
    remaining=$((total_steps - current_steps))
    run_steps=$chunk_steps
    if [[ $remaining -lt $chunk_steps ]]; then
        run_steps=$remaining
    fi

    seg_idx=$(( (current_steps + chunk_steps - 1) / chunk_steps ))

    # pick previous restart: prefer current md if present, else fall back to eqnpt_appear
    rst_prev="eqnpt_appear.rst7"
    if [[ -s md-current.rst7 ]]; then
        rst_prev="md-current.rst7"
    fi

    if [[ ! -f $rst_prev ]]; then
        echo "[ERROR] Missing restart file $rst_prev; cannot continue."
        exit 1
    fi

    # archive prior restart if present
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

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_prev -o ${out_tag}.out -r $rst_out -x ${out_tag}.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "$rst_out"

    current_steps=$((current_steps + run_steps))
    last_rst="$rst_out"
done

print_and_run "cpptraj -p $PRMTOP -y ${last_rst} -x output.pdb >> \"$log_file\" 2>&1"

# check output.pdb exists
# to catch cases where the simulation did not run to completion
if [[ -s output.pdb ]]; then
    echo "EQFINISHED" > FINISHED
    exit 0
fi
