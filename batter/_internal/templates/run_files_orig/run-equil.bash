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

# ---- load helpers FIRST ----
source check_run.bash

if [[ -f FINISHED ]]; then
    echo "Simulation is complete."
    exit 0
fi

# remove FAILED file if it exists
if [[ -f FAILED ]]; then
    rm -f FAILED
fi

tmpl="mdin-template"
mdin_current="mdin-current"

# sanity check template exists
if [[ ! -f $tmpl ]]; then
    echo "[ERROR] Missing mdin template: $tmpl"
    exit 1
fi

# template-driven MD params
dt_ps=$(parse_dt_ps "$tmpl")
total_steps=$(parse_total_steps "$tmpl")
chunk_steps=$(parse_nstlim "$tmpl")
total_ps=$(awk -v s="$total_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')
chunk_ps=$(awk -v s="$chunk_steps" -v dt="$dt_ps" 'BEGIN{printf "%.6f\n", s*dt}')

# ---------------- Minimization ----------------
if [[ $overwrite -eq 0 && -s eqnpt_appear.rst7 ]]; then
    echo "Skipping EM steps."
else
    print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    check_sim_failure "Minimization" "$log_file" mini.rst7

    print_and_run "$PMEMD_DPFP_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
    check_sim_failure "Minimization 2" "$log_file" mini2.rst7

    if ! check_min_energy "mini2.out" -1000; then
        echo "Minimization not passed with cuda; trying CPU"
        rm -f "$log_file" mini.rst7 mini.nc mini.out mini2.rst7 mini2.nc mini2.out

        if [[ ${SLURM_JOB_CPUS_PER_NODE:-1} -gt 1 ]]; then
            print_and_run "$MPI_EXEC --oversubscribe -np ${SLURM_JOB_CPUS_PER_NODE:-1} $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$MPI_EXEC --oversubscribe -np ${SLURM_JOB_CPUS_PER_NODE:-1} $PMEMD_CPU_MPI_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        else
            print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c $INPCRD -o mini.out -r mini.rst7 -x mini.nc -ref $INPCRD >> \"$log_file\" 2>&1"
            print_and_run "$PMEMD_CPU_EXEC -O -i mini.in -p $PRMTOP -c mini.rst7 -o mini2.out -r mini2.rst7 -x mini2.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        fi

        check_sim_failure "Minimization" "$log_file" mini.rst7
        check_sim_failure "Minimization 2" "$log_file" mini2.rst7

        if ! check_min_energy "mini2.out" -1000; then
            echo "Minimization with CPU also failed, exiting."
            rm -f mini.rst7 mini.nc mini.out mini2.rst7 mini2.nc mini2.out
            exit 1
        fi
    fi
fi

# ---------------- Equilibration ----------------
if [[ $overwrite -eq 0 && -s eqnpt_appear.rst7 ]]; then
    echo "Skipping equilibration steps."
else
    python check_penetration.py mini2.rst7

    if [[ -f RING_PENETRATION ]]; then
        echo "Ligand ring penetration detected previously; using longer equilibration."

        print_and_run "$PMEMD_DPFP_EXEC -O -i eqnvt.in -p $PRMTOP -c mini2.rst7 -o eqnvt.out -r eqnvt.rst7 -x eqnvt.nc -ref $INPCRD >> \"$log_file\" 2>&1"
        check_sim_failure "NVT" "$log_file" eqnvt.rst7

        python check_penetration.py eqnvt.rst7
        if [[ -f RING_PENETRATION ]]; then
            echo "Ligand ring penetration still detected after NVT; exiting."
            exit 1
        fi
    else
        cp mini2.rst7 eqnvt.rst7
    fi

    # Equilibration with protein and ligand restrained (CPU for stability)
    if [[ ${SLURM_JOB_CPUS_PER_NODE:-1} -gt 1 ]]; then
        print_and_run "$MPI_EXEC --oversubscribe -np ${SLURM_JOB_CPUS_PER_NODE:-1} $PMEMD_CPU_MPI_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    else
        print_and_run "$PMEMD_CPU_EXEC -O -i eqnpt0.in -p $PRMTOP -c eqnvt.rst7 -o eqnpt_pre.out -r eqnpt_pre.rst7 -x eqnpt_pre.nc -ref eqnvt.rst7 >> \"$log_file\" 2>&1"
    fi
    check_sim_failure "Pre equilibration" "$log_file" eqnpt_pre.rst7

    # Equilibration with C-alpha restrained
    print_and_run "$PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c eqnpt_pre.rst7 -o eqnpt00.out -r eqnpt00.rst7 -x traj00.nc -ref eqnpt_pre.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "Equilibration stage 0" "$log_file" eqnpt00.rst7

    for step in {1..4}; do
        prev=$(printf "eqnpt%02d.rst7" $((step - 1)))
        curr=$(printf "eqnpt%02d" $step)
        print_and_run "$PMEMD_DPFP_EXEC -O -i eqnpt.in -p $PRMTOP -c $prev -o ${curr}.out -r ${curr}.rst7 -x traj${step}.nc -ref $prev >> \"$log_file\" 2>&1"
        check_sim_failure "Equilibration stage $step" "$log_file" "${curr}.rst7" "$prev"
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

# ---------------- Production MD (progress = restart) ----------------

# current progress (ps) from restart
current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
[[ -z $current_ps ]] && current_ps=0

echo "Current completed time (from restart): $current_ps ps / $total_ps ps (dt=$dt_ps ps)"

# pick previous restart: prefer current md if present, else fall back to eqnpt_appear
rst_in="eqnpt_appear.rst7"
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

# determine current segment index from OUT files (not from time)
seg_idx=$(latest_md_index "md-*.out")
if [[ $seg_idx -lt 0 ]]; then
    seg_idx=0
fi

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

    # first run if no md-*.out exists
    first_run=0
    if [[ $(latest_md_index "md-*.out") -lt 0 ]]; then
        first_run=1
    fi

    out_tag=$(printf "md-%02d" $((seg_idx + 1)))
    echo "[INFO] Running segment $((seg_idx + 1)) -> ${out_tag}.out for ${run_steps} steps (${run_ps} ps); restart_in=$rst_in"

    write_mdin_current "$tmpl" "$run_steps" "$first_run" > "$mdin_current"

    # Preflight: ensure directory is writable (avoid Fortran OPEN failures)
    : > .write_test.$$ 2>/dev/null || {
        echo "[ERROR] Cannot write in $(pwd). Check permissions/quota."
        df -h . || true
        exit 1
    }
    rm -f .write_test.$$

    # archive prior restart if present
    if [[ -f md-current.rst7 ]]; then
        [[ -s md-current.rst7 ]] || { echo "[ERROR] Found md-current.rst7 but empty; aborting."; exit 1; }
        mv -f md-current.rst7 md-previous.rst7
    fi

    print_and_run "$PMEMD_EXEC -O -i $mdin_current -p $PRMTOP -c $rst_in -o ${out_tag}.out -r md-current.rst7 -x ${out_tag}.nc -ref eqnpt04.rst7 >> \"$log_file\" 2>&1"
    check_sim_failure "MD segment $((seg_idx + 1))" "$log_file" "md-current.rst7"

    # Update progress from restart
    current_ps=$(completed_steps "$tmpl" 2>/dev/null | tail -n 1)
    [[ -z $current_ps ]] && current_ps=0
    echo "[INFO] Updated completed time (from restart): $current_ps ps / $total_ps ps"

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
    echo "FAILED" > FAILED
    exit 1
fi
echo "[INFO] Not finished yet; rerun to continue."
exit 0
