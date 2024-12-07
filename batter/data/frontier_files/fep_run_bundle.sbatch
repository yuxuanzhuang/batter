#!/usr/bin/env bash
#SBATCH -A BIP235
#SBATCH -J FEP_SIM
#SBATCH -o %x-%j.out
#SBATCH -t 0:30:00
#SBATCH -p batch               ##partition (queue) to use
#SBATCH -N 3600
#SBATCH -S 0

#SBATCH --open-mode=append
#SBATCH --dependency=singleton

source ~/env.amber
echo $AMBERHOME

if [ -z "${AMBERHOME}" ]; then echo "AMBERHOME is not set" && exit 0; fi

echo "running FEP simulations"
#

# Number of simulations
num_simulations=1800

# Nodes per simulation
nodes_per_simulation=1

# Run simulations in parallel
for i in $(seq 1 $num_simulations)
do
  cd lig_$i
  echo "Starting simulation $i"
  srun -N 2 -n 16 pmemd.hip_DPFP.MPI -rem 3 -ng 16 -groupfile inputs/ti.groupfile &
  # Allow a brief pause between job submissions to avoid overloading the scheduler
  sleep 0.2
  cd ..
done

# Wait for all background jobs to complete
wait