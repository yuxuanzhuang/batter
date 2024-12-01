#!/usr/bin/env bash
#SBATCH -A BIP235
#SBATCH -J FEP_SIM
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch               ##partition (queue) to use
#SBATCH -N NUM_NODES           ##Number of nodes
#SBATCH -S 0

#SBATCH --open-mode=append
#SBATCH --dependency=singleton

source ~/env.amber
echo $AMBERHOME

if [ -z "${AMBERHOME}" ]; then echo "AMBERHOME is not set" && exit 0; fi