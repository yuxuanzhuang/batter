#!/usr/bin/env bash
#SBATCH -A BIP235
#SBATCH -J FEP_SIM_XXX
#SBATCH -o %x.out
#SBATCH -t 00:30:00
#SBATCH -p batch               ##partition (queue) to use
#SBATCH -N NUM_NODES           ##Number of nodes
#SBATCH -S 0

#SBATCH --open-mode=append
#SBATCH --dependency=singleton

source ~/env.amber > /dev/null 2>&1
echo $AMBERHOME

if [ -z "${AMBERHOME}" ]; then echo "AMBERHOME is not set" && exit 0; fi