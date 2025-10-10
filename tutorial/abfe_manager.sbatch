#!/usr/bin/env bash 
#SBATCH --job-name=abfe_manager
#SBATCH --partition=rondror
#SBATCH --time=72:00:00
#SBATCH --output=abfe_manager-%j.out
#SBATCH --error=abfe_manager-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4 # use 4 CPUs
#SBATCH --cpus-per-task=1

#SBATCH --open-mode=append 

# GPU parameters 
#SBATCH --gres=gpu:0
#SBATCH --mem=60GB

# print out node information
scontrol show job $SLURM_JOB_ID

# usage
# modify the input json, replicate name, only-equil flag as needed
python pipeline.py \
  --input INPUT.json \
  --param abfe.in \
  --replicate REP_XXX \
  --ligand-ff openff-2.2.1 \
  --only-equil