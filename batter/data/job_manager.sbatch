#!/usr/bin/env bash 
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=rondror
#SBATCH --time=72:00:00
#SBATCH --output=__JOB_LOG_BASE__-%j.out
#SBATCH --error=__JOB_LOG_BASE__-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4 # use 4 CPUs
#SBATCH --cpus-per-task=1

#SBATCH --open-mode=append 

# GPU parameters 
#SBATCH --gres=gpu:0
#SBATCH --mem=60GB

# print out node information
scontrol show job $SLURM_JOB_ID
