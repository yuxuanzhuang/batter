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

/scratch/users/yuzhuang/miniforge3_0808/envs/batter_new/bin/batter run /oak/stanford/groups/rondror/users/yuzhuang/software/batter_dev/examples/mabfe_nonmembrane.yaml --on-failure raise
echo 'Job completed.'

