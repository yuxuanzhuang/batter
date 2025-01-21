#!/usr/bin/env bash 
#SBATCH --job-name=MOR_abfe
#SBATCH --partition=rondror
#SBATCH --time=96:00:00
#SBATCH --output=sim.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --open-mode=append 
#SBATCH --dependency=singleton 

# GPU parameters 
#SBATCH --gres=gpu:0

# conda activate batter
python rbfe_pipeline.py &
#python abfe_pipeline.py &
wait