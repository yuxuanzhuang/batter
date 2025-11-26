#SBATCH --dummy

/Users/scottzhuang/mambaforge/envs/batter_env/bin/python3.12 -m batter.cli run /private/var/folders/fn/_k4nj2nd6zv85t7_r3417nb40000gn/T/pytest-of-scottzhuang/pytest-378/test_cli_run_slurm_submit_uses0/run.yaml --on-failure raise
echo 'Job completed.'

