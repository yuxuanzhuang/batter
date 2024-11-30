for folder in e[0-9]*; do
    if [ -d "$folder" ]; then # Check if it's a directory
        cd "$folder"
        sbatch SLURMM-run
        cd ../
    fi
done

for folder in v[0-9]*; do
    if [ -d "$folder" ]; then # Check if it's a directory
        cd "$folder"
        sbatch SLURMM-run
        cd ../
    fi
done