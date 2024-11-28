for folder in m[0-9]*; do
    if [ -d "$folder" ]; then # Check if it's a directory
        cd "$folder"
        sbatch SLURMM-run
        cd ../
    fi
done

for folder in n[0-9]*; do
    if [ -d "$folder" ]; then # Check if it's a directory
        cd "$folder"
        sbatch SLURMM-run
        cd ../
    fi
done