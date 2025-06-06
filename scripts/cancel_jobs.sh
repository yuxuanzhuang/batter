myname=$(whoami)

# Fetch all eq job IDs with names matching the pattern
job_ids=$(squeue -u "$myname" --format="%A %j" --noheader | grep -E "fep" | awk '{print $1}')

# Check if any jobs were found
if [[ -z "$job_ids" ]]; then
    echo "No jobs matching the pattern were found."
    exit 0
fi

# Cancel each job
for job_id in $job_ids; do
    scancel "$job_id"
    echo "Canceled job ID: $job_id"
done

echo "All matching jobs have been canceled."
