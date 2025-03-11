myname=$(whoami)

# Fetch all eq job IDs with names matching the pattern
job_eq_ids=$(squeue -u "$myname" --format="%A %j" --noheader | grep -E "equil-pose[0-9]" | awk '{print $1}')
# Fetch all fe job IDs with names matching the pattern
job_fe_ids=$(squeue -u "$myname" --format="%A %j" --noheader | grep -E "pose[0-9]-[altmncrevfwx][0-9]{2}" | awk '{print $1}')
job_fe_ids_2=$(squeue -u "$myname" --format="%A %j" --noheader | grep -E "pose[0-9]{2}-[altmncrevfwx][0-9]{2}" | awk '{print $1}')

job_ids="$job_eq_ids $job_fe_ids $job_fe_ids_2"

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
