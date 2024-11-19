#!/bin/bash
# cleanup_jobs.sh

REGION="us-central1"
PROJECT_ID="betterdata-441921"

echo "Starting cleanup of unfinished Dataproc jobs..."

# Get all jobs with ACTIVE state
active_jobs=$(gcloud dataproc jobs list \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --state-filter=ACTIVE \
    --format="value(reference.jobId)")

if [ -z "$active_jobs" ]; then
    echo "No active jobs found."
else
    echo "Found active jobs. Terminating..."
    
    for job_id in $active_jobs; do
        echo "Terminating job: $job_id"
        gcloud dataproc jobs kill $job_id \
            --region=${REGION} \
            --project=${PROJECT_ID}
        
        # Wait for job to be killed
        echo "Waiting for job $job_id to terminate..."
        while true; do
            status=$(gcloud dataproc jobs describe $job_id \
                --region=${REGION} \
                --project=${PROJECT_ID} \
                --format="value(status.state)" 2>/dev/null)
            
            if [[ "$status" != "RUNNING" && "$status" != "PENDING" ]]; then
                echo "Job $job_id terminated with status: $status"
                break
            fi
            sleep 2
        done
    done
    
    echo "All active jobs have been terminated"
fi

# Double check for any remaining jobs
remaining_jobs=$(gcloud dataproc jobs list \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --state-filter=ACTIVE \
    --format="value(reference.jobId)")

if [ -z "$remaining_jobs" ]; then
    echo "Cleanup completed successfully."
else
    echo "Warning: Some jobs might still be active. Please check manually."
fi