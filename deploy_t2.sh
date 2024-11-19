#!/bin/bash

# Configuration
PROJECT_ID="betterdata-441921"
BUCKET_NAME="gcs_buck_1"
REGION="us-central1"
CLUSTER_NAME="gmm-cluster-local"
INPUT_PATH="gs://${BUCKET_NAME}/data/scaled_data"
OUTPUT_PATH="gs://${BUCKET_NAME}/data/transformed_data"

# Function to cleanup jobs
cleanup_jobs() {
    echo "Cleaning up any existing jobs..."
    active_jobs=$(gcloud dataproc jobs list \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --filter="status.state=RUNNING OR status.state=PENDING OR status.state=SETUP_DONE" \
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
            echo "Job $job_id terminated"
        done
        
        echo "All active jobs have been terminated"
        sleep 10
    fi
}

echo "Starting VGM deployment process..."

# Clean up existing jobs before starting
cleanup_jobs

# Create GCS bucket if it doesn't exist
if ! gsutil ls -b gs://${BUCKET_NAME} > /dev/null 2>&1; then
    echo "Creating GCS bucket..."
    gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${BUCKET_NAME}
fi

# Package the source code
echo "Packaging source code..."
zip -r vgm_src.zip src/

# Upload code to GCS
echo "Uploading code to GCS..."
gsutil -m cp vgm_src.zip gs://${BUCKET_NAME}/code/
gsutil -m cp src/vgm_main.py gs://${BUCKET_NAME}/code/

# Create Dataproc cluster if it doesn't exist
if ! gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${REGION} > /dev/null 2>&1; then
    echo "Creating Dataproc cluster..."
    gcloud dataproc clusters create ${CLUSTER_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --zone=${REGION}-a \
        --master-machine-type=n1-standard-8 \
        --master-boot-disk-size=500GB \
        --num-workers=10 \
        --worker-machine-type=n1-standard-8 \
        --worker-boot-disk-size=500GB \
        --image-version=2.0-debian10 \
        --properties="spark:spark.driver.memory=4g,spark:spark.executor.memory=8g,spark:spark.executor.cores=4,spark:spark.sql.adaptive.enabled=true,spark:spark.dynamicAllocation.enabled=true"
fi

# Submit PySpark job and capture the job ID
echo "Submitting VGM processing job..."
JOB_ID=$(gcloud dataproc jobs submit pyspark \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --py-files=gs://${BUCKET_NAME}/code/vgm_src.zip \
    gs://${BUCKET_NAME}/code/vgm_main.py \
    -- \
    --bucket_name=${BUCKET_NAME} \
    --input_path=${INPUT_PATH} \
    --output_path=${OUTPUT_PATH} \
    --batch_size=1000 \
    --sample_size=1000 \
    --validation_threshold=0.05 \
    --validation_sample_size=10000 \
    2>&1 | grep -o 'Job \[.*\] submitted' | cut -d '[' -f2 | cut -d ']' -f1)

if [ -z "${JOB_ID}" ]; then
    echo "Failed to get Job ID"
    exit 1
fi

echo "Job submitted with ID: ${JOB_ID}"

# Clean up local zip file
rm vgm_src.zip

# Wait for job completion and stream logs
echo "Streaming job logs..."
gcloud dataproc jobs wait ${JOB_ID} --region=${REGION}

# Check job status
JOB_STATUS=$(gcloud dataproc jobs describe ${JOB_ID} \
    --region=${REGION} \
    --format="value(status.state)")

if [ "${JOB_STATUS}" = "DONE" ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with status: ${JOB_STATUS}"
    exit 1
fi

# Function to handle script interruption
cleanup_on_interrupt() {
    echo "Script interrupted. Cleaning up..."
    cleanup_jobs
    exit 1
}

# Set up interrupt handler
trap cleanup_on_interrupt SIGINT SIGTERM