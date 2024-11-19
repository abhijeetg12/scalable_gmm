#!/bin/bash

# Configuration
PROJECT_ID="betterdata-441921"
REGION="us-central1"
CLUSTER_NAME="gmm-cluster-local"
BUCKET_NAME="gcs_buck_1"
LOCAL_DATA_PATH="Data/Credit.csv"

echo "Starting deployment process..."

# Create GCS bucket if it doesn't exist
if ! gsutil ls -b gs://${BUCKET_NAME} > /dev/null 2>&1; then
    echo "Creating GCS bucket..."
    gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${BUCKET_NAME}
fi

# Package the source code
echo "Packaging source code..."
zip -r src.zip src/

# Upload data and code to GCS
echo "Uploading data and code to GCS..."
gsutil -m cp ${LOCAL_DATA_PATH} gs://${BUCKET_NAME}/data/
gsutil -m cp src.zip gs://${BUCKET_NAME}/code/
gsutil -m cp src/main.py gs://${BUCKET_NAME}/code/
gsutil -m cp config.py gs://${BUCKET_NAME}/code/

# Create Dataproc cluster if it doesn't exist
if ! gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${REGION} > /dev/null 2>&1; then
    echo "Creating Dataproc cluster..."
    gcloud dataproc clusters create ${CLUSTER_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --zone=${REGION}-a \
        --master-machine-type=n1-standard-4 \
        --master-boot-disk-size=500GB \
        --num-workers=2 \
        --worker-machine-type=n1-standard-4 \
        --worker-boot-disk-size=500GB \
        --image-version=2.0-debian10 \
        --properties="spark:spark.driver.memory=4g,spark:spark.executor.memory=8g,spark:spark.executor.cores=4"
fi

# Submit PySpark job and capture the job ID
echo "Submitting PySpark job..."
JOB_ID=$(gcloud dataproc jobs submit pyspark \
    --cluster=${CLUSTER_NAME} \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --files=gs://${BUCKET_NAME}/code/config.py \
    --py-files=gs://${BUCKET_NAME}/code/src.zip \
    gs://${BUCKET_NAME}/code/main.py \
    -- \
    --bucket_name=${BUCKET_NAME} \
    --region=${REGION} \
    2>&1 | grep -o 'Job \[.*\] submitted' | cut -d '[' -f2 | cut -d ']' -f1)

if [ -z "${JOB_ID}" ]; then
    echo "Failed to get Job ID"
    exit 1
fi

echo "Job submitted with ID: ${JOB_ID}"

# Clean up local zip file
rm src.zip

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