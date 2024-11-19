# config.py

# GCP Configuration
PROJECT_ID = "betterdata-441921"
REGION = "us-central1"
DATASET_ID = "credit_data"
CLUSTER_NAME = "gmm-cluster-local"
BUCKET_NAME = 'gcs_buck_1'

# Data Configuration
SAMPLE_DATA_PATH = "gs://{PROJECT_ID}/data/Credit.csv"
OUTPUT_PATH = f"gs://{PROJECT_ID}/output"
TARGET_ROWS = 1_000_000_000
BATCH_SIZE = 10_000_000
TABLE_NAME = "credit_data"

# Spark Configuration
SPARK_CONFIG = {
    "spark.executor.memory": "8g",
    "spark.driver.memory": "4g",
    "spark.executor.cores": "4",
    "spark.sql.shuffle.partitions": "1000"
}