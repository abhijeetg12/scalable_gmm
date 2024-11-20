# src/main.py

import argparse
from pyspark.sql import SparkSession
import logging
import os
import sys
import time
from datetime import datetime

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.generator import DataGenerator
from model.scalable_gmm import ScalableGMM, GMMConfig
from src.model.transformer import DataTransformer, TransformerConfig
from src.utils.metrics import MetricsCollector

def setup_logging():
    """Configure logging with timestamps and detailed formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Handle command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', required=True, help='GCS bucket name')
    parser.add_argument('--region', required=True, help='GCP region')
    return parser.parse_args()

# src/main.py

def main():
    start_time = time.time()
    print("="*80)
    print(f"Starting GMM Processing Job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup
    logger = setup_logging()
    args = parse_arguments()
    
    print("\n1. Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("GMM_Processing") \
        .config("spark.sql.broadcastTimeout", "3600") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    
    print("   ✓ Spark session created")
    print(f"   ✓ Using bucket: {args.bucket_name}")
    
    try:
        # Start with a very small test
        TEST_SIZE = 100_000_000  # Start with just 100K records
        BATCH_SIZE = 10_000_000
        
        print(f"\n2. Starting test run with {TEST_SIZE:,} records")
        input_path = f"gs://{args.bucket_name}/data/Credit.csv"
        scaled_data_path = f"gs://{args.bucket_name}/data/scaled_data"
        
        print(f"   - Input path: {input_path}")
        print(f"   - Output path: {scaled_data_path}")
        
        generator = DataGenerator(spark=spark, input_path=input_path)
        
        generator.generate_scaled_data(
            output_path=scaled_data_path,
            target_size=TEST_SIZE,
            batch_size=BATCH_SIZE
        )
        
        print("\n3. Verifying generated data...")
        result_df = spark.read.parquet(scaled_data_path)
        final_count = result_df.count()
        print(f"   ✓ Generated {final_count:,} records")
        
        print("\n4. Basic statistics of generated data:")
        result_df.describe().show()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        logger.error(f"Error in processing: {str(e)}")
        raise
    finally:
        print("\nCleaning up...")
        spark.stop()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total job duration: {total_time:.2f} seconds")
        print("="*80)

if __name__ == "__main__":
    main()