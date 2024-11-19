# src/data/generator.py

from typing import Optional
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, expr, mean, stddev, skewness, count
import logging
import subprocess
import time

class DataGenerator:
    def __init__(self, spark: SparkSession, input_path: str):
        self.spark = spark
        self.input_path = input_path
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _verify_data_exists(self):
        """Verify if input data exists and is readable."""
        try:
            result = subprocess.run(['gsutil', 'ls', self.input_path], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Input file not found: {self.input_path}")
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error checking input file: {str(e)}")
            raise

    def generate_scaled_data(self, 
                           output_path: str,
                           target_size: int = 1_000_000,  # Reduced for testing
                           batch_size: int = 100_000) -> None:
        try:
            print(f"\nStarting data generation process...")
            print(f"Input path: {self.input_path}")
            print(f"Output path: {output_path}")
            print(f"Target size: {target_size:,} records")
            
            # Verify input file exists
            print("Verifying input file...")
            self._verify_data_exists()
            
            # Read input data
            print("Reading input CSV file...")
            df = self.spark.read.csv(
                self.input_path,
                header=True,
                inferSchema=True
            )
            
            # Verify data was read
            input_count = df.count()
            print(f"Read {input_count:,} records from input file")
            
            # Select and verify Amount column
            amount_df = df.select("Amount")
            print("\nVerifying Amount column statistics:")
            amount_df.describe().show()
            
            # Calculate basic statistics
            stats = amount_df.select(
                mean("Amount").alias("mean"),
                stddev("Amount").alias("std")
            ).collect()[0]
            
            mean_val = float(stats["mean"])
            std_val = float(stats["std"])
            print(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            
            print("\nStarting data multiplication process...")
            start_time = time.time()
            
            # Create base DataFrame with random variation
            base_expr = (col("Amount") + (expr("rand()") - 0.5) * std_val * 0.1)
            result_df = amount_df.select(base_expr.alias("Amount"))
            
            # Calculate required copies
            copies_needed = target_size // input_count
            print(f"Will create {copies_needed:,} copies of data")
            
            # Initialize with first batch
            current_size = input_count
            batch_counter = 1
            last_print_time = time.time()
            
            while current_size < target_size:
                batch_df = amount_df.select(base_expr.alias("Amount"))
                result_df = result_df.union(batch_df)
                current_size += input_count
                
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_print_time >= 5:
                    elapsed_time = current_time - start_time
                    progress = (current_size / target_size) * 100
                    print(f"Progress: {progress:.1f}% - Generated {current_size:,} records in {elapsed_time:.1f} seconds")
                    last_print_time = current_time
                
                batch_counter += 1
            
            print("\nWriting data to parquet...")
            print(f"Output path: {output_path}")
            
            # Write with more partitions for better performance
            num_partitions = max(100, target_size // 100_000)
            result_df = result_df.repartition(num_partitions)
            
            result_df.write.mode("overwrite").parquet(output_path)
            
            # Verify output
            output_count = self.spark.read.parquet(output_path).count()
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\nGeneration completed:")
            print(f"- Total records: {output_count:,}")
            print(f"- Time taken: {total_time:.2f} seconds")
            print(f"- Average speed: {output_count/total_time:.0f} records/second")
            
        except Exception as e:
            print(f"\nError in data generation: {str(e)}")
            self.logger.error(f"Error in data generation: {str(e)}")
            raise