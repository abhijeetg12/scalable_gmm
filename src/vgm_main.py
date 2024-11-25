#!/usr/bin/env python3
# vgm_main.py

import os
import sys
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from src.model.scalable_gmm import ScalableGMM, GMMConfig
from src.utils.metrics import MetricsCollector, PerformanceMetrics
from src.utils.validations import DataValidator

def setup_logging(log_level=logging.INFO):
    """Configure logging with detailed formatting."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='VGM Processing Job')
    parser.add_argument('--bucket_name', required=True, help='GCS bucket name')
    parser.add_argument('--input_path', required=True, help='Input data path')
    parser.add_argument('--output_path', required=True, help='Output path for transformed data')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--sample_size', type=int, default=100000, help='Sample size for GMM fitting')
    parser.add_argument('--validation_threshold', type=float, default=0.05, 
                       help='Threshold for validation tests')
    parser.add_argument('--validation_sample_size', type=int, default=10000,
                       help='Sample size for validation')
    return parser.parse_args()

def create_spark_session():
    """Create and configure Spark session with optimized settings."""
    return SparkSession.builder \
        .appName("VGM_Processing") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.shuffle.service.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "1000") \
        .getOrCreate()

def validate_transformation(validator: DataValidator, 
                          original_data: np.ndarray,
                          transformed_data: np.ndarray,
                          inverse_transformed_data: np.ndarray,
                          logger: logging.Logger) -> float:
    """Validate the transformation process using multiple metrics."""
    # Distribution validation
    dist_valid, p_value = validator.validate_distribution(
        original_data, 
        inverse_transformed_data
    )
    logger.info(f"Distribution validation - Valid: {dist_valid}, p-value: {p_value:.4f}")
    
    # Bounds validation
    bounds_valid = validator.validate_bounds(
        inverse_transformed_data,
        original_data.min(),
        original_data.max()
    )
    logger.info(f"Bounds validation - Valid: {bounds_valid}")
    
    # Moments validation
    moments_valid, moment_diffs = validator.validate_moments(
        original_data,
        inverse_transformed_data
    )
    logger.info(f"Moments validation - Valid: {moments_valid}")
    logger.info(f"Moment differences: {moment_diffs}")
    
    # Calculate overall accuracy
    validations = [dist_valid, bounds_valid, moments_valid]
    accuracy = sum(validations) / len(validations)
    
    return accuracy

def main():
    """Main execution function with comprehensive error handling and validation."""
    logger = setup_logging()
    args = parse_arguments()
    metrics_collector = MetricsCollector()
    validator = DataValidator(threshold=args.validation_threshold)
    
    try:
        # Initialize Spark
        spark = create_spark_session()
        logger.info("Initialized Spark session")
        
        # Load and prepare data
        logger.info(f"Loading data from {args.input_path}")
        data = spark.read.parquet(args.input_path) \
            .select("Amount") \
            .repartition(1000)
        data.cache()
        
        total_records = data.count()
        logger.info(f"Loaded {total_records:,} records")
        
        # Initialize model
        gmm_config = GMMConfig(
            n_components=10,
            batch_size=args.batch_size,
            eps=0.005
        )
        
        # Start metrics collection
        # metrics_collector.start_operation()
        
        # Initialize and fit VGM model
        logger.info("Initializing and fitting VGM model")
        vgm = ScalableGMM(gmm_config)
        vgm.fit(data)
        
        # Transform data
        logger.info("Transforming full dataset")
        transformed_data = vgm.transform(data)
        
        # Validate transformation
        logger.info(f"Validating transformation with {args.validation_sample_size:,} samples")
        
        # Get samples ensuring we don't exceed the data size
        sample_size = min(args.validation_sample_size, data.count())
        sample_original = data.limit(sample_size).toPandas()["Amount"].values
        sample_transformed = transformed_data.limit(sample_size)
        sample_inverse = vgm.inverse_transform(sample_transformed).toPandas()["Amount"].values
        
        if len(sample_original) != len(sample_inverse):
            logger.warning(f"Sample size mismatch: original={len(sample_original)}, inverse={len(sample_inverse)}")
            min_size = min(len(sample_original), len(sample_inverse))
            sample_original = sample_original[:min_size]
            sample_inverse = sample_inverse[:min_size]
        
        # Perform validation

        # # Perform validation
        accuracy = validate_transformation(
            validator,
            sample_original,
            None,
            sample_inverse,
            logger
        )
        
        # # Record metrics (without memory tracking)
        transform_metrics = metrics_collector.end_operation(
            records_processed=total_records,
            accuracy=accuracy
        )
        
        # Save transformed data
        logger.info(f"Saving transformed data to {args.output_path}")
        transformed_data.write \
            .mode("overwrite") \
            .parquet(args.output_path)
        
        # Log final metrics
        logger.info("Job completed successfully")
        logger.info(f"Transform metrics: {transform_metrics}")
        logger.info(f"Performance summary: {metrics_collector.get_summary()}")
        
    except Exception as e:
        logger.error(f"Error in VGM processing: {str(e)}", exc_info=True)
        raise
    finally:
        if 'data' in locals():
            data.unpersist()
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()