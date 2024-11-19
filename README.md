# Scalable Gaussian Mixture Model Implementation

This project implements a scalable version of Variational Gaussian Mixture Models (VGM) for processing large-scale financial data. It's designed to handle datasets up to 1 billion rows while maintaining accuracy and reasonable processing speed.

## Overview

The system transforms financial data (like transaction amounts) using mode-specific normalization through Variational Gaussian Mixture Models. This is particularly useful for tasks like synthetic data generation and anomaly detection in financial datasets.

## Key Features

### 1. Scalability
- Uses Apache Spark for distributed processing
- Implements batch processing with configurable chunk sizes
- Employs stratified sampling for model fitting
- Handles memory constraints through streaming and partitioning
- Supports processing of 100M to 1B rows through Dataproc cluster deployment

### 2. Speed Optimizations
- Batch processing with configurable sizes
- Efficient data partitioning strategy
- Uses Spark's adaptive query execution
- Leverages Pandas UDFs for faster processing
- Memory-efficient operations through proper garbage collection

### 3. Correctness Guarantees
- Comprehensive validation framework
- Statistical tests for distribution preservation
- Bounds checking for extreme values
- Moment matching validation
- Logging and monitoring at each step

## Architecture

The project is organized into several key components:

```
project/
├── src/
│   ├── model/
│   │   ├── scalable_gmm.py      # Core GMM implementation
│   │   └── transformer.py        # Data transformation logic
│   ├── data/
│   │   ├── generator.py         # Data generation utilities
│   │   └── loader.py           # Data loading utilities
│   └── utils/
│       ├── metrics.py          # Performance monitoring
│       └── validation.py       # Data validation
├── deploy/
│   ├── deploy.sh              # Development deployment
│   └── deploy_t2.sh           # Production deployment
└── configs/
    ├── spark.conf             # Spark configuration
    └── cluster.yaml           # Cluster configuration
```

## Trade-offs and Design Decisions

1. **Scalability vs. Speed**
   - Used stratified sampling for model fitting instead of full data
   - Compromise: Slight loss in model accuracy for massive gain in processing speed
   - Mitigation: Validation framework ensures results remain within acceptable bounds

2. **Memory vs. Speed**
   - Implemented batch processing instead of full data loading
   - Compromise: Slower processing for better memory management
   - Mitigation: Optimized batch sizes based on available resources

3. **Accuracy vs. Speed**
   - Used approximate methods for large-scale statistical computations
   - Compromise: Small reduction in precision for significant speed gains
   - Mitigation: Comprehensive validation ensures results meet quality thresholds

## Tools and Technologies

- **Framework**: Apache Spark for distributed processing
- **Language**: Python 3.7+
- **ML Libraries**: scikit-learn for GMM implementation
- **Cloud**: Google Cloud Platform (GCP) with Dataproc
- **Monitoring**: Custom metrics collection system
- **Data Validation**: Statistical testing framework


## Getting Started

1. **Prerequisites**
   ```bash
   - Python 3.7+
   - Apache Spark 3.0+
   - Google Cloud SDK
   ```

2. **Installation**
   ```bash
   # Clone the repository
   git clone [repository-url]
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Deployment**
   ```bash
   # Development deployment
   ./deploy.sh
   
   # Production deployment
   ./deploy_t2.sh
   ```

## Configuration

Key configuration parameters can be adjusted in `configs/`:
- Cluster size and resources
- Batch processing parameters
- Validation thresholds
- Performance monitoring settings