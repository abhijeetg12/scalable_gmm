# src/data/loader.py

from typing import Optional
from pyspark.sql import SparkSession, DataFrame
import logging

class DataLoader:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def load_parquet(self, path: str, partition_count: Optional[int] = None) -> DataFrame:
        df = self.spark.read.parquet(path)
        if partition_count:
            df = df.repartition(partition_count)
        return df

    def save_parquet(self, df: DataFrame, path: str, mode: str = 'overwrite') -> None:
        df.write.mode(mode).parquet(path)

    def load_sample(self, path: str, fraction: float = 0.1) -> DataFrame:
        return self.load_parquet(path).sample(fraction)