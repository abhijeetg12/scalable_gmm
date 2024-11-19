# src/model/transformer.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType
import logging
from .scalable_gmm import GMMConfig

@dataclass
class TransformerConfig:
    gmm_config: GMMConfig
    batch_size: int = 100000
    categorical_columns: List[int] = None
    mixed_columns: Dict[int, List[float]] = None

class DataTransformer:
    """Distributed implementation of data transformation for large-scale datasets."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.metadata = []
        self.output_info = []
        self.output_dimensions = 0
        self.gmm_models = {}
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fit(self, data: DataFrame) -> None:
        """Fit transformer on input DataFrame."""
        try:
            self.metadata = []
            self.output_info = []
            self.output_dimensions = 0

            for col_idx, col_name in enumerate(data.columns):
                if col_idx in (self.config.categorical_columns or []):
                    self._fit_categorical(data, col_idx, col_name)
                elif col_idx in (self.config.mixed_columns or {}):
                    self._fit_mixed(data, col_idx, col_name)
                else:
                    self._fit_continuous(data, col_idx, col_name)

            self.logger.info(f"Successfully fit transformer with {len(self.metadata)} columns")

        except Exception as e:
            self.logger.error(f"Error during transformer fitting: {str(e)}")
            raise

    def _fit_continuous(self, data: DataFrame, col_idx: int, col_name: str) -> None:
        """Fit continuous column."""
        stats = data.select(col_name).summary().collect()
        min_val = float(stats[3][1])  # min
        max_val = float(stats[7][1])  # max

        self.metadata.append({
            'name': col_name,
            'type': 'continuous',
            'min_val': min_val,
            'max_val': max_val
        })

        self.output_info.append((1, 'tanh'))
        self.output_dimensions += 1

    def _fit_categorical(self, data: DataFrame, col_idx: int, col_name: str) -> None:
        """Fit categorical column."""
        distinct_values = data.select(col_name).distinct().collect()
        n_categories = len(distinct_values)

        self.metadata.append({
            'name': col_name,
            'type': 'categorical',
            'size': n_categories
        })

        self.output_info.append((n_categories, 'softmax'))
        self.output_dimensions += n_categories

    def _fit_mixed(self, data: DataFrame, col_idx: int, col_name: str) -> None:
        """Fit mixed-type column."""
        modal_values = self.config.mixed_columns[col_idx]
        stats = data.select(col_name).summary().collect()
        min_val = float(stats[3][1])
        max_val = float(stats[7][1])

        self.metadata.append({
            'name': col_name,
            'type': 'mixed',
            'min_val': min_val,
            'max_val': max_val,
            'modal_values': modal_values
        })

        self.output_info.append((1, 'tanh'))
        self.output_info.append((len(modal_values), 'softmax'))
        self.output_dimensions += 1 + len(modal_values)

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform the input DataFrame."""
        try:
            schema = StructType([
                StructField(f"col_{i}", DoubleType(), True) 
                for i in range(self.output_dimensions)
            ])

            @pandas_udf(schema)
            def transform_batch(pdf):
                return self._transform_batch(pdf.values)

            return data.select(transform_batch('*'))

        except Exception as e:
            self.logger.error(f"Error during transformation: {str(e)}")
            raise

    def _transform_batch(self, batch: np.ndarray) -> np.ndarray:
        """Transform a batch of data."""
        result = []
        col_idx = 0

        for meta in self.metadata:
            if meta['type'] == 'continuous':
                normalized = self._normalize_continuous(
                    batch[:, col_idx],
                    meta['min_val'],
                    meta['max_val']
                )
                result.append(normalized)
            
            elif meta['type'] == 'categorical':
                one_hot = self._transform_categorical(
                    batch[:, col_idx],
                    meta['size']
                )
                result.append(one_hot)
            
            elif meta['type'] == 'mixed':
                mixed_transform = self._transform_mixed(
                    batch[:, col_idx],
                    meta
                )
                result.extend(mixed_transform)
            
            col_idx += 1

        return np.concatenate(result, axis=1)

    def _normalize_continuous(self, data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Normalize continuous data."""
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(data).reshape(-1, 1)
        normalized = (data - min_val) / range_val * 2 - 1
        return np.clip(normalized, -1, 1).reshape(-1, 1)

    def _transform_categorical(self, data: np.ndarray, n_categories: int) -> np.ndarray:
        """One-hot encode categorical data."""
        one_hot = np.zeros((len(data), n_categories))
        for i, val in enumerate(data):
            if not np.isnan(val):
                one_hot[i, int(val)] = 1
        return one_hot

    def _transform_mixed(self, data: np.ndarray, meta: dict) -> List[np.ndarray]:
        """Transform mixed-type data."""
        modal_values = set(meta['modal_values'])
        is_continuous = np.array([x not in modal_values for x in data])
        
        continuous_normalized = self._normalize_continuous(
            data[is_continuous],
            meta['min_val'],
            meta['max_val']
        )
        
        one_hot = np.zeros((len(data), len(modal_values)))
        for i, val in enumerate(data):
            if val in modal_values:
                one_hot[i, list(modal_values).index(val)] = 1
                
        return [continuous_normalized, one_hot]