# src/model/scalable_gmm.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType
import pandas as pd
import logging
import gc

@dataclass
class GMMConfig:
    n_components: int = 10
    weight_concentration_prior: float = 0.001
    max_iter: int = 100
    n_init: int = 1
    random_state: int = 42
    chunk_size: int = 1000
    batch_size: int = 1000
    eps: float = 0.005
    clip_min: float = -0.99
    clip_max: float = 0.99
    normalization_factor: float = 4.0

class ScalableGMM:
    def __init__(self, config: GMMConfig):
        self.config = config
        self.model: Optional[BayesianGaussianMixture] = None
        self.components: List[bool] = []
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_model_state(self):
        if self.model is None:
            raise ValueError("Model must be fit before transform/inverse_transform operations")

    def fit(self, data: DataFrame) -> None:
        try:
            sample_size = min(100000, data.count())
            sampled_data = data.sample(False, fraction=sample_size/data.count())
            sample_array = np.array(sampled_data.select("Amount").collect()).reshape(-1, 1)
            
            self.model = BayesianGaussianMixture(
                n_components=self.config.n_components,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.config.weight_concentration_prior,
                max_iter=self.config.max_iter,
                n_init=self.config.n_init,
                random_state=self.config.random_state
            )
            
            self.model.fit(sample_array)
            self.means_ = self.model.means_
            self.covariances_ = self.model.covariances_
            self.weights_ = self.model.weights_
            self._identify_components(sample_array)
            
            del sample_array
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error in GMM fitting: {str(e)}")
            raise

    def _identify_components(self, data: np.ndarray) -> None:
        old_comp = self.weights_ > self.config.eps
        predictions = self.model.predict(data)
        mode_freq = np.unique(predictions)
        
        self.components = [
            (i in mode_freq) and old_comp[i]
            for i in range(self.config.n_components)
        ]

    def transform_batch(self, data: pd.Series) -> pd.DataFrame:
        """Transform a batch of data with consistent array dimensions."""
        try:
            self._validate_model_state()
            
            # Preserve the index and ensure we're working with the right size
            original_index = data.index
            values = data.values.reshape(-1, 1)
            n_samples = len(values)
            
            # Get probabilities for active components
            probs = self.model.predict_proba(values)
            active_component_indices = np.where(self.components)[0]
            active_probs = probs[:, active_component_indices]
            
            # Normalize probabilities
            pp = active_probs + 1e-6
            pp = pp / pp.sum(axis=1, keepdims=True)
            
            # Select modes
            n_active_components = len(active_component_indices)
            opt_sel = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                opt_sel[i] = np.random.choice(n_active_components, p=pp[i])
            
            # Get selected means and stds
            selected_means = self.means_[active_component_indices][opt_sel].reshape(-1, 1)
            selected_stds = np.sqrt(self.covariances_[active_component_indices].flatten())[opt_sel].reshape(-1, 1)
            
            # Normalize values
            normalized_data = (values - selected_means) / (self.config.normalization_factor * selected_stds)
            normalized_data = np.clip(normalized_data, self.config.clip_min, self.config.clip_max)
            
            # Create mode encoding
            mode_encoding = np.zeros((n_samples, n_active_components))
            mode_encoding[np.arange(n_samples), opt_sel] = 1
            
            # Ensure we're creating a DataFrame with matching dimensions
            result_df = pd.DataFrame(
                index=original_index,
                data={
                    'normalized_value': normalized_data.ravel(),
                    'mode_encoding': [row.tolist() for row in mode_encoding]
                }
            )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in transform_batch: {str(e)}")
            raise

    def inverse_transform_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform a batch of data."""
        try:
            # Work with the actual data dimensions
            original_index = data.index
            n_samples = len(data)
            
            # Ensure proper dimensions
            normalized = np.array(data["normalized_value"]).reshape(n_samples, 1)
            encoding = np.stack([np.array(x) for x in data["mode_encoding"]])
            
            # Get active components
            active_component_indices = np.where(self.components)[0]
            
            # Get selected modes
            mode_idx = np.argmax(encoding, axis=1)
            
            # Get corresponding means and stds
            selected_means = self.means_[active_component_indices][mode_idx].reshape(n_samples, 1)
            selected_stds = np.sqrt(self.covariances_[active_component_indices].flatten())[mode_idx].reshape(n_samples, 1)
            
            # Inverse normalize ensuring proper dimensions
            original = (normalized * self.config.normalization_factor * selected_stds + selected_means)
            
            # Create DataFrame ensuring dimensions match
            return pd.DataFrame(
                data={"Amount": original.ravel()},
                index=original_index
            )
            
        except Exception as e:
            self.logger.error(f"Error in inverse_transform_batch: {str(e)}")
            raise

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform data using distributed processing."""
        self._validate_model_state()
        
        schema = StructType([
            StructField("normalized_value", DoubleType(), True),
            StructField("mode_encoding", ArrayType(DoubleType(), False), True)
        ])
        
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def transform_group(pdf: pd.DataFrame) -> pd.DataFrame:
            # Process only the rows we have
            result = self.transform_batch(pdf["Amount"])
            return result
        
        return data.repartition(data.count() // self.config.chunk_size + 1) \
                  .groupBy((col("Amount").cast("long") / self.config.chunk_size).cast("long")) \
                  .apply(transform_group)

    def inverse_transform(self, data: DataFrame) -> DataFrame:
        """Inverse transform data using distributed processing."""
        self._validate_model_state()
        
        inverse_schema = StructType([
            StructField("Amount", DoubleType(), True)
        ])
        
        @pandas_udf(inverse_schema, PandasUDFType.GROUPED_MAP)
        def inverse_transform_group(pdf: pd.DataFrame) -> pd.DataFrame:
            # Process only the rows we have
            result = self.inverse_transform_batch(pdf)
            return result
        
        return data.repartition(data.count() // self.config.chunk_size + 1) \
                  .groupBy((col("normalized_value").cast("long") / self.config.chunk_size).cast("long")) \
                  .apply(inverse_transform_group)