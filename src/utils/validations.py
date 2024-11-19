# src/utils/validation.py

from typing import Tuple
import numpy as np
from scipy import stats

class DataValidator:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def validate_distribution(self, original: np.ndarray, 
                            generated: np.ndarray) -> Tuple[bool, float]:
        ks_statistic, p_value = stats.ks_2samp(original, generated)
        return p_value > self.threshold, p_value

    def validate_bounds(self, data: np.ndarray, 
                       original_min: float, original_max: float) -> bool:
        return np.all((data >= original_min) & (data <= original_max))

    def validate_moments(self, original: np.ndarray, 
                        generated: np.ndarray) -> Tuple[bool, dict]:
        orig_moments = self._calculate_moments(original)
        gen_moments = self._calculate_moments(generated)
        
        differences = {
            k: abs(orig_moments[k] - gen_moments[k]) 
            for k in orig_moments.keys()
        }
        
        valid = all(diff < self.threshold for diff in differences.values())
        return valid, differences

    def _calculate_moments(self, data: np.ndarray) -> dict:
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skew': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }