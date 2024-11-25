# src/utils/metrics.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np

@dataclass
class PerformanceMetrics:
    processing_time: float
    memory_usage: Optional[float]
    accuracy: float
    throughput: float

class MetricsCollector:
    def __init__(self):
        self.start_time = None
        self.metrics: Dict[str, List[float]] = {
            'processing_time': [],
            'accuracy': [],
            'throughput': []
        }

    def start_operation(self):
        """Start timing an operation"""
        self.start_time = time.time()

    def end_operation(self, records_processed: int, accuracy: float) -> PerformanceMetrics:
        """End timing an operation and collect metrics"""
        if self.start_time is None:
            raise ValueError("start_operation() must be called before end_operation()")
            
        end_time = time.time()
        processing_time = end_time - self.start_time
        throughput = records_processed / processing_time if processing_time > 0 else 0

        metrics = PerformanceMetrics(
            processing_time=processing_time,
            memory_usage=None,  # We're not tracking memory usage
            accuracy=accuracy,
            throughput=throughput
        )

        self._update_metrics(metrics)
        return metrics

    def _update_metrics(self, metrics: PerformanceMetrics):
        """Update internal metrics storage"""
        self.metrics['processing_time'].append(metrics.processing_time)
        self.metrics['accuracy'].append(metrics.accuracy)
        self.metrics['throughput'].append(metrics.throughput)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of collected metrics"""
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in self.metrics.items()
        }