# src/utils/metrics.py

from dataclasses import dataclass
from typing import Dict, List
import time
import numpy as np

@dataclass
class PerformanceMetrics:
    processing_time: float
    memory_usage: float
    accuracy: float
    throughput: float

class MetricsCollector:
    def __init__(self):
        self.start_time = None
        self.metrics: Dict[str, List[float]] = {
            'processing_time': [],
            'memory_usage': [],
            'accuracy': [],
            'throughput': []
        }

    def start_operation(self):
        self.start_time = time.time()

    def end_operation(self, records_processed: int, accuracy: float, memory_used: float = None) -> PerformanceMetrics:
        end_time = time.time()
        processing_time = end_time - self.start_time
        throughput = records_processed / processing_time


        metrics = PerformanceMetrics(
            processing_time=processing_time,
            memory_usage=memory_used,
            accuracy=accuracy,
            throughput=throughput
        )

        self._update_metrics(metrics)
        return metrics

    def _update_metrics(self, metrics: PerformanceMetrics):
        for key, value in metrics.__dict__.items():
            self.metrics[key].append(value)

    def get_summary(self) -> Dict[str, float]:
        return {
            key: np.mean(values) for key, values in self.metrics.items()
        }