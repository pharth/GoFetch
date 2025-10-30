"""Metrics collection and logging utilities."""

import csv
import logging
import time
from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and export prefetcher metrics."""
    
    def __init__(self, output_file: Optional[str] = None, window_size: int = 100):
        self.output_file = output_file
        self.window_size = window_size
        self.metrics = {
            'predictions': deque(maxlen=window_size),
            'prefetches': deque(maxlen=window_size),
            'hits': deque(maxlen=window_size),
            'misses': deque(maxlen=window_size),
            'latencies': deque(maxlen=window_size),
        }
        self.csv_writer = None
        self.csv_file = None
        
        if output_file:
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file for metrics output."""
        try:
            self.csv_file = open(self.output_file, 'w', newline='')
            fieldnames = ['timestamp', 'prediction_latency_ms', 'prefetch_count', 
                         'cache_hit_rate', 'prediction_accuracy']
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
        except Exception as e:
            logger.warning(f"Failed to initialize CSV output: {e}")
    
    def record_prediction(self, latency_ms: float, accuracy: float = 0.0):
        """Record a prediction event."""
        self.metrics['predictions'].append({
            'timestamp': time.time(),
            'latency': latency_ms,
            'accuracy': accuracy
        })
    
    def record_prefetch(self, count: int):
        """Record prefetch operations."""
        self.metrics['prefetches'].append({
            'timestamp': time.time(),
            'count': count
        })
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['hits'].append(time.time())
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['misses'].append(time.time())
    
    def record_latency(self, latency_ms: float):
        """Record read latency."""
        self.metrics['latencies'].append(latency_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        predictions = list(self.metrics['predictions'])
        prefetches = list(self.metrics['prefetches'])
        hits = list(self.metrics['hits'])
        misses = list(self.metrics['misses'])
        latencies = list(self.metrics['latencies'])
        
        total_accesses = len(hits) + len(misses)
        cache_hit_rate = len(hits) / total_accesses if total_accesses > 0 else 0.0
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0.0
        p99_latency = sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0.0
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'total_hits': len(hits),
            'total_misses': len(misses),
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p99_latency_ms': p99_latency,
            'predictions_count': len(predictions),
            'prefetches_count': sum(p['count'] for p in prefetches),
        }
    
    def flush(self):
        """Flush metrics to CSV if enabled."""
        if self.csv_writer and self.csv_file:
            try:
                stats = self.get_stats()
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction_latency_ms': stats.get('avg_latency_ms', 0),
                    'prefetch_count': stats.get('prefetches_count', 0),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0),
                    'prediction_accuracy': 0.0  # TODO: calculate from actual data
                }
                self.csv_writer.writerow(row)
                self.csv_file.flush()
            except Exception as e:
                logger.warning(f"Failed to write metrics row: {e}")
    
    def close(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

