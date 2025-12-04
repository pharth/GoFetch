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
    def __init__(self, output_file: Optional[str] = None, window_size: int = 100):
        self.output_file = output_file
        self.metrics = {
            'predictions': deque(maxlen=window_size),
            'prefetches': deque(maxlen=window_size),
            'hits': 0,      # Changed to counter
            'misses': 0,    # Changed to counter
            'latencies': deque(maxlen=window_size),
        }
        self.csv_file = None
        self.csv_writer = None
        if output_file:
            self._init_csv()

    def _init_csv(self):
        try:
            self.csv_file = open(self.output_file, 'w', newline='')
            fieldnames = ['timestamp', 'prediction_latency_ms', 'prefetch_count', 
                          'cache_hit_rate', 'prediction_accuracy']
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
        except Exception as e:
            logger.warning(f"Failed to initialize CSV output: {e}")

    def record_prediction(self, latency_ms: float):
        self.metrics['predictions'].append(latency_ms)

    def record_prefetch(self, count: int):
        self.metrics['prefetches'].append(count)

    def record_hit(self):
        """Record a successful prefetch (prediction came true)."""
        self.metrics['hits'] += 1

    def record_miss(self):
        """Record a missed prediction."""
        self.metrics['misses'] += 1

    def flush(self):
        if self.csv_writer and self.csv_file:
            try:
                hits = self.metrics['hits']
                misses = self.metrics['misses']
                total_events = hits + misses
                
                # Calculate Rates
                accuracy = (hits / total_events) if total_events > 0 else 0.0
                hit_rate = accuracy # In this simulation, accuracy == hit rate
                
                # Get average latency
                lats = list(self.metrics['predictions'])
                avg_lat = sum(lats)/len(lats) if lats else 0.0
                
                # Get recent prefetch count
                prefetches = list(self.metrics['prefetches'])
                recent_prefetch_count = sum(prefetches)

                row = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction_latency_ms': round(avg_lat, 3),
                    'prefetch_count': recent_prefetch_count,
                    'cache_hit_rate': round(hit_rate, 4),
                    'prediction_accuracy': round(accuracy, 4)
                }
                self.csv_writer.writerow(row)
                self.csv_file.flush()
                
                # Clear deque to prevent double counting in next flush
                self.metrics['prefetches'].clear()
                
            except Exception as e:
                logger.warning(f"Failed to write metrics row: {e}")

    def close(self):
        if self.csv_file:
            self.csv_file.close()
