"""Main prefetch daemon."""

import os
import sys
import time
import signal
import argparse
from typing import Optional
from utils.config import Config
from utils.metrics import MetricsCollector
from predictor.inference import HybridPredictor
from prefetcher.syscalls import advise_file
from prefetcher.policy import PrefetchPolicy
from storage.trace_db import TraceDatabase
from collector.trace_schema import TraceEvent


class PrefetchDaemon:
    """Main daemon for intelligent file prefetching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # Initialize components
        self.db = TraceDatabase(config.get('storage.db_path', 'traces.db'))
        
        # Initialize predictor
        baseline_model = config.get('predictor.baseline.enabled', True)
        ml_enabled = config.get('predictor.ml.enabled', False)
        
        self.predictor = HybridPredictor(
            db_path=config.get('storage.db_path', 'traces.db'),
            baseline_model_path='models/baseline.json',
            ml_model_path=config.get('predictor.ml.model_path', 'models/prefetch_gru.pt'),
            use_ml=ml_enabled
        )
        
        # Initialize policy
        self.policy = PrefetchPolicy(
            confidence_threshold=config.get('prefetcher.confidence_threshold', 0.7),
            rate_limit_per_sec=config.get('prefetcher.rate_limit_per_sec', 10),
            max_prefetch_size_mb=config.get('prefetcher.max_prefetch_size_mb', 16)
        )
        
        # Initialize metrics
        metrics_enabled = config.get('metrics.enabled', True)
        metrics_file = config.get('metrics.output_file', 'metrics.csv') if metrics_enabled else None
        self.metrics = MetricsCollector(output_file=metrics_file)
        
        # Daemon state
        self.last_poll_time_ns = None
        self.prefetch_interval = 0.1  # 100ms
        self.top_k = config.get('prefetcher.top_k_predictions', 5)
    
    def start(self):
        """Start the daemon."""
        self.running = True
        print("Starting prefetch daemon...")
        print(f"  Confidence threshold: {self.policy.confidence_threshold}")
        print(f"  Rate limit: {self.policy.rate_limit_per_sec} ops/sec")
        print(f"  Top-K predictions: {self.top_k}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def _run_loop(self):
        """Main daemon loop."""
        iteration = 0
        
        while self.running:
            start_time = time.time()
            
            # Poll for new traces
            traces = self.db.get_recent(since_ns=self.last_poll_time_ns)
            
            if traces:
                print(f"Processing {len(traces)} new traces...")
                self._process_traces(traces)
                self.last_poll_time_ns = traces[-1].timestamp
            else:
                # Still update predictor to maintain state
                self.predictor.update_from_db(since_ns=self.last_poll_time_ns)
            
            # Get active workers
            active_workers = self.predictor.get_active_workers()
            
            # Make predictions and prefetch
            if active_workers and not self.policy.is_rate_limited():
                self._prefetch_for_workers(active_workers)
            
            # Flush metrics
            if iteration % 10 == 0:
                self.metrics.flush()
            
            # Sleep to limit CPU usage
            elapsed = time.time() - start_time
            sleep_time = max(0, self.prefetch_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            iteration += 1
    
    def _process_traces(self, traces: list):
        """Process trace events and update predictor state."""
        for trace in traces:
            self.predictor.observe_trace(trace)
            
            # Record in metrics
            # (could track cache hits/misses here if we detect them)
    
    def _prefetch_for_workers(self, worker_ids: list):
        """Generate predictions and prefetch for active workers."""
        total_prefetched = 0
        
        for worker_id in worker_ids:
            try:
                # Get predictions
                predictions = self.predictor.predict(worker_id, top_k=self.top_k)
                
                # Apply policy filters
                filtered = self.policy.filter_predictions(predictions)
                
                # Issue prefetch requests
                for file_path, confidence in filtered:
                    if advise_file(file_path, offset=0, length=self.policy.get_prefetch_size(file_path)):
                        self.policy.record_prefetch(file_path)
                        total_prefetched += 1
                        
                        # Check rate limit
                        if self.policy.is_rate_limited():
                            break
            except Exception as e:
                print(f"Error prefetching for worker {worker_id}: {e}")
        
        if total_prefetched > 0:
            self.metrics.record_prefetch(total_prefetched)
            print(f"Prefetched {total_prefetched} files")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
    
    def stop(self):
        """Stop the daemon and cleanup."""
        self.running = False
        self.metrics.flush()
        self.metrics.close()
        self.predictor.close()
        self.db.close()
        print("Daemon stopped.")


def main():
    """Main entry point for daemon."""
    parser = argparse.ArgumentParser(description="DL File Prefetcher Daemon")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--confidence", type=float, help="Confidence threshold")
    parser.add_argument("--rate-limit", type=int, help="Rate limit per second")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_args(args)
    
    # Create and start daemon
    daemon = PrefetchDaemon(config)
    daemon.start()


if __name__ == "__main__":
    main()

