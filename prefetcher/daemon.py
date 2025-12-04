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
from prefetcher.policy import PolicyFactory
from storage.trace_db import TraceDatabase

class PrefetchDaemon:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        self.db = TraceDatabase(config.get('storage.db_path', 'traces.db'))
        
        self.predictor = HybridPredictor(
            db_path=config.get('storage.db_path', 'traces.db'),
            baseline_model_path='models/baseline.json',
            ml_model_path=config.get('predictor.ml.model_path', 'models/prefetch_gru.pt'),
            use_ml=config.get('predictor.ml.enabled', False)
        )
        
        policy_type = config.get('prefetcher.policy_type', 'fifo')
        print(f"Initializing Policy Engine: {policy_type.upper()}")

        self.policy = PolicyFactory.get_policy(
            policy_type,
            confidence_threshold=config.get('prefetcher.confidence_threshold', 0.6), # Lowered slightly for demo
            rate_limit_per_sec=config.get('prefetcher.rate_limit_per_sec', 20),
            max_prefetch_size_mb=config.get('prefetcher.max_prefetch_size_mb', 16)
        )
        
        self.metrics = MetricsCollector(output_file='metrics.csv')
        self.last_poll_time_ns = None
        self.prefetch_interval = 0.1
        self.top_k = config.get('prefetcher.top_k_predictions', 5)
        
        # MEMORY OF WHAT WE PREFETCHED
        # Key: file_path, Value: timestamp
        self.prefetched_cache = {} 
        
    def start(self):
        self.running = True
        print("Starting prefetch daemon...")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def _run_loop(self):
        iteration = 0
        while self.running:
            loop_start = time.time()
            
            # 1. Poll for new traces
            traces = self.db.get_recent(since_ns=self.last_poll_time_ns)
            
            if traces:
                print(f"Processing {len(traces)} new traces...")
                self._process_traces_and_check_hits(traces)
                self.last_poll_time_ns = traces[-1].timestamp
            else:
                self.predictor.update_from_db(since_ns=self.last_poll_time_ns)
            
            # 2. Push Predictions
            active_workers = self.predictor.get_active_workers()
            for worker_id in active_workers:
                predict_start = time.time()
                predictions = self.predictor.predict(worker_id, top_k=self.top_k)
                latency = (time.time() - predict_start) * 1000
                self.metrics.record_prediction(latency)
                
                for file_path, confidence in predictions:
                    self.policy.push(file_path, confidence)

            # 3. Pop Batch and Execute
            batch = self.policy.pop_batch(batch_size=5)
            
            for file_path in batch:
                if advise_file(file_path):
                    print(f"[{'HEAP' if 'Heap' in self.policy.__class__.__name__ else 'FIFO'}] Prefetching: {file_path}")
                    self.metrics.record_prefetch(1)
                    
                    # Store in our "Short Term Memory" to verify hits later
                    self.prefetched_cache[file_path] = time.time()
            
            # Flush metrics
            if iteration % 10 == 0:
                self.metrics.flush()
                self._cleanup_cache() # Remove old entries from memory
            
            # Sleep
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.prefetch_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            iteration += 1
    
    def _process_traces_and_check_hits(self, traces: list):
        """Update predictor AND check if our previous prefetches were useful."""
        current_time = time.time()
        
        for trace in traces:
            # 1. Update Predictor
            self.predictor.observe_trace(trace)
            
            # 2. Check Hit/Miss
            # If we prefetched this file recently (e.g., last 10 seconds), it's a HIT
            if trace.file_path in self.prefetched_cache:
                self.metrics.record_hit()
                # print(f"  -> Cache HIT! {trace.file_path}")
            else:
                self.metrics.record_miss()

    def _cleanup_cache(self):
        """Remove prefetches older than 30 seconds from memory."""
        now = time.time()
        expired = [k for k, v in self.prefetched_cache.items() if now - v > 30]
        for k in expired:
            del self.prefetched_cache[k]
    
    def _signal_handler(self, signum, frame):
        self.running = False
    
    def stop(self):
        self.running = False
        self.metrics.close()
        self.predictor.close()
        self.db.close()
        print("Daemon stopped.")

def main():
    parser = argparse.ArgumentParser(description="DL File Prefetcher Daemon")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    args = parser.parse_args()
    config = Config.from_args(args)
    daemon = PrefetchDaemon(config)
    daemon.start()

if __name__ == "__main__":
    main()
