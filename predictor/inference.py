"""Inference engine for combining baseline and ML predictors."""

from typing import List, Tuple, Optional
from predictor.baseline import MarkovPredictor
from predictor.ml_model import FileAccessPredictor
from storage.trace_db import TraceDatabase
from collector.trace_schema import TraceEvent


class HybridPredictor:
    """Combines baseline and ML predictors with fallback logic."""
    
    def __init__(self, db_path: str = "traces.db", baseline_model_path: str = "models/baseline.json",
                 ml_model_path: Optional[str] = None, use_ml: bool = False):
        self.db = TraceDatabase(db_path)
        self.use_ml = use_ml
        
        # Initialize baseline predictor
        self.baseline = MarkovPredictor()
        try:
            self.baseline.load(baseline_model_path)
            print(f"Loaded baseline model from {baseline_model_path}")
        except Exception as e:
            print(f"Could not load baseline model: {e}")
        
        # Initialize ML predictor if enabled
        self.ml_model = None
        if use_ml and ml_model_path:
            try:
                self.ml_model = FileAccessPredictor(ml_model_path)
                print(f"Loaded ML model from {ml_model_path}")
            except Exception as e:
                print(f"Could not load ML model: {e}")
        
        # Per-worker state
        self.worker_sequences = {}  # worker_id -> [file1, file2, ...]
        self.last_predictions = {}  # worker_id -> [(file, confidence), ...]
    
    def observe_trace(self, trace: TraceEvent):
        """Observe a trace event and update predictor state."""
        worker_id = trace.worker_id
        file_path = trace.file_path
        
        # Update worker sequence
        if worker_id not in self.worker_sequences:
            self.worker_sequences[worker_id] = []
        
        self.worker_sequences[worker_id].append(file_path)
        
        # Keep only recent history
        max_history = 20
        if len(self.worker_sequences[worker_id]) > max_history:
            self.worker_sequences[worker_id] = self.worker_sequences[worker_id][-max_history:]
        
        # Update baseline predictor
        self.baseline.observe(worker_id, file_path, epoch=trace.epoch)
    
    def update_from_db(self, since_ns: Optional[int] = None):
        """Update predictor state from recent traces."""
        traces = self.db.get_recent(since_ns=since_ns)
        
        for trace in traces:
            self.observe_trace(trace)
    
    def predict(self, worker_id: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next files for a worker."""
        predictions = []
        
        # Try ML model first if enabled
        if self.ml_model and worker_id in self.worker_sequences:
            try:
                recent_files = self.worker_sequences[worker_id]
                ml_predictions = self.ml_model.predict(worker_id, recent_files, top_k)
                
                if ml_predictions:
                    predictions = ml_predictions
            except Exception as e:
                print(f"ML prediction failed: {e}, falling back to baseline")
        
        # Fallback to baseline
        if not predictions:
            baseline_predictions = self.baseline.predict(worker_id, top_k)
            predictions = baseline_predictions
        
        self.last_predictions[worker_id] = predictions
        return predictions
    
    def get_active_workers(self) -> List[int]:
        """Get list of active workers from database."""
        return self.db.get_active_workers()
    
    def should_use_prediction(self, file_path: str, confidence: float, 
                              threshold: float = 0.7) -> bool:
        """Determine if a prediction should be used for prefetching."""
        return confidence >= threshold
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

