"""Prefetch policy engine with rate limiting and confidence thresholds."""

import time
from typing import List, Tuple, Dict
from collections import deque


class PrefetchPolicy:
    """Policy engine for prefetch decisions."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 rate_limit_per_sec: int = 10,
                 max_prefetch_size_mb: int = 16):
        self.confidence_threshold = confidence_threshold
        self.rate_limit_per_sec = rate_limit_per_sec
        self.max_prefetch_size = max_prefetch_size_mb * 1024 * 1024
        
        self.prefetch_history = deque(maxlen=100)  # Track recent prefetches
        self.last_prefetch_time = {}  # Track rate limiting per file
        
    def should_prefetch(self, file_path: str, confidence: float) -> bool:
        """
        Determine if a file should be prefetched based on policy.
        
        Args:
            file_path: Path to file
            confidence: Prediction confidence score
        
        Returns:
            True if should prefetch, False otherwise
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check rate limiting
        now = time.time()
        if file_path in self.last_prefetch_time:
            time_since_last = now - self.last_prefetch_time[file_path]
            if time_since_last < (1.0 / self.rate_limit_per_sec):
                return False
        
        return True
    
    def record_prefetch(self, file_path: str):
        """Record a prefetch operation."""
        self.last_prefetch_time[file_path] = time.time()
        self.prefetch_history.append({
            'timestamp': time.time(),
            'file_path': file_path
        })
    
    def get_prefetch_size(self, file_path: str) -> int:
        """Get the prefetch size for a file."""
        return self.max_prefetch_size
    
    def is_rate_limited(self) -> bool:
        """Check if we're rate limited globally."""
        if len(self.prefetch_history) < self.rate_limit_per_sec:
            return False
        
        recent_prefetches = [p for p in self.prefetch_history if time.time() - p['timestamp'] < 1.0]
        return len(recent_prefetches) >= self.rate_limit_per_sec
    
    def filter_predictions(self, predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Filter predictions based on policy.
        
        Args:
            predictions: List of (file_path, confidence) tuples
        
        Returns:
            Filtered list of predictions
        """
        filtered = []
        
        for file_path, confidence in predictions:
            if self.should_prefetch(file_path, confidence):
                filtered.append((file_path, confidence))
        
        return filtered
    
    def should_pause_prefetch(self) -> bool:
        """Determine if prefetching should be paused (e.g., due to high I/O load)."""
        # TODO: Check system I/O utilization
        # This would require reading /proc/vmstat or using psutil
        return False


class CacheAwarePolicy(PrefetchPolicy):
    """Extended policy with cache awareness."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked_files = set()  # Track which files we've checked
    
    def should_prefetch(self, file_path: str, confidence: float) -> bool:
        """Enhanced decision with cache checks."""
        # Call parent logic first
        if not super().should_prefetch(file_path, confidence):
            return False
        
        # TODO: Check if file is already in cache
        # This would use mincore or /proc to check page residency
        # For now, skip this check
        
        return True

