import time
import heapq
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
from abc import ABC, abstractmethod

class BasePrefetchPolicy(ABC):
    """Abstract base class for all prefetch policies."""
    
    def __init__(self, confidence_threshold: float, rate_limit_per_sec: int, max_prefetch_size_mb: int):
        self.confidence_threshold = confidence_threshold
        self.rate_limit_per_sec = rate_limit_per_sec
        self.max_prefetch_size = max_prefetch_size_mb * 1024 * 1024
        self.prefetch_history = deque(maxlen=100)
        self.last_prefetch_time = {}
        self.pending_files = set() # Track what is currently in the queue

    def should_prefetch(self, file_path: str, confidence: float) -> bool:
        """Global checks: Threshold and Per-File Rate Limiting."""
        if confidence < self.confidence_threshold:
            return False
        
        # Don't fetch the same file if we just fetched it 5 seconds ago
        now = time.time()
        if file_path in self.last_prefetch_time:
            if now - self.last_prefetch_time[file_path] < 5.0:
                return False
        return True

    def record_success(self, file_path: str):
        """Log that a file was actually sent to the kernel."""
        self.last_prefetch_time[file_path] = time.time()
        self.prefetch_history.append({'timestamp': time.time()})
        self.pending_files.discard(file_path)

    def is_global_rate_limited(self) -> bool:
        """Check if we are exceeding global ops/sec."""
        if len(self.prefetch_history) < self.rate_limit_per_sec:
            return False
        # Count prefetches in the last 1 second
        recent = [p for p in self.prefetch_history if time.time() - p['timestamp'] < 1.0]
        return len(recent) >= self.rate_limit_per_sec

    def get_prefetch_size(self, file_path: str) -> int:
        return self.max_prefetch_size

    @abstractmethod
    def push(self, file_path: str, confidence: float):
        """Add a prediction to the policy's internal queue."""
        pass

    @abstractmethod
    def pop_batch(self, batch_size: int) -> List[str]:
        """Get the next batch of files to prefetch."""
        pass


class FIFOPolicy(BasePrefetchPolicy):
    """
    Standard Implementation (Like the Paper).
    First predicted = First prefetched.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = deque()

    def push(self, file_path: str, confidence: float):
        if file_path in self.pending_files:
            return
        
        if self.should_prefetch(file_path, confidence):
            self.queue.append(file_path)
            self.pending_files.add(file_path)

    def pop_batch(self, batch_size: int) -> List[str]:
        batch = []
        while self.queue and len(batch) < batch_size:
            if self.is_global_rate_limited():
                break
            
            file_path = self.queue.popleft() # FIFO: Pop from left
            batch.append(file_path)
            self.record_success(file_path)
            
        return batch


class HeapPolicy(BasePrefetchPolicy):
    """
    Your Implementation.
    Highest Confidence = First prefetched.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = [] # This will be a heap

    def push(self, file_path: str, confidence: float):
        if file_path in self.pending_files:
            return

        if self.should_prefetch(file_path, confidence):
            # Heapq is a min-heap. We store negative confidence to simulate max-heap.
            # Tuple: (-confidence, timestamp, file_path)
            # Timestamp ensures stability (FIFO for equal confidence)
            entry = (-confidence, time.time(), file_path)
            heapq.heappush(self.queue, entry)
            self.pending_files.add(file_path)

    def pop_batch(self, batch_size: int) -> List[str]:
        batch = []
        while self.queue and len(batch) < batch_size:
            if self.is_global_rate_limited():
                break
            
            # Pop highest confidence
            _, _, file_path = heapq.heappop(self.queue)
            
            # Double check if we should still prefetch (in case time passed)
            # We don't have the confidence score readily handy here without unpacking, 
            # but usually once queued, we commit to it unless file cached.
            
            batch.append(file_path)
            self.record_success(file_path)
            
        return batch

class PolicyFactory:
    @staticmethod
    def get_policy(config_name: str, **kwargs) -> BasePrefetchPolicy:
        if config_name.lower() == "heap":
            return HeapPolicy(**kwargs)
        return FIFOPolicy(**kwargs)
