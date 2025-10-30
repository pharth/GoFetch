"""Baseline n-gram/Markov chain predictor for file access patterns."""

import json
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import os


class MarkovPredictor:
    """Markov chain predictor for file access patterns."""
    
    def __init__(self, ngram_size: int = 3, decay_factor: float = 0.95):
        self.ngram_size = ngram_size
        self.decay_factor = decay_factor
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.file_counts = defaultdict(int)
        self.worker_sequences = defaultdict(list)  # worker_id -> [file1, file2, ...]
        self.current_epoch = 0
    
    def observe(self, worker_id: int, file_path: str, epoch: Optional[int] = None):
        """Record a file access observation."""
        if epoch is not None and epoch != self.current_epoch:
            # New epoch - reset sequences to adapt to shuffling
            self.current_epoch = epoch
            self.worker_sequences = defaultdict(list)
        
        # Update worker sequence
        self.worker_sequences[worker_id].append(file_path)
        if len(self.worker_sequences[worker_id]) > self.ngram_size:
            self.worker_sequences[worker_id].pop(0)
        
        # Build transitions from recent history
        sequence = self.worker_sequences[worker_id]
        for i in range(len(sequence) - 1):
            context = tuple(sequence[i:])
            next_file = sequence[-1]
            self.transitions[context][next_file] += 1
        
        self.file_counts[file_path] += 1
    
    def predict(self, worker_id: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next files for a worker."""
        sequence = self.worker_sequences.get(worker_id, [])
        
        if not sequence:
            # No history - predict by global frequency
            total = sum(self.file_counts.values())
            if total == 0:
                return []
            
            predictions = sorted(
                self.file_counts.items(),
                key=lambda x: -x[1]
            )[:top_k]
            
            return [(file, count / total) for file, count in predictions]
        
        # Try different context lengths (longest to shortest)
        for context_len in range(len(sequence), 0, -1):
            context = tuple(sequence[-context_len:])
            
            if context in self.transitions:
                dests = self.transitions[context]
                total = sum(dests.values())
                if total > 0:
                    # Calculate probabilities
                    predictions = sorted(
                        dests.items(),
                        key=lambda x: -x[1]
                    )[:top_k]
                    
                    return [(file, count / total) for file, count in predictions]
        
        # Fallback to global frequency
        total = sum(self.file_counts.values())
        if total == 0:
            return []
        
        predictions = sorted(
            self.file_counts.items(),
            key=lambda x: -x[1]
        )[:top_k]
        
        return [(file, count / total) for file, count in predictions]
    
    def apply_decay(self):
        """Apply decay to old transitions."""
        for context in self.transitions:
            for file in self.transitions[context]:
                self.transitions[context][file] *= self.decay_factor
    
    def reset_epoch(self):
        """Reset for new epoch."""
        self.worker_sequences = defaultdict(list)
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'ngram_size': self.ngram_size,
            'decay_factor': self.decay_factor,
            'transitions': {str(k): dict(v) for k, v in self.transitions.items()},
            'file_counts': dict(self.file_counts),
            'current_epoch': self.current_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.ngram_size = model_data['ngram_size']
        self.decay_factor = model_data['decay_factor']
        self.transitions = defaultdict(lambda: defaultdict(int))
        
        for k, v in model_data['transitions'].items():
            context = tuple(eval(k))  # Convert string back to tuple
            self.transitions[context] = defaultdict(int, v)
        
        self.file_counts = defaultdict(int, model_data['file_counts'])
        self.current_epoch = model_data.get('current_epoch', 0)


class FrequencyPredictor:
    """Simple frequency-based predictor (baseline of baseline)."""
    
    def __init__(self):
        self.file_counts = defaultdict(int)
        self.total_accesses = 0
    
    def observe(self, worker_id: int, file_path: str, epoch: Optional[int] = None):
        """Record file access."""
        self.file_counts[file_path] += 1
        self.total_accesses += 1
    
    def predict(self, worker_id: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict by frequency."""
        if self.total_accesses == 0:
            return []
        
        predictions = sorted(
            self.file_counts.items(),
            key=lambda x: -x[1]
        )[:top_k]
        
        return [(file, count / self.total_accesses) for file, count in predictions]
    
    def save(self, filepath: str):
        """Save model."""
        model_data = {
            'file_counts': dict(self.file_counts),
            'total_accesses': self.total_accesses
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load model."""
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.file_counts = defaultdict(int, model_data['file_counts'])
        self.total_accesses = model_data['total_accesses']

