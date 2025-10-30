"""GRU-based ML predictor for file access patterns."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import os


class FileAccessGRU(nn.Module):
    """Lightweight GRU model for predicting next file access."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding for file IDs
        self.file_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Additional features: offset bin, time delta, worker ID
        self.extra_feat_dim = 8  # 2 (offset bin) + 2 (time delta) + 4 (worker)
        self.extra_features = nn.Linear(self.extra_feat_dim, embedding_dim)
        
        # GRU layer
        self.gru = nn.GRU(
            embedding_dim * 2,  # file embedding + extra features
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, file_ids: torch.Tensor, extra_features: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            file_ids: [batch, seq_len] file ID tokens
            extra_features: [batch, seq_len, extra_feat_dim] additional features
            hidden: optional hidden state
        """
        # Embed file IDs
        file_embeds = self.file_embedding(file_ids)  # [batch, seq_len, embedding_dim]
        
        # Process extra features
        extra_embeds = self.extra_features(extra_features)  # [batch, seq_len, embedding_dim]
        
        # Concatenate embeddings
        combined = torch.cat([file_embeds, extra_embeds], dim=-1)  # [batch, seq_len, embedding_dim*2]
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Pass through GRU
        output, hidden = self.gru(combined, hidden)  # output: [batch, seq_len, hidden_dim]
        
        # Take last output
        last_output = output[:, -1, :]  # [batch, hidden_dim]
        
        # Predict next file
        logits = self.fc(last_output)  # [batch, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state."""
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device
        )


class FileAccessPredictor:
    """Wrapper for GRU predictor with vocabulary management."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.vocab = {}  # file_path -> id
        self.id_to_path = {}  # id -> file_path
        self.next_id = 0
        self.model = None
        self.model_path = model_path
        self.sequence_window = 10
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def add_file_to_vocab(self, file_path: str) -> int:
        """Add file to vocabulary."""
        if file_path not in self.vocab:
            self.vocab[file_path] = self.next_id
            self.id_to_path[self.next_id] = file_path
            self.next_id += 1
        return self.vocab[file_path]
    
    def get_file_id(self, file_path: str) -> int:
        """Get file ID from path."""
        if file_path not in self.vocab:
            return self.add_file_to_vocab(file_path)
        return self.vocab[file_path]
    
    def get_file_path(self, file_id: int) -> Optional[str]:
        """Get file path from ID."""
        return self.id_to_path.get(file_id)
    
    def build_model(self, hidden_dim: int = 128, num_layers: int = 1):
        """Build GRU model."""
        vocab_size = len(self.vocab) or 1000  # Default if empty
        self.model = FileAccessGRU(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        self.model.eval()
    
    def train(self, training_data: List[Tuple], epochs: int = 10, lr: float = 0.001):
        """Train model on trace data."""
        if not self.model:
            self.build_model()
        
        # Convert traces to training data
        # Format: (file_ids, extra_features, target_file_id)
        # This is a simplified version - full implementation would need
        # proper data loading, batching, etc.
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            # Training loop would go here
            # For now, this is a placeholder
            pass
    
    def predict(self, worker_id: int, recent_files: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next files using the model."""
        if not self.model or len(recent_files) == 0:
            return []
        
        # Convert files to IDs
        file_ids = [self.get_file_id(f) for f in recent_files[-self.sequence_window:]]
        
        # Create dummy extra features (offset, time, worker)
        # In real implementation, these would be computed from trace data
        import numpy as np
        batch_size = 1
        seq_len = len(file_ids)
        extra_features = np.zeros((batch_size, seq_len, 8))
        extra_features[0, :, 6] = worker_id  # worker ID feature
        
        # Convert to tensors
        file_ids_tensor = torch.tensor([file_ids], dtype=torch.long)
        extra_features_tensor = torch.tensor(extra_features, dtype=torch.float32)
        
        with torch.no_grad():
            logits, _ = self.model(file_ids_tensor, extra_features_tensor)
            probs = torch.softmax(logits[0], dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.vocab)))
        
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            file_path = self.get_file_path(idx.item())
            if file_path:
                predictions.append((file_path, prob.item()))
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model and vocabulary."""
        if not self.model:
            return
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'id_to_path': self.id_to_path,
            'next_id': self.next_id,
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model and vocabulary."""
        if not os.path.exists(filepath):
            return
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.vocab = checkpoint['vocab']
        self.id_to_path = checkpoint['id_to_path']
        self.next_id = checkpoint['next_id']
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

