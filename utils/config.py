"""Configuration management for dl-file-prefetcher."""

import os
import yaml
import argparse
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager with YAML and CLI override support."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create config from CLI arguments."""
        config = cls(args.config if hasattr(args, 'config') else "config.yaml")
        
        # Override with CLI args
        if hasattr(args, 'model_path'):
            config.set('predictor.ml.model_path', args.model_path)
        if hasattr(args, 'confidence'):
            config.set('prefetcher.confidence_threshold', args.confidence)
        if hasattr(args, 'rate_limit'):
            config.set('prefetcher.rate_limit_per_sec', args.rate_limit)
            
        return config

