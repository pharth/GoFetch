"""PyTorch/TensorFlow DataLoader instrumentation hook for trace collection."""

import os
import functools
from typing import Optional
from collector.trace_schema import TraceEvent, AccessType
from storage.trace_db import TraceDatabase


class DataLoaderHook:
    """Hook for instrumenting PyTorch DataLoaders to emit traces."""
    
    def __init__(self, db_path: str = "traces.db", worker_id: Optional[int] = None):
        self.db = TraceDatabase(db_path)
        self.worker_id = worker_id or 0
        self.pid = os.getpid()
        self.epoch = 0
        self.hooked = False
    
    def set_epoch(self, epoch: int):
        """Set current training epoch."""
        self.epoch = epoch
    
    def hook_pytorch_dataloader(self):
        """Hook PyTorch DataLoader to intercept file accesses."""
        if self.hooked:
            return
        
        try:
            import torch
            from torch.utils.data import DataLoader
            
            # Monkey-patch Dataset to intercept __getitem__
            original_getitem = None
            
            def patched_getitem(self, idx):
                result = original_getitem(self, idx)
                
                # Try to extract file path from the result
                if isinstance(result, (tuple, list)):
                    for item in result:
                        if hasattr(item, 'path'):  # Some dataset types store path
                            self._log_file_access(item.path)
                elif hasattr(result, 'path'):
                    self._log_file_access(result.path)
                
                return result
            
            # Store original before patching
            import torch.utils.data
            DataLoader.__getitem__ = original_getitem
            
            # This is a simplified version - full implementation would
            # need to detect PIL.Image.open, etc.
            self.hooked = True
            
        except ImportError:
            print("PyTorch not available, skipping DataLoader hook")
    
    def hook_file_operations(self):
        """Hook common file operations to trace accesses."""
        self._hook_pil()
        self._hook_open()
    
    def _hook_pil(self):
        """Hook PIL/Pillow Image.open."""
        try:
            from PIL import Image
            original_open = Image.open
            
            @functools.wraps(original_open)
            def traced_open(fp, mode='r'):
                self._log_file_access(fp if isinstance(fp, str) else getattr(fp, 'name', ''))
                return original_open(fp, mode)
            
            Image.open = traced_open
            
        except ImportError:
            pass
    
    def _hook_open(self):
        """Hook Python's built-in open() function."""
        import builtins
        original_open = builtins.open
        
        @functools.wraps(original_open)
        def traced_open(name, mode='r', buffering=-1, encoding=None, **kwargs):
            if 'r' in mode or 'a' in mode or 'w' in mode:  # Any access
                self._log_file_access(name)
            return original_open(name, mode, buffering, encoding, **kwargs)
        
        builtins.open = traced_open
    
    def _log_file_access(self, file_path: str, offset: int = 0, length: int = 0):
        """Log a file access event."""
        event = TraceEvent(
            timestamp=TraceEvent.now_ns(),
            pid=self.pid,
            worker_id=self.worker_id,
            file_path=file_path,
            offset=offset,
            length=length,
            access_type=AccessType.READ,
            epoch=self.epoch
        )
        try:
            self.db.add_event(event)
        except Exception as e:
            print(f"Failed to log trace event: {e}")
    
    def setup_torch_dataloader(self):
        """Setup instrumentation for PyTorch training scripts."""
        # Hook DataLoader's __iter__ to track worker/epoch
        import torch
        
        original_iter = torch.utils.data.DataLoader.__iter__
        
        @functools.wraps(original_iter)
        def traced_iter(self):
            if hasattr(self, 'worker_init_fn'):
                # Store original worker_init_fn
                old_worker_init = self.worker_init_fn
                
                def new_worker_init(worker_id):
                    # Update hook's worker_id for this worker
                    if old_worker_init:
                        old_worker_init(worker_id)
                self.worker_init_fn = new_worker_init
            
            return original_iter(self)
        
        torch.utils.data.DataLoader.__iter__ = traced_iter
        
        # Enable hooks
        self.hook_file_operations()
        print("DataLoader hooks enabled")


# Global instance for easy import
_global_hook: Optional[DataLoaderHook] = None


def get_hook(db_path: str = "traces.db") -> DataLoaderHook:
    """Get or create global DataLoaderHook instance."""
    global _global_hook
    if _global_hook is None:
        _global_hook = DataLoaderHook(db_path)
    return _global_hook


def enable_hook(db_path: str = "traces.db"):
    """Enable DataLoader tracing hook."""
    hook = get_hook(db_path)
    hook.hook_file_operations()
    hook.setup_torch_dataloader()
    return hook


if __name__ == "__main__":
    # Example usage
    print("DataLoader hook - import this module in your training script")
    print("Usage: from collector.dataloader_hook import enable_hook; enable_hook()")

