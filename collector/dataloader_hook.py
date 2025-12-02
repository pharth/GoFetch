"""PyTorch/TensorFlow DataLoader instrumentation hook for trace collection."""

import os
import functools
from typing import Optional
from collector.trace_schema import TraceEvent, AccessType
from storage.trace_db import TraceDatabase

# --- Global Helper for Pickling Support ---
def _worker_init_wrapper(worker_id, original_init_fn=None, db_path="traces.db"):
    """
    Top-level wrapper to initialize the hook in a child process.
    Must be top-level to be pickleable on Windows.
    """
    # 1. Get/Create the global hook instance in this new process
    hook = get_hook(db_path)
    
    # 2. Set the worker ID for this process
    hook.worker_id = worker_id
    
    # 3. Call the user's original init function if it existed
    if original_init_fn:
        original_init_fn(worker_id)

class DataLoaderHook:
    """Hook for instrumenting PyTorch DataLoaders to emit traces."""
    
    def __init__(self, db_path: str = "traces.db", worker_id: Optional[int] = None):
        self.db_path = db_path
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
                # (This logic depends heavily on dataset implementation)
                if isinstance(result, (tuple, list)):
                    for item in result:
                        if hasattr(item, 'path'):  # Some dataset types store path
                            # We access the global hook to ensure we use the instance 
                            # specific to this process/worker
                            get_hook()._log_file_access(item.path)
                elif hasattr(result, 'path'):
                    get_hook()._log_file_access(result.path)
                
                return result
            
            # Store original before patching
            # Note: In a robust production system, we'd patch specific dataset classes
            # rather than the abstract base class to avoid side effects.
            # For this demo, we rely on the file system hooks mainly.
            
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
                # Log access using the global hook instance
                # This ensures we pick up the correct worker_id in child processes
                get_hook()._log_file_access(fp if isinstance(fp, str) else getattr(fp, 'name', ''))
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
            # FIX: Convert name to string to handle pathlib.Path objects
            name_str = str(name)
            
            # Log read accesses
            if ('r' in mode or 'a' in mode): 
                # Avoid infinite recursion loop with self-logging
                if "traces.db" not in name_str and ".py" not in name_str:
                    get_hook()._log_file_access(name_str)
                    
            return original_open(name, mode, buffering, encoding, **kwargs)
        
        builtins.open = traced_open
    
    def _log_file_access(self, file_path: str, offset: int = 0, length: int = 0):
        """Log a file access event."""
        # Filter out internal python files or db files to reduce noise
        if not file_path or "traces.db" in file_path or ".py" in file_path:
            return

        event = TraceEvent(
            timestamp=TraceEvent.now_ns(),
            pid=os.getpid(),
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
            # Silently fail on logging errors to not crash training
            pass
    
    def setup_torch_dataloader(self):
        """Setup instrumentation for PyTorch training scripts."""
        import torch
        from torch.utils.data import DataLoader

        original_iter = DataLoader.__iter__
        
        @functools.wraps(original_iter)
        def traced_iter(loader_self):
            # This runs in the main process when the iterator is created.
            
            # We wrap the existing worker_init_fn with our pickleable global wrapper.
            # This ensures that when workers start, they set up their own Hook instance.
            old_worker_init = getattr(loader_self, 'worker_init_fn', None)
            
            # Use functools.partial to create a pickleable callable
            loader_self.worker_init_fn = functools.partial(
                _worker_init_wrapper, 
                original_init_fn=old_worker_init,
                db_path=self.db_path
            )
            
            return original_iter(loader_self)
        
        DataLoader.__iter__ = traced_iter
        
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
    hook.setup_torch_dataloader()
    return hook


if __name__ == "__main__":
    # Example usage
    print("DataLoader hook - import this module in your training script")
    print("Usage: from collector.dataloader_hook import enable_hook; enable_hook()")
