"""
Progress tracking utilities for long-running operations
"""

import time
from tqdm import tqdm
from typing import Optional, Callable, Any
import contextlib

class ProgressTracker:
    """Progress tracking for pipeline operations."""
    
    def __init__(self, description: str = "Processing", total: Optional[int] = None):
        self.description = description
        self.total = total
        self.pbar = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total, desc=self.description, 
                        unit="items", ncols=100, leave=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"⏱️  {self.description} completed in {elapsed:.2f} seconds")
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """Update progress."""
        if self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(n)
    
    def set_description(self, description: str):
        """Set progress description."""
        if self.pbar:
            self.pbar.set_description(description)

@contextlib.contextmanager
def track_progress(description: str, total: Optional[int] = None):
    """Context manager for progress tracking."""
    with ProgressTracker(description, total) as tracker:
        yield tracker

def with_progress(description: str, total: Optional[int] = None):
    """Decorator for progress tracking."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with track_progress(description, total) as tracker:
                return func(*args, **kwargs)
        return wrapper
    return decorator
