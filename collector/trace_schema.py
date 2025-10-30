"""Common trace event schema and utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


class AccessType(Enum):
    """File access type enumeration."""
    READ = "read"
    MMAP = "mmap"
    OPEN = "open"


@dataclass
class TraceEvent:
    """Single trace event representing a file access."""
    
    timestamp: int  # nanoseconds
    pid: int
    worker_id: int  # DataLoader worker ID
    file_path: str
    offset: int  # file offset in bytes
    length: int  # bytes read
    access_type: AccessType = AccessType.READ
    epoch: int = 0  # training epoch
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'pid': self.pid,
            'worker_id': self.worker_id,
            'file_path': self.file_path,
            'offset': self.offset,
            'length': self.length,
            'access_type': self.access_type.value,
            'epoch': self.epoch
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TraceEvent':
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            pid=data['pid'],
            worker_id=data['worker_id'],
            file_path=data['file_path'],
            offset=data['offset'],
            length=data['length'],
            access_type=AccessType(data['access_type']),
            epoch=data.get('epoch', 0)
        )
    
    @staticmethod
    def now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()

