"""Syscall wrappers for posix_fadvise and readahead."""

import os
import ctypes
import errno
from typing import Optional
from pathlib import Path
import sys


# Linux-specific: might need adjustment for different platforms
POSIX_FADV_WILLNEED = 3  # From bits/fcntl-linux.h


def posix_fadvise(fd: int, offset: int, length: int, advice: int = POSIX_FADV_WILLNEED) -> int:
    """
    Advise the kernel about expected file access pattern.
    
    Args:
        fd: File descriptor
        offset: File offset in bytes
        length: Number of bytes
        advice: Advice value (POSIX_FADV_WILLNEED, etc.)
    
    Returns:
        0 on success, error code on failure
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
        libc.posix_fadvise.restype = ctypes.c_int
        
        result = libc.posix_fadvise(
            ctypes.c_int(fd),
            ctypes.c_uint64(offset),
            ctypes.c_uint64(length),
            ctypes.c_int(advice)
        )
        
        if result != 0:
            err = ctypes.get_errno()
            print(f"posix_fadvise failed: errno={err}")
        
        return result
    except Exception as e:
        print(f"posix_fadvise error: {e}")
        return -1


def readahead_syscall(fd: int, offset: int, count: int) -> int:
    """
    Initiate readahead into page cache (Linux-specific syscall).
    
    Args:
        fd: File descriptor
        offset: File offset in bytes
        count: Number of bytes to read ahead
    
    Returns:
        0 on success, error code on failure
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        
        # readahead signature: int readahead(int fd, off64_t offset, size_t count)
        libc.readahead.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t]
        libc.readahead.restype = ctypes.c_int
        
        result = libc.readahead(
            ctypes.c_int(fd),
            ctypes.c_uint64(offset),
            ctypes.c_size_t(count)
        )
        
        if result != 0:
            err = ctypes.get_errno()
            print(f"readahead failed: errno={err}")
        
        return result
    except Exception as e:
        print(f"readahead error: {e}")
        return -1


def advise_file(file_path: str, offset: int = 0, length: int = 16 * 1024 * 1024) -> bool:
    """
    Issue posix_fadvise for a file path (Linux) or simulate it (Windows).
    """
    if not os.path.exists(file_path):
        return False

    # --- Windows Fallback ---
    if os.name == 'nt':
        # Windows doesn't have posix_fadvise. 
        # We simulate a "prefetch" by doing a tiny 1-byte read 
        # which triggers the OS to cache the file metadata and first page.
        try:
            with open(file_path, 'rb') as f:
                f.seek(offset)
                f.read(1)
            return True
        except Exception:
            return False

    # --- Linux Implementation ---
    try:
        fd = os.open(file_path, os.O_RDONLY)
        ret = posix_fadvise(fd, offset, length, POSIX_FADV_WILLNEED)
        os.close(fd)
        return ret == 0
    except Exception as e:
        print(f"Failed to advise file {file_path}: {e}")
        return False


def readahead_file(file_path: str, offset: int = 0, count: int = 16 * 1024 * 1024) -> bool:
    """
    Issue readahead syscall for a file path.
    
    Args:
        file_path: Path to file
        offset: Offset in bytes
        count: Count in bytes
    
    Returns:
        True on success, False on failure
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        fd = os.open(file_path, os.O_RDONLY)
        result = readahead_syscall(fd, offset, count)
        os.close(fd)
        return result == 0
    except Exception as e:
        print(f"Failed to readahead file {file_path}: {e}")
        return False


def is_file_cached(file_path: str, offset: int = 0, length: int = 1024 * 1024) -> bool:
    """
    Check if file pages are in cache using mincore.
    
    Args:
        file_path: Path to file
        offset: Offset in bytes
        length: Length to check
    
    Returns:
        True if pages are cached, False otherwise
    """
    try:
        # This is a simplified check - full implementation would use
        # mmap + mincore to check page residency
        return False  # Placeholder
    except Exception:
        return False

