"""SQLite trace database for storing file access events."""

import sqlite3
import os
import sys
import time
from typing import List, Optional, Iterator
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from collector.trace_schema import TraceEvent


class TraceDatabase:
    """SQLite database for storing trace events."""
    
    def __init__(self, db_path: str = "traces.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                pid INTEGER NOT NULL,
                worker_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL,
                access_type TEXT NOT NULL,
                epoch INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON traces(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pid 
            ON traces(pid)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_worker_id 
            ON traces(worker_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pid_timestamp 
            ON traces(pid, timestamp)
        """)
        
        self.conn.commit()
    
    def add_event(self, event: TraceEvent) -> int:
        """Add a trace event to the database."""
        cursor = self.conn.cursor()
        data = event.to_dict()
        cursor.execute("""
            INSERT INTO traces (timestamp, pid, worker_id, file_path, offset, length, access_type, epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['timestamp'],
            data['pid'],
            data['worker_id'],
            data['file_path'],
            data['offset'],
            data['length'],
            data['access_type'],
            data['epoch']
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent(self, since_ns: Optional[int] = None, limit: int = 1000) -> List[TraceEvent]:
        """Get recent trace events since a timestamp."""
        cursor = self.conn.cursor()
        
        if since_ns is not None:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE timestamp > ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """, (since_ns, limit))
        else:
            cursor.execute("""
                SELECT * FROM traces 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        return [TraceEvent.from_dict(dict(row)) for row in rows]
    
    def get_by_pid(self, pid: int, since_ns: Optional[int] = None) -> List[TraceEvent]:
        """Get traces for a specific process."""
        cursor = self.conn.cursor()
        
        if since_ns is not None:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE pid = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (pid, since_ns))
        else:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE pid = ?
                ORDER BY timestamp ASC
            """, (pid,))
        
        rows = cursor.fetchall()
        return [TraceEvent.from_dict(dict(row)) for row in rows]
    
    def get_by_worker(self, worker_id: int, since_ns: Optional[int] = None) -> List[TraceEvent]:
        """Get traces for a specific worker."""
        cursor = self.conn.cursor()
        
        if since_ns is not None:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE worker_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (worker_id, since_ns))
        else:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE worker_id = ?
                ORDER BY timestamp ASC
            """, (worker_id,))
        
        rows = cursor.fetchall()
        return [TraceEvent.from_dict(dict(row)) for row in rows]
    
    def get_active_pids(self, since_ns: Optional[int] = None) -> List[int]:
        """Get list of active process IDs."""
        cursor = self.conn.cursor()
        
        if since_ns is not None:
            cursor.execute("""
                SELECT DISTINCT pid FROM traces 
                WHERE timestamp > ?
            """, (since_ns,))
        else:
            cursor.execute("""
                SELECT DISTINCT pid FROM traces
            """)
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_active_workers(self, since_ns: Optional[int] = None) -> List[int]:
        """Get list of active worker IDs."""
        cursor = self.conn.cursor()
        
        if since_ns is not None:
            cursor.execute("""
                SELECT DISTINCT worker_id FROM traces 
                WHERE timestamp > ?
            """, (since_ns,))
        else:
            cursor.execute("""
                SELECT DISTINCT worker_id FROM traces
            """)
        
        return [row[0] for row in cursor.fetchall()]
    
    def cleanup_old(self, retention_days: int = 7):
        """Clean up traces older than retention_days."""
        cursor = self.conn.cursor()
        cutoff_ns = time.time_ns() - (retention_days * 24 * 60 * 60 * 1_000_000_000)
        
        cursor.execute("""
            DELETE FROM traces WHERE timestamp < ?
        """, (cutoff_ns,))
        
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM traces")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT pid) FROM traces")
        unique_pids = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT worker_id) FROM traces")
        unique_workers = cursor.fetchone()[0]
        
        return {
            'total_events': total,
            'unique_pids': unique_pids,
            'unique_workers': unique_workers
        }
    
    def export_to_parquet(self, output_path: str):
        """Export traces to Parquet format for ML training."""
        import pandas as pd
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM traces ORDER BY timestamp")
        rows = cursor.fetchall()
        
        df = pd.DataFrame([dict(row) for row in rows])
        df.to_parquet(output_path)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

