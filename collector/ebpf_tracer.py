"""eBPF-based file access tracer (requires root)."""

import os
import sys
import argparse
import time
from storage.trace_db import TraceDatabase
from collector.trace_schema import TraceEvent, AccessType

# Try to import BCC (only available on Linux with BCC installed)
try:
    from bcc import BPF
    BCC_AVAILABLE = True
except ImportError:
    BCC_AVAILABLE = False
    print("Warning: bcc not available. eBPF tracing disabled.")
    print("Install with: sudo apt-get install python3-bpfcc")


# eBPF program to trace file access
BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>

struct trace_event {
    u64 timestamp;
    u32 pid;
    u32 worker_id;
    int fd;
    u64 offset;
    u64 length;
    char file_path[256];
};

BPF_PERF_OUTPUT(events);
BPF_HASH(fd_map, int, char[256]);

int trace_pread64_enter(struct pt_regs *ctx, int fd, void __user *buf, size_t count, loff_t pos) {
    struct trace_event event = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    
    event.timestamp = bpf_ktime_get_ns();
    event.pid = pid_tgid >> 32;
    event.fd = fd;
    event.offset = pos;
    event.length = count;
    event.worker_id = 0;  // Would need additional logic to determine worker
    
    // Try to get file path from fd map (populated by open hook)
    char *path = fd_map.lookup(&fd);
    if (path) {
        __builtin_memcpy(event.file_path, path, sizeof(event.file_path));
    }
    
    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}

int trace_openat_enter(struct pt_regs *ctx, int dfd, const char __user *filename, int flags) {
    // Store mapping of fd to path (simplified)
    // In production, would need more robust fd tracking
    return 0;
}
"""


class EBPFTracer:
    """eBPF-based file access tracer."""
    
    def __init__(self, db_path: str = "traces.db", target_processes: list = None):
        self.db = TraceDatabase(db_path)
        self.target_processes = target_processes or ["python"]
        self.bpf = None
        self.running = False
        
        if not BCC_AVAILABLE:
            raise RuntimeError("BCC not available. Cannot use eBPF tracing.")
    
    def start(self):
        """Start eBPF tracing."""
        print("Starting eBPF tracer...")
        
        try:
            self.bpf = BPF(text=BPF_PROGRAM)
            
            # Attach to syscalls
            self.bpf.attach_kprobe(event="sys_pread64", fn_name="trace_pread64_enter")
            # Note: syscall names vary by kernel version
            # May need: __x64_sys_pread64, __se_sys_pread64, etc.
            
            print("eBPF tracer started. Press Ctrl+C to stop.")
            self.running = True
            
            # Open perf buffer
            self.bpf["events"].open_perf_buffer(self._process_event)
            
            # Poll loop
            while self.running:
                self.bpf.perf_buffer_poll(timeout=1000)
                
        except Exception as e:
            print(f"Failed to start eBPF tracer: {e}")
            print("Note: eBPF requires root privileges and Linux kernel")
            sys.exit(1)
    
    def _process_event(self, cpu, data, size):
        """Process trace event from eBPF."""
        try:
            event = self.bpf["events"].event(data)
            
            # Check if process matches target
            proc_name = self._get_process_name(event.pid)
            if not any(target in proc_name for target in self.target_processes):
                return
            
            # Convert to trace event
            trace = TraceEvent(
                timestamp=event.timestamp,
                pid=event.pid,
                worker_id=event.worker_id,
                file_path=event.file_path.decode('utf-8', errors='ignore'),
                offset=event.offset,
                length=event.length,
                access_type=AccessType.READ
            )
            
            # Store in database
            self.db.add_event(trace)
            
        except Exception as e:
            print(f"Error processing event: {e}")
    
    def _get_process_name(self, pid: int) -> str:
        """Get process name for PID."""
        try:
            with open(f"/proc/{pid}/comm", 'r') as f:
                return f.read().strip()
        except:
            return ""
    
    def stop(self):
        """Stop tracing."""
        self.running = False
        self.db.close()
        print("Tracer stopped.")


def main():
    """Main entry point for eBPF tracer."""
    parser = argparse.ArgumentParser(description="eBPF File Access Tracer")
    parser.add_argument("--db", default="traces.db", help="Trace database path")
    parser.add_argument("--target", action='append', default=["python"], 
                       help="Target processes to trace")
    
    args = parser.parse_args()
    
    if not BCC_AVAILABLE:
        print("ERROR: BCC not available.")
        print("Install with:")
        print("  sudo apt-get install python3-bpfcc bpfcc-tools")
        print("Or use the DataLoader hook instead (no root required)")
        sys.exit(1)
    
    tracer = EBPFTracer(db_path=args.db, target_processes=args.target)
    
    try:
        tracer.start()
    except KeyboardInterrupt:
        print("\nStopping tracer...")
    finally:
        tracer.stop()


if __name__ == "__main__":
    main()

