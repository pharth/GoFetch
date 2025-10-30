# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Trace Collection Layer                    │
├──────────────────────┬──────────────────────────────────────┤
│   eBPF Tracer        │    DataLoader Hook                   │
│   (kernel-level)     │    (userspace)                       │
│   • Syscall hooks    │    • PIL.Image.open                   │
│   • No code changes  │    • Python open()                   │
│   • Requires root    │    • No root needed                 │
└──────────┬───────────┴────────────┬─────────────────────────┘
           │                        │
           └──────────┬─────────────┘
                      │
           ┌──────────▼──────────┐
           │   Trace Database    │
           │   (SQLite)          │
           │   • Event storage   │
           │   • Indexed queries │
           │   • Retention mgmt │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │   Predictor Engine │
           ├────────────────────┤
           │   • Baseline       │  ┌──────────────┐
           │     (Markov chain)  │  │ Fast path    │
           │   • ML Model        │  │ O(1) lookup  │
           │     (GRU)           │  │              │
           │   • Hybrid logic    │  │              │
           └──────────┬──────────┘  └──────────────┘
                      │
           ┌──────────▼──────────┐
           │   Policy Engine     │
           ├────────────────────┤
           │   • Confidence     │
           │   • Rate limiting   │
           │   • Size limits     │
           │   • Cache checks    │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │   Prefetch Daemon   │
           │   • Main loop       │
           │   • Predictions     │
           │   • Syscall calls   │
           │   • Metrics         │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │   Kernel (Linux)    │
           │   • posix_fadvise   │
           │   • Page cache      │
           │   • Prefetch hint   │
           └─────────────────────┘
```

## Data Flow

### 1. Training → Tracing
```
Training Script
  └─ File.read() / DataLoader
      └─ [Tracer hooks] → TraceEvent
          └─ SQLite Database
```

### 2. Daemon → Prefetch
```
Daemon (100ms loop)
  ├─ Read recent traces from DB
  ├─ Update predictor state
  ├─ Get predictions (top-K)
  ├─ Apply policy filters
  └─ Call posix_fadvise()
      └─ Kernel page cache warmed
```

### 3. Prediction → Cache Hit
```
Predictor
  ├─ Observed: [file_A, file_B, file_C]
  ├─ Learned: After [A, B, C] → likely file_D
  └─ Action: posix_fadvise(file_D)
      └─ When training reads file_D → Cache hit! ✓
```

## Component Details

### Trace Collector
**Role**: Capture file access events with minimal overhead

**Methods**:
- **eBPF**: Kernel probes (`sys_pread64`, `sys_read`) → ring buffer → userspace
- **Hook**: Monkey-patch Python functions → direct emission

**Output**: `TraceEvent` (timestamp, pid, worker_id, file_path, offset, length, access_type, epoch)

### Predictor
**Role**: Learn access patterns and predict next files

**Baseline Algorithm**:
```
State: {context_tuple → {file → count}}
Example: (file_A, file_B) → {file_C: 10, file_D: 5, ...}

Prediction:
1. Look up last N files
2. Find contexts matching recent history
3. Score next files by transition counts
4. Return top-K with confidence scores
```

**ML Algorithm**:
```
Input: [file_ids, offset_bins, time_deltas, worker_id]
  ↓
Embeddings: file_emb + feature_emb
  ↓
GRU: hidden state → context
  ↓
Linear: context → vocab_logits
  ↓
Output: top-K file predictions with probabilities
```

### Policy Engine
**Role**: Decide which predictions to act on

**Filters**:
- Confidence > threshold (default 0.7)
- Rate limit: max N ops/sec (default 10)
- File size limit (default 16MB)
- Already cached? Skip
- System overloaded? Pause

**Rate Limiting**:
```
last_prefetch_time[file_path] = now
if (now - last_prefetch < 1.0 / rate_limit_per_sec):
    skip this file
```

### Prefetch Daemon
**Role**: Orchestrate the entire system

**Loop** (every 100ms):
1. Poll trace database for new events
2. Update predictor state per worker
3. Get predictions for active workers
4. Apply policy filters
5. Call posix_fadvise() for approved predictions
6. Record metrics
7. Sleep until next poll

**Metrics**:
- Cache hit/miss counts
- Prediction accuracy
- Prefetch latency
- Total prefetches issued
- Active workers count

## State Management

### Per-Worker State
```
worker_sequences: {
    worker_0: [file_001.bin, file_005.bin, file_003.bin, ...],
    worker_1: [file_042.bin, file_088.bin, ...],
    ...
}

last_predictions: {
    worker_0: [(file_001.bin, 0.95), (file_007.bin, 0.82), ...],
    ...
}
```

### Global State
```
transitions: {
    (file_A, file_B): {file_C: 10, file_D: 5},
    (file_B, file_C): {file_D: 8},
    ...
}

file_counts: {
    "file_A": 1234,
    "file_B": 987,
    ...
}
```

## Integration Points

### With Training Code
**Option 1 (Recommended)**: DataLoader Hook
```python
from collector.dataloader_hook import enable_hook
enable_hook("traces.db")

# Your training code (unchanged)
train(model, dataloader)
```

**Option 2**: eBPF (transparent)
```bash
# No code changes needed
sudo python -m collector.ebpf_tracer &
python your_training.py
```

### With Prefetcher Daemon
```bash
# Start daemon (before training)
python -m prefetcher.daemon --config config.yaml

# Run training
python your_training.py

# Check metrics
cat metrics.csv
```

## Performance Characteristics

### Latency Budget
- Trace collection: <1µs per event (eBPF) / <10µs (hook)
- Database insert: <1ms per event
- Prediction: <1ms (baseline) / <10ms (ML)
- posix_fadvise: <100µs
- **Total per prediction: <15ms**

### Resource Usage
- Memory: ~100MB (daemon) + model size
- CPU: <5% for daemon loop
- Disk: Trace DB grows ~1MB/1000 events
- Network: None (local page cache only)

### Scalability
- Traces: Millions of events (SQLite handles well)
- Files: Unlimited (vocabulary grows dynamically)
- Workers: Supports 100+ concurrent workers
- Processes: eBPF can trace 100+ PIDs

## Failure Modes & Recovery

### Tracer Failure
- **Symptom**: No traces collected
- **Fallback**: Continue training normally (no tracing)

### Database Full
- **Symptom**: Inserts fail
- **Recovery**: Automatic cleanup (retention days)

### Predictor Failure
- **Symptom**: No predictions
- **Fallback**: Baseline predictor or skip prefetching

### Syscall Failure
- **Symptom**: posix_fadvise returns error
- **Recovery**: Log and continue (non-fatal)

### Daemon Crash
- **Symptom**: Prefetching stops
- **Recovery**: Training continues normally, just without prefetching

## Extensibility

### Add New Predictor
1. Create class in `predictor/`
2. Implement `observe()`, `predict()` methods
3. Add to `HybridPredictor` in `predictor/inference.py`

### Add New Trace Source
1. Create collector in `collector/`
2. Emit `TraceEvent` objects
3. Write to database via `TraceDatabase`

### Add New Policy
1. Extend `PrefetchPolicy` in `prefetcher/policy.py`
2. Implement custom decision logic
3. Plug into daemon's filter pipeline

