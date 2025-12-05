# 🐕 GoFetch

A smart file prefetching system for deep learning workloads that predicts and pre-loads data files into memory before they're needed, reducing I/O bottlenecks during model training.

![Python](https://img. shields.io/badge/Python-3.8%2B-blue)
![License](https://img. shields.io/badge/License-MIT-green)

## 📖 Overview

**GoFetch** is an intelligent prefetching daemon designed to accelerate deep learning training pipelines.  It learns file access patterns during training and proactively loads files into the kernel page cache, minimizing disk I/O wait times.

### Key Features

- **🔍 Access Pattern Learning** — Uses Markov chains and optional GRU neural networks to predict which files will be accessed next
- **⚡ Kernel-Level Prefetching** — Leverages `posix_fadvise()` to hint the kernel about upcoming file reads
- **🔧 Flexible Tracing** — Supports both eBPF (kernel-level) and Python hook-based trace collection
- **📊 Real-Time Metrics** — Tracks cache hits, prediction accuracy, and prefetch latency
- **🎛️ Configurable Policies** — Choose between FIFO or Heap-based prefetch scheduling

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Linux OS (for eBPF and `posix_fadvise` support)
- Root access (only for eBPF tracing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pharth/GoFetch.git
   cd GoFetch
   ```

2.  **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **For eBPF support (Ubuntu/Debian)**
   ```bash
   sudo apt-get install python3-bpfcc
   ```

## 📝 Usage

### Option 1: DataLoader Hook (Recommended)

Add the hook to your training script — no root access required:

```python
from collector.dataloader_hook import enable_hook

# Enable tracing
enable_hook("traces.db")

# Your training code (unchanged)
train(model, dataloader)
```

### Option 2: eBPF Tracing (Transparent)

Run the tracer in the background — no code changes needed:

```bash
# Terminal 1: Start the eBPF tracer (requires root)
sudo python -m collector.ebpf_tracer &

# Terminal 2: Run your training script normally
python your_training. py
```

### Running the Prefetch Daemon

Start the daemon before your training begins:

```bash
# Start the prefetcher daemon
python -m prefetcher.daemon --config config.yaml

# Check metrics after training
cat metrics_heap.csv
```

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
storage:
  db_path: "traces.db"

prefetcher:
  policy_type: "heap"          # Options: "heap" or "fifo"
  confidence_threshold: 0.6     # Minimum confidence to prefetch
  rate_limit_per_sec: 20        # Max prefetch operations per second
  top_k_predictions: 5          # Number of files to predict
  max_prefetch_size_mb: 16      # Maximum file size to prefetch

predictor:
  baseline:
    enabled: true               # Markov chain predictor
  ml:
    enabled: false              # GRU neural network predictor
    model_path: "models/prefetch_gru.pt"
```

## 🔧 How It Works

1.  **Trace Collection** — Captures file access events (path, offset, timestamp) during training
2. **Pattern Learning** — Builds transition probabilities: "After accessing file A then B, file C is likely next"
3. **Prediction** — Uses learned patterns to predict upcoming file accesses
4. **Prefetching** — Calls `posix_fadvise()` to hint the kernel to load predicted files into cache
5. **Cache Hit** — When training reads the file, it's already in memory!  ✓

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📚 Learn More

- See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design
- Check the `examples/` directory for sample usage

## 📄 License

This project is open source and available under the [MIT License](LICENSE). 

---

Made with ❤️ for faster deep learning training
