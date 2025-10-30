"""Training pipeline for ML predictor."""

import argparse
from storage.trace_db import TraceDatabase
from predictor.baseline import MarkovPredictor
from predictor.ml_model import FileAccessPredictor


def train_baseline(db_path: str = "traces.db", output_path: str = "models/baseline.json"):
    """Train baseline Markov predictor from traces."""
    print("Training baseline predictor...")
    
    db = TraceDatabase(db_path)
    predictor = MarkovPredictor(ngram_size=3, decay_factor=0.95)
    
    # Get all traces
    traces = db.get_recent(limit=100000)
    print(f"Loaded {len(traces)} trace events")
    
    # Process traces
    for trace in traces:
        predictor.observe(
            worker_id=trace.worker_id,
            file_path=trace.file_path,
            epoch=trace.epoch
        )
    
    # Save model
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictor.save(output_path)
    print(f"Saved baseline model to {output_path}")
    
    db.close()


def train_ml(db_path: str = "traces.db", output_path: str = "models/prefetch_gru.pt", epochs: int = 10):
    """Train ML predictor from traces."""
    print("Training ML predictor...")
    print("Note: Full ML training requires additional implementation")
    print(f"Would train for {epochs} epochs and save to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train predictor models")
    parser.add_argument("--db", default="traces.db", help="Trace database path")
    parser.add_argument("--output", default="models/baseline.json", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (for ML)")
    parser.add_argument("--model-type", choices=["baseline", "ml", "both"], default="baseline",
                       help="Which model to train")
    
    args = parser.parse_args()
    
    if args.model_type in ["baseline", "both"]:
        train_baseline(args.db, args.output)
    
    if args.model_type in ["ml", "both"]:
        train_ml(args.db, args.output.replace(".json", ".pt"), args.epochs)


if __name__ == "__main__":
    main()

