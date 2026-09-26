#!/usr/bin/env python3
"""
CLI Entry point for Business Entity Resolution Pipeline.

Usage:
    # Full end-to-end (Full Dataset Train -> Tune -> Test Inference):
    python run_pipeline.py --mode all

    # Fast development run:
    python run_pipeline.py --mode all --dev

    # Inference only using saved model artifact:
    python run_pipeline.py --mode predict
"""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="OptiResolve Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["all", "train", "predict"], default="all", help="Pipeline execution mode")
    parser.add_argument("--dev", action="store_true", help="Development mode (subsamples train/val for rapid execution)")
    parser.add_argument("--train-limit", type=int, default=None, help="Explicit max S1 entities for training (None = full)")
    parser.add_argument("--val-limit", type=int, default=None, help="Explicit max S1 entities for validation (None = full)")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit threshold override")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for streaming test inference")
    args = parser.parse_args()

    config = PipelineConfig()

    if args.dev:
        config.train_s1_limit = 50000
        config.val_s1_limit = 10000
    else:
        config.train_s1_limit = args.train_limit
        config.val_s1_limit = args.val_limit

    if args.threshold is not None:
        config.default_threshold = args.threshold

    pipeline = EntityResolutionPipeline(config)

    if args.mode in ("all", "train"):
        pipeline.fit()

    if args.mode in ("all", "predict"):
        if args.mode == "predict" and config.paths.model_path.exists():
            pipeline.model.load(config.paths.model_path)
        pipeline.predict_test(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
