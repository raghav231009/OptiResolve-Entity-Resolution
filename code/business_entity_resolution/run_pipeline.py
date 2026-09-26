#!/usr/bin/env python3
"""
CLI Entry point for Business Entity Resolution Pipeline.

Usage:
    # Full end-to-end (Train -> Tune -> Test Inference):
    python run_pipeline.py --mode all

    # Train and tune only:
    python run_pipeline.py --mode train

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
    parser.add_argument("--train-limit", type=int, default=80000, help="Max S1 entities for training (None for full)")
    parser.add_argument("--val-limit", type=int, default=15000, help="Max S1 entities for validation")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit threshold override")
    parser.add_argument("--batch-size", type=int, default=25000, help="Batch size for streaming test inference")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.train_limit is not None:
        config.train_s1_limit = args.train_limit
    if args.val_limit is not None:
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
