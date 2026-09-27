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
    parser.add_argument("--mode", choices=["all", "train", "predict"], default=None, help="Pipeline execution mode (all, train, predict)")
    parser.add_argument("--train", action="store_true", help="Train the LightGBM entity resolution model")
    parser.add_argument("--eval", action="store_true", help="Run validation and threshold optimization")
    parser.add_argument("--predict", action="store_true", help="Run streaming test inference and output generation")

    # Explicit training modes (Requirement 3)
    parser.add_argument(
        "--training-mode",
        choices=["production", "experiment", "development"],
        default=None,
        help="Explicit training mode profile (production: 100% full dataset; experiment: custom limits; development: subsampled quick run)",
    )
    parser.add_argument(
        "--prod", "--production",
        action="store_true",
        dest="prod",
        help="Final production training (enforces 100% of available train_source1 rows without silent truncation)",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Validation/training experiment mode (allows custom S1 limits via --train-limit and --val-limit)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development training mode (subsamples train/val for rapid execution: 50k train / 10k val)",
    )

    parser.add_argument("--train-limit", type=int, default=None, help="Explicit max S1 entities for training (None = 100% full dataset in production)")
    parser.add_argument("--val-limit", type=int, default=None, help="Explicit max S1 entities for validation (None = proportional 20% holdout)")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit threshold override")
    parser.add_argument(
        "--max-candidates",
        "--max-candidates-per-entity",
        "-k",
        type=int,
        default=None,
        help="Explicit candidate cap K per entity override (default: configured in BlockingConfig)",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for streaming test inference")
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help="In development mode, allow pipeline to continue even if required ground-truth target entities are missing from target sources",
    )
    args = parser.parse_args()

    # Determine execution phases from flags or --mode
    do_train = False
    do_eval = False
    do_predict = False

    if args.mode is not None:
        if args.mode in ("all", "train"):
            do_train = True
            do_eval = True
        if args.mode in ("all", "predict"):
            do_predict = True
    elif args.train or args.eval or args.predict:
        do_train = args.train
        do_eval = args.eval or args.train  # training includes validation tuning
        do_predict = args.predict
    else:
        # Default when run with no arguments: run complete pipeline
        do_train = True
        do_eval = True
        do_predict = True

    config = PipelineConfig()

    # Explicitly configure training mode and dataset limits (Requirements 1, 2, 3)
    if args.dev or args.training_mode == "development":
        config.training_mode = "development"
        config.train_s1_limit = args.train_limit or 50000
        config.val_s1_limit = args.val_limit or 10000
    elif args.experiment or args.train_limit is not None or args.val_limit is not None or args.training_mode == "experiment":
        config.training_mode = "experiment"
        config.train_s1_limit = args.train_limit
        config.val_s1_limit = args.val_limit
    else:
        # Default: Final Production Training (100% of available dataset used)
        config.training_mode = "production"
        config.train_s1_limit = None
        config.val_s1_limit = None

    if args.threshold is not None:
        config.default_threshold = args.threshold

    if args.max_candidates is not None:
        config.blocking.max_candidates_per_entity = args.max_candidates

    if args.allow_missing_targets:
        if config.is_production:
            parser.error("--allow-missing-targets cannot be used in production mode! Production training requires 100% of ground-truth target entities to prevent incomplete positive labels.")
        config.allow_missing_targets = True

    pipeline = EntityResolutionPipeline(config)

    if do_train or do_eval:
        pipeline.fit()

    if do_predict:
        if not do_train and config.paths.model_path.exists():
            pipeline.model.load(config.paths.model_path)
        pipeline.predict_test(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
