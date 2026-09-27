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
    parser.add_argument(
        "--mode",
        choices=["all", "dev-train", "final-train", "train", "predict"],
        default=None,
        help="Pipeline execution mode: dev-train (Phase A: split/tune/early-stop), final-train (Phase B: 100% data/freeze), predict (Phase C: inference), all, train",
    )
    parser.add_argument("--train", action="store_true", help="Train the LightGBM entity resolution model")
    parser.add_argument("--eval", action="store_true", help="Run validation and threshold optimization")
    parser.add_argument("--predict", action="store_true", help="Run streaming test inference and output generation")

    # Explicit training modes
    parser.add_argument(
        "--training-mode",
        choices=["production", "final-train", "experiment", "development", "dev-train"],
        default=None,
        help="Explicit training mode profile",
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

    # Path overrides
    parser.add_argument("--dataset-root", type=str, default=None, help="Explicit dataset root directory override")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory for TSV results override")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Explicit artifacts directory override")

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

    config = PipelineConfig()

    # Apply path overrides if provided
    if args.dataset_root:
        config.paths.dataset_root = Path(args.dataset_root).resolve()
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir).resolve()
    if args.artifacts_dir:
        config.paths.artifacts_dir = Path(args.artifacts_dir).resolve()

    # Configure training mode and limits
    if args.mode == "final-train" or (args.mode == "train" and args.prod):
        config.training_mode = "production"
        config.train_s1_limit = args.train_limit
        config.val_s1_limit = args.val_limit
    elif args.mode == "dev-train" or args.dev or args.training_mode in ("development", "dev-train"):
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

    # Startup diagnostics
    print("=" * 64)
    print("OptiResolve Entity Resolution Pipeline Startup Diagnostics")
    print("=" * 64)
    print(f"  Dataset Root:      {config.paths.dataset_root}")
    print(f"  Output Directory:  {config.paths.output_dir}")
    print(f"  Artifacts Dir:     {config.paths.artifacts_dir}")
    print(f"  Model Path:        {config.paths.model_path}")
    print(f"  Execution Mode:    {args.mode or ('predict' if args.predict else 'train')}")
    print(f"  Training Mode:     {config.training_mode}")
    print(f"  Candidate Cap K:   {config.blocking.max_candidates_per_entity}")
    print(f"  Default Threshold: {config.default_threshold:.3f}")
    print("=" * 64)

    pipeline = EntityResolutionPipeline(config)

    # Route execution based on mode and flags
    if args.mode == "dev-train":
        pipeline.fit_dev()
    elif args.mode == "final-train":
        pipeline.fit_final()
    elif args.mode == "predict":
        if config.paths.model_path.exists():
            pipeline.model.load(config.paths.model_path)
        pipeline.predict_test(batch_size=args.batch_size)
    elif args.mode == "all":
        # Full end-to-end: Dev tune -> Final 100% train -> Test predict
        pipeline.fit_dev()
        pipeline.fit_final()
        pipeline.predict_test(batch_size=args.batch_size)
    elif args.mode == "train":
        if args.prod:
            pipeline.fit_final()
        else:
            pipeline.fit()
    else:
        # Flag-based or default execution
        do_train = args.train or (not args.predict and not args.eval)
        do_predict = args.predict or (not args.train and not args.eval)

        if do_train:
            if args.prod:
                pipeline.fit_final()
            else:
                pipeline.fit()

        if do_predict:
            if not do_train and config.paths.model_path.exists():
                pipeline.model.load(config.paths.model_path)
            pipeline.predict_test(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
