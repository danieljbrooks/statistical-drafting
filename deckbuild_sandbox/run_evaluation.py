#!/usr/bin/env python3
"""
Command-line tool for evaluating deckbuilding models.

Usage:
    python run_evaluation.py --set FDN --mode Premier --max-examples 100
    python run_evaluation.py --set FDN --mode Premier --full
    python run_evaluation.py --set FDN --mode Premier --max-examples 1000 --plot
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import statisticaldeckbuild as sdb


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate deckbuilding model performance on validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on 10 examples
  python run_evaluation.py --set FDN --mode Premier --max-examples 10

  # Evaluate 100 examples with plots
  python run_evaluation.py --set FDN --mode Premier --max-examples 100 --plot

  # Full evaluation on all validation data
  python run_evaluation.py --set FDN --mode Premier --full

  # Custom paths
  python run_evaluation.py --set FDN --mode Premier --max-examples 100 \\
      --data-folder ./my_data/training_sets \\
      --model-folder ./my_data/models
        """
    )

    parser.add_argument(
        "--set",
        type=str,
        required=True,
        help="Set abbreviation (e.g., FDN, BLB, DSK)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="Premier",
        help="Draft mode (default: Premier)"
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all examples (same as --max-examples None)"
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N examples (default: 10)"
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        default="../data/training_sets/",
        help="Folder containing validation datasets (default: ../data/training_sets/)"
    )

    parser.add_argument(
        "--model-folder",
        type=str,
        default="../data/models/",
        help="Folder containing trained models (default: ../data/models/)"
    )

    parser.add_argument(
        "--cards-folder",
        type=str,
        default="../data/cards/",
        help="Folder containing card data (default: ../data/cards/)"
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="evaluation_results/",
        help="Folder to save results (default: evaluation_results/)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save distribution plot"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Handle --full flag
    if args.full:
        max_examples = None
    else:
        max_examples = args.max_examples

    # Build file paths
    val_dataset_path = os.path.join(
        args.data_folder,
        f"{args.set}_{args.mode}_deckbuild_val.pth"
    )

    # Print configuration
    print("=" * 70)
    print("DECKBUILDING MODEL EVALUATION")
    print("=" * 70)
    print(f"Set:                {args.set}")
    print(f"Draft Mode:         {args.mode}")
    print(f"Max Examples:       {max_examples or 'All'}")
    print(f"Validation Data:    {val_dataset_path}")
    print("=" * 70)
    print()

    # Load validation dataset
    if not os.path.exists(val_dataset_path):
        print(f"ERROR: Validation dataset not found at {val_dataset_path}")
        print(f"Please check your --set, --mode, and --data-folder arguments.")
        sys.exit(1)

    print("Loading validation dataset...")
    val_dataset = torch.load(val_dataset_path, weights_only=False)
    print(f"  ✓ Loaded {len(val_dataset)} validation examples")
    print(f"  ✓ Number of cards: {len(val_dataset.cardnames)}")
    print()

    # Initialize builder
    print("Initializing IterativeDeckBuilder...")
    try:
        builder = sdb.IterativeDeckBuilder(
            set_abbreviation=args.set,
            draft_mode=args.mode,
            model_folder=args.model_folder,
            cards_folder=args.cards_folder,
        )
        print(f"  ✓ Loaded model successfully")
        print(f"  ✓ Device: {builder.device}")
        print()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Please check your --model-folder and --cards-folder arguments.")
        sys.exit(1)

    # Prepare output path
    if not args.no_save:
        os.makedirs(args.output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"eval_{args.set}_{args.mode}_{timestamp}.json"
        output_path = os.path.join(args.output_folder, output_filename)
    else:
        output_path = None

    # Run evaluation
    print("Running evaluation...")
    print()
    results = sdb.evaluate_deckbuilder(
        val_dataset=val_dataset,
        builder=builder,
        max_examples=max_examples,
        progress_interval=args.progress_interval,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_path=output_path,
    )

    # Print summary
    print()
    sdb.print_summary(results)
    print()

    # Generate plot if requested
    if args.plot:
        plot_filename = f"eval_{args.set}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_distribution.png"
        plot_path = os.path.join(args.output_folder, plot_filename)
        print(f"Generating distribution plot...")
        sdb.plot_difference_distribution(results, save_path=plot_path)
        print()

    # Print file locations
    if not args.no_save:
        print("=" * 70)
        print("Results saved to:")
        print(f"  {output_path}")
        if args.plot:
            print(f"  {plot_path}")
        print("=" * 70)

    return results


if __name__ == "__main__":
    main()
