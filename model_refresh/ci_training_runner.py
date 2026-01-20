#!/usr/bin/env python3
"""
CI Training Pipeline Runner

Runs the training pipeline based on environment variables set by GitHub Actions.
Produces structured output for reporting.
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from refresh_models import (
    load_data_tracker, save_data_tracker,
    check_and_download_cards, check_and_download_draft_data,
    run_training_pipeline
)
from get_latest_set import get_latest_set_info


def set_output(name, value):
    """Set a GitHub Actions output."""
    output_file = os.environ.get('GITHUB_OUTPUT')
    if output_file:
        with open(output_file, 'a') as f:
            f.write(f"{name}={value}\n")
    print(f"Output: {name}={value}")


def main():
    # Set PyTorch environment variables to avoid threading issues in CI
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting training pipeline...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

    # Import PyTorch and check version
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    # Load current state
    tracker_data = load_data_tracker()
    latest_set_info = get_latest_set_info()

    if not latest_set_info.get("success"):
        print(f"ERROR: Failed to get latest set info")
        sys.exit(1)

    latest_set = latest_set_info.get("most_recent_set")
    print(f"Latest set: {latest_set}")

    # Check and download data
    print("\nChecking and downloading data...")
    cards_updated = check_and_download_cards(tracker_data, latest_set_info)
    premier_updated, traditional_updated, picktwodraft_updated = check_and_download_draft_data(
        tracker_data, latest_set_info
    )

    # Track training results
    training_results = []
    models_trained = False

    # Train ONE model per set to reduce memory usage
    # Priority: PickTwo > Premier (PickTwo is more important for the website)
    if picktwodraft_updated:
        print(f"\n{'='*50}")
        print(f"Training PickTwoDraft model for {latest_set}")
        print(f"{'='*50}")
        try:
            success, training_info = run_training_pipeline(latest_set, "PickTwo")
            if success:
                training_results.append(training_info)
                models_trained = True
                print(f"PickTwoDraft training completed successfully")
            else:
                print(f"WARNING: PickTwoDraft training failed")
        except Exception as e:
            print(f"ERROR: PickTwoDraft training crashed: {e}")
            import traceback
            traceback.print_exc()
    elif premier_updated:
        print(f"\n{'='*50}")
        print(f"Training Premier Draft model for {latest_set}")
        print(f"{'='*50}")
        try:
            success, training_info = run_training_pipeline(latest_set, "Premier")
            if success:
                training_results.append(training_info)
                models_trained = True
                print(f"Premier Draft training completed successfully")
            else:
                print(f"WARNING: Premier Draft training failed")
        except Exception as e:
            print(f"ERROR: Premier Draft training crashed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No updates detected, skipping training")

    # Save tracker with training results
    if training_results:
        tracker_data["last_training_logs"] = training_results
        tracker_data["last_training_timestamp"] = datetime.now().isoformat()

    save_data_tracker(tracker_data)

    # Write training report for release notes and notifications
    report = {
        "timestamp": datetime.now().isoformat(),
        "latest_set": latest_set,
        "cards_updated": cards_updated,
        "premier_updated": premier_updated,
        "picktwodraft_updated": picktwodraft_updated,
        "training_results": training_results,
        "models_trained": models_trained
    }

    with open("training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTraining report written to training_report.json")

    # Set output for release step
    set_output("models_trained", str(models_trained).lower())

    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"Models trained: {len(training_results)}")
    for result in training_results:
        print(f"  - {result['experiment_name']}: {result['validation_accuracy']:.2f}% accuracy")
        print(f"    Training picks: {result['training_picks']:,}")
        print(f"    Validation picks: {result['validation_picks']:,}")
        print(f"    Epochs: {result['num_epochs']}")
    print(f"{'='*50}\n")

    if not models_trained:
        print("WARNING: No models were trained!")


if __name__ == "__main__":
    main()
