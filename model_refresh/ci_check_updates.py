#!/usr/bin/env python3
"""
CI Check for Updates Script

Checks for data updates without downloading or training.
Outputs GitHub Actions outputs for the workflow.
"""

import json
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_latest_set import get_latest_set_info, get_file_last_modified

DATA_TRACKER_PATH = "data_tracker.json"


def load_data_tracker():
    """Load the data tracker JSON file."""
    if os.path.exists(DATA_TRACKER_PATH):
        with open(DATA_TRACKER_PATH, 'r') as f:
            return json.load(f)
    return {
        "most_recent_set": None,
        "premier_draft_last_updated": None,
        "traditional_draft_last_updated": None,
        "picktwodraft_last_updated": None,
    }


def set_output(name, value):
    """Set a GitHub Actions output."""
    output_file = os.environ.get('GITHUB_OUTPUT')
    if output_file:
        with open(output_file, 'a') as f:
            f.write(f"{name}={value}\n")
    print(f"Output: {name}={value}")


def main():
    force_training = os.environ.get('FORCE_TRAINING', 'false').lower() == 'true'

    print("Checking for data updates...")
    tracker_data = load_data_tracker()
    latest_set_info = get_latest_set_info()

    if not latest_set_info.get("success"):
        print(f"ERROR: Failed to get latest set info: {latest_set_info}")
        set_output("has_updates", "false")
        sys.exit(1)

    latest_set = latest_set_info.get("most_recent_set")
    set_output("latest_set", latest_set)

    # Check for new set
    current_set = tracker_data.get("most_recent_set")
    new_set_detected = current_set != latest_set

    # Check Premier Draft updates
    premier_url = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{latest_set}.PremierDraft.csv.gz"
    premier_last_modified = get_file_last_modified(premier_url)
    current_premier_date = tracker_data.get("premier_draft_last_updated")
    premier_updated = premier_last_modified and (
        current_premier_date is None or premier_last_modified != current_premier_date
    )

    # Check Traditional Draft updates
    trad_url = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{latest_set}.TradDraft.csv.gz"
    trad_last_modified = get_file_last_modified(trad_url)
    current_trad_date = tracker_data.get("traditional_draft_last_updated")
    trad_updated = trad_last_modified and (
        current_trad_date is None or trad_last_modified != current_trad_date
    )

    # Check PickTwoDraft updates (check multiple recent sets)
    picktwodraft_updated = False
    all_sets = latest_set_info.get("all_available_sets", [])[:5]
    for set_code in all_sets:
        picktwodraft_url = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{set_code}.PickTwoDraft.csv.gz"
        picktwodraft_last_modified = get_file_last_modified(picktwodraft_url)
        if picktwodraft_last_modified:
            current_picktwodraft_date = tracker_data.get("picktwodraft_last_updated")
            if current_picktwodraft_date is None or picktwodraft_last_modified != current_picktwodraft_date:
                picktwodraft_updated = True
                break

    has_updates = new_set_detected or premier_updated or trad_updated or picktwodraft_updated or force_training

    # Set outputs
    set_output("has_updates", str(has_updates).lower())
    set_output("premier_updated", str(premier_updated).lower())
    set_output("traditional_updated", str(trad_updated).lower())
    set_output("picktwodraft_updated", str(picktwodraft_updated).lower())

    # Summary
    print(f"\n{'='*50}")
    print("UPDATE CHECK SUMMARY")
    print(f"{'='*50}")
    print(f"Latest Set: {latest_set}")
    print(f"New Set Detected: {new_set_detected}")
    print(f"Premier Draft Updated: {premier_updated}")
    print(f"Traditional Draft Updated: {trad_updated}")
    print(f"PickTwoDraft Updated: {picktwodraft_updated}")
    print(f"Force Training: {force_training}")
    print(f"Has Updates (will train): {has_updates}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
