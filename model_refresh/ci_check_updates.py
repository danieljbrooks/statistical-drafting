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


def generate_issue_body(latest_set, premier_updated, picktwodraft_updated):
    """Generate the GitHub Issue body content."""
    # Determine update status
    premier_status = "Updated" if premier_updated else "No change"
    picktwodraft_status = "Updated" if picktwodraft_updated else "No change"

    # Get workflow URL from environment
    github_server = os.environ.get('GITHUB_SERVER_URL', 'https://github.com')
    github_repo = os.environ.get('GITHUB_REPOSITORY', '')
    github_run_id = os.environ.get('GITHUB_RUN_ID', '')
    workflow_url = f"{github_server}/{github_repo}/actions/runs/{github_run_id}" if github_run_id else "N/A"

    body = f"""New draft data has been detected on 17lands.com.

## Data Updates
- **Set:** {latest_set}
- **Premier Draft:** {premier_status}
- **PickTwoDraft:** {picktwodraft_status}

## To Update Models

Run from the statistical-drafting directory:
```bash
./deploy.sh
```

This will train new models and deploy them to the website.

---
Workflow run: {workflow_url}
"""
    return body


def main():
    force_notification = os.environ.get('FORCE_NOTIFICATION', 'false').lower() == 'true'

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

    has_updates = new_set_detected or premier_updated or picktwodraft_updated or force_notification

    # Set outputs
    set_output("has_updates", str(has_updates).lower())
    set_output("premier_updated", str(premier_updated).lower())
    set_output("picktwodraft_updated", str(picktwodraft_updated).lower())

    # Generate and set issue body if there are updates
    if has_updates:
        issue_body = generate_issue_body(latest_set, premier_updated, picktwodraft_updated)
        # Escape newlines and special characters for GitHub Actions output
        escaped_body = issue_body.replace('%', '%25').replace('\n', '%0A').replace('\r', '%0D')
        set_output("issue_body", escaped_body)

        # Update tracker data
        if new_set_detected:
            tracker_data["most_recent_set"] = latest_set
        if premier_updated and premier_last_modified:
            tracker_data["premier_draft_last_updated"] = premier_last_modified
        if picktwodraft_updated:
            # Update with the latest picktwodraft timestamp found
            for set_code in all_sets:
                picktwodraft_url = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{set_code}.PickTwoDraft.csv.gz"
                picktwodraft_last_modified = get_file_last_modified(picktwodraft_url)
                if picktwodraft_last_modified:
                    tracker_data["picktwodraft_last_updated"] = picktwodraft_last_modified
                    break

        # Save updated tracker
        with open(DATA_TRACKER_PATH, 'w') as f:
            json.dump(tracker_data, f, indent=2)
        print(f"Updated tracker saved to {DATA_TRACKER_PATH}")

    # Summary
    print(f"\n{'='*50}")
    print("UPDATE CHECK SUMMARY")
    print(f"{'='*50}")
    print(f"Latest Set: {latest_set}")
    print(f"New Set Detected: {new_set_detected}")
    print(f"Premier Draft Updated: {premier_updated}")
    print(f"PickTwoDraft Updated: {picktwodraft_updated}")
    print(f"Force Notification: {force_notification}")
    print(f"Has Updates (will notify): {has_updates}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
