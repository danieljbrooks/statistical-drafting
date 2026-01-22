#!/usr/bin/env python3
"""
Model Refresh Automation Script

This script automates the process of checking for and downloading new MTG draft data
from 17lands, then retraining models when updates are detected.
"""

import json
import os
import sys
import requests
import gzip
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add the parent directory to sys.path to import statisticaldrafting
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd

# Import get_latest_set from the same directory
from get_latest_set import get_latest_set_info, get_file_last_modified

# Configuration
CARDS_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv"
DATA_TRACKER_PATH = "data_tracker.json"
CARDS_DATA_PATH = "../data/cards/cards.csv"
DRAFT_DATA_PATH = "../data/17lands/"


def load_data_tracker() -> Dict:
    """Load the data tracker JSON file."""
    if os.path.exists(DATA_TRACKER_PATH):
        with open(DATA_TRACKER_PATH, 'r') as f:
            return json.load(f)
    else:
        # Return default structure if file doesn't exist
        return {
            "most_recent_set": None,
            "last_cards_update": None,
            "premier_draft_last_updated": None,
            "picktwodraft_last_updated": None,
            "last_check_timestamp": None,
            "notes": "This file tracks the most recent MTG set and update dates for Premier and PickTwoDraft data to avoid redundant downloads and processing. Only Premier models are trained, with PickTwo as fallback."
        }


def save_data_tracker(data: Dict) -> None:
    """Save the data tracker JSON file."""
    data["last_check_timestamp"] = datetime.now().isoformat()
    with open(DATA_TRACKER_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated data tracker: {DATA_TRACKER_PATH}")


def download_file(url: str, destination: str) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        print(f"✅ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def extract_gz_file(gz_path: str, destination: str) -> bool:
    """Extract a .gz file to destination."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(destination, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Extracted {gz_path} to {destination}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {gz_path}: {e}")
        return False


def check_and_download_cards(tracker_data: Dict, latest_set_info: Dict) -> bool:
    """Check if cards.csv needs updating and download if necessary."""
    print("\n🔍 Checking cards.csv for updates...")
    
    # Check if we have a new set
    current_set = tracker_data.get("most_recent_set")
    latest_set = latest_set_info.get("most_recent_set")
    
    if current_set != latest_set:
        print(f"🆕 New set detected: {current_set} -> {latest_set}")
        
        # Download updated cards.csv
        if download_file(CARDS_URL, CARDS_DATA_PATH):
            tracker_data["most_recent_set"] = latest_set
            tracker_data["last_cards_update"] = datetime.now().strftime('%Y-%m-%d')
            
            # Reset draft data tracking for new set (they will be downloaded fresh)
            tracker_data["premier_draft_last_updated"] = None
            tracker_data["picktwodraft_last_updated"] = None
            print("🔄 Reset draft data tracking for new set")
            
            return True
        else:
            print("❌ Failed to download cards.csv")
            return False
    else:
        print(f"✅ No new set detected (current: {current_set})")
        return False


def get_draft_data_url(set_code: str, draft_mode: str) -> str:
    """Construct the draft data URL for a given set and mode."""
    return f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{set_code}.{draft_mode}Draft.csv.gz"


def check_and_download_draft_data(tracker_data: Dict, latest_set_info: Dict) -> Tuple[bool, str, bool]:
    """
    Check if draft data needs updating and download if necessary.
    
    Strategy: Only download and train Premier draft models. If Premier is not available,
    fall back to PickTwo draft. Traditional draft is no longer downloaded or trained.
    
    Returns:
        Tuple of (data_updated, draft_mode, is_picktwodraft_fallback)
        - data_updated: bool indicating if new data was downloaded
        - draft_mode: str indicating which mode to train ("Premier" or "PickTwo"), or None
        - is_picktwodraft_fallback: bool indicating if we're using PickTwo as fallback
    """
    print("\n🔍 Checking draft data for updates...")
    
    latest_set = latest_set_info.get("most_recent_set")
    all_sets = latest_set_info.get("all_available_sets", [])
    if not latest_set:
        print("❌ No latest set information available")
        return False, None, False
    
    # Check Premier Draft data first (preferred)
    premier_url = get_draft_data_url(latest_set, "Premier")
    premier_last_modified = get_file_last_modified(premier_url)
    current_premier_date = tracker_data.get("premier_draft_last_updated")
    
    # If Premier Draft is available
    if premier_last_modified:
        # Check if it needs updating
        if current_premier_date is None or premier_last_modified != current_premier_date:
            if current_premier_date is None:
                print(f"🆕 First run - Premier Draft data needs download: {premier_last_modified}")
            else:
                print(f"🆕 Premier Draft data updated: {current_premier_date} -> {premier_last_modified}")
            
            # Download Premier Draft data
            gz_path = os.path.join(DRAFT_DATA_PATH, f"draft_data_public.{latest_set}.PremierDraft.csv.gz")
            if download_file(premier_url, gz_path):
                # Update tracker immediately after successful download
                tracker_data["premier_draft_last_updated"] = premier_last_modified
                print(f"✅ Premier Draft download tracked: {premier_last_modified}")
                return True, "Premier", False
            else:
                print("❌ Failed to download Premier Draft data")
                return False, None, False
        else:
            print(f"✅ Premier Draft data up to date ({premier_last_modified})")
            return False, "Premier", False
    
    # Premier Draft not available - check for PickTwoDraft as fallback
    print("⚠️  Premier Draft not available, checking for PickTwoDraft fallback...")
    
    current_picktwodraft_date = tracker_data.get("picktwodraft_last_updated")
    picktwodraft_last_modified = None
    picktwodraft_set = None
    
    # Check the most recent sets for PickTwoDraft data (limit to first 5 sets)
    sets_to_check = all_sets[:5] if all_sets else [latest_set]
    
    for set_code in sets_to_check:
        picktwodraft_url = get_draft_data_url(set_code, "PickTwo")
        last_modified = get_file_last_modified(picktwodraft_url)
        
        if last_modified:
            print(f"🔍 Found PickTwoDraft data for {set_code}: {last_modified}")
            # Use the most recent PickTwoDraft data found
            if picktwodraft_last_modified is None or last_modified > picktwodraft_last_modified:
                picktwodraft_last_modified = last_modified
                picktwodraft_set = set_code
        else:
            print(f"🔍 No PickTwoDraft data found for {set_code}")
    
    # If we found PickTwoDraft data and it needs updating
    if picktwodraft_last_modified and picktwodraft_set:
        if current_picktwodraft_date is None or picktwodraft_last_modified != current_picktwodraft_date:
            if current_picktwodraft_date is None:
                print(f"🆕 First run - Using PickTwoDraft as fallback: {picktwodraft_set} ({picktwodraft_last_modified})")
            else:
                print(f"🆕 PickTwoDraft data updated: {current_picktwodraft_date} -> {picktwodraft_last_modified} ({picktwodraft_set})")
            
            # Download PickTwoDraft data
            picktwodraft_url = get_draft_data_url(picktwodraft_set, "PickTwo")
            gz_path = os.path.join(DRAFT_DATA_PATH, f"draft_data_public.{picktwodraft_set}.PickTwoDraft.csv.gz")
            if download_file(picktwodraft_url, gz_path):
                # Update tracker immediately after successful download
                tracker_data["picktwodraft_last_updated"] = picktwodraft_last_modified
                tracker_data["picktwodraft_set"] = picktwodraft_set  # Track which set we downloaded
                print(f"✅ PickTwoDraft download tracked: {picktwodraft_set} ({picktwodraft_last_modified})")
                return True, "PickTwo", True
            else:
                print("❌ Failed to download PickTwoDraft data")
                return False, None, True
        else:
            print(f"✅ PickTwoDraft data up to date ({picktwodraft_set}: {picktwodraft_last_modified})")
            return False, "PickTwo", True
    
    print("❌ No draft data available (neither Premier nor PickTwo)")
    return False, None, False


def run_training_pipeline(set_code: str, draft_mode: str) -> Tuple[bool, Dict]:
    """Run the training pipeline for a given set and draft mode."""
    try:
        print(f"\n🚀 Starting training pipeline for {set_code} {draft_mode} Draft...")

        # Change to notebooks directory to run training (expected context for relative paths)
        original_cwd = os.getcwd()
        notebooks_dir = os.path.join(os.path.dirname(os.getcwd()), "notebooks")
        os.chdir(notebooks_dir)
        print(f"📁 Changed working directory to: {os.getcwd()}")

        try:
            print(f"📦 About to call sd.default_training_pipeline...")
            print(f"   Parameters: set={set_code}, mode={draft_mode}, overwrite=True")
            sys.stdout.flush()  # Force flush before potential crash

            training_info = sd.default_training_pipeline(
                set_abbreviation=set_code,
                draft_mode=draft_mode,
                overwrite_dataset=True
            )
            
            # Log training information
            print(f"📊 Training Summary:")
            print(f"   • Experiment: {training_info['experiment_name']}")
            print(f"   • Training date: {training_info['training_date']}")
            print(f"   • Training picks: {training_info['training_picks']:,}")
            print(f"   • Validation picks: {training_info['validation_picks']:,}")
            print(f"   • Best validation accuracy: {training_info['validation_accuracy']:.2f}%")
            print(f"   • Best epoch: {training_info['num_epochs']}")
            
            print(f"✅ Training completed for {set_code} {draft_mode} Draft")
            return True, training_info
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Training failed for {set_code} {draft_mode} Draft: {e}")
        return False, {}


def main():
    """Main automation function."""
    print("🤖 Starting Model Refresh Automation")
    print("=" * 50)
    
    # Load current tracker data
    tracker_data = load_data_tracker()
    print(f"📊 Loaded tracker data: {tracker_data}")
    
    # Get latest set information
    print("\n🔍 Fetching latest set information from 17lands...")
    latest_set_info = get_latest_set_info()
    
    if not latest_set_info.get("success"):
        print(f"❌ Failed to get latest set info: {latest_set_info}")
        return
    
    print(f"✅ Latest set info: {latest_set_info['most_recent_set']} (Premier: {latest_set_info['premier_draft_last_updated']})")
    
    # Check and download cards.csv if needed
    cards_updated = check_and_download_cards(tracker_data, latest_set_info)
    
    # Check and download draft data if needed
    data_updated, draft_mode, is_picktwodraft_fallback = check_and_download_draft_data(tracker_data, latest_set_info)
    
    # Run training pipeline if data was updated (only one model per set)
    latest_set = latest_set_info.get("most_recent_set")
    training_logs = []
    
    if data_updated and draft_mode:
        if is_picktwodraft_fallback:
            print(f"\n⚠️  Training PickTwo model as fallback (Premier not available)")
        
        success, training_info = run_training_pipeline(latest_set, draft_mode)
        if success:
            training_logs.append(training_info)
            if is_picktwodraft_fallback:
                training_info['fallback_mode'] = True
        else:
            print(f"⚠️  Training failed for {latest_set} {draft_mode} Draft")
    
    # Add training logs to tracker data
    if training_logs:
        tracker_data["last_training_logs"] = training_logs
        tracker_data["last_training_timestamp"] = datetime.now().isoformat()
    
    # Save updated tracker data
    save_data_tracker(tracker_data)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"🎯 Latest Set: {latest_set}")
    print(f"📦 Cards Downloaded: {'✅' if cards_updated else '❌'}")
    print(f"🏆 Draft Data Downloaded: {'✅' if data_updated else '❌'}")
    if data_updated and draft_mode:
        mode_display = f"{draft_mode} {'(fallback)' if is_picktwodraft_fallback else ''}"
        print(f"🎮 Draft Mode: {mode_display}")
    
    # Show training logs if any models were trained
    if training_logs:
        print(f"\n📊 Training Results:")
        for log in training_logs:
            fallback_note = " [PickTwo fallback]" if log.get('fallback_mode') else ""
            print(f"   • {log['experiment_name']}: {log['validation_accuracy']:.2f}% accuracy{fallback_note}")
            print(f"     {log['training_picks']:,} training picks, {log['num_epochs']} epochs")
            print(f"     Trained on: {log['training_date']}")
    
    if not any([cards_updated, data_updated]):
        print("✨ All data is up to date - no action needed!")
    else:
        print("🚀 Automation completed with updates!")


if __name__ == "__main__":
    main()