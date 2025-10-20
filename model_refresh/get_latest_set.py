#!/usr/bin/env python3
"""
Get Latest Set Information from 17lands

This script uses Playwright to handle the dynamic React content on the 
17lands public datasets page to extract S3 dataset links and determine
the most recent set and Premier Draft update information.
"""

import re
import requests
from datetime import datetime
from playwright.sync_api import sync_playwright
from typing import List, Dict, Optional, Tuple

URL = "https://www.17lands.com/public_datasets"

def get_s3_links():
    """Get S3 dataset links from the 17lands public datasets page."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL)
        page.wait_for_load_state("networkidle")  # wait for React to finish
        links = [a.get_attribute("href") for a in page.query_selector_all("a")]
        s3_links = [l for l in links if l and "s3.amazonaws.com" in l]
        browser.close()
        return s3_links

def extract_set_info_from_links(s3_links: List[str]) -> Dict:
    """Extract set information from S3 links."""
    premier_draft_links = []
    picktwo_draft_links = []
    
    # Pattern to match: .SET_CODE.PremierDraft.csv.gz
    premier_pattern = r'\.([A-Z0-9]{3,4})\.PremierDraft\.csv\.gz'
    # Pattern to match: .SET_CODE.PickTwoDraft.csv.gz
    picktwo_pattern = r'\.([A-Z0-9]{3,4})\.PickTwoDraft\.csv\.gz'
    
    for link in s3_links:
        # Check for Premier Draft links
        premier_match = re.search(premier_pattern, link)
        if premier_match:
            set_code = premier_match.group(1)
            premier_draft_links.append((set_code, link))
        
        # Check for PickTwoDraft links
        picktwo_match = re.search(picktwo_pattern, link)
        if picktwo_match:
            set_code = picktwo_match.group(1)
            picktwo_draft_links.append((set_code, link))
    
    if not premier_draft_links and not picktwo_draft_links:
        return {"error": "No Premier Draft or PickTwoDraft links found"}
    
    # Combine all sets from both Premier and PickTwoDraft, preserving original order
    all_sets_with_links = []
    
    # Process links in original order to preserve chronological order
    for link in s3_links:
        # Check for Premier Draft links
        premier_match = re.search(premier_pattern, link)
        if premier_match:
            set_code = premier_match.group(1)
            all_sets_with_links.append((set_code, link))
        
        # Check for PickTwoDraft links
        picktwo_match = re.search(picktwo_pattern, link)
        if picktwo_match:
            set_code = picktwo_match.group(1)
            all_sets_with_links.append((set_code, link))
    
    # Get unique sets in the order they appear (preserving order is important!)
    seen = set()
    unique_sets_ordered = []
    for set_code, _ in all_sets_with_links:
        if set_code not in seen:
            unique_sets_ordered.append(set_code)
            seen.add(set_code)
    
    # The first set in the original list is typically the most recent
    if unique_sets_ordered:
        most_recent_set = unique_sets_ordered[0]
        
        # Find the first Premier Draft link for this set (prefer Premier for "most recent")
        premier_draft_link = next((link for set_code, link in premier_draft_links if set_code == most_recent_set), None)
        
        return {
            "success": True,
            "most_recent_set": most_recent_set,
            "premier_draft_link": premier_draft_link,
            "all_sets": unique_sets_ordered,
            "total_premier_draft_links": len(premier_draft_links),
            "total_picktwo_draft_links": len(picktwo_draft_links)
        }
    
    return {"error": "No sets found"}

def get_file_last_modified(url: str) -> Optional[str]:
    """Get the last modified date of an S3 file."""
    try:
        # Send HEAD request to get metadata without downloading the file
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            # S3 returns Last-Modified header
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                # Parse the date and convert to YYYY-MM-DD format
                dt = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
                return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Could not get last modified date: {e}")
    
    return None

def get_latest_set_info() -> Dict:
    """
    Get the latest MTG set information from 17lands.
    
    Returns:
        Dictionary with latest set info and Premier Draft update date
    """
    print("Fetching S3 dataset links from 17lands...")
    s3_links = get_s3_links()
    
    print(f"Found {len(s3_links)} S3 dataset links")
    
    # Extract set information
    set_info = extract_set_info_from_links(s3_links)
    
    if not set_info.get("success"):
        return set_info
    
    # Get the last modified date for the Premier Draft file (if it exists)
    premier_draft_link = set_info["premier_draft_link"]
    last_updated = None
    if premier_draft_link:
        last_updated = get_file_last_modified(premier_draft_link)
    
    result = {
        "success": True,
        "most_recent_set": set_info["most_recent_set"],
        "premier_draft_last_updated": last_updated,
        "formatted_entry": f"{set_info['most_recent_set']} PremierDraft {last_updated}" if last_updated else f"{set_info['most_recent_set']} PremierDraft (date unknown)",
        "premier_draft_link": premier_draft_link,
        "all_available_sets": set_info["all_sets"][:10],  # Show top 10
        "data_source": "17lands_s3_direct"
    }
    
    return result

if __name__ == "__main__":
    print("Getting latest MTG set information from 17lands...")
    
    info = get_latest_set_info()
    
    print("\n" + "="*60)
    print("17LANDS LATEST SET INFORMATION")
    print("="*60)
    
    if info.get("success"):
        print("✅ Successfully retrieved data!")
        print(f"\n📦 Most Recent Set: {info['most_recent_set']}")
        print(f"📅 Premier Draft Last Updated: {info['premier_draft_last_updated']}")
        print(f"🎯 Formatted Entry: {info['formatted_entry']}")
        print(f"📊 Data Source: {info['data_source']}")
        print(f"🔗 Premier Draft Link: {info['premier_draft_link']}")
        
        if 'all_available_sets' in info:
            print(f"\n📋 All Available Sets (recent first):")
            for i, set_code in enumerate(info['all_available_sets'], 1):
                marker = "🔥" if i == 1 else f"{i:2d}."
                print(f"  {marker} {set_code}")
    else:
        print("❌ Failed to retrieve data")
        print(f"Error: {info.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
