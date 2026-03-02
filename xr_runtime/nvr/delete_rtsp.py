#!/usr/bin/env python3
import sys
import time
import json
import urllib.request
from urllib.error import URLError
import os

# Load environment variables from .env
def load_env(file_path=".env"):
    env_vars = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    except FileNotFoundError:
        print(f"Error: {file_path} file not found.")
        sys.exit(1)
    return env_vars

env = load_env()
GROUP_KEY = env.get("GROUP_KEY")
API_KEY = env.get("API_KEY")

def delete_existing_monitors(base_url, api_key, group_key):
    """Deletes existing monitors with 'xr_stream' in their name."""
    try:
        # API endpoint to get all monitors in a group
        list_url = f"{base_url}/{api_key}/monitor/{group_key}"
        with urllib.request.urlopen(list_url) as response:
            monitors = json.loads(response.read().decode('utf-8'))
        
        monitors_to_delete = [m for m in monitors if 'name' in m and 'Port' in m['name']]
        
        if not monitors_to_delete:
            print("No existing 'Port' monitors found to delete.")
            return

        deleted_names = []
        for monitor in monitors_to_delete:
            monitor_id = monitor['mid']
            # Construct delete URL as per documentation
            delete_url = f"{base_url}/{api_key}/configureMonitor/{group_key}/{monitor_id}/delete"
            with urllib.request.urlopen(delete_url) as delete_response:
                result = json.loads(delete_response.read().decode('utf-8'))
                if result.get("ok"):
                    deleted_names.append(monitor['name'])
        
        if deleted_names:
            print(f"Deleted monitors: {', '.join(deleted_names)}")
        else:
            print("No monitors were deleted.")

        # Wait a moment for changes to apply
        time.sleep(2)

    except URLError as e:
        print(f"Could not connect to Shinobi to list monitors: {e}")
    except Exception as e:
        print(f"An error occurred while trying to delete monitors: {e}")


def delete(api_key, group_key):
    """Delete RTSP streams from Shinobi via API."""
    base_url = "http://localhost:8088"
    
    # Wait longer for Shinobi to be fully ready
    for attempt in range(30):  # Try for 30 seconds
        try:
            urllib.request.urlopen(f"{base_url}")
            break
        except URLError:
            time.sleep(1)
    
    # Extra wait for API readiness
    time.sleep(10)

    delete_existing_monitors(base_url, api_key, group_key)


if __name__ == "__main__":
    print("\n========== Shinobi setting up... ==========")
    delete(API_KEY, GROUP_KEY)