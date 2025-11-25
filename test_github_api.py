#!/usr/bin/env python
"""
Test script for GitHubFileProvider

This script tests fetching metadata from a GitHub repository using the
GitHub API (no cloning required).

Usage:
    export GITHUB_TOKEN="your_token_here"
    python test_github_api.py
"""

import os
import json
from file_providers import fetch_metadata_from_github_repo, get_github_token_from_env

def main():
    # Test repository URL
    repo_url = "https://github.com/OpenNeuroDatasets/ds001971"

    print("=" * 60)
    print("Testing GitHubFileProvider")
    print("=" * 60)
    print(f"Repository: {repo_url}")
    print()

    # Get token
    try:
        token = get_github_token_from_env()
        print("✓ GitHub token found")
    except ValueError as e:
        print(f"✗ Error: {e}")
        print()
        print("Please set GITHUB_TOKEN environment variable:")
        print("  export GITHUB_TOKEN='your_token_here'")
        print()
        print("See file_providers.py for setup instructions.")
        return 1

    # Fetch metadata
    print("\nFetching metadata via GitHub API...")
    print()
    try:
        metadata = fetch_metadata_from_github_repo(repo_url, token, verbose=True)
        print()
        print("✓ Successfully fetched metadata")
        print()

        # Print summary
        print("Dataset Summary:")
        print(f"  ID: {metadata.get('dataset_id', 'N/A')}")
        print(f"  Title: {metadata.get('title', 'N/A')}")
        print(f"  Recording Modality: {metadata.get('recording_modality', 'N/A')}")
        print(f"  Tasks: {len(metadata.get('tasks', []))} tasks")
        print(f"  Events: {len(metadata.get('events', []))} event types")
        print()

        # Print full JSON
        print("Full JSON output:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
