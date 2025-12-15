"""Dataset enrichment with metadata from GitHub and OpenNeuro."""

from typing import List, Dict, Any, Optional


def enrich_with_metadata(
    rows: List[Dict[str, Any]],
    build_metadata_func,
    fetch_openneuro_metadata_func,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    For each row, fetch metadata by cloning GitHub repositories.

    This is a consolidated version of the enrich_with_metadata() function that
    was previously duplicated in eegdash_scraper.py and fetch_complete_datasets.py.

    Args:
        rows: List of dataset dicts from collect_incomplete_datasets() or
              collect_complete_datasets()
        build_metadata_func: Function to build metadata from GitHub URL
                             (typically build_metadata_for_dataset)
        fetch_openneuro_metadata_func: Function to fetch OpenNeuro metadata
                                        (typically fetch_openneuro_metadata)
        verbose: If True, print progress messages

    Returns:
        List of dicts with 'metadata' field added (or 'error' on failure)
    """
    results = []

    for i, row in enumerate(rows, 1):
        if verbose:
            print(f"\n[{i}/{len(rows)}] Fetching metadata for {row['dataset_id']} ({row['openneuro_id']})...")

        try:
            # Fetch BIDS metadata by cloning repository
            bids_meta = build_metadata_func(row['github_url'], token=None, verbose=verbose)

            # Start from BIDS metadata
            metadata = dict(bids_meta)

            # Ensure recording_modality prefers EEGDash record_modality if available
            rec_mod = row.get("record_modality") or metadata.get("recording_modality")
            if rec_mod:
                metadata["recording_modality"] = rec_mod

            # Fetch and attach OpenNeuro metadata (only in metadata object, not at parent level)
            if verbose:
                print(f"  Fetching OpenNeuro metadata...")
            openneuro_info = fetch_openneuro_metadata_func(row.get('openneuro_id'), verbose=verbose)

            # Add OpenNeuro fields to metadata
            if openneuro_info.get("openneuro_name"):
                metadata["openneuro_name"] = openneuro_info["openneuro_name"]
            if openneuro_info.get("openneuro_authors"):
                metadata["openneuro_authors"] = openneuro_info["openneuro_authors"]
            if openneuro_info.get("openneuro_doi"):
                metadata["openneuro_doi"] = openneuro_info["openneuro_doi"]
            if openneuro_info.get("openneuro_license"):
                metadata["openneuro_license"] = openneuro_info["openneuro_license"]
            if openneuro_info.get("openneuro_modalities"):
                metadata["openneuro_modalities"] = openneuro_info["openneuro_modalities"]
            if openneuro_info.get("openneuro_tasks"):
                metadata["openneuro_tasks"] = openneuro_info["openneuro_tasks"]

            # Attach EEGDash info to metadata (subjects count is useful)
            if row.get("eegdash_subjects") is not None:
                metadata["eegdash_subjects"] = row["eegdash_subjects"]

            row['metadata'] = metadata

            if verbose:
                print(f"  ✓ Successfully extracted metadata")
                print(f"    Title: {metadata.get('title', 'N/A')}")
                print(f"    Tasks: {len(metadata.get('tasks', []))} tasks")
                print(f"    Events: {len(metadata.get('events', []))} event types")

        except Exception as e:
            row['error'] = str(e)

            if verbose:
                print(f"  ✗ Error: {e}")

        results.append(row)

    return results
