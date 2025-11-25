#!/usr/bin/env python
"""
Fetch Metadata for Complete EEGDash Datasets

This script fetches metadata for datasets that already have all three tags
(pathology, modality, type) filled in. It reuses functions from eegdash_scraper.py
to minimize code duplication.

The only difference from eegdash_scraper.py is the filtering logic:
- eegdash_scraper.py: Fetches datasets with missing/unknown tags
- fetch_complete_datasets.py: Fetches datasets with all tags filled
"""

from pathlib import Path
from typing import List, Dict, Any
import argparse
import json
import sys

# Import reusable functions from eegdash_scraper
from eegdash_scraper import (
    fetch_dataset_table,
    fetch_openneuro_id,
    fetch_eegdash_detail_info,
    build_metadata_for_dataset,
    fetch_openneuro_metadata,
    EEGDASH_BASE_URL,
    OPENNEURO_BASE_URL,
    GITHUB_BASE_URL
)


def has_complete_tagging(row: Dict[str, str]) -> bool:
    """
    Return True if this dataset has ALL three key tags filled:
    pathology, modality, and type.

    This is the opposite of needs_tagging() from eegdash_scraper.py.

    Args:
        row: Dataset dict with pathology, modality, and type fields

    Returns:
        True if all tags are present and not 'unknown'
    """
    for key in ['pathology', 'modality', 'type']:
        value = row.get(key, '').strip().lower()
        # If any value is empty or 'unknown', return False
        if not value or value == 'unknown':
            return False
    return True


def collect_complete_datasets(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    For each dataset with complete tags, fetch OpenNeuro ID and construct URLs.

    Args:
        verbose: If True, print progress messages

    Returns:
        List of dicts with dataset info and URLs:
        {
            "dataset_id": "DS001971",
            "pathology": "Healthy",
            "modality": "Visual",
            "type": "Perception",
            "record_modality": "EEG",
            "openneuro_id": "ds001971",
            "openneuro_url": "...",
            "github_url": "...",
            "detail_url": "...",
            "eegdash_subjects": 18
        }
    """
    if verbose:
        print("Fetching dataset table from EEGDash...")

    all_datasets = fetch_dataset_table()

    if verbose:
        print(f"Found {len(all_datasets)} total datasets")

    # Filter to complete datasets (opposite of incomplete)
    complete = [d for d in all_datasets if has_complete_tagging(d)]

    if verbose:
        print(f"Found {len(complete)} datasets with complete tags")

    # Enrich with OpenNeuro info
    enriched = []
    for i, dataset in enumerate(complete, 1):
        if verbose:
            print(f"[{i}/{len(complete)}] Processing {dataset['dataset_id']}...")

        # Fetch OpenNeuro ID from detail page
        openneuro_id = fetch_openneuro_id(dataset['detail_url'])

        if not openneuro_id:
            if verbose:
                print(f"  Warning: Could not find OpenNeuro ID, skipping")
            continue

        # Construct URLs
        dataset['openneuro_id'] = openneuro_id
        dataset['openneuro_url'] = f"{OPENNEURO_BASE_URL}/{openneuro_id}"
        dataset['github_url'] = f"{GITHUB_BASE_URL}/{openneuro_id}"

        if verbose:
            print(f"  Found OpenNeuro ID: {openneuro_id}")

        # Fetch additional info from EEGDash detail page
        if verbose:
            print(f"  Fetching EEGDash detail info...")
        eegdash_info = fetch_eegdash_detail_info(dataset['detail_url'], verbose=verbose)
        # Only keep eegdash_subjects at parent level
        if 'eegdash_subjects' in eegdash_info:
            dataset['eegdash_subjects'] = eegdash_info['eegdash_subjects']

        enriched.append(dataset)

    return enriched


def enrich_with_metadata(rows: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    For each row, fetch metadata by cloning GitHub repositories.

    Args:
        rows: List of dataset dicts from collect_complete_datasets()
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
            bids_meta = build_metadata_for_dataset(row['github_url'], token=None, verbose=verbose)

            # Start from BIDS metadata
            metadata = dict(bids_meta)

            # Ensure recording_modality prefers EEGDash record_modality if available
            rec_mod = row.get("record_modality") or metadata.get("recording_modality")
            if rec_mod:
                metadata["recording_modality"] = rec_mod

            # Fetch and attach OpenNeuro metadata (only in metadata object)
            if verbose:
                print(f"  Fetching OpenNeuro metadata...")
            openneuro_info = fetch_openneuro_metadata(row.get('openneuro_id'), verbose=verbose)

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

            # Attach EEGDash info to metadata
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


def main():
    """CLI entry point for fetching complete datasets."""
    parser = argparse.ArgumentParser(
        description="Fetch metadata for EEGDash datasets with complete tags (pathology, modality, type)"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages"
    )

    args = parser.parse_args()

    # Step 1: Collect complete datasets
    if args.verbose:
        print("=" * 60)
        print("Fetch Complete EEGDash Datasets (Git Clone Mode)")
        print("=" * 60)

    complete_datasets = collect_complete_datasets(verbose=args.verbose)

    if not complete_datasets:
        print("No datasets found with complete tags")
        return 0

    # Step 2: Apply limit if specified
    if args.limit:
        complete_datasets = complete_datasets[:args.limit]
        if args.verbose:
            print(f"\nLimiting to first {args.limit} dataset(s)")

    # Step 3: Enrich with metadata by cloning GitHub repos
    if args.verbose:
        print(f"\nFetching metadata for {len(complete_datasets)} dataset(s) by cloning GitHub repos...")

    enriched_datasets = enrich_with_metadata(complete_datasets, verbose=args.verbose)

    # Step 4: Write output
    output_path = Path(args.output_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_datasets, f, indent=2, ensure_ascii=False)

    # Step 5: Print summary
    successful = sum(1 for d in enriched_datasets if 'metadata' in d)
    failed = sum(1 for d in enriched_datasets if 'error' in d)

    if args.verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total processed:    {len(enriched_datasets)}")
        print(f"Successful:         {successful}")
        print(f"Failed:             {failed}")
        print(f"Output written to:  {output_path}")
    else:
        print(f"Processed {len(enriched_datasets)} datasets ({successful} successful, {failed} failed)")
        print(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
