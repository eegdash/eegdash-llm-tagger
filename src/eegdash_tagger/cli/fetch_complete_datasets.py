#!/usr/bin/env python3
"""
CLI wrapper for fetching complete EEGDash datasets.

This script finds datasets with complete tags and extracts their metadata.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from eegdash_tagger.scraping import (
    fetch_dataset_table,
    fetch_openneuro_id,
    fetch_eegdash_detail_info,
    build_metadata_for_dataset,
    fetch_openneuro_metadata,
    has_complete_tagging,
    EEGDASH_BASE_URL,
    OPENNEURO_BASE_URL,
    GITHUB_BASE_URL,
)
from eegdash_tagger.scraping.enrichment import enrich_with_metadata


def collect_complete_datasets(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    For each dataset with complete tags, fetch OpenNeuro ID and construct URLs.

    Args:
        verbose: If True, print progress messages

    Returns:
        List of dicts with dataset info and URLs
    """
    if verbose:
        print("Fetching dataset table from EEGDash...")

    all_datasets = fetch_dataset_table()

    if verbose:
        print(f"Found {len(all_datasets)} total datasets")

    # Filter to complete datasets
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
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Step 1: Collect complete datasets
    if args.verbose:
        print("=" * 60)
        print("Fetch Complete EEGDash Datasets")
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

    # Step 3: Enrich with metadata
    if args.verbose:
        print(f"\nFetching metadata for {len(complete_datasets)} dataset(s)...")

    enriched_datasets = enrich_with_metadata(
        complete_datasets,
        build_metadata_for_dataset,
        fetch_openneuro_metadata,
        verbose=args.verbose
    )

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
