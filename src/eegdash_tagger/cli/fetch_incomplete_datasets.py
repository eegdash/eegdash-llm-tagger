#!/usr/bin/env python3
"""
CLI wrapper for fetching incomplete EEGDash datasets.

This script finds datasets with missing tags and extracts their metadata.
"""

import argparse
import json
import sys
from pathlib import Path

from eegdash_tagger.scraping import (
    collect_incomplete_datasets,
    build_metadata_for_dataset,
    fetch_openneuro_metadata,
)
from eegdash_tagger.scraping.enrichment import enrich_with_metadata


def main():
    """CLI entry point for the scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape EEGDash for datasets with missing tags and extract metadata"
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

    # Step 1: Collect incomplete datasets
    if args.verbose:
        print("=" * 60)
        print("EEGDash Dataset Scraper (Incomplete Datasets)")
        print("=" * 60)

    incomplete_datasets = collect_incomplete_datasets(verbose=args.verbose)

    if not incomplete_datasets:
        print("No datasets found with missing tags")
        return 0

    # Step 2: Apply limit if specified
    if args.limit:
        incomplete_datasets = incomplete_datasets[:args.limit]
        if args.verbose:
            print(f"\nLimiting to first {args.limit} dataset(s)")

    # Step 3: Enrich with metadata by cloning GitHub repos
    if args.verbose:
        print(f"\nFetching metadata for {len(incomplete_datasets)} dataset(s)...")

    enriched_datasets = enrich_with_metadata(
        incomplete_datasets,
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
