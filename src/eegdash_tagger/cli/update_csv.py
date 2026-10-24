#!/usr/bin/env python3
"""
CLI wrapper for updating CSV with LLM predictions.

This script reads LLM predictions and updates a CSV file with predicted values.
"""

import argparse
import sys
from pathlib import Path

from eegdash_tagger.utils import (
    load_llm_predictions,
    update_csv_with_predictions,
)


def main():
    """CLI entry point for CSV updater."""
    parser = argparse.ArgumentParser(
        description="Update EEGDash CSV with LLM predictions"
    )
    parser.add_argument(
        "--llm-json",
        default="llm_output.json",
        help="Path to LLM predictions JSON file (default: llm_output.json)"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to dataset_summary.csv file"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to apply prediction (default: 0.5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Validate paths
    llm_json_path = Path(args.llm_json)
    csv_path = Path(args.csv)

    if not llm_json_path.exists():
        print(f"Error: LLM JSON file not found: {llm_json_path}", file=sys.stderr)
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loading LLM predictions from: {llm_json_path}")

    # Load predictions
    predictions = load_llm_predictions(llm_json_path)

    if args.verbose:
        print(f"Loaded {len(predictions)} predictions")
        print()

    # Update CSV
    update_csv_with_predictions(
        csv_path,
        predictions,
        confidence_threshold=args.confidence_threshold,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
