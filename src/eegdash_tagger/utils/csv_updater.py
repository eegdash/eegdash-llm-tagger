#!/usr/bin/env python3
"""
Update EEGDash CSV with LLM predictions

This script reads LLM predictions from llm_output.json and updates the
dataset_summary.csv file with predicted values for pathology, modality, and type
columns, but only if:
1. The confidence for that column is >= 0.5
2. The column is currently empty/missing
3. There's no existing value (never override existing values)
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd


def load_llm_predictions(json_path: Path) -> dict:
    """
    Load LLM predictions from JSON file.

    Args:
        json_path: Path to llm_output.json

    Returns:
        Dict mapping dataset_id -> {pathology, modality, type, confidence}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = {}
    for result in data.get('results', []):
        dataset_id = result.get('dataset_id')
        if not dataset_id:
            continue

        # Extract predictions (lists) and flatten to single values
        pathology = result.get('pathology', [])
        modality = result.get('modality', [])
        exp_type = result.get('type', [])

        # Get first element if list, otherwise use as-is
        pathology_val = pathology[0] if isinstance(pathology, list) and pathology else pathology
        modality_val = modality[0] if isinstance(modality, list) and modality else modality
        type_val = exp_type[0] if isinstance(exp_type, list) and exp_type else exp_type

        # Get confidence scores
        confidence = result.get('confidence', {})

        predictions[dataset_id] = {
            'pathology': pathology_val,
            'modality': modality_val,
            'type': type_val,
            'confidence': confidence
        }

    return predictions


def is_empty_value(val) -> bool:
    """
    Check if a value is considered empty/missing.

    Args:
        val: Value to check

    Returns:
        True if value is empty, None, NaN, or whitespace-only string
    """
    if pd.isna(val):
        return True
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def update_csv_with_predictions(
    csv_path: Path,
    predictions: dict,
    confidence_threshold: float = 0.5,
    dry_run: bool = False,
    verbose: bool = False
) -> None:
    """
    Update CSV file with LLM predictions.

    Values below confidence threshold are marked with " (low confidence)" suffix.

    Args:
        csv_path: Path to dataset_summary.csv
        predictions: Dict of predictions from load_llm_predictions()
        confidence_threshold: Minimum confidence to mark as high confidence (default 0.5)
        dry_run: If True, don't write changes, just show what would be updated
        verbose: If True, print detailed progress
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Map column names
    # CSV columns: "Type Subject" (pathology), "modality of exp" (modality), "type of exp" (type)
    # We need to find these columns
    pathology_col = None
    modality_col = None
    type_col = None

    # Find the correct column names (case-insensitive partial match)
    for col in df.columns:
        col_lower = col.lower()
        if 'type' in col_lower and 'subject' in col_lower:
            pathology_col = col
        elif 'modality' in col_lower and 'exp' in col_lower:
            modality_col = col
        elif 'type' in col_lower and 'exp' in col_lower:
            type_col = col

    if not all([pathology_col, modality_col, type_col]):
        print("Error: Could not find required columns in CSV", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        print(f"Pathology column: {pathology_col}", file=sys.stderr)
        print(f"Modality column: {modality_col}", file=sys.stderr)
        print(f"Type column: {type_col}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"CSV columns mapped:")
        print(f"  Pathology: '{pathology_col}'")
        print(f"  Modality: '{modality_col}'")
        print(f"  Type: '{type_col}'")
        print()

    # Track statistics
    stats = {
        'total_datasets': len(predictions),
        'pathology_updated_high_conf': 0,
        'pathology_updated_low_conf': 0,
        'modality_updated_high_conf': 0,
        'modality_updated_low_conf': 0,
        'type_updated_high_conf': 0,
        'type_updated_low_conf': 0,
        'pathology_skipped_existing': 0,
        'modality_skipped_existing': 0,
        'type_skipped_existing': 0,
    }

    # Find dataset column
    dataset_col = 'dataset' if 'dataset' in df.columns else 'DatasetID'
    if dataset_col not in df.columns:
        print("Error: Could not find dataset ID column", file=sys.stderr)
        sys.exit(1)

    # Update each row
    for dataset_id, pred in predictions.items():
        # Find row for this dataset (case-insensitive match)
        mask = df[dataset_col].str.lower() == dataset_id.lower()
        matching_rows = df[mask]

        if matching_rows.empty:
            if verbose:
                print(f"Warning: Dataset {dataset_id} not found in CSV")
            continue

        row_idx = matching_rows.index[0]

        if verbose:
            print(f"Processing {dataset_id}:")

        # Update pathology
        current_pathology = df.at[row_idx, pathology_col]
        pathology_confidence = pred['confidence'].get('pathology', 0.0)

        if not is_empty_value(current_pathology):
            stats['pathology_skipped_existing'] += 1
            if verbose:
                print(f"  Pathology: Keeping existing value '{current_pathology}'")
        elif pred['pathology'] and pathology_confidence >= confidence_threshold:
            # Only update if confidence meets threshold
            df.at[row_idx, pathology_col] = pred['pathology']
            stats['pathology_updated_high_conf'] += 1
            if verbose:
                print(f"  Pathology: Updated to '{pred['pathology']}' (confidence {pathology_confidence:.2f})")
        elif pred['pathology'] and pathology_confidence < confidence_threshold:
            # Low confidence - skip
            stats['pathology_updated_low_conf'] += 1
            if verbose:
                print(f"  Pathology: Skipped (low confidence {pathology_confidence:.2f})")

        # Update modality
        current_modality = df.at[row_idx, modality_col]
        modality_confidence = pred['confidence'].get('modality', 0.0)

        if not is_empty_value(current_modality):
            stats['modality_skipped_existing'] += 1
            if verbose:
                print(f"  Modality: Keeping existing value '{current_modality}'")
        elif pred['modality'] and modality_confidence >= confidence_threshold:
            # Only update if confidence meets threshold
            df.at[row_idx, modality_col] = pred['modality']
            stats['modality_updated_high_conf'] += 1
            if verbose:
                print(f"  Modality: Updated to '{pred['modality']}' (confidence {modality_confidence:.2f})")
        elif pred['modality'] and modality_confidence < confidence_threshold:
            # Low confidence - skip
            stats['modality_updated_low_conf'] += 1
            if verbose:
                print(f"  Modality: Skipped (low confidence {modality_confidence:.2f})")

        # Update type - DISABLED
        # Type field updates are currently disabled due to low prediction accuracy.
        # This functionality has been removed. See git history for previous implementation.
        if verbose:
            print(f"  Type: Skipped (updates disabled)")
            print()

    # Print summary
    print("=" * 60)
    print("Update Summary")
    print("=" * 60)
    print(f"Total datasets in LLM output: {stats['total_datasets']}")
    print()
    print(f"Pathology:")
    print(f"  Updated:              {stats['pathology_updated_high_conf']}")
    print(f"  Skipped (low conf):   {stats['pathology_updated_low_conf']}")
    print(f"  Skipped (existing):   {stats['pathology_skipped_existing']}")
    print()
    print(f"Modality:")
    print(f"  Updated:              {stats['modality_updated_high_conf']}")
    print(f"  Skipped (low conf):   {stats['modality_updated_low_conf']}")
    print(f"  Skipped (existing):   {stats['modality_skipped_existing']}")
    print()
    print(f"Type:")
    print(f"  Updated:              {stats['type_updated_high_conf']}")
    print(f"  Skipped (low conf):   {stats['type_updated_low_conf']}")
    print(f"  Skipped (existing):   {stats['type_skipped_existing']}")
    print()

    total_updates = (stats['pathology_updated_high_conf'] +
                     stats['modality_updated_high_conf'] +
                     stats['type_updated_high_conf'])

    if dry_run:
        print(f"DRY RUN: Would have updated {total_updates} values")
        print(f"No changes written to {csv_path}")
    else:
        # Write updated CSV
        df.to_csv(csv_path, index=False)
        print(f"✓ Updated {total_updates} values in {csv_path}")


def main():
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
