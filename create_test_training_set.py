#!/usr/bin/env python3
"""
Create Test/Training Dataset

This script takes the last 25 datasets from complete_metadata.json and:
1. Creates test_training.json with pathology, modality, type set to empty strings
2. Creates result_training.json with the original values for those fields

This simulates having incomplete datasets to test the LLM tagger.
"""

import json
import sys
from pathlib import Path


def create_test_training_set(
    input_file: str = "complete_metadata.json",
    test_output: str = "test_training.json",
    result_output: str = "result_training.json",
    num_datasets: int = 25
):
    """
    Extract last N datasets and create test/result files.

    Args:
        input_file: Path to complete metadata JSON file
        test_output: Path to output test file (with empty tags)
        result_output: Path to output result file (with correct tags)
        num_datasets: Number of datasets to extract from the end
    """
    # Load complete metadata
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1

    print(f"Loading {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        complete_data = json.load(f)

    total_datasets = len(complete_data)
    print(f"Total datasets in file: {total_datasets}")

    if total_datasets < num_datasets:
        print(f"Warning: Only {total_datasets} datasets available, using all of them")
        num_datasets = total_datasets

    # Extract last N datasets
    last_n_datasets = complete_data[-num_datasets:]
    print(f"Extracting last {num_datasets} datasets...")

    # Create test data (with empty tags) and result data (with original tags)
    test_data = []
    result_data = []

    for dataset in last_n_datasets:
        # Extract original tag values
        original_tags = {
            "dataset_id": dataset.get("dataset_id"),
            "pathology": dataset.get("pathology", ""),
            "modality": dataset.get("modality", ""),
            "type": dataset.get("type", "")
        }
        result_data.append(original_tags)

        # Create test version with empty tags
        test_dataset = dict(dataset)
        test_dataset["pathology"] = ""
        test_dataset["modality"] = ""
        test_dataset["type"] = ""
        test_data.append(test_dataset)

    # Write test data (for LLM to predict)
    test_path = Path(test_output)
    print(f"\nWriting test data to {test_output}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Created {test_output} with {len(test_data)} datasets (empty tags)")

    # Write result data (ground truth for evaluation)
    result_path = Path(result_output)
    print(f"\nWriting result data to {result_output}...")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Created {result_output} with {len(result_data)} ground truth labels")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input file:       {input_file} ({total_datasets} datasets)")
    print(f"Datasets used:    Last {num_datasets} datasets")
    print()
    print(f"Test file:        {test_output}")
    print(f"  - Contains {len(test_data)} datasets with empty pathology, modality, type")
    print(f"  - Use this file as input to LLM tagger")
    print()
    print(f"Result file:      {result_output}")
    print(f"  - Contains {len(result_data)} ground truth labels")
    print(f"  - Use this to evaluate LLM predictions")
    print()
    print("Example result entry:")
    if result_data:
        print(json.dumps(result_data[0], indent=2))

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create test/training dataset from complete metadata"
    )
    parser.add_argument(
        "--input",
        default="complete_metadata.json",
        help="Input file with complete metadata (default: complete_metadata.json)"
    )
    parser.add_argument(
        "--test-output",
        default="test_training.json",
        help="Output file for test data with empty tags (default: test_training.json)"
    )
    parser.add_argument(
        "--result-output",
        default="result_training.json",
        help="Output file for ground truth labels (default: result_training.json)"
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=25,
        help="Number of datasets to extract from the end (default: 25)"
    )

    args = parser.parse_args()

    return create_test_training_set(
        input_file=args.input,
        test_output=args.test_output,
        result_output=args.result_output,
        num_datasets=args.num_datasets
    )


if __name__ == "__main__":
    sys.exit(main())
