#!/usr/bin/env python
"""
Show the JSON-formatted metadata output for a BIDS dataset
"""

import argparse
import json
from eegdash_metadata import build_dataset_summary_from_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract and display metadata from a BIDS dataset"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to BIDS dataset directory"
    )

    args = parser.parse_args()

    # Build the summary using backward-compatible wrapper
    summary = build_dataset_summary_from_path(args.dataset)

    # Get JSON output from parser
    llm_json = summary.to_llm_json()

    # Pretty print JSON
    print(json.dumps(llm_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
