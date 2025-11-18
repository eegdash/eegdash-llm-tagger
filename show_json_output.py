#!/usr/bin/env python
"""
Show the JSON-formatted output for LLM consumption
"""

from eegdash_metadata import build_dataset_summary
import json

# Test both datasets
datasets = [
    "/home/ad-kkokate/BIDS-LLM/ds006923",
    "/home/ad-kkokate/BIDS-LLM/ds004841"
]

for dataset_path in datasets:
    print("=" * 80)
    print(f"Dataset: {dataset_path.split('/')[-1]}")
    print("=" * 80)

    # Build the summary
    summary = build_dataset_summary(dataset_path)

    # Get JSON output
    llm_json = summary.to_llm_json()

    # Pretty print JSON
    print(json.dumps(llm_json, indent=2, ensure_ascii=False))
    print()
