#!/usr/bin/env python
"""
Show the JSON-formatted output for LLM consumption with tagging results
"""

from eegdash_metadata import build_dataset_summary
from eegdash_tagger import tag_from_summary
import json

# Test dataset
datasets = [
    "/Users/kuntalkokate/neuroscience_work/ds001971"
]

for dataset_path in datasets:
    # Build the summary
    summary = build_dataset_summary(dataset_path)

    # Get JSON output from parser
    llm_json = summary.to_llm_json()

    # Run tagger on the parsed metadata
    tagging_result = tag_from_summary(llm_json)

    # Combine parser output and tagging results
    combined = {**llm_json, **tagging_result}

    # Pretty print combined JSON
    print(json.dumps(combined, indent=2, ensure_ascii=False))
