#!/usr/bin/env python
"""
Show the LLM-ready output from eegdash_metadata module
"""

from eegdash_metadata import build_dataset_summary

# Path to the dataset
dataset_path = "/home/ad-kkokate/BIDS-LLM/ds006923"

# Build the summary
summary = build_dataset_summary(dataset_path)

# Get LLM-ready prompt dictionary
prompt_dict = summary.to_prompt_dict()

# Print in a format ready for LLM consumption
print("="*80)
print("LLM-READY METADATA OUTPUT")
print("="*80)
print()

for key, value in prompt_dict.items():
    print(f"[{key.upper()}]")
    print(value)
    print()
    print("-"*80)
    print()
