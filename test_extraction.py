#!/usr/bin/env python
"""
Test script for eegdash_metadata module on ds006923 dataset
"""

from eegdash_metadata import build_dataset_summary
import json

# Path to the dataset
dataset_path = "/home/ad-kkokate/BIDS-LLM/ds004841"

print("=" * 80)
print("Testing eegdash_metadata on ds004841")
print("=" * 80)
print()

# Build the summary
print("Building dataset summary...")
summary = build_dataset_summary(dataset_path)

print("\n" + "=" * 80)
print("DATASET SUMMARY - Raw Fields")
print("=" * 80)
print(f"\nDataset ID: {summary.dataset_id}")
print(f"Title: {summary.title}")
print(f"Dataset Type: {summary.dataset_type}")
print(f"Modalities: {summary.modalities}")
print(f"\nTask Names: {summary.task_names}")
print(f"Number of Event Types: {len(summary.event_types)}")
print(f"Participants Columns: {list(summary.participants_columns.keys())}")

print("\n" + "=" * 80)
print("LLM-READY PROMPT DICTIONARY")
print("=" * 80)
prompt_dict = summary.to_prompt_dict()

for key, value in prompt_dict.items():
    print(f"\n### {key.upper()} ###")
    print(value)
    print()

print("\n" + "=" * 80)
print("JSON OUTPUT (for LLM consumption)")
print("=" * 80)
print(json.dumps(prompt_dict, indent=2))
