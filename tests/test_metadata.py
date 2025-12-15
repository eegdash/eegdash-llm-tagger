#!/usr/bin/env python3
"""
Quick test script to verify JSON metadata summarization feature.
"""
import json
import subprocess
import tempfile
from pathlib import Path

from eegdash_tagger.metadata import build_dataset_summary_from_path

def test_json_metadata_feature():
    """Test the new JSON metadata feature on a cloned dataset."""

    # Use a small dataset for testing
    openneuro_id = "ds002718"
    github_url = f"https://github.com/OpenNeuroDatasets/{openneuro_id}.git"

    print(f"Testing JSON metadata feature on {openneuro_id}...")
    print(f"Cloning from: {github_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone the repository
        clone_path = Path(tmpdir) / openneuro_id
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr}")
            return False

        print(f"✓ Repository cloned to {clone_path}")

        # Build dataset summary
        print("Building dataset summary...")
        summary = build_dataset_summary_from_path(clone_path)

        # Convert to LLM JSON format
        llm_json = summary.to_llm_json()

        # Check if new field is present
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)

        if "json_metadata_summary" in llm_json:
            print("✓ json_metadata_summary field is present in LLM output")
            print(f"\nContent preview (first 500 chars):")
            print("-" * 60)
            content = llm_json["json_metadata_summary"]
            print(content[:500])
            if len(content) > 500:
                print("...")
            print("-" * 60)
            print(f"\nTotal length: {len(content)} characters")
        else:
            print("✗ json_metadata_summary field is NOT present")
            print("This could mean no candidate JSON files were found.")

        # Check internal tracking
        if summary.json_metadata_files:
            print(f"\n✓ Found {len(summary.json_metadata_files)} JSON files:")
            for f in summary.json_metadata_files:
                print(f"  - {f}")
        else:
            print("\n✗ No JSON metadata files were discovered")

        # Verify files list is NOT exposed to LLM
        if "json_metadata_files" not in llm_json:
            print("\n✓ json_metadata_files is correctly NOT exposed to LLM")
        else:
            print("\n✗ WARNING: json_metadata_files should NOT be in LLM output!")

        print("\n" + "="*60)
        print("Full LLM JSON keys:")
        print("="*60)
        for key in sorted(llm_json.keys()):
            if isinstance(llm_json[key], str):
                length = len(llm_json[key])
                print(f"  {key}: <string, {length} chars>")
            elif isinstance(llm_json[key], list):
                length = len(llm_json[key])
                print(f"  {key}: <list, {length} items>")
            else:
                print(f"  {key}: {type(llm_json[key]).__name__}")

        return True

if __name__ == "__main__":
    try:
        success = test_json_metadata_feature()
        if success:
            print("\n✓ Test completed successfully")
        else:
            print("\n✗ Test failed")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
