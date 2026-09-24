#!/usr/bin/env python3
"""
Test script for OpenRouter LLM Tagger.

This script tests the LLM tagger with a single dataset to verify
API integration before batch processing.
"""

import json
import os
import sys
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.tagging.llm_tagger import OpenRouterTagger
from eegdash_tagger.tagging.tagger import ParsedMetadata


def main():
    """Test LLM tagger with single dataset."""
    print("=" * 60)
    print("OpenRouter LLM Tagger - Single Dataset Test")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("\n❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
        return 1

    print(f"\n✓ API key found: {api_key[:20]}...")

    # Load incomplete metadata
    incomplete_path = Path(__file__).parent.parent / "data" / "processed" / "incomplete_metadata.json"

    if not incomplete_path.exists():
        print(f"\n❌ Error: Incomplete metadata not found at: {incomplete_path}")
        return 1

    print(f"✓ Loading datasets from: {incomplete_path}")

    with open(incomplete_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    if not datasets:
        print("\n❌ Error: No datasets found in incomplete_metadata.json")
        return 1

    # Use first dataset
    first_dataset = datasets[0]
    dataset_id = first_dataset.get('dataset_id', 'unknown')
    metadata = first_dataset.get('metadata', {})

    print(f"\n✓ Loaded {len(datasets)} datasets")
    print(f"✓ Testing with dataset: {dataset_id}")
    print(f"  Title: {metadata.get('title', 'N/A')[:60]}...")

    # Initialize tagger
    print(f"\n{'=' * 60}")
    print("Initializing OpenRouter Tagger")
    print("=" * 60)

    try:
        tagger = OpenRouterTagger(
            api_key=api_key,
            model="openai/gpt-4-turbo",  # Using turbo for lower cost
            verbose=True
        )
    except Exception as e:
        print(f"\n❌ Error initializing tagger: {e}")
        return 1

    # Convert to ParsedMetadata
    print(f"\n{'=' * 60}")
    print("Preparing Metadata")
    print("=" * 60)

    parsed_meta = ParsedMetadata(
        title=metadata.get('title', ''),
        dataset_description=metadata.get('dataset_description', ''),
        readme=metadata.get('readme', ''),
        participants_overview=metadata.get('participants_overview', ''),
        tasks=metadata.get('tasks', []),
        events=metadata.get('events', [])
    )

    print(f"✓ Metadata fields:")
    print(f"  - Title: {len(parsed_meta.get('title', ''))} chars")
    print(f"  - Description: {len(parsed_meta.get('dataset_description', ''))} chars")
    print(f"  - README: {len(parsed_meta.get('readme', ''))} chars")
    print(f"  - Tasks: {len(parsed_meta.get('tasks', []))} tasks")
    print(f"  - Events: {len(parsed_meta.get('events', []))} events")

    # Tag dataset
    print(f"\n{'=' * 60}")
    print("Calling LLM API")
    print("=" * 60)

    try:
        result = tagger.tag_with_details(parsed_meta, dataset_id=dataset_id)
    except Exception as e:
        print(f"\n❌ Error calling API: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display results
    print(f"\n{'=' * 60}")
    print("Tagging Results")
    print("=" * 60)

    print(f"\nDataset ID: {result.get('dataset_id')}")
    print(f"\nPathology: {result.get('pathology')}")
    print(f"Modality: {result.get('modality')}")
    print(f"Type: {result.get('type')}")

    confidence = result.get('confidence', {})
    print(f"\nConfidence Scores:")
    print(f"  - Pathology: {confidence.get('pathology', 0.0):.2f}")
    print(f"  - Modality: {confidence.get('modality', 0.0):.2f}")
    print(f"  - Type: {confidence.get('type', 0.0):.2f}")

    reasoning = result.get('reasoning', {})
    if reasoning:
        print(f"\nReasoning Summary:")
        print(f"  - Few-shot: {reasoning.get('few_shot_analysis', 'N/A')[:100]}...")
        print(f"  - Metadata: {reasoning.get('metadata_analysis', 'N/A')[:100]}...")
        print(f"  - Decision: {reasoning.get('decision_summary', 'N/A')[:100]}...")

    # Save output
    output_path = Path(__file__).parent.parent / "data" / "processed" / "test_llm_output.json"
    output_data = {
        "results": [result]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("✓ Test Complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review the output to verify correctness")
    print("  2. Try batch processing with: python scripts/tag_with_llm.py --limit 5")

    return 0


if __name__ == "__main__":
    sys.exit(main())
