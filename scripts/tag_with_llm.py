#!/usr/bin/env python3
"""
Batch LLM Tagging Script.

This script processes all incomplete datasets using OpenRouter.ai LLM API
and generates llm_output.json for CSV updating.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, rely on system environment
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.tagging.llm_tagger import OpenRouterTagger
from eegdash_tagger.tagging.tagger import ParsedMetadata


def load_datasets(input_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load datasets from JSON file.

    Args:
        input_path: Path to incomplete_metadata.json
        limit: Maximum number of datasets to process

    Returns:
        List of dataset dictionaries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    if limit:
        datasets = datasets[:limit]

    return datasets


def convert_to_parsed_metadata(metadata: Dict[str, Any]) -> ParsedMetadata:
    """
    Convert metadata dict to ParsedMetadata.

    Args:
        metadata: Raw metadata from dataset

    Returns:
        ParsedMetadata object
    """
    return ParsedMetadata(
        title=metadata.get('title', ''),
        dataset_description=metadata.get('dataset_description', ''),
        readme=metadata.get('readme', ''),
        participants_overview=metadata.get('participants_overview', ''),
        tasks=metadata.get('tasks', []),
        events=metadata.get('events', []),
        paper_abstract=metadata.get('paper_abstract', '')
    )


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """
    Save results to JSON file.

    Args:
        results: List of tagging results
        output_path: Path to output file
    """
    output_data = {"results": results}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point for batch LLM tagging."""
    parser = argparse.ArgumentParser(
        description="Tag EEG datasets using OpenRouter.ai LLM API"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incomplete_metadata.json"),
        help="Path to incomplete_metadata.json (default: data/processed/incomplete_metadata.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/llm_output.json"),
        help="Path to output JSON file (default: data/processed/llm_output.json)"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4-turbo",
        help="OpenRouter model identifier (default: openai/gpt-4-turbo)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save partial results every N datasets (default: 10)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OpenRouter LLM Batch Tagging")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("\n❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
        return 1

    print(f"\n✓ API key found")
    print(f"✓ Model: {args.model}")

    # Load datasets
    if not args.input.exists():
        print(f"\n❌ Error: Input file not found: {args.input}")
        return 1

    print(f"✓ Loading datasets from: {args.input}")
    datasets = load_datasets(args.input, args.limit)

    print(f"✓ Loaded {len(datasets)} datasets")

    if args.limit:
        print(f"  (Limited to {args.limit} datasets)")

    # Initialize tagger
    print("\n" + "=" * 60)
    print("Initializing Tagger")
    print("=" * 60)

    try:
        tagger = OpenRouterTagger(
            api_key=api_key,
            model=args.model,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\n❌ Error initializing tagger: {e}")
        return 1

    # Process datasets
    print("\n" + "=" * 60)
    print("Processing Datasets")
    print("=" * 60)

    results = []
    failed = []
    start_time = time.time()

    for i, dataset in enumerate(datasets, 1):
        dataset_id = dataset.get('dataset_id', f'unknown_{i}')
        metadata = dataset.get('metadata', {})

        if args.verbose or i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = avg_time * (len(datasets) - i)

            print(f"\n[{i}/{len(datasets)}] Processing {dataset_id}")
            print(f"  Elapsed: {elapsed:.1f}s | Avg: {avg_time:.1f}s/dataset | Est. remaining: {remaining:.1f}s")

        try:
            # Convert to ParsedMetadata
            parsed_meta = convert_to_parsed_metadata(metadata)

            # Tag dataset
            result = tagger.tag_with_details(parsed_meta, dataset_id=dataset_id)
            results.append(result)

            if args.verbose:
                print(f"  ✓ Tagged: {result.get('pathology')} | {result.get('modality')} | {result.get('type')}")
                conf = result.get('confidence', {})
                print(f"    Confidence: P={conf.get('pathology', 0):.2f} M={conf.get('modality', 0):.2f} T={conf.get('type', 0):.2f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append({"dataset_id": dataset_id, "error": str(e)})

            # Add fallback result
            results.append({
                "dataset_id": dataset_id,
                "pathology": ["Unknown"],
                "modality": ["Unknown"],
                "type": ["Unknown"],
                "confidence": {"pathology": 0.0, "modality": 0.0, "type": 0.0},
                "reasoning": {
                    "few_shot_analysis": f"Error: {str(e)}",
                    "metadata_analysis": "N/A",
                    "citation_analysis": "N/A",
                    "decision_summary": "Processing failed"
                }
            })

        # Save partial results
        if i % args.save_interval == 0:
            if args.verbose:
                print(f"  💾 Saving partial results ({i} datasets)...")
            save_results(results, args.output)

    # Save final results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    save_results(results, args.output)

    # Print summary
    total_time = time.time() - start_time
    successful = len(results) - len(failed)

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Total processed:  {len(datasets)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {len(failed)}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Average time:     {total_time / len(datasets):.1f}s per dataset")
    print(f"\nOutput saved to:  {args.output}")

    if failed:
        print(f"\nFailed datasets:")
        for fail in failed[:10]:  # Show first 10 failures
            print(f"  - {fail['dataset_id']}: {fail['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print("\nNext step:")
    print(f"  python scripts/update_csv.py --llm-json {args.output} --csv dataset_summary.csv --dry-run --verbose")

    return 0


if __name__ == "__main__":
    sys.exit(main())
