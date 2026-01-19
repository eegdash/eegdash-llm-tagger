#!/usr/bin/env python3
"""
Example script showing how to integrate the LLM Tagger API
with the EEGDash ingestion pipeline.

Usage:
    uv run python scripts/pipeline_example.py --api-url http://localhost:8000 --limit 2
"""
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.client import EEGDashTaggerClient


def tag_incomplete_datasets(
    api_url: str = "http://localhost:8000",
    input_file: str = "data/processed/incomplete_metadata.json",
    output_file: str = "data/processed/api_tagged_output.json",
    limit: int = None,
    verbose: bool = True
):
    """
    Tag incomplete datasets using the API service.

    Args:
        api_url: URL of the LLM Tagger API
        input_file: Path to incomplete_metadata.json
        output_file: Path to save results
        limit: Max datasets to process (None for all)
        verbose: Print progress
    """
    # Initialize client
    client = EEGDashTaggerClient(api_url)

    # Check API health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Cache entries: {health['cache_entries']}")
        print(f"Config hash: {health['config_hash']}")
    except Exception as e:
        print(f"ERROR: API not available at {api_url}")
        print(f"Error: {e}")
        print("\nMake sure the Docker container is running:")
        print("  cd /path/to/eegdash-llm-api && docker-compose up -d")
        return

    # Load incomplete datasets
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        print("\nTo create the input file, run:")
        print("  uv run python scripts/fetch_incomplete_datasets.py --output-json data/processed/incomplete_metadata.json")
        return

    print(f"\nLoading datasets from {input_file}...")
    with open(input_path, 'r') as f:
        datasets = json.load(f)

    if limit:
        datasets = datasets[:limit]

    print(f"Processing {len(datasets)} datasets...")

    # Prepare dataset list for batch processing
    dataset_list = []
    for ds in datasets:
        dataset_id = ds.get('dataset_id')
        # Construct GitHub URL from OpenNeuro ID
        source_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}"
        dataset_list.append({
            "dataset_id": dataset_id,
            "source_url": source_url
        })

    # Tag datasets (skip cached by default)
    print("\nTagging datasets...")
    results = client.batch_tag_datasets(dataset_list, skip_cached=True, verbose=verbose)

    # Convert to output format
    output = []
    for result in results:
        output.append({
            "dataset_id": result.dataset_id,
            "pathology": result.pathology,
            "modality": result.modality,
            "type": result.type,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "from_cache": result.from_cache,
            "stale": result.stale,
            "error": result.error
        })

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Results saved to: {output_file}")
    print(f"Total processed: {len(results)}")
    print(f"From cache: {sum(1 for r in results if r.from_cache)}")
    print(f"Fresh LLM calls: {sum(1 for r in results if not r.from_cache)}")
    print(f"Errors: {sum(1 for r in results if r.error)}")

    # Print sample result
    if results:
        print(f"\nSample result for {results[0].dataset_id}:")
        print(f"  Pathology: {results[0].pathology}")
        print(f"  Modality: {results[0].modality}")
        print(f"  Type: {results[0].type}")


def main():
    parser = argparse.ArgumentParser(
        description="Tag EEG/MEG datasets using the LLM Tagger API"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="LLM Tagger API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--input",
        default="data/processed/incomplete_metadata.json",
        help="Input JSON file with datasets to tag"
    )
    parser.add_argument(
        "--output",
        default="data/processed/api_tagged_output.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max datasets to process (for testing)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    tag_incomplete_datasets(
        api_url=args.api_url,
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
