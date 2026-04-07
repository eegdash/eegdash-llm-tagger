#!/usr/bin/env python3
"""
Batch LLM Tagging Script.

This script processes all incomplete datasets using OpenRouter.ai LLM API
and generates llm_output.json for CSV updating. Supports parallel requests
via --workers for much faster throughput.
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.tagging.llm_tagger import OpenRouterTagger
from eegdash_tagger.tagging.tagger import ParsedMetadata


def load_datasets(input_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(input_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)
    if limit:
        datasets = datasets[:limit]
    return datasets


def convert_to_parsed_metadata(metadata: Dict[str, Any]) -> ParsedMetadata:
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
    output_data = {"results": results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def tag_single_dataset(tagger: OpenRouterTagger, dataset: Dict[str, Any], index: int, total: int, verbose: bool) -> Dict[str, Any]:
    """Tag a single dataset. Thread-safe — tagger._call_api is stateless."""
    dataset_id = dataset.get('dataset_id', f'unknown_{index}')
    metadata = dataset.get('metadata', {})

    try:
        parsed_meta = convert_to_parsed_metadata(metadata)
        result = tagger.tag_with_details(parsed_meta, dataset_id=dataset_id)

        if verbose:
            conf = result.get('confidence', {})
            print(f"  [{index}/{total}] ✓ {dataset_id}: "
                  f"{result.get('pathology')} | {result.get('modality')} | {result.get('type')} "
                  f"(P={conf.get('pathology', 0):.2f} M={conf.get('modality', 0):.2f} T={conf.get('type', 0):.2f})")

        return result

    except Exception as e:
        print(f"  [{index}/{total}] ✗ {dataset_id}: {e}")
        return {
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
        }


def main():
    parser = argparse.ArgumentParser(
        description="Tag EEG datasets using OpenRouter.ai LLM API"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/incomplete_metadata.json"),
        help="Path to incomplete_metadata.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/llm_output.json"),
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--model", default="openai/gpt-5.2",
        help="OpenRouter model identifier (default: openai/gpt-5.2)"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of datasets to process"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--save-interval", type=int, default=20,
        help="Save partial results every N completed datasets (default: 20)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OpenRouter LLM Batch Tagging (Parallel)")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("\nError: OPENROUTER_API_KEY environment variable not set")
        return 1

    print(f"\n✓ API key found")
    print(f"✓ Model: {args.model}")
    print(f"✓ Workers: {args.workers}")

    # Load datasets
    if not args.input.exists():
        print(f"\nError: Input file not found: {args.input}")
        return 1

    datasets = load_datasets(args.input, args.limit)
    print(f"✓ Loaded {len(datasets)} datasets")

    # Initialize tagger
    try:
        tagger = OpenRouterTagger(
            api_key=api_key,
            model=args.model,
            verbose=False  # per-request verbosity handled in tag_single_dataset
        )
    except Exception as e:
        print(f"\nError initializing tagger: {e}")
        return 1

    # Process datasets in parallel
    print(f"\n{'=' * 60}")
    print(f"Processing {len(datasets)} datasets with {args.workers} workers")
    print("=" * 60)

    results = [None] * len(datasets)  # Pre-allocate to preserve order
    completed = 0
    failed_count = 0
    lock = threading.Lock()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                tag_single_dataset, tagger, ds, i + 1, len(datasets), args.verbose
            ): i
            for i, ds in enumerate(datasets)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result

            with lock:
                completed += 1
                if result.get("confidence", {}).get("pathology", 1) == 0.0:
                    failed_count += 1

                # Progress update
                if completed % 5 == 0 or completed == len(datasets):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(datasets) - completed) / rate if rate > 0 else 0
                    print(f"\n  Progress: {completed}/{len(datasets)} "
                          f"({completed/len(datasets)*100:.0f}%) | "
                          f"{rate:.1f} datasets/s | "
                          f"ETA: {remaining:.0f}s")

                # Save partial results
                if completed % args.save_interval == 0:
                    # Collect completed results (skip None entries)
                    partial = [r for r in results if r is not None]
                    save_results(partial, args.output)
                    print(f"  Saved {len(partial)} partial results")

    # Save final results
    save_results(results, args.output)

    total_time = time.time() - start_time
    successful = len(results) - failed_count

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Total processed:  {len(datasets)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed_count}")
    print(f"Total time:       {total_time:.1f}s")
    print(f"Throughput:       {len(datasets)/total_time:.1f} datasets/s")
    print(f"Avg per dataset:  {total_time/len(datasets):.1f}s (wall clock: {total_time/len(datasets)*1:.1f}s)")
    print(f"\nOutput saved to:  {args.output}")

    print("\nNext step:")
    print(f"  python scripts/update_csv.py --llm-json {args.output} --csv ground-truth-data/dataset_summary.csv --dry-run --verbose")

    return 0


if __name__ == "__main__":
    sys.exit(main())
