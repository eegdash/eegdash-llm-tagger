#!/usr/bin/env python3
"""
Push LLM tagging results to the EEGDash API.

Reads llm_output.json and updates each dataset's `tags` and `tagger_meta`
fields via the EEGDash admin API.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Add parent project to path for eegdash imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eegdash.api import EEGDash


def build_tagger_meta(result: dict, model: str, config_hash: str) -> dict:
    """Build tagger_meta matching the existing format in the DB."""
    # Hash the metadata that was sent to the LLM
    reasoning = result.get("reasoning", {})
    meta_str = json.dumps(reasoning, sort_keys=True)
    metadata_hash = hashlib.blake2b(meta_str.encode(), digest_size=8).hexdigest()

    return {
        "config_hash": config_hash,
        "metadata_hash": metadata_hash,
        "model": model,
        "tagged_at": datetime.now(timezone.utc).isoformat(),
    }


def build_tags(result: dict) -> dict:
    """Build tags dict matching the existing format in the DB."""
    return {
        "pathology": result.get("pathology", ["Unknown"]),
        "modality": result.get("modality", ["Unknown"]),
        "type": result.get("type", ["Unknown"]),
        "confidence": result.get("confidence", {}),
        "reasoning": result.get("reasoning", {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Push LLM tagging results to the EEGDash API"
    )
    parser.add_argument(
        "--llm-json",
        type=Path,
        default=Path("data/processed/llm_output.json"),
        help="Path to LLM output JSON",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="Model used for tagging (stored in tagger_meta)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Skip datasets where all confidences are below this",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have tags in the DB",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel API requests (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--database",
        default="eegdash",
        help="Database name (default: eegdash)",
    )

    args = parser.parse_args()

    # Load LLM results
    with open(args.llm_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"Loaded {len(results)} LLM results from {args.llm_json}")

    # Compute a config hash from the prompt file
    prompt_path = Path(__file__).parent.parent / "prompt.md"
    if prompt_path.exists():
        config_hash = hashlib.blake2b(
            prompt_path.read_bytes(), digest_size=8
        ).hexdigest()
    else:
        config_hash = "unknown"

    # Filter by confidence
    eligible = []
    skipped_low_conf = 0
    for r in results:
        conf = r.get("confidence", {})
        max_conf = max(conf.values()) if conf else 0
        if max_conf >= args.confidence_threshold:
            eligible.append(r)
        else:
            skipped_low_conf += 1

    print(f"Eligible (conf >= {args.confidence_threshold}): {len(eligible)}")
    print(f"Skipped (low confidence): {skipped_low_conf}")

    if args.dry_run:
        print(f"\nDRY RUN — would update {len(eligible)} datasets")
        for r in eligible[:10]:
            print(f"  {r['dataset_id']}: {r.get('pathology')} | {r.get('modality')} | {r.get('type')}")
        if len(eligible) > 10:
            print(f"  ... and {len(eligible) - 10} more")
        return 0

    # Connect to API
    eegdash = EEGDash(database=args.database)

    # Push updates
    updated = 0
    skipped_existing = 0
    failed = 0
    start_time = time.time()

    def push_one(result):
        dataset_id = result["dataset_id"]
        tags = build_tags(result)
        tagger_meta = build_tagger_meta(result, args.model, config_hash)

        if args.skip_existing:
            ds = eegdash._client.get_dataset(dataset_id)
            if ds:
                existing_tags = ds.get("tags") or {}
                if existing_tags.get("pathology") and existing_tags.get("modality") and existing_tags.get("type"):
                    return ("skipped", dataset_id)

        try:
            modified = eegdash.update_dataset(
                dataset_id,
                {"tags": tags, "tagger_meta": tagger_meta},
            )
            return ("ok", dataset_id, modified)
        except Exception as e:
            return ("error", dataset_id, str(e))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(push_one, r): r for r in eligible}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result[0]

            if status == "ok":
                updated += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] ✓ {result[1]} (modified={result[2]})")
            elif status == "skipped":
                skipped_existing += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] — {result[1]} (already tagged)")
            else:
                failed += 1
                print(f"  [{i}/{len(eligible)}] ✗ {result[1]}: {result[2]}")

            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i}/{len(eligible)} ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Updated:          {updated}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed:           {failed}")
    print(f"Total time:       {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
