#!/usr/bin/env python3
"""
Push LLM tagging results directly to MongoDB via SSH tunnel.

Connects to the SCCN server's MongoDB through an SSH tunnel and updates
the `tags` and `tagger_meta` fields on each dataset document.

Usage:
    # First open SSH tunnel in another terminal:
    ssh -L 27018:localhost:27017 sccn

    # Then run this script:
    python scripts/push_tags_to_mongo.py --llm-json data/processed/llm_output.json --verbose
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymongo


def build_tagger_meta(result: dict, model: str, config_hash: str) -> dict:
    """Build tagger_meta matching the existing format in the DB."""
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
        description="Push LLM tags directly to MongoDB via SSH tunnel"
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
        help="Skip datasets that already have non-empty tags",
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://competition_admin:CompAdmin2025Secure@localhost:27018",
        help="MongoDB URI (default uses SSH tunnel on port 27018)",
    )
    parser.add_argument(
        "--database",
        default="eegdash",
        help="Database name (default: eegdash)",
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

    args = parser.parse_args()

    # Load LLM results
    with open(args.llm_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"Loaded {len(results)} LLM results")

    # Compute config hash from prompt
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

    # Connect to MongoDB
    print(f"\nConnecting to MongoDB at {args.mongo_uri.split('@')[1] if '@' in args.mongo_uri else args.mongo_uri}...")
    try:
        client = pymongo.MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print("✓ Connected to MongoDB")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the SSH tunnel is open:")
        print("  ssh -L 27018:localhost:27017 sccn")
        return 1

    db = client[args.database]
    datasets_col = db["datasets"]

    # Push updates
    updated = 0
    skipped_existing = 0
    not_found = 0
    failed = 0

    for i, result in enumerate(eligible, 1):
        dataset_id = result["dataset_id"]
        tags = build_tags(result)
        tagger_meta = build_tagger_meta(result, args.model, config_hash)

        try:
            if args.skip_existing:
                existing = datasets_col.find_one({"dataset_id": dataset_id})
                if existing:
                    et = existing.get("tags") or {}
                    if et.get("pathology") and et.get("modality") and et.get("type"):
                        skipped_existing += 1
                        if args.verbose:
                            print(f"  [{i}/{len(eligible)}] — {dataset_id} (already tagged)")
                        continue

            res = datasets_col.update_one(
                {"dataset_id": dataset_id},
                {"$set": {"tags": tags, "tagger_meta": tagger_meta}},
            )

            if res.matched_count == 0:
                not_found += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] ? {dataset_id} (not found in DB)")
            else:
                updated += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] ✓ {dataset_id}: "
                          f"{result.get('pathology')} | {result.get('modality')} | {result.get('type')}")

        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(eligible)}] ✗ {dataset_id}: {e}")

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(eligible)}")

    client.close()

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Updated:          {updated}")
    print(f"Not found in DB:  {not_found}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed:           {failed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
