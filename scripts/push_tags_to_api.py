#!/usr/bin/env python3
"""
Push LLM tagging results to the EEGDash API.

Reads llm_output.json and updates each dataset's `tags` and `tagger_meta`
fields via the EEGDash admin API.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

from eegdash_tagger.tagging.provenance import verified_result_provenance


def build_tagger_meta(result: dict, model: str | None = None) -> dict:
    """Use captured request provenance; never hash generated reasoning."""
    return verified_result_provenance(result, expected_model=model)


def build_tags(result: dict) -> dict:
    """Build tags dict matching the existing format in the DB."""
    return {
        "pathology": result.get("pathology", ["Unknown"]),
        "modality": result.get("modality", ["Unknown"]),
        "type": result.get("type", ["Unknown"]),
        "confidence": result.get("confidence", {}),
        "reasoning": result.get("reasoning", {}),
    }


def push_one(result: dict, *, api_url: str, database: str, token: str,
             skip_existing: bool = False, model: str | None = None) -> tuple:
    """Check existence, then use the catalogue's authenticated HTTP upsert.

    The remote protocol has no atomic update-only endpoint. The existence check
    reduces accidental inserts but cannot prevent a concurrent deletion race.
    A confirmed zero-update result is an unchanged write, not an error.
    """
    dataset_id = result["dataset_id"]
    base = api_url.rstrip("/")
    database_path = quote(database, safe="")
    try:
        response = requests.get(
            f"{base}/api/{database_path}/datasets/{quote(dataset_id, safe='')}",
            timeout=30,
        )
        response.raise_for_status()
        envelope = response.json()
        if not isinstance(envelope, dict) or envelope.get("success") is not True or not isinstance(envelope.get("data"), dict):
            return ("error", dataset_id, "Catalogue did not confirm an existing dataset")
        existing = envelope["data"]
        if skip_existing:
            tags = existing.get("tags") or {}
            if all(tags.get(axis) for axis in ("pathology", "modality", "type")):
                return ("skipped", dataset_id)
        payload = {"dataset_id": dataset_id, "tags": build_tags(result),
                   "tagger_meta": build_tagger_meta(result, model)}
        response = requests.post(
            f"{base}/admin/{database_path}/datasets", json=payload,
            headers={"Authorization": f"Bearer {token}"}, timeout=30,
        )
        response.raise_for_status()
        envelope = response.json()
        if (not isinstance(envelope, dict) or envelope.get("success") is not True
                or type(envelope.get("inserted_count")) is not int or envelope["inserted_count"] != 0
                or type(envelope.get("updated_count")) is not int or envelope["updated_count"] not in (0, 1)):
            return ("error", dataset_id, "Catalogue did not confirm an update-only write")
        return ("ok", dataset_id, envelope["updated_count"])
    except (requests.RequestException, ValueError, TypeError) as exc:
        # Do not print endpoint credentials/tokens from provider exception text.
        return ("error", dataset_id, type(exc).__name__)


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
        default=None,
        help="Optional check against the requested model already recorded at generation",
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

    parser.add_argument("--api-url", default=os.getenv("EEGDASH_API_URL", "https://data.eegdash.org"),
                        help="Catalogue API base URL (or EEGDASH_API_URL)")
    parser.add_argument("--admin-token", default=os.getenv("EEGDASH_ADMIN_TOKEN"),
                        help="Admin bearer token (prefer EEGDASH_ADMIN_TOKEN environment variable)")
    args = parser.parse_args()

    # Load LLM results
    with open(args.llm_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"Loaded {len(results)} LLM results from {args.llm_json}")

    # Filter by confidence
    eligible = []
    skipped_low_conf = 0
    for r in results:
        if r.get("status") == "error":
            continue
        conf = r.get("confidence", {})
        max_conf = max(conf.values()) if conf else 0
        if max_conf >= args.confidence_threshold:
            eligible.append(r)
        else:
            skipped_low_conf += 1

    print(f"Eligible (conf >= {args.confidence_threshold}): {len(eligible)}")
    print(f"Skipped (low confidence): {skipped_low_conf}")

    # Fail before any connection/write if the input cannot support provenance.
    try:
        for result in eligible:
            build_tagger_meta(result, args.model)
    except ValueError as exc:
        print(f"Invalid input provenance: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\nDRY RUN — would update {len(eligible)} datasets")
        for r in eligible[:10]:
            print(f"  {r['dataset_id']}: {r.get('pathology')} | {r.get('modality')} | {r.get('type')}")
        if len(eligible) > 10:
            print(f"  ... and {len(eligible) - 10} more")
        return 0

    if not eligible:
        return 0
    if not args.admin_token:
        print("An explicit EEGDASH_ADMIN_TOKEN or --admin-token is required", file=sys.stderr)
        return 1

    updated = 0
    skipped_existing = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(push_one, r, api_url=args.api_url, database=args.database,
                                   token=args.admin_token, skip_existing=args.skip_existing,
                                   model=args.model): r for r in eligible}

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
    print(f"Confirmed writes (updated or unchanged): {updated}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed:           {failed}")
    print(f"Total time:       {elapsed:.1f}s")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
