#!/usr/bin/env python3
"""Delete dataset(s) from the EEGDash database (records + dataset doc).

Targets the admin mongodb-eegdash-server endpoints:

  DELETE /admin/{database}/records?filter={"dataset": "<id>"}
  DELETE /admin/{database}/datasets/{id}

Both require an ``EEGDASH_API_TOKEN`` admin token.

Usage:
    export EEGDASH_API_TOKEN=...
    python scripts/delete_datasets.py nm000202 nm000203 nm000156 --dry-run
    python scripts/delete_datasets.py nm000202 nm000203 nm000156  # real run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

DEFAULT_API_URL = os.getenv("EEGDASH_API_URL", "https://data.eegdash.org")


def get_dataset(api_url: str, database: str, dataset_id: str) -> dict | None:
    r = requests.get(
        f"{api_url}/api/{database}/datasets/{dataset_id}", timeout=30
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def count_records(api_url: str, database: str, dataset_id: str) -> int:
    r = requests.get(
        f"{api_url}/api/{database}/count",
        params={"filter": f'{{"dataset": "{dataset_id}"}}'},
        timeout=30,
    )
    r.raise_for_status()
    return int(r.json().get("count", 0))


def delete_records(api_url: str, database: str, dataset_id: str, token: str) -> int:
    r = requests.delete(
        f"{api_url}/admin/{database}/records",
        params={
            "filter": f'{{"dataset": "{dataset_id}"}}',
            "compute_stats": "false",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    r.raise_for_status()
    return int(r.json().get("deleted_count", 0))


def delete_dataset_doc(
    api_url: str, database: str, dataset_id: str, token: str
) -> bool:
    r = requests.delete(
        f"{api_url}/admin/{database}/datasets/{dataset_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return int(r.json().get("deleted_count", 0)) > 0


def main() -> int:
    p = argparse.ArgumentParser(description="Delete datasets from EEGDash DB")
    p.add_argument("dataset_ids", nargs="+", help="Dataset IDs to delete")
    p.add_argument(
        "--database", default="eegdash", help="Database name (default: eegdash)"
    )
    p.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Admin API URL (default: $EEGDASH_API_URL or public endpoint)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted, no writes",
    )
    p.add_argument(
        "--skip-records",
        action="store_true",
        help="Only delete the dataset doc; leave records alone (dangerous)",
    )
    args = p.parse_args()

    token = os.getenv("EEGDASH_API_TOKEN")
    if not args.dry_run and not token:
        print(
            "Error: EEGDASH_API_TOKEN env var not set (needed for admin writes).",
            file=sys.stderr,
        )
        return 1

    print(f"API: {args.api_url}")
    print(f"Database: {args.database}")
    print(f"Targets: {args.dataset_ids}")
    print(f"Dry run: {args.dry_run}")
    print()

    for did in args.dataset_ids:
        print(f"=== {did} ===")
        doc = get_dataset(args.api_url, args.database, did)
        if doc is None:
            print(f"  [skip] dataset doc not found in {args.database}")
            continue
        title = doc.get("name") or doc.get("dataset_title") or "<no title>"
        print(f"  title: {title!r:.80}")
        try:
            rec_count = count_records(args.api_url, args.database, did)
        except Exception as e:
            rec_count = -1
            print(f"  (record count failed: {e})")
        print(f"  records: {rec_count}")

        if args.dry_run:
            print("  DRY RUN — no deletions performed.")
            continue

        if not args.skip_records:
            deleted = delete_records(args.api_url, args.database, did, token)
            print(f"  ✓ deleted {deleted} records")
        removed = delete_dataset_doc(args.api_url, args.database, did, token)
        if removed:
            print(f"  ✓ deleted dataset doc")
        else:
            print(f"  ? dataset doc deletion returned no-op")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
