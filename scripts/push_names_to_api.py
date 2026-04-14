#!/usr/bin/env python3
"""Push canonical-name suggestions to the EEGDash database.

Reads ``name_suggestions_all.json`` (or any file with the same schema) and
sets ``canonical_name``, ``name_source``, ``name_confidence``, and
``name_meta`` on each dataset document.

Duplicate names across datasets are NOT filtered out here — the registry
resolves collisions at load time with WARNING logs, and the user wants
the raw LLM output preserved on the DB side.

Usage:
    # Set admin token and run (API path)
    export EEGDASH_API_TOKEN=...
    python scripts/push_names_to_api.py \
        --names-json data/processed/name_suggestions_all.json --verbose

    # Dry-run (no writes)
    python scripts/push_names_to_api.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add parent project to path for eegdash imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eegdash.api import EEGDash  # noqa: E402


def build_name_meta(result: dict, model: str, config_hash: str) -> dict:
    """Provenance sidecar, analogous to ``tagger_meta`` for tags."""
    reasoning = result.get("reasoning", "")
    meta_hash = hashlib.blake2b(
        str(reasoning).encode(), digest_size=8
    ).hexdigest()
    return {
        "config_hash": config_hash,
        "metadata_hash": meta_hash,
        "model": model,
        "suggested_at": datetime.now(timezone.utc).isoformat(),
        "name_source": result.get("name_source", "none"),
        "name_confidence": float(result.get("name_confidence", 0.0)),
        "reasoning": str(reasoning)[:500],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push canonical-name suggestions to the EEGDash API"
    )
    parser.add_argument(
        "--names-json",
        type=Path,
        default=Path("data/processed/name_suggestions_all.json"),
        help="Path to the name-suggestions JSON",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2 + openai/gpt-5.4-mini",
        help="Model(s) used for suggestion (recorded in name_meta)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Skip datasets whose name_confidence is below this value "
        "(default: 0.0 — push everything non-empty)",
    )
    parser.add_argument(
        "--only-named",
        action="store_true",
        default=True,
        help="Skip datasets where canonical_name is empty (default: on)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have a non-empty canonical_name in the DB",
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

    with open(args.names_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"Loaded {len(results)} name suggestions from {args.names_json}")

    # Hash the prompt as a config fingerprint.
    prompt_src = Path(__file__).parent.parent / "src" / "eegdash_tagger" / "naming" / "name_suggester.py"
    config_hash = (
        hashlib.blake2b(prompt_src.read_bytes(), digest_size=8).hexdigest()
        if prompt_src.exists()
        else "unknown"
    )

    # Filter candidates.
    eligible = []
    skipped_empty = 0
    skipped_low_conf = 0
    for r in results:
        names = r.get("canonical_name") or []
        if args.only_named and not names:
            skipped_empty += 1
            continue
        if float(r.get("name_confidence", 0.0)) < args.confidence_threshold:
            skipped_low_conf += 1
            continue
        eligible.append(r)

    print(
        f"Eligible: {len(eligible)} | empty-skipped: {skipped_empty} | "
        f"low-conf-skipped: {skipped_low_conf}"
    )

    if args.dry_run:
        print(f"\nDRY RUN — would update {len(eligible)} datasets. Sample:")
        for r in eligible[:15]:
            print(
                f"  {r['dataset_id']:15} {r.get('name_source'):12} "
                f"conf={r.get('name_confidence', 0.0):.2f} -> {r.get('canonical_name')}"
            )
        if len(eligible) > 15:
            print(f"  ... and {len(eligible) - 15} more")
        return 0

    eegdash = EEGDash(database=args.database)

    updated = 0
    skipped_existing = 0
    failed = 0
    start = time.time()

    def push_one(result: dict) -> tuple:
        dataset_id = result["dataset_id"]
        names = result.get("canonical_name") or []
        source = result.get("name_source", "none")
        conf = float(result.get("name_confidence", 0.0))
        name_meta = build_name_meta(result, args.model, config_hash)

        if args.skip_existing:
            existing = eegdash._client.get_dataset(dataset_id)
            if existing and (existing.get("canonical_name") or []):
                return ("skipped", dataset_id, None)

        payload = {
            "canonical_name": names,
            "name_source": source,
            "name_confidence": conf,
            "name_meta": name_meta,
        }
        try:
            modified = eegdash.update_dataset(dataset_id, payload)
            return ("ok", dataset_id, modified)
        except Exception as e:
            return ("error", dataset_id, str(e))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(push_one, r) for r in eligible]
        for i, future in enumerate(as_completed(futures), 1):
            status, ds_id, info = future.result()
            if status == "ok":
                updated += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] ✓ {ds_id} (modified={info})")
            elif status == "skipped":
                skipped_existing += 1
                if args.verbose:
                    print(f"  [{i}/{len(eligible)}] — {ds_id} (already has name)")
            else:
                failed += 1
                print(f"  [{i}/{len(eligible)}] ✗ {ds_id}: {info}")
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(eligible)} ({time.time() - start:.0f}s)")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Updated:          {updated}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed:           {failed}")
    print(f"Total time:       {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
