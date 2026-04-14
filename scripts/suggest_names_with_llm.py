#!/usr/bin/env python3
"""Batch canonical-name suggestion for EEG datasets.

Reads a metadata JSON (same format as ``incomplete_metadata.json``),
calls :class:`NameSuggester` in parallel via a thread pool, and writes a
JSON file with per-dataset canonical-name suggestions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

try:  # Auto-load .env like tag_with_llm.py does.
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.naming import NameSuggester, NameSuggestion  # noqa: E402


def load_datasets(
    input_path: Path, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if limit:
        datasets = datasets[:limit]
    return datasets


def suggest_one(
    suggester: NameSuggester,
    dataset: dict[str, Any],
    index: int,
    total: int,
    verbose: bool,
) -> NameSuggestion:
    dataset_id = dataset.get("dataset_id", f"unknown_{index}")
    metadata = dataset.get("metadata", {}) or {}
    result = suggester.suggest(metadata, dataset_id=dataset_id)
    if verbose:
        names = ", ".join(result.get("canonical_name", [])) or "(none)"
        print(
            f"  [{index}/{total}] {dataset_id}: {names} "
            f"[{result.get('name_source')}, "
            f"conf={result.get('name_confidence', 0.0):.2f}]"
        )
    return result


def save_results(results: list[NameSuggestion], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suggest canonical names for EEG datasets via OpenRouter LLM"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/incomplete_metadata.json"),
        help="Path to metadata JSON (same schema as incomplete_metadata.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/name_suggestions.json"),
        help="Path to write the per-dataset suggestions",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="OpenRouter model identifier (default: openai/gpt-5.2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N datasets (useful for smoke tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Thread pool size (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Per-dataset progress output",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY env var not set (and no .env found).",
            file=sys.stderr,
        )
        return 1

    if not args.input.exists():
        print(f"Error: input file does not exist: {args.input}", file=sys.stderr)
        return 1

    datasets = load_datasets(args.input, limit=args.limit)
    total = len(datasets)
    if total == 0:
        print("No datasets to process.")
        return 0

    print(f"Loaded {total} dataset(s) from {args.input}")
    print(f"Model: {args.model} | workers: {args.workers}")
    print(f"Output: {args.output}")
    print()

    suggester = NameSuggester(
        api_key=api_key, model=args.model, verbose=args.verbose
    )

    results: list[NameSuggestion] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(
                suggest_one, suggester, ds, i + 1, total, args.verbose
            ): i
            for i, ds in enumerate(datasets)
        }
        # list.append is atomic under the GIL — no lock needed.
        for future in as_completed(future_to_index):
            results.append(future.result())

    elapsed = time.time() - start
    # Keep the order stable (by dataset_id) so diffs are readable.
    results.sort(key=lambda r: r.get("dataset_id", ""))

    save_results(results, args.output)

    # Summary stats.
    with_name = sum(1 for r in results if r.get("canonical_name"))
    by_source: dict[str, int] = {}
    for r in results:
        src = r.get("name_source", "none")
        by_source[src] = by_source.get(src, 0) + 1
    print()
    print(f"Done in {elapsed:.1f}s — wrote {args.output}")
    print(f"  with-name: {with_name}/{total}")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
