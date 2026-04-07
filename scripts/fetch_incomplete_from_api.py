#!/usr/bin/env python3
"""
Fetch incomplete datasets directly from the EEGDash API.

Instead of scraping the HTML page and cloning GitHub repos, this script
pulls metadata (including READMEs, tasks, demographics, DOIs) directly
from https://data.eegdash.org/api, which is much faster and covers all
dataset sources (OpenNeuro, NEMAR, etc.).
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"


def fetch_all_datasets(database: str = DEFAULT_DATABASE) -> list[dict[str, Any]]:
    """Fetch all dataset summaries from the EEGDash API."""
    all_datasets = []
    skip = 0
    limit = 1000

    while True:
        url = f"{API_BASE_URL}/{database}/datasets/summary?limit={limit}&skip={skip}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        if not data.get("success"):
            raise ValueError(f"API returned error: {data}")

        datasets = data.get("data", [])
        all_datasets.extend(datasets)

        if len(datasets) < limit:
            break
        skip += limit

    return all_datasets


def needs_tagging(ds: dict[str, Any]) -> bool:
    """Check if a dataset is missing any of the three tags."""
    tags = ds.get("tags") or {}
    for key in ("pathology", "modality", "type"):
        val = tags.get(key, [])
        if not val:
            return True
        # Handle both list and string
        if isinstance(val, list):
            label = val[0].strip().lower() if val else ""
        else:
            label = str(val).strip().lower()
        if not label or label == "unknown":
            return True
    return False


def build_metadata_from_api(ds: dict[str, Any]) -> dict[str, Any]:
    """Convert an API dataset record into the metadata format expected by the tagger."""
    # Build dataset_description similar to what the BIDS parser produces
    authors = ds.get("authors", [])
    authors_str = ", ".join(authors) if authors else ""
    doi = ds.get("dataset_doi", "") or ""
    paper_doi = ds.get("associated_paper_doi", "") or ""

    desc_parts = [f"Name: {ds.get('name', '')}"]
    if authors_str:
        desc_parts.append(f"Authors: {authors_str}")
    if doi:
        desc_parts.append(f"DOI: {doi}")
    if paper_doi:
        desc_parts.append(f"References: doi: {paper_doi}")

    # Build participants overview from demographics
    demographics = ds.get("demographics", {})
    participants_parts = []
    if isinstance(demographics, dict):
        if demographics.get("subjects_count"):
            participants_parts.append(
                f"Subjects: {demographics['subjects_count']}"
            )
        if demographics.get("sex_distribution"):
            participants_parts.append(
                f"Sex: {demographics['sex_distribution']}"
            )
        if demographics.get("age_min") is not None and demographics.get("age_max") is not None:
            participants_parts.append(
                f"Age range: {demographics['age_min']}-{demographics['age_max']}"
            )
        if demographics.get("species"):
            participants_parts.append(f"Species: {demographics['species']}")
        if demographics.get("handedness_distribution"):
            participants_parts.append(
                f"Handedness: {demographics['handedness_distribution']}"
            )

    # Tasks
    tasks = ds.get("tasks", []) or []

    metadata = {
        "title": ds.get("computed_title") or ds.get("name", ""),
        "dataset_description": "\n".join(desc_parts),
        "readme": ds.get("readme", "") or "",
        "participants_overview": "; ".join(participants_parts) if participants_parts else "",
        "tasks": tasks,
        "events": [],  # Not available in summary API
        "paper_abstract": "",  # Will be fetched separately if needed
    }

    # Add extra fields that the tagger can use
    if ds.get("recording_modality"):
        rec_mod = ds["recording_modality"]
        if isinstance(rec_mod, list):
            metadata["recording_modality"] = ", ".join(rec_mod)
        else:
            metadata["recording_modality"] = str(rec_mod)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Fetch incomplete datasets from the EEGDash API"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets to process",
    )
    parser.add_argument(
        "--fetch-abstracts",
        action="store_true",
        help="Also fetch paper abstracts (slower but better metadata)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    if args.verbose:
        print("=" * 60)
        print("EEGDash API — Fetch Incomplete Datasets")
        print("=" * 60)

    # Step 1: Fetch all datasets from API
    if args.verbose:
        print("Fetching datasets from EEGDash API...")

    all_datasets = fetch_all_datasets()

    if args.verbose:
        print(f"Fetched {len(all_datasets)} total datasets")

    # Step 2: Filter to incomplete
    incomplete = [ds for ds in all_datasets if needs_tagging(ds)]

    if args.verbose:
        print(f"Found {len(incomplete)} datasets needing tagging")

    if not incomplete:
        print("No datasets found with missing tags")
        return 0

    # Step 3: Apply limit
    if args.limit:
        incomplete = incomplete[: args.limit]
        if args.verbose:
            print(f"Limiting to first {args.limit} dataset(s)")

    # Step 4: Build metadata for each dataset
    results = []

    # Optionally set up abstract fetching
    fetch_abstract_fn = None
    cache_path = None
    if args.fetch_abstracts:
        try:
            from eegdash_tagger.scraping.abstract_fetcher import (
                extract_dois_from_references,
                fetch_abstract_with_cache,
            )

            cache_path = (
                Path(__file__).parent.parent
                / "data"
                / "processed"
                / "abstract_cache.json"
            )
            fetch_abstract_fn = fetch_abstract_with_cache
        except ImportError:
            print("Warning: Could not import abstract fetcher, skipping abstracts")

    for i, ds in enumerate(incomplete, 1):
        dataset_id = ds.get("dataset_id", "")
        if args.verbose:
            print(f"[{i}/{len(incomplete)}] {dataset_id}: {ds.get('name', '')[:60]}")

        try:
            metadata = build_metadata_from_api(ds)

            # Fetch paper abstract if requested
            if fetch_abstract_fn and cache_path:
                paper_doi = ds.get("associated_paper_doi") or ""
                dataset_doi = ds.get("dataset_doi") or ""
                # Try paper DOI first, then extract from description
                dois_to_try = []
                if paper_doi:
                    dois_to_try.append(paper_doi)
                if dataset_doi:
                    # Extract referenced papers from dataset description
                    try:
                        from eegdash_tagger.scraping.abstract_fetcher import (
                            extract_dois_from_references,
                        )

                        desc = metadata.get("dataset_description", "")
                        extracted = extract_dois_from_references(
                            desc, dataset_doi
                        )
                        dois_to_try.extend(extracted)
                    except Exception:
                        pass

                abstracts = []
                for doi in dois_to_try:
                    abstract = fetch_abstract_fn(
                        doi, cache_path=cache_path, verbose=False
                    )
                    if abstract:
                        abstracts.append(f"[DOI: {doi}]\n{abstract}")
                if abstracts:
                    metadata["paper_abstract"] = "\n\n---\n\n".join(abstracts)
                    if args.verbose:
                        print(f"  Fetched {len(abstracts)} abstract(s)")

            entry = {
                "dataset_id": dataset_id,
                "source": ds.get("source", ""),
                "metadata": metadata,
            }
            results.append(entry)

            if args.verbose and metadata.get("readme"):
                print(f"  README: {len(metadata['readme'])} chars")

        except Exception as e:
            if args.verbose:
                print(f"  Error: {e}")
            results.append({"dataset_id": dataset_id, "error": str(e)})

    # Step 5: Write output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in results if "metadata" in r)
    failed = sum(1 for r in results if "error" in r)
    with_readme = sum(
        1
        for r in results
        if "metadata" in r and len(r["metadata"].get("readme", "")) > 10
    )

    if args.verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
    print(f"Total processed:    {len(results)}")
    print(f"Successful:         {successful}")
    print(f"  With README:      {with_readme}")
    print(f"  Without README:   {successful - with_readme}")
    print(f"Failed:             {failed}")
    print(f"Output written to:  {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
