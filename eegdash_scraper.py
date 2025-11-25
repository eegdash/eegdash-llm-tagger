#!/usr/bin/env python
"""
EEGDash Dataset Scraper

This module scrapes the EEGDash website to find datasets with missing or
unknown tags, discovers their OpenNeuro and GitHub repositories, clones them
into temporary directories, and extracts metadata using our existing metadata
parser. Cloned repositories are automatically cleaned up after parsing.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
import subprocess
import tempfile
import sys

import requests
from bs4 import BeautifulSoup

# Import our existing metadata parser
from eegdash_metadata import build_dataset_summary, build_dataset_summary_from_path


# ============================================================================
# Constants
# ============================================================================

EEGDASH_BASE_URL = "https://eegdash.org"
DATASET_SUMMARY_URL = f"{EEGDASH_BASE_URL}/dataset_summary.html"
OPENNEURO_BASE_URL = "https://openneuro.org/datasets"
OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
GITHUB_BASE_URL = "https://github.com/OpenNeuroDatasets"


# ============================================================================
# Core Scraping Functions
# ============================================================================

def fetch_dataset_table() -> List[Dict[str, str]]:
    """
    Scrape the dataset summary table from EEGDash and return a list of dicts.

    Returns:
        List of dicts with keys:
        - dataset_id: e.g., "DS001971"
        - pathology: Pathology tag from table
        - modality: "Modality of experiment" tag
        - type: "Type of experiment" tag
        - record_modality: "Record modality" column
        - detail_url: URL to dataset detail page
    """
    try:
        response = requests.get(DATASET_SUMMARY_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching dataset summary page: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main data table
    table = soup.find('table')
    if not table:
        print("Error: Could not find dataset table on page", file=sys.stderr)
        return []

    # Parse header row to get column indices
    headers = []
    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

    # Map column names to indices
    col_map = {}
    for i, header in enumerate(headers):
        header_lower = header.lower().strip()
        if header_lower == 'dataset':
            col_map['dataset_id'] = i
        elif header_lower == 'pathology':
            col_map['pathology'] = i
        elif header_lower == 'modality':
            col_map['modality'] = i
        elif header_lower == 'type':
            col_map['type'] = i
        elif header_lower == 'record modality':
            col_map['record_modality'] = i

    # Parse data rows
    datasets = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all(['td', 'th'])
        if len(cells) < len(col_map):
            continue

        # Extract dataset ID
        dataset_id = None
        if 'dataset_id' in col_map:
            cell = cells[col_map['dataset_id']]
            # Look for link first
            link = cell.find('a')
            if link:
                dataset_id = link.get_text(strip=True)
            else:
                dataset_id = cell.get_text(strip=True)

        if not dataset_id:
            continue

        # Build dataset entry
        dataset = {
            'dataset_id': dataset_id,
            'pathology': cells[col_map.get('pathology', 0)].get_text(strip=True) if 'pathology' in col_map else '',
            'modality': cells[col_map.get('modality', 0)].get_text(strip=True) if 'modality' in col_map else '',
            'type': cells[col_map.get('type', 0)].get_text(strip=True) if 'type' in col_map else '',
            'record_modality': cells[col_map.get('record_modality', 0)].get_text(strip=True) if 'record_modality' in col_map else '',
            'detail_url': f"{EEGDASH_BASE_URL}/api/dataset/eegdash.dataset.{dataset_id}.html"
        }

        datasets.append(dataset)

    return datasets


def needs_tagging(row: Dict[str, str]) -> bool:
    """
    Return True if this dataset is missing any of the three key tags:
    pathology, modality, or type.

    Treats empty strings or case-insensitive 'unknown' as missing.

    Args:
        row: Dataset dict with pathology, modality, and type fields

    Returns:
        True if any tag is missing or unknown
    """
    for key in ['pathology', 'modality', 'type']:
        value = row.get(key, '').strip().lower()
        if not value or value == 'unknown':
            return True
    return False


def fetch_openneuro_id(detail_url: str) -> Optional[str]:
    """
    Fetch the EEGDash dataset detail page and extract the OpenNeuro dataset ID.

    Args:
        detail_url: URL to EEGDash dataset detail page

    Returns:
        OpenNeuro ID (e.g., 'ds001971') or None if not found
    """
    try:
        response = requests.get(detail_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Warning: Could not fetch detail page {detail_url}: {e}", file=sys.stderr)
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Strategy 1: Look for text containing "OpenNeuro:" or "openneuro.org"
    text = soup.get_text()
    lines = text.split('\n')
    for line in lines:
        if 'openneuro' in line.lower():
            # Look for ds followed by digits
            import re
            match = re.search(r'\b(ds\d{6})\b', line, re.IGNORECASE)
            if match:
                return match.group(1).lower()

    # Strategy 2: Look for links to openneuro.org
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'openneuro.org' in href:
            import re
            match = re.search(r'/(ds\d{6})', href, re.IGNORECASE)
            if match:
                return match.group(1).lower()

    # Strategy 3: Look for any ds\d{6} pattern in the page
    import re
    match = re.search(r'\b(ds\d{6})\b', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    print(f"Warning: Could not find OpenNeuro ID on {detail_url}", file=sys.stderr)
    return None


def fetch_eegdash_detail_info(detail_url: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Fetch the EEGDash dataset detail page and extract summary information.

    Args:
        detail_url: URL to EEGDash dataset detail page
        verbose: If True, print progress messages

    Returns:
        Dict with keys like:
        - eegdash_summary: short textual summary
        - eegdash_subjects: number of subjects (if found)
        - eegdash_sampling_rate: sampling rate in Hz (if found)
        Empty dict if parsing fails.
    """
    try:
        response = requests.get(detail_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        if verbose:
            print(f"  Warning: Could not fetch EEGDash detail page: {e}", file=sys.stderr)
        return {}

    soup = BeautifulSoup(response.text, 'html.parser')
    result = {}

    try:
        # Look for dataset information in the page
        text = soup.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Try to extract structured info
        for line in lines:
            line_lower = line.lower()

            # Extract subject count
            if 'subject' in line_lower:
                import re
                match = re.search(r'(\d+)\s*subject', line_lower)
                if match:
                    result['eegdash_subjects'] = int(match.group(1))

            # Extract sampling rate
            if 'sampling' in line_lower or 'sample rate' in line_lower:
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*hz', line_lower)
                if match:
                    result['eegdash_sampling_rate'] = float(match.group(1))

        # Build a summary from key-value pairs found on the page
        summary_parts = []
        for line in lines[:50]:  # Check first 50 lines
            if ':' in line and len(line) < 200:
                # Lines like "Modality: EEG" or "Type: Resting state"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key and val and len(key) < 30:
                        summary_parts.append(f"{key}: {val}")

        if summary_parts:
            result['eegdash_summary'] = "; ".join(summary_parts[:5])

    except Exception as e:
        if verbose:
            print(f"  Warning: Error parsing EEGDash detail page: {e}", file=sys.stderr)

    return result


def fetch_openneuro_metadata(openneuro_id: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Fetch dataset metadata from OpenNeuro GraphQL API.

    Args:
        openneuro_id: OpenNeuro dataset ID (e.g., "ds001971")
        verbose: If True, print progress messages

    Returns:
        Dict with keys like:
        - openneuro_name: dataset name
        - openneuro_description: dataset description
        - openneuro_keywords: list of keywords
        - openneuro_modalities: list of modalities
        - openneuro_tasks: list of task names
        Empty dict if API call fails.
    """
    # GraphQL query to fetch dataset metadata
    query = """
    query($datasetId: ID!) {
      dataset(id: $datasetId) {
        id
        name
        public
        latestSnapshot {
          tag
          description {
            Name
            BIDSVersion
            Authors
            DatasetDOI
            License
            Acknowledgements
            HowToAcknowledge
            Funding
            ReferencesAndLinks
            DatasetType
          }
          summary {
            modalities
            tasks
            subjects
            sessions
            size
          }
        }
      }
    }
    """

    variables = {"datasetId": openneuro_id}

    try:
        response = requests.post(
            OPENNEURO_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Check for GraphQL errors
        if 'errors' in data:
            if verbose:
                print(f"  Warning: GraphQL errors: {data['errors']}", file=sys.stderr)
            return {}

        result = {}

        # Extract dataset info
        dataset = data.get('data', {}).get('dataset', {})
        if not dataset:
            return {}

        # Basic info
        if 'name' in dataset:
            result['openneuro_name'] = dataset['name']

        # Snapshot metadata
        snapshot = dataset.get('latestSnapshot', {})
        if snapshot:
            # Description fields
            description = snapshot.get('description', {})
            if description:
                if 'Name' in description:
                    result['openneuro_name'] = description['Name']  # Override with full name
                if 'Authors' in description and description['Authors']:
                    result['openneuro_authors'] = description['Authors']
                if 'DatasetDOI' in description:
                    result['openneuro_doi'] = description['DatasetDOI']
                if 'License' in description:
                    result['openneuro_license'] = description['License']

            # Summary fields
            summary = snapshot.get('summary', {})
            if summary:
                if 'modalities' in summary and summary['modalities']:
                    result['openneuro_modalities'] = summary['modalities']
                if 'tasks' in summary and summary['tasks']:
                    result['openneuro_tasks'] = summary['tasks']
                # Note: 'subjects' field contains list of subject IDs, not useful for tagging

        return result

    except requests.RequestException as e:
        if verbose:
            print(f"  Warning: Could not fetch OpenNeuro metadata: {e}", file=sys.stderr)
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        if verbose:
            print(f"  Warning: Error parsing OpenNeuro metadata: {e}", file=sys.stderr)
        return {}


# ============================================================================
# Repository Handling
# ============================================================================

def get_github_token() -> str:
    """
    Get GitHub token from GITHUB_TOKEN environment variable.

    Returns:
        GitHub token string

    Raises:
        ValueError: If GITHUB_TOKEN is not set
    """
    from file_providers import get_github_token_from_env
    return get_github_token_from_env()


def clone_repo_to_temp(github_url: str) -> Path:
    """
    [DEPRECATED] Clone the given GitHub repo into a temporary directory.

    This function is deprecated in favor of direct GitHub API access via
    build_metadata_for_dataset(), which doesn't require cloning.

    Kept for potential manual testing or special cases.

    Args:
        github_url: GitHub repository URL

    Returns:
        Path to cloned repository root

    Raises:
        RuntimeError: If cloning fails
    """
    temp_dir = tempfile.mkdtemp(prefix='eegdash_')
    temp_path = Path(temp_dir)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(temp_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 minute timeout
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git clone failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Git clone timed out after 5 minutes")
    except FileNotFoundError:
        raise RuntimeError("Git command not found. Please install git.")

    return temp_path


def build_metadata_for_dataset(github_url: str, token: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract metadata from a GitHub repository by shallow cloning into a
    temporary directory and running build_dataset_summary_from_path().

    Args:
        github_url: GitHub repository URL
        token: UNUSED (kept for backward compatibility; cloning public repos
               doesn't require authentication)
        verbose: If True, print progress messages

    Returns:
        Metadata dict from DatasetSummary.to_llm_json()

    Raises:
        RuntimeError: If cloning or parsing fails
    """
    if verbose:
        print(f"  Cloning {github_url} into temporary directory...")

    with tempfile.TemporaryDirectory(prefix="eegdash_") as tmpdir:
        repo_path = Path(tmpdir) / "repo"

        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", github_url, str(repo_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minutes
            )
        except FileNotFoundError:
            raise RuntimeError("Git command not found. Please install git.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git clone timed out after 5 minutes")

        if proc.returncode != 0:
            raise RuntimeError(f"Git clone failed for {github_url}: {proc.stderr}")

        # Now parse the local repo
        try:
            summary = build_dataset_summary_from_path(repo_path)
            metadata = summary.to_llm_json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse metadata from cloned repo: {e}")

        if verbose:
            print(f"  ✓ Successfully extracted metadata")

        return metadata


# ============================================================================
# Data Collection and Enrichment
# ============================================================================

def collect_incomplete_datasets(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    For each dataset with missing/Unknown tags, fetch OpenNeuro ID,
    construct URLs, and gather additional info from EEGDash and OpenNeuro.

    Args:
        verbose: If True, print progress messages

    Returns:
        List of dicts with dataset info, URLs, and enriched metadata:
        {
            "dataset_id": "DS001971",
            "pathology": "Unknown",
            "modality": "Unknown",
            "type": "Unknown",
            "record_modality": "EEG",
            "openneuro_id": "ds001971",
            "openneuro_url": "...",
            "github_url": "...",
            "eegdash_summary": "...",  # if available
            "eegdash_subjects": 18,    # if available
            "openneuro_name": "...",   # if available
            "openneuro_description": "...",  # if available
            ...
        }
    """
    if verbose:
        print("Fetching dataset table from EEGDash...")

    all_datasets = fetch_dataset_table()

    if verbose:
        print(f"Found {len(all_datasets)} total datasets")

    # Filter to incomplete datasets
    incomplete = [d for d in all_datasets if needs_tagging(d)]

    if verbose:
        print(f"Found {len(incomplete)} datasets with missing/unknown tags")

    # Enrich with OpenNeuro info
    enriched = []
    for i, dataset in enumerate(incomplete, 1):
        if verbose:
            print(f"[{i}/{len(incomplete)}] Processing {dataset['dataset_id']}...")

        # Fetch OpenNeuro ID from detail page
        openneuro_id = fetch_openneuro_id(dataset['detail_url'])

        if not openneuro_id:
            if verbose:
                print(f"  Warning: Could not find OpenNeuro ID, skipping")
            continue

        # Construct URLs
        dataset['openneuro_id'] = openneuro_id
        dataset['openneuro_url'] = f"{OPENNEURO_BASE_URL}/{openneuro_id}"
        dataset['github_url'] = f"{GITHUB_BASE_URL}/{openneuro_id}"

        if verbose:
            print(f"  Found OpenNeuro ID: {openneuro_id}")

        # Fetch additional info from EEGDash detail page
        if verbose:
            print(f"  Fetching EEGDash detail info...")
        eegdash_info = fetch_eegdash_detail_info(dataset['detail_url'], verbose=verbose)
        # Only keep eegdash_subjects at parent level (other info will be in metadata)
        if 'eegdash_subjects' in eegdash_info:
            dataset['eegdash_subjects'] = eegdash_info['eegdash_subjects']

        enriched.append(dataset)

    return enriched


def enrich_with_metadata(rows: List[Dict[str, Any]], token: Optional[str] = None, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    For each row, fetch metadata by cloning GitHub repositories.

    Args:
        rows: List of dataset dicts from collect_incomplete_datasets()
        token: UNUSED (kept for backward compatibility)
        verbose: If True, print progress messages

    Returns:
        List of dicts with 'metadata' field added (or 'error' on failure)
    """
    results = []

    for i, row in enumerate(rows, 1):
        if verbose:
            print(f"\n[{i}/{len(rows)}] Fetching metadata for {row['dataset_id']} ({row['openneuro_id']})...")

        try:
            # Fetch BIDS metadata by cloning repository
            bids_meta = build_metadata_for_dataset(row['github_url'], token, verbose=verbose)

            # Start from BIDS metadata
            metadata = dict(bids_meta)

            # Ensure recording_modality prefers EEGDash record_modality if available
            rec_mod = row.get("record_modality") or metadata.get("recording_modality")
            if rec_mod:
                metadata["recording_modality"] = rec_mod

            # Fetch and attach OpenNeuro metadata (only in metadata object, not at parent level)
            if verbose:
                print(f"  Fetching OpenNeuro metadata...")
            openneuro_info = fetch_openneuro_metadata(row.get('openneuro_id'), verbose=verbose)

            # Add OpenNeuro fields to metadata
            if openneuro_info.get("openneuro_name"):
                metadata["openneuro_name"] = openneuro_info["openneuro_name"]
            if openneuro_info.get("openneuro_authors"):
                metadata["openneuro_authors"] = openneuro_info["openneuro_authors"]
            if openneuro_info.get("openneuro_doi"):
                metadata["openneuro_doi"] = openneuro_info["openneuro_doi"]
            if openneuro_info.get("openneuro_license"):
                metadata["openneuro_license"] = openneuro_info["openneuro_license"]
            if openneuro_info.get("openneuro_modalities"):
                metadata["openneuro_modalities"] = openneuro_info["openneuro_modalities"]
            if openneuro_info.get("openneuro_tasks"):
                metadata["openneuro_tasks"] = openneuro_info["openneuro_tasks"]

            # Attach EEGDash info to metadata (subjects count is useful)
            if row.get("eegdash_subjects") is not None:
                metadata["eegdash_subjects"] = row["eegdash_subjects"]

            row['metadata'] = metadata

            if verbose:
                print(f"  ✓ Successfully extracted metadata")
                print(f"    Title: {metadata.get('title', 'N/A')}")
                print(f"    Tasks: {len(metadata.get('tasks', []))} tasks")
                print(f"    Events: {len(metadata.get('events', []))} event types")

        except Exception as e:
            row['error'] = str(e)

            if verbose:
                print(f"  ✗ Error: {e}")

        results.append(row)

    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI entry point for the scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape EEGDash for datasets with missing tags and extract metadata"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of datasets to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages"
    )

    args = parser.parse_args()

    # Step 1: Collect incomplete datasets
    if args.verbose:
        print("=" * 60)
        print("EEGDash Dataset Scraper (Git Clone Mode)")
        print("=" * 60)

    incomplete_datasets = collect_incomplete_datasets(verbose=args.verbose)

    if not incomplete_datasets:
        print("No datasets found with missing tags")
        return 0

    # Step 2: Apply limit if specified
    if args.limit:
        incomplete_datasets = incomplete_datasets[:args.limit]
        if args.verbose:
            print(f"\nLimiting to first {args.limit} dataset(s)")

    # Step 3: Enrich with metadata by cloning GitHub repos
    if args.verbose:
        print(f"\nFetching metadata for {len(incomplete_datasets)} dataset(s) by cloning GitHub repos...")

    enriched_datasets = enrich_with_metadata(incomplete_datasets, token=None, verbose=args.verbose)

    # Step 4: Write output
    output_path = Path(args.output_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_datasets, f, indent=2, ensure_ascii=False)

    # Step 5: Print summary
    successful = sum(1 for d in enriched_datasets if 'metadata' in d)
    failed = sum(1 for d in enriched_datasets if 'error' in d)

    if args.verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total processed:    {len(enriched_datasets)}")
        print(f"Successful:         {successful}")
        print(f"Failed:             {failed}")
        print(f"Output written to:  {output_path}")
    else:
        print(f"Processed {len(enriched_datasets)} datasets ({successful} successful, {failed} failed)")
        print(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
