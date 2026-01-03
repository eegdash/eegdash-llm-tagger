"""
Paper abstract fetching for EEGDash datasets.

This module extracts DOIs from dataset references and fetches abstracts
from CrossRef and Semantic Scholar APIs with persistent caching.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import requests


def extract_dois_from_references(
    dataset_description: str,
    openneuro_doi: Optional[str] = None
) -> List[str]:
    """
    Extract paper DOIs from dataset_description References field.

    Handles multiple DOI formats:
    - Full URLs: https://doi.org/10.1038/sdata.2015.1
    - doi: prefix: doi:10.1038/sdata.2015.1
    - Bare DOIs: 10.1038/sdata.2015.1

    Filters out OpenNeuro dataset DOIs (format: doi:10.18112/openneuro.ds*)

    Args:
        dataset_description: Full dataset_description text from metadata
        openneuro_doi: Dataset's own DOI to exclude (e.g., "doi:10.18112/openneuro.ds004844.v1.0.0")

    Returns:
        List of normalized DOIs (format: "10.XXXX/..."), excluding dataset DOI
        Empty list if no paper DOIs found
    """
    if not dataset_description:
        return []

    # Find References line
    references_line = ""
    for line in dataset_description.split('\n'):
        if line.startswith("References:"):
            references_line = line
            break

    if not references_line:
        return []

    # Extract text after "References:" marker
    ref_text = references_line.replace("References:", "").strip()

    if not ref_text:
        return []

    # DOI extraction patterns (order matters - try URL format first)
    patterns = [
        r'https?://doi\.org/(10\.\d+/[^\s,;]+)',     # doi.org URL format
        r'https?://dx\.doi\.org/(10\.\d+/[^\s,;]+)', # dx.doi.org URL format
        r'doi:\s*(10\.\d+/[^\s,;]+)',                # doi: prefix format
        r'\b(10\.\d+/[^\s,;]+)\b',                   # Bare DOI format
    ]

    found_dois = set()

    for pattern in patterns:
        matches = re.findall(pattern, ref_text, re.IGNORECASE)
        for match in matches:
            # Normalize DOI (remove trailing punctuation)
            doi = match.strip().rstrip('.,;')
            found_dois.add(doi)

    # Filter out OpenNeuro dataset DOIs
    paper_dois = []
    for doi in found_dois:
        # Skip OpenNeuro DOIs
        if '10.18112/openneuro.ds' in doi.lower():
            continue
        # Skip if it matches the dataset's own DOI
        if openneuro_doi and doi in openneuro_doi:
            continue
        paper_dois.append(doi)

    return sorted(paper_dois)  # Return sorted for consistency


def _clean_abstract(text: str) -> str:
    """
    Clean abstract text (remove XML tags, normalize whitespace).

    Args:
        text: Raw abstract text

    Returns:
        Cleaned abstract text
    """
    if not text:
        return ""

    # Remove JATS XML tags commonly found in CrossRef abstracts
    text = re.sub(r'<jats:[^>]+>', '', text)
    text = re.sub(r'</jats:[^>]+>', '', text)
    text = re.sub(r'</?p>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Truncate if too long (keep first 2000 chars)
    if len(text) > 2000:
        text = text[:2000] + "..."

    return text


def _fetch_from_crossref(doi: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch abstract from CrossRef API.

    Args:
        doi: Normalized DOI (format: "10.XXXX/...")
        timeout: Request timeout in seconds

    Returns:
        Abstract text or None if not found/error
    """
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "EEGDash-LLM-Tagger/1.0 (mailto:research@eegdash.org)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        # Extract abstract from response
        abstract = data.get('message', {}).get('abstract')

        if abstract:
            return _clean_abstract(abstract)

        return None

    except requests.RequestException:
        return None


def _fetch_from_semantic_scholar(doi: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch abstract from Semantic Scholar API.

    Args:
        doi: Normalized DOI (format: "10.XXXX/...")
        timeout: Request timeout in seconds

    Returns:
        Abstract text or None if not found/error
    """
    # Semantic Scholar uses DOI in URL
    url = f"https://api.semanticscholar.org/v1/paper/{doi}"
    headers = {
        "User-Agent": "EEGDash-LLM-Tagger/1.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        # Extract abstract from response
        abstract = data.get('abstract')

        if abstract:
            return _clean_abstract(abstract)

        return None

    except requests.RequestException:
        return None


def _fetch_from_pubmed(doi: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch abstract from PubMed API using DOI.

    Uses PubMed E-utilities API:
    1. ESearch to find PMID from DOI
    2. EFetch to get abstract from PMID

    Args:
        doi: Normalized DOI (format: "10.XXXX/...")
        timeout: Request timeout in seconds

    Returns:
        Abstract text or None if not found/error
    """
    try:
        # Step 1: Search for PMID using DOI
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"{doi}[DOI]",
            "retmode": "json"
        }

        response = requests.get(search_url, params=search_params, timeout=timeout)

        if response.status_code != 200:
            return None

        search_data = response.json()
        pmid_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmid_list:
            return None

        pmid = pmid_list[0]

        # Step 2: Fetch abstract using PMID
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract"
        }

        response = requests.get(fetch_url, params=fetch_params, timeout=timeout)

        if response.status_code != 200:
            return None

        # Parse XML to extract abstract
        # Look for <AbstractText> tags (can be single or multiple for structured abstracts)
        xml_text = response.text
        abstract_matches = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml_text, re.DOTALL)

        if abstract_matches:
            # Combine all abstract sections (for structured abstracts)
            abstract = ' '.join(abstract_matches)
            return _clean_abstract(abstract)

        return None

    except requests.RequestException:
        return None


def fetch_abstract(doi: str, timeout: int = 30, verbose: bool = False) -> Optional[str]:
    """
    Fetch paper abstract for a given DOI.

    Tries multiple sources in order:
    1. CrossRef API (best for general academic papers)
    2. Semantic Scholar API (good for CS/AI papers)
    3. PubMed API (best for biomedical/IEEE papers)

    Args:
        doi: Normalized DOI (format: "10.XXXX/...")
        timeout: Request timeout in seconds (default: 30)
        verbose: Print debug information

    Returns:
        Abstract text as string, or None if:
        - DOI not found in any API
        - All APIs fail
        - No abstract available in any source
    """
    if not doi:
        return None

    # Try CrossRef first
    if verbose:
        print(f"      Trying CrossRef API...", file=sys.stderr)

    abstract = _fetch_from_crossref(doi, timeout)

    if abstract:
        if verbose:
            print(f"      ✓ Found in CrossRef", file=sys.stderr)
        return abstract

    # Fallback to Semantic Scholar
    if verbose:
        print(f"      Trying Semantic Scholar API...", file=sys.stderr)

    abstract = _fetch_from_semantic_scholar(doi, timeout)

    if abstract:
        if verbose:
            print(f"      ✓ Found in Semantic Scholar", file=sys.stderr)
        return abstract

    # Fallback to PubMed
    if verbose:
        print(f"      Trying PubMed API...", file=sys.stderr)

    abstract = _fetch_from_pubmed(doi, timeout)

    if abstract:
        if verbose:
            print(f"      ✓ Found in PubMed", file=sys.stderr)
        return abstract

    if verbose:
        print(f"      ✗ Abstract not found in any API", file=sys.stderr)

    return None


def _load_cache(cache_path: Path) -> dict:
    """
    Load cache from JSON file.

    Args:
        cache_path: Path to cache file

    Returns:
        Cache dictionary or empty dict if file doesn't exist/is corrupted
    """
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Corrupted cache - start fresh
        return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """
    Save cache to JSON file.

    Args:
        cache: Cache dictionary
        cache_path: Path to cache file
    """
    try:
        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically (write to temp file, then rename)
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(cache_path)

    except IOError:
        # Cache save failed - continue without caching
        pass


def fetch_abstract_with_cache(
    doi: str,
    cache_path: Path = Path("data/processed/abstract_cache.json"),
    timeout: int = 30,
    verbose: bool = False
) -> str:
    """
    Fetch abstract with persistent caching.

    Checks cache first. If not found, fetches from APIs and updates cache.
    Cache stores both successful fetches (abstract text) and failures (empty string).

    Args:
        doi: Normalized DOI
        cache_path: Path to cache JSON file
        timeout: API request timeout
        verbose: Print debug info

    Returns:
        Abstract text, or empty string "" if:
        - DOI not found
        - No abstract available
        - API fetch failed

    Never returns None - always returns a string (possibly empty).
    """
    if not doi:
        return ""

    # Load cache
    cache = _load_cache(cache_path)

    # Check cache
    if doi in cache:
        if verbose:
            print(f"      ℹ Cache hit for DOI: {doi}", file=sys.stderr)
        return cache[doi]

    # Fetch from APIs
    if verbose:
        print(f"      Fetching abstract for DOI: {doi}", file=sys.stderr)

    abstract = fetch_abstract(doi, timeout, verbose)

    # Convert None to empty string
    result = abstract if abstract else ""

    # Update cache
    cache[doi] = result
    _save_cache(cache, cache_path)

    return result
