"""
Client for EEGDash LLM Tagger API.

This module provides a Python client for integrating with the
EEGDash LLM Tagger API service.

Example:
    client = EEGDashTaggerClient("http://localhost:8000")
    result = client.tag_dataset("ds001234", "https://github.com/OpenNeuroDatasets/ds001234")
    print(result.pathology)  # ["Healthy"]
"""
import os
from urllib.parse import quote, urlsplit

import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class _KeyScopedSession(requests.Session):
    """requests only strips Authorization on redirects; do the same for X-API-Key."""

    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        old, new = urlsplit(response.request.url), urlsplit(prepared_request.url)
        if (old.hostname, old.scheme) != (new.hostname, new.scheme):
            prepared_request.headers.pop("X-API-Key", None)


@dataclass
class TaggingResult:
    """Result from tagging API."""
    dataset_id: str
    pathology: List[str]
    modality: List[str]
    type: List[str]
    confidence: Dict[str, float]
    reasoning: Dict[str, str]
    from_cache: bool
    stale: bool = False
    error: Optional[str] = None


class EEGDashTaggerClient:
    """
    Client for EEGDash LLM Tagger API.

    This client provides methods to interact with the EEGDash LLM Tagger
    API service for tagging EEG/MEG datasets.

    See ``__init__`` for arguments (``base_url``, ``timeout``, ``api_key``).

    Example:
        >>> client = EEGDashTaggerClient("http://localhost:8000")
        >>> result = client.tag_dataset(
        ...     "ds001234",
        ...     "https://github.com/OpenNeuroDatasets/ds001234"
        ... )
        >>> print(result.pathology)
        ['Healthy']
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 300,
        api_key: Optional[str] = None,
    ):
        """
        Initialize client.

        Args:
            base_url: API base URL (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds (default 5 minutes for LLM calls)
            api_key: X-API-Key for tagging and cache-deletion routes
                (default: ``TAGGER_API_KEY`` environment variable)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # api_key="" explicitly opts out of the environment key.
        key = (api_key if api_key is not None else os.getenv("TAGGER_API_KEY", "")).strip()
        # One session: connections are reused, and the key rides on every
        # call (the Caddy edge in front of the API gates GETs too).
        self._session = _KeyScopedSession()
        if key:
            self._session.headers["X-API-Key"] = key

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.

        Returns:
            Health status dict with status, cache_entries, config_hash

        Raises:
            requests.RequestException: If API is unavailable
        """
        response = self._session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def tag_dataset(
        self,
        dataset_id: str,
        source_url: str,
        force_refresh: bool = False
    ) -> TaggingResult:
        """
        Tag a dataset with LLM classification.

        Args:
            dataset_id: Dataset identifier (e.g., "ds001234")
            source_url: GitHub/OpenNeuro repository URL
            force_refresh: Skip cache and force new LLM call

        Returns:
            TaggingResult with classification labels and metadata

        Raises:
            requests.RequestException: If API call fails
        """
        response = self._session.post(
            f"{self.base_url}/api/v1/tag",
            json={
                "dataset_id": dataset_id,
                "source_url": source_url,
                "force_refresh": force_refresh
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        return TaggingResult(
            dataset_id=data["dataset_id"],
            pathology=data["pathology"],
            modality=data["modality"],
            type=data["type"],
            confidence=data.get("confidence", {}),
            reasoning=data.get("reasoning", {}),
            from_cache=data["from_cache"],
            stale=data.get("stale", False),
            error=data.get("error")
        )

    def get_cached_tags(self, dataset_id: str) -> Optional[TaggingResult]:
        """
        Get cached tags for a dataset (no LLM call).

        Args:
            dataset_id: Dataset identifier

        Returns:
            TaggingResult if cached, None if not found
        """
        response = self._session.get(
            f"{self.base_url}/api/v1/tags/{dataset_id}",
            timeout=10
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        return TaggingResult(
            dataset_id=data["dataset_id"],
            pathology=data["pathology"],
            modality=data["modality"],
            type=data["type"],
            confidence=data.get("confidence", {}),
            reasoning=data.get("reasoning", {}),
            from_cache=True,
            stale=data.get("stale", False)
        )

    def batch_tag_datasets(
        self,
        datasets: List[Dict[str, str]],
        skip_cached: bool = True,
        verbose: bool = False
    ) -> List[TaggingResult]:
        """
        Tag multiple datasets.

        Args:
            datasets: List of dicts with 'dataset_id' and 'source_url'
            skip_cached: If True, skip datasets already cached
            verbose: If True, print progress

        Returns:
            List of TaggingResult for each dataset
        """
        results = []

        for i, ds in enumerate(datasets, 1):
            dataset_id = ds["dataset_id"]
            source_url = ds["source_url"]

            if verbose:
                print(f"[{i}/{len(datasets)}] Processing {dataset_id}...")

            # Check cache first if skip_cached enabled
            if skip_cached:
                cached = self.get_cached_tags(dataset_id)
                if cached:
                    if verbose:
                        print(f"  -> Using cached result")
                    results.append(cached)
                    continue

            # Tag the dataset
            try:
                result = self.tag_dataset(dataset_id, source_url)
                if verbose:
                    status = "cached" if result.from_cache else "fresh"
                    print(f"  -> Tagged ({status}): {result.pathology}")
                results.append(result)
            except requests.HTTPError as e:
                # Auth/config errors hit every dataset the same way: stop
                # instead of recording N "Unknown" results.
                if e.response is not None and e.response.status_code in (401, 403, 503):
                    raise
                if verbose:
                    print(f"  -> Error: {e}")
                results.append(self._error_result(dataset_id, e))
            except Exception as e:
                if verbose:
                    print(f"  -> Error: {e}")
                # Create error result
                results.append(TaggingResult(
                    dataset_id=dataset_id,
                    pathology=["Unknown"],
                    modality=["Unknown"],
                    type=["Unknown"],
                    confidence={},
                    reasoning={},
                    from_cache=False,
                    error=str(e)
                ))

        return results

    @staticmethod
    def _error_result(dataset_id: str, error: Exception) -> "TaggingResult":
        return TaggingResult(
            dataset_id=dataset_id,
            pathology=["Unknown"],
            modality=["Unknown"],
            type=["Unknown"],
            confidence={},
            reasoning={},
            from_cache=False,
            error=str(error),
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with total_entries, config_hash, unique_datasets, datasets
        """
        response = self._session.get(f"{self.base_url}/api/v1/cache/stats", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_cache_entries(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List cache entries, optionally filtered by dataset.

        Args:
            dataset_id: Optional filter by dataset ID

        Returns:
            List of cache entry summaries
        """
        url = f"{self.base_url}/api/v1/cache/entries"
        if dataset_id:
            url += f"?dataset_id={dataset_id}"

        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def clear_cache(self) -> None:
        """Clear entire cache."""
        response = self._session.delete(
            f"{self.base_url}/api/v1/cache", timeout=10
        )
        response.raise_for_status()

    def delete_cache_entry(self, cache_key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            cache_key: Full cache key to delete

        Returns:
            True if deleted, False if not found
        """
        response = self._session.delete(
            f"{self.base_url}/api/v1/cache/{quote(cache_key, safe='')}",
            timeout=10
        )

        if response.status_code == 404:
            return False

        response.raise_for_status()
        return True
