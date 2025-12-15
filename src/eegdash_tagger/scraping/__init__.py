"""Web scraping and dataset collection modules."""

from .scraper import (
    fetch_dataset_table,
    fetch_openneuro_id,
    fetch_eegdash_detail_info,
    fetch_openneuro_metadata,
    build_metadata_for_dataset,
    collect_incomplete_datasets,
    EEGDASH_BASE_URL,
    OPENNEURO_BASE_URL,
    OPENNEURO_GRAPHQL_URL,
    GITHUB_BASE_URL,
)
from .dataset_filters import check_tagging_status, needs_tagging, has_complete_tagging
from .enrichment import enrich_with_metadata

__all__ = [
    "fetch_dataset_table",
    "fetch_openneuro_id",
    "fetch_eegdash_detail_info",
    "fetch_openneuro_metadata",
    "build_metadata_for_dataset",
    "collect_incomplete_datasets",
    "check_tagging_status",
    "needs_tagging",
    "has_complete_tagging",
    "enrich_with_metadata",
    "EEGDASH_BASE_URL",
    "OPENNEURO_BASE_URL",
    "OPENNEURO_GRAPHQL_URL",
    "GITHUB_BASE_URL",
]
