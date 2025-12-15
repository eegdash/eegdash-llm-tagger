#!/usr/bin/env python
"""
File Provider Abstraction for BIDS Metadata Extraction

This module provides an abstraction layer for accessing files from different
sources (local filesystem, GitHub API, etc.) without requiring git cloning.

GitHub Personal Access Token Setup
-----------------------------------

To use GitHubFileProvider, you need a GitHub Personal Access Token (PAT)
with permission to read public repositories.

1. Go to: https://github.com/settings/tokens

2. Choose token type:
   - For "Personal access tokens (classic)":
     * Click "Generate new token (classic)"
     * Give it a descriptive name: "EEGDash metadata fetcher"
     * Set expiration (recommend: 90 days or longer)
     * Under "Select scopes", enable:
       - 'public_repo' (read access to public repositories)
     * You do NOT need write, admin, or other scopes

   - For "Fine-grained personal access tokens":
     * Click "Generate new token"
     * Give it a descriptive name: "EEGDash metadata fetcher"
     * Set expiration and repository access
     * Under "Permissions", grant:
       - Repository permissions → Contents: Read-only
     * You can limit access to specific organizations (e.g., OpenNeuroDatasets)

3. Click "Generate token" and copy the generated token immediately
   (you won't be able to see it again!)

4. Store the token securely:
   - NEVER commit it to git
   - NEVER share it publicly
   - On your machine, set it as an environment variable:

     # Linux/macOS (add to ~/.bashrc or ~/.zshrc):
     export GITHUB_TOKEN="ghp_your_very_long_token_string_here"

     # Windows Command Prompt:
     set GITHUB_TOKEN=ghp_your_very_long_token_string_here

     # Windows PowerShell:
     $env:GITHUB_TOKEN="ghp_your_very_long_token_string_here"

5. Verify it's set:

     # Linux/macOS/Windows:
     echo $GITHUB_TOKEN

6. When running scripts that use GitHub API:
   - The Python code will automatically read: os.environ["GITHUB_TOKEN"]
   - If missing, you'll get a clear error message

Security Notes:
- Treat your token like a password
- If compromised, revoke it immediately at github.com/settings/tokens
- Regenerate tokens periodically for security
"""

from pathlib import Path
from typing import Protocol, List, Optional, Dict, Any
from urllib.parse import urlparse
import json
import os
import sys
import time

import requests


# ============================================================================
# FileProvider Protocol
# ============================================================================

class FileProvider(Protocol):
    """
    Protocol defining the interface for file access abstraction.

    Implementations can provide access to files from various sources
    (local filesystem, GitHub API, etc.) through a unified interface.
    """

    def list_files(self, prefix: str = "") -> List[str]:
        """
        Return a list of file paths (POSIX-style, relative to dataset root).

        Args:
            prefix: Optional prefix filter (e.g., "sub-01/" to list only
                    files in that subdirectory)

        Returns:
            List of relative file paths, e.g.:
            ["dataset_description.json", "README.md", "sub-01/eeg/data.eeg"]
        """
        ...

    def read_text(self, path: str) -> Optional[str]:
        """
        Return the text content of the given path.

        Args:
            path: Relative path to file (POSIX-style)

        Returns:
            File contents as string, or None if file doesn't exist or
            cannot be read as text
        """
        ...


# ============================================================================
# LocalFileProvider Implementation
# ============================================================================

class LocalFileProvider:
    """
    FileProvider implementation for local filesystem access.

    This provider is used for working with locally cloned or downloaded
    dataset repositories.
    """

    def __init__(self, root: Path):
        """
        Initialize provider with a local directory path.

        Args:
            root: Path to dataset root directory
        """
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

    def list_files(self, prefix: str = "") -> List[str]:
        """List all files recursively under root directory."""
        try:
            all_files = []
            for path in self.root.rglob("*"):
                if path.is_file():
                    # Convert to relative POSIX-style path
                    rel_path = path.relative_to(self.root).as_posix()
                    if not prefix or rel_path.startswith(prefix):
                        all_files.append(rel_path)
            return sorted(all_files)
        except Exception:
            return []

    def read_text(self, path: str) -> Optional[str]:
        """Read text file from local filesystem."""
        try:
            file_path = self.root / path
            if not file_path.exists() or not file_path.is_file():
                return None
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return None


# ============================================================================
# GitHubFileProvider Implementation
# ============================================================================

class GitHubFileProvider:
    """
    FileProvider implementation using GitHub REST API.

    This provider fetches files directly from GitHub without cloning,
    using the GitHub Contents API and Git Trees API.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        branch: str = "main",
        base_url: str = "https://api.github.com",
        verbose: bool = False
    ):
        """
        Initialize GitHub file provider.

        Args:
            owner: Repository owner (e.g., "OpenNeuroDatasets")
            repo: Repository name (e.g., "ds001971")
            token: GitHub Personal Access Token
            branch: Branch name (default: "main", fallback to "master" if needed)
            base_url: GitHub API base URL
            verbose: If True, print progress messages to stderr
        """
        self.owner = owner
        self.repo = repo
        self.token = token
        self.branch = branch
        self.base_url = base_url
        self.verbose = verbose

        # Internal caches to avoid repeated API calls
        self._tree_cache: Optional[List[str]] = None
        self._file_cache: Dict[str, str] = {}

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json"
        })

    def _fetch_tree(self) -> List[str]:
        """
        Fetch complete file tree from GitHub using Git Trees API.

        Returns:
            List of file paths (blobs only, no directories)
        """
        if self.verbose:
            print(f"  Fetching file tree from GitHub API...", file=sys.stderr)

        start_time = time.time()

        # Try main branch first, then master if it fails
        for branch_name in [self.branch, "master", "main"]:
            try:
                if self.verbose and branch_name != self.branch:
                    print(f"  Branch '{self.branch}' not found, trying '{branch_name}'...", file=sys.stderr)

                url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/trees/{branch_name}"
                params = {"recursive": "1"}

                response = self._session.get(url, params=params, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    tree = data.get("tree", [])

                    # Filter to only blob entries (files, not directories)
                    file_paths = [
                        item["path"]
                        for item in tree
                        if item.get("type") == "blob"
                    ]

                    # Update branch if we had to use a different one
                    if branch_name != self.branch:
                        self.branch = branch_name

                    elapsed = time.time() - start_time
                    if self.verbose:
                        print(f"  ✓ Found {len(file_paths)} files in repository ({elapsed:.1f} seconds)", file=sys.stderr)

                    return sorted(file_paths)
                elif response.status_code == 404:
                    # Try next branch
                    continue
                else:
                    response.raise_for_status()

            except requests.RequestException as e:
                # Try next branch
                if branch_name == "main":  # Last attempt
                    raise RuntimeError(f"Failed to fetch tree from GitHub: {e}")
                continue

        raise RuntimeError(f"Could not fetch tree from any branch (tried: {self.branch}, master, main)")

    def list_files(self, prefix: str = "") -> List[str]:
        """List all files in the repository."""
        # Fetch tree on first call
        if self._tree_cache is None:
            self._tree_cache = self._fetch_tree()
        elif self.verbose:
            print(f"  Using cached file tree ({len(self._tree_cache)} files)", file=sys.stderr)

        # Filter by prefix if provided
        if prefix:
            return [p for p in self._tree_cache if p.startswith(prefix)]

        return list(self._tree_cache)

    def read_text(self, path: str) -> Optional[str]:
        """
        Read file contents from GitHub using Contents API.

        Uses raw content mode to avoid base64 decoding.
        Caches results to minimize API calls.
        """
        # Check cache first
        if path in self._file_cache:
            if self.verbose:
                print(f"  Reading {path} (cached)", file=sys.stderr)
            return self._file_cache[path]

        try:
            if self.verbose:
                print(f"  Fetching: {path}", file=sys.stderr)

            # Use Contents API with raw accept header
            url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.v3.raw"  # Get raw content, not JSON
            }
            params = {"ref": self.branch}

            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                # Cache and return
                text = response.text
                self._file_cache[path] = text

                if self.verbose:
                    size_kb = len(text) / 1024
                    print(f"  ✓ Fetched {path} ({size_kb:.1f} KB)", file=sys.stderr)

                return text
            elif response.status_code == 404:
                # File not found
                return None
            else:
                # Other error
                return None

        except requests.RequestException:
            # Network error or API failure
            return None


# ============================================================================
# Helper Functions
# ============================================================================

def build_github_provider_from_url(repo_url: str, token: str, verbose: bool = False) -> GitHubFileProvider:
    """
    Parse a GitHub repository URL and create a GitHubFileProvider.

    Args:
        repo_url: GitHub repository URL, e.g.:
                  - https://github.com/OpenNeuroDatasets/ds001971
                  - https://github.com/OpenNeuroDatasets/ds001971.git
        token: GitHub Personal Access Token
        verbose: If True, print progress messages

    Returns:
        GitHubFileProvider instance

    Raises:
        ValueError: If URL format is invalid

    Example:
        >>> provider = build_github_provider_from_url(
        ...     "https://github.com/OpenNeuroDatasets/ds001971",
        ...     token="ghp_...",
        ...     verbose=True
        ... )
    """
    # Parse URL
    parsed = urlparse(repo_url)

    if parsed.netloc != "github.com":
        raise ValueError(f"Not a GitHub URL: {repo_url}")

    # Extract owner and repo from path
    # Path format: /owner/repo or /owner/repo.git
    path_parts = parsed.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")

    owner = path_parts[0]
    repo = path_parts[1]

    # Strip .git suffix if present
    if repo.endswith(".git"):
        repo = repo[:-4]

    return GitHubFileProvider(owner=owner, repo=repo, token=token, verbose=verbose)


def fetch_metadata_from_github_repo(repo_url: str, token: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Fetch and parse BIDS metadata from a GitHub repository without cloning.

    This is the main entry point for remote metadata extraction. It creates
    a GitHubFileProvider and passes it to the metadata parser.

    Args:
        repo_url: GitHub repository URL
        token: GitHub Personal Access Token
        verbose: If True, print progress messages

    Returns:
        Dictionary with parsed metadata (from build_dataset_summary().to_llm_json())

    Raises:
        ValueError: If URL is invalid or token is empty
        RuntimeError: If GitHub API calls fail

    Example:
        >>> metadata = fetch_metadata_from_github_repo(
        ...     "https://github.com/OpenNeuroDatasets/ds001971",
        ...     token=os.environ["GITHUB_TOKEN"],
        ...     verbose=True
        ... )
        >>> print(metadata["title"])
        'Audiocue walking study'
    """
    if not token:
        raise ValueError(
            "GitHub token is required. Please set GITHUB_TOKEN environment variable. "
            "See file_providers.py docstring for setup instructions."
        )

    # Create provider with verbose flag
    provider = build_github_provider_from_url(repo_url, token, verbose=verbose)

    # Extract dataset ID from URL (e.g., "ds001971")
    parsed = urlparse(repo_url)
    dataset_id = parsed.path.strip("/").split("/")[-1]
    if dataset_id.endswith(".git"):
        dataset_id = dataset_id[:-4]

    # Import here to avoid circular dependency (PEP8 allows this)
    from .parser import build_dataset_summary

    # Build metadata using provider
    summary = build_dataset_summary(provider, dataset_id=dataset_id)

    return summary.to_llm_json()


def get_github_token_from_env() -> str:
    """
    Get GitHub token from GITHUB_TOKEN environment variable.

    Returns:
        GitHub token string

    Raises:
        ValueError: If GITHUB_TOKEN is not set
    """
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise ValueError(
            "GITHUB_TOKEN environment variable is not set.\n"
            "\n"
            "To fix this:\n"
            "1. Create a GitHub Personal Access Token at: https://github.com/settings/tokens\n"
            "2. Grant 'public_repo' access (for classic tokens)\n"
            "3. Set the environment variable:\n"
            "   export GITHUB_TOKEN='your_token_here'\n"
            "\n"
            "See file_providers.py for detailed setup instructions."
        )

    return token
