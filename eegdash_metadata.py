"""
BIDS Dataset Metadata Extraction Module

This module provides functionality to extract and summarize metadata from
BIDS-style EEG/MEG datasets using a FileProvider abstraction that works
with both local and remote (GitHub API) data sources.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, Optional, List
from io import StringIO
import json
import pandas as pd
import fnmatch

# Import FileProvider protocol (avoid circular import by importing at function level where needed)
try:
    from file_providers import FileProvider
except ImportError:
    # Fallback for type checking
    from typing import Protocol
    class FileProvider(Protocol):
        def list_files(self, prefix: str = "") -> List[str]: ...
        def read_text(self, path: str) -> Optional[str]: ...


@dataclass
class DatasetSummary:
    """Summary of a BIDS dataset's metadata."""

    dataset_id: str | None

    # High-level metadata
    title: str | None
    dataset_description: str | None  # from dataset_description.json
    modalities: list[str]            # e.g. ["EEG", "MEG"]
    dataset_type: str | None         # e.g. "raw"

    # Text sources (for LLM prompt)
    readme_text: str | None
    participants_summary: str | None
    tasks_summary: str | None
    events_summary: str | None
    extra_docs_text: str | None        # concatenated extra README-like docs from code/, stimuli/
    tasks_detailed_text: str | None    # combined detailed task descriptions

    # Raw structured bits (for debugging / future use)
    task_names: list[str]
    task_descriptions: list[str]
    participants_columns: dict[str, list[str]]  # column → unique values (as strings)
    event_types: list[str]                      # deduped trial_type / event label values

    def to_prompt_dict(self) -> dict[str, str]:
        """
        Return a dict of short-ish strings suitable for feeding into an LLM.
        Long texts should be truncated (e.g. 4000 characters).
        """
        result = {}

        # Basic info
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id
        if self.title:
            result["title"] = self.title
        if self.dataset_type:
            result["dataset_type"] = self.dataset_type

        # Modalities
        if self.modalities:
            result["modalities"] = ", ".join(self.modalities)

        # Dataset description (truncate if needed)
        if self.dataset_description:
            result["dataset_description"] = _truncate_text(self.dataset_description, 2000)

        # README (truncate to reasonable size)
        if self.readme_text:
            result["readme"] = _truncate_text(self.readme_text, 4000)

        # Participants summary
        if self.participants_summary:
            result["participants"] = _truncate_text(self.participants_summary, 1000)

        # Tasks summary
        if self.tasks_summary:
            result["tasks"] = _truncate_text(self.tasks_summary, 2000)
        elif self.task_names:
            result["tasks"] = f"Task names: {', '.join(self.task_names)}"

        # Events summary
        if self.events_summary:
            result["events"] = _truncate_text(self.events_summary, 1000)
        elif self.event_types:
            result["events"] = f"Event types: {', '.join(self.event_types[:50])}"

        return result

    def to_llm_json(self) -> dict:
        """
        Return a JSON-serializable dict optimized for LLM consumption.
        Tasks and events are returned as lists.
        Format:
        {
            "dataset_id": "ds001971",
            "title": "...",
            "recording_modality": "EEG" or "EEG+MRI",
            "dataset_description": "...",
            "readme": "...",
            "participants_overview": "...",
            "tasks": ["task1", "task2", ...],
            "events": ["event1", "event2", ...]
        }
        """
        result = {}

        # Dataset ID
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id

        # Title
        if self.title:
            result["title"] = self.title

        # Recording modality (join multiple modalities with +)
        if self.modalities:
            result["recording_modality"] = "+".join(self.modalities)
        else:
            result["recording_modality"] = None

        # Dataset description (truncate if needed)
        if self.dataset_description:
            result["dataset_description"] = _truncate_text(self.dataset_description, 2000)

        # README (truncate to reasonable size)
        if self.readme_text:
            result["readme"] = _truncate_text(self.readme_text, 4000)

        # Participants overview
        if self.participants_summary:
            result["participants_overview"] = _truncate_text(self.participants_summary, 1000)

        # Tasks as list
        if self.task_names:
            result["tasks"] = self.task_names
        else:
            result["tasks"] = []

        # Events as list
        if self.event_types:
            result["events"] = self.event_types[:50]  # Cap at 50 event types
        else:
            result["events"] = []

        # Task details (extended task information)
        if self.tasks_detailed_text:
            result["task_details"] = _truncate_text(self.tasks_detailed_text, 4000)

        # Extra docs (from code/, stimuli/, etc.)
        if self.extra_docs_text:
            result["extra_docs"] = _truncate_text(self.extra_docs_text, 4000)

        return result


# Helper functions

def find_first_file(provider: FileProvider, names: List[str]) -> Optional[str]:
    """
    Return the first existing file with any of the given names.
    First checks root directory, then immediate subdirectories.

    Args:
        provider: FileProvider instance
        names: List of possible filenames to search for

    Returns:
        Relative path to first found file, or None if not found
    """
    all_files = provider.list_files()

    # Check root directory first (exact filename match)
    for name in names:
        if name in all_files:
            return name

    # Check immediate subdirectories (one level deep: "subdir/filename")
    for name in names:
        for file_path in all_files:
            # Match pattern: exactly one slash, ends with the target name
            if file_path.count("/") == 1 and file_path.endswith("/" + name):
                return file_path

    return None


def glob_files(provider: FileProvider, pattern: str, max_files: Optional[int] = None) -> List[str]:
    """
    Find all files matching the given pattern (e.g., 'task-*.json' or '*events.tsv').

    Args:
        provider: FileProvider instance
        pattern: Glob-style pattern to match against filenames
        max_files: Optional limit on number of files to return

    Returns:
        List of relative paths matching the pattern
    """
    try:
        all_files = provider.list_files()
        matched = []

        for file_path in all_files:
            # Match against the full path
            if fnmatch.fnmatch(file_path, f"**/{pattern}") or fnmatch.fnmatch(file_path, pattern):
                matched.append(file_path)
            # Also try matching just the filename
            elif "/" in file_path:
                filename = file_path.split("/")[-1]
                if fnmatch.fnmatch(filename, pattern):
                    matched.append(file_path)

        if max_files is not None:
            matched = matched[:max_files]

        return sorted(matched)
    except Exception:
        return []


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def _shorten_json_description(raw_json: dict) -> str:
    """
    Convert selected interesting keys from dataset_description.json
    into a compact multi-line string.
    """
    parts = []

    if "Name" in raw_json:
        parts.append(f"Name: {raw_json['Name']}")

    if "Authors" in raw_json:
        authors = raw_json["Authors"]
        if isinstance(authors, list):
            authors_str = ", ".join(str(a) for a in authors[:5])
            if len(authors) > 5:
                authors_str += f" (and {len(authors) - 5} more)"
            parts.append(f"Authors: {authors_str}")
        else:
            parts.append(f"Authors: {authors}")

    if "HowToAcknowledge" in raw_json:
        ack = str(raw_json["HowToAcknowledge"])
        parts.append(f"Acknowledgment: {_truncate_text(ack, 200)}")

    if "ReferencesAndLinks" in raw_json:
        refs = raw_json["ReferencesAndLinks"]
        if isinstance(refs, list):
            refs_str = "; ".join(str(r) for r in refs[:3])
            if len(refs) > 3:
                refs_str += f" (and {len(refs) - 3} more)"
            parts.append(f"References: {refs_str}")
        else:
            parts.append(f"References: {refs}")

    if "DatasetDOI" in raw_json:
        parts.append(f"DOI: {raw_json['DatasetDOI']}")

    result = "\n".join(parts)
    return _truncate_text(result, 1500)


# Parsing functions

def parse_dataset_description(provider: FileProvider, rel_path: str) -> dict[str, Any]:
    """
    Load and parse dataset_description.json.

    Args:
        provider: FileProvider instance
        rel_path: Relative path to dataset_description.json

    Returns:
        Dict with keys:
        - 'title': str | None
        - 'dataset_type': str | None
        - 'modalities': list[str]
        - 'raw_json': dict
    """
    try:
        content = provider.read_text(rel_path)
        if content is None:
            raise FileNotFoundError(f"File not found: {rel_path}")

        data = json.loads(content)

        title = data.get("Name")
        dataset_type = data.get("DatasetType")

        # Handle both singular and plural modality fields
        modalities = []
        if "Modality" in data:
            mod = data["Modality"]
            modalities = [mod] if isinstance(mod, str) else mod
        elif "modalities" in data:
            mod = data["modalities"]
            modalities = [mod] if isinstance(mod, str) else mod

        # Ensure modalities is a list of strings
        modalities = [str(m) for m in modalities] if modalities else []

        return {
            "title": title,
            "dataset_type": dataset_type,
            "modalities": modalities,
            "raw_json": data
        }
    except Exception:
        # Fail softly
        return {
            "title": None,
            "dataset_type": None,
            "modalities": [],
            "raw_json": {}
        }


def parse_readme(provider: FileProvider, rel_path: str, max_chars: int = 8000) -> str:
    """
    Read and normalize README text.

    Args:
        provider: FileProvider instance
        rel_path: Relative path to README file
        max_chars: Maximum characters to return

    Returns:
        Normalized and truncated README text
    """
    try:
        text = provider.read_text(rel_path)
        if text is None:
            return ""

        # Normalize newlines and strip
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.strip()

        return _truncate_text(text, max_chars)
    except Exception:
        return ""


def parse_participants(provider: FileProvider, rel_path: str, max_unique_per_column: int = 20) -> tuple[dict[str, list[str]], str]:
    """
    Parse participants.tsv file.

    Args:
        provider: FileProvider instance
        rel_path: Relative path to participants.tsv
        max_unique_per_column: Maximum unique values to keep per column

    Returns:
        Tuple of (columns_map, summary_str):
        - columns_map: dict[column_name -> list[str]]
        - summary_str: human-readable text summarizing key columns
    """
    try:
        content = provider.read_text(rel_path)
        if content is None:
            return {}, None

        # Parse TSV using pandas
        df = pd.read_csv(StringIO(content), sep='\t', dtype=str, na_values=['n/a', 'N/A', ''])

        columns_map = {}
        summary_parts = []

        # Priority columns for summary
        priority_cols = ['diagnosis', 'group', 'sex', 'gender', 'age', 'site',
                        'condition', 'treatment', 'status']

        for col in df.columns:
            # Skip participant_id column
            if col.lower() in ['participant_id', 'subject_id']:
                continue

            # Get unique non-null values
            unique_vals = df[col].dropna().unique()
            unique_vals = sorted([str(v) for v in unique_vals])

            # Limit number of unique values
            unique_vals = unique_vals[:max_unique_per_column]
            columns_map[col] = unique_vals

            # Add to summary if it's a priority column or has few unique values
            if col.lower() in priority_cols or len(unique_vals) <= 10:
                if unique_vals:
                    vals_str = ", ".join(f'"{v}"' for v in unique_vals[:10])
                    if len(unique_vals) > 10:
                        vals_str += f" (and {len(unique_vals) - 10} more)"
                    summary_parts.append(f"{col}: [{vals_str}]")

        summary_str = "; ".join(summary_parts) if summary_parts else None
        return columns_map, summary_str

    except Exception:
        return {}, None


def parse_tasks(provider: FileProvider, rel_paths: List[str]) -> tuple[list[str], list[str], str]:
    """
    Parse multiple task JSON files.

    Args:
        provider: FileProvider instance
        rel_paths: List of relative paths to task-*.json files

    Returns:
        Tuple of (task_names, task_descriptions, summary_str):
        - task_names: list of task names
        - task_descriptions: list of textual descriptions
        - summary_str: human-readable summary
    """
    task_names = []
    task_descriptions = []
    summary_parts = []

    for rel_path in rel_paths:
        try:
            content = provider.read_text(rel_path)
            if content is None:
                continue

            data = json.loads(content)

            # Extract task name
            task_name = data.get("TaskName")
            if not task_name:
                # Try to infer from filename: task-<name>_bold.json -> <name>
                filename = rel_path.split("/")[-1]
                if "task-" in filename:
                    task_name = filename.split("task-")[1].split("_")[0].split(".")[0]

            if task_name:
                task_names.append(task_name)

            # Extract descriptions
            desc_parts = []
            for key in ["TaskDescription", "CognitiveDescription", "Instructions"]:
                if key in data:
                    val = str(data[key])
                    desc_parts.append(_truncate_text(val, 500))

            if desc_parts:
                desc = " | ".join(desc_parts)
                task_descriptions.append(desc)
                if task_name:
                    summary_parts.append(f"{task_name}: {_truncate_text(desc, 300)}")
            elif task_name:
                summary_parts.append(task_name)

        except Exception:
            # Skip malformed task files
            continue

    summary_str = "\n".join(summary_parts) if summary_parts else None
    return task_names, task_descriptions, summary_str


def parse_events(provider: FileProvider, rel_paths: List[str], max_unique: int = 40) -> tuple[list[str], str]:
    """
    Parse multiple events.tsv files.

    Args:
        provider: FileProvider instance
        rel_paths: List of relative paths to *events.tsv files
        max_unique: Maximum unique event types to keep

    Returns:
        Tuple of (event_labels, summary_str):
        - event_labels: list of unique event type/condition values
        - summary_str: human-readable summary
    """
    priority_columns = ['trial_type', 'event_type', 'stim_type', 'condition']
    all_values = {col: set() for col in priority_columns}

    for rel_path in rel_paths:
        try:
            content = provider.read_text(rel_path)
            if content is None:
                continue

            # Read limited rows to avoid memory issues
            df = pd.read_csv(StringIO(content), sep='\t', dtype=str, nrows=2000,
                           na_values=['n/a', 'N/A', ''])

            for col in priority_columns:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    all_values[col].update(str(v) for v in unique_vals)

        except Exception:
            # Skip malformed event files
            continue

    # Aggregate all unique event labels
    event_labels = set()
    summary_parts = []

    for col in priority_columns:
        if all_values[col]:
            event_labels.update(all_values[col])
            sorted_vals = sorted(all_values[col])[:max_unique]
            vals_str = ", ".join(f'"{v}"' for v in sorted_vals[:20])
            if len(sorted_vals) > 20:
                vals_str += f" (and {len(sorted_vals) - 20} more)"
            summary_parts.append(f"{col}: [{vals_str}]")

    event_labels = sorted(event_labels)[:max_unique]
    summary_str = "; ".join(summary_parts) if summary_parts else None

    return event_labels, summary_str


# Main entry point

def collect_extra_docs(provider: FileProvider, max_file_size: int = 100_000) -> Optional[str]:
    """
    Collect extra documentation files from code/, stimuli/ directories and
    *description*.md/txt files.

    Args:
        provider: FileProvider instance
        max_file_size: Maximum file size in bytes to include (default 100KB)

    Returns:
        Concatenated text with headers, or None if no extra docs found
    """
    all_files = provider.list_files()

    # Patterns to look for
    extra_doc_patterns = [
        "code/README*",
        "code/*.md",
        "code/*.txt",
        "stimuli/README*",
        "stimuli/*.md",
        "stimuli/*.txt",
        "*description*.md",
        "*description*.txt",
    ]

    # Exclude the main README (already processed separately)
    exclude_patterns = ["README.md", "README.MD", "README.txt", "README"]

    extra_docs = []

    for pattern in extra_doc_patterns:
        matches = glob_files(provider, pattern, max_files=20)
        for file_path in matches:
            # Skip if it's the main README
            filename = file_path.split("/")[-1]
            if filename in exclude_patterns:
                continue

            try:
                content = provider.read_text(file_path)
                if content is None:
                    continue

                # Skip if too large
                if len(content) > max_file_size:
                    continue

                # Skip if appears to be binary
                if '\x00' in content[:1000]:
                    continue

                # Add with header
                extra_docs.append(f"=== {file_path} ===\n{_truncate_text(content, 10000)}")

            except Exception:
                continue

    if extra_docs:
        return "\n\n".join(extra_docs)

    return None


def collect_detailed_task_info(provider: FileProvider, task_files: List[str],
                                task_names: List[str], task_descriptions: List[str]) -> Optional[str]:
    """
    Collect detailed task information from task-*.json files.

    Args:
        provider: FileProvider instance
        task_files: List of task JSON file paths
        task_names: List of task names (from parse_tasks)
        task_descriptions: List of task descriptions (from parse_tasks)

    Returns:
        Combined detailed task text, or None if no tasks
    """
    if not task_files:
        return None

    detailed_parts = []

    for i, rel_path in enumerate(task_files):
        try:
            content = provider.read_text(rel_path)
            if content is None:
                continue

            data = json.loads(content)

            # Get task name
            task_name = task_names[i] if i < len(task_names) else "Unknown"

            # Collect all descriptive fields
            desc_fields = []
            for key in ["TaskName", "TaskDescription", "CognitiveDescription",
                       "Instructions", "CogAtlasID", "TaskDesignType"]:
                if key in data:
                    val = str(data[key])
                    if val and len(val) > 5:  # Skip very short values
                        desc_fields.append(f"{key}: {_truncate_text(val, 800)}")

            if desc_fields:
                detailed_parts.append(f"Task '{task_name}':\n" + "\n".join(desc_fields))

        except Exception:
            continue

    if detailed_parts:
        return "\n\n".join(detailed_parts)

    return None


def build_dataset_summary(provider: FileProvider, dataset_id: Optional[str] = None) -> DatasetSummary:
    """
    Build a DatasetSummary from a BIDS dataset using a FileProvider.

    This function works with any FileProvider implementation (local filesystem,
    GitHub API, etc.) to extract metadata without requiring a local clone.

    Args:
        provider: FileProvider instance for accessing dataset files
        dataset_id: Optional dataset identifier (e.g., "ds001971")

    Returns:
        DatasetSummary object containing extracted metadata
    """
    # dataset_description.json
    dd_path = find_first_file(provider, ["dataset_description.json"])
    dd_info = parse_dataset_description(provider, dd_path) if dd_path else {
        "title": None, "dataset_type": None, "modalities": [], "raw_json": {}
    }

    # README
    readme_path = find_first_file(provider, ["README.md", "README.MD", "README.txt", "README"])
    readme_text = parse_readme(provider, readme_path) if readme_path else None

    # participants
    participants_path = find_first_file(provider, ["participants.tsv"])
    if participants_path:
        participants_columns, participants_summary = parse_participants(provider, participants_path)
    else:
        participants_columns, participants_summary = {}, None

    # tasks
    task_files = glob_files(provider, "task-*.json")
    task_names, task_descriptions, tasks_summary = parse_tasks(provider, task_files) if task_files else ([], [], None)

    # events
    event_files = glob_files(provider, "*events.tsv")
    event_types, events_summary = parse_events(provider, event_files) if event_files else ([], None)

    # extra docs (code/, stimuli/, description files)
    extra_docs_text = collect_extra_docs(provider)

    # detailed task information
    tasks_detailed_text = collect_detailed_task_info(provider, task_files, task_names, task_descriptions)

    return DatasetSummary(
        dataset_id=dataset_id,
        title=dd_info["title"],
        dataset_description=_shorten_json_description(dd_info["raw_json"]),
        modalities=dd_info["modalities"],
        dataset_type=dd_info["dataset_type"],
        readme_text=readme_text,
        participants_summary=participants_summary,
        tasks_summary=tasks_summary,
        events_summary=events_summary,
        extra_docs_text=extra_docs_text,
        tasks_detailed_text=tasks_detailed_text,
        task_names=task_names,
        task_descriptions=task_descriptions,
        participants_columns=participants_columns,
        event_types=event_types,
    )


def build_dataset_summary_from_path(root: Union[str, Path]) -> DatasetSummary:
    """
    Build a DatasetSummary from a local BIDS dataset repository.

    This is a backward-compatible wrapper that uses LocalFileProvider internally.

    Args:
        root: Path to the local dataset root directory

    Returns:
        DatasetSummary object containing extracted metadata

    Example:
        >>> summary = build_dataset_summary_from_path("/path/to/ds001971")
        >>> print(summary.to_llm_json())
    """
    # Import LocalFileProvider here to avoid circular dependency
    from file_providers import LocalFileProvider

    root_path = Path(root).resolve()
    dataset_id = root_path.name

    provider = LocalFileProvider(root_path)
    return build_dataset_summary(provider, dataset_id=dataset_id)
