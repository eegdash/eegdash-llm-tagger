"""
BIDS Dataset Metadata Extraction Module

This module provides functionality to extract and summarize metadata from a
local BIDS-style EEG/MEG dataset repository for use in LLM-based tagging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
import json
import pandas as pd


@dataclass
class DatasetSummary:
    """Summary of a BIDS dataset's metadata."""

    root: Path
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

        return result


# Helper functions

def find_first(root: Path, names: list[str]) -> Path | None:
    """
    Return the first existing file with any of the given names under root.
    First checks root directory, then searches recursively (limited depth).
    """
    # Check root directory first
    for name in names:
        candidate = root / name
        if candidate.exists() and candidate.is_file():
            return candidate

    # Check immediate subdirectories
    for name in names:
        for candidate in root.glob(f"*/{name}"):
            if candidate.is_file():
                return candidate

    return None


def glob_relative(root: Path, pattern: str, max_files: int | None = None) -> list[Path]:
    """
    Glob recursively under root for pattern (e.g. '**/*events.tsv').
    Optionally cap number of files.
    """
    try:
        files = list(root.glob(f"**/{pattern}"))
        if max_files is not None:
            files = files[:max_files]
        return files
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

def parse_dataset_description(path: Path) -> dict[str, Any]:
    """
    Load JSON. Extract relevant fields:
        - Name
        - DatasetType
        - Modality / modalities
        - Authors (optional)
        - HowToAcknowledge (optional)
    Return a dict with at least keys:
        - 'title': str | None
        - 'dataset_type': str | None
        - 'modalities': list[str]
        - 'raw_json': dict
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

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
    except Exception as e:
        # Fail softly
        return {
            "title": None,
            "dataset_type": None,
            "modalities": [],
            "raw_json": {}
        }


def parse_readme(path: Path, max_chars: int = 8000) -> str:
    """
    Read text, normalize newlines, strip leading/trailing whitespace.
    Truncate to max_chars.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Normalize newlines and strip
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.strip()

        return _truncate_text(text, max_chars)
    except Exception:
        return ""


def parse_participants(path: Path, max_unique_per_column: int = 20) -> tuple[dict[str, list[str]], str]:
    """
    Use pandas.read_csv(sep='\t').

    For each column:
      - compute unique non-null values (as strings)
      - sort them
      - keep at most max_unique_per_column values
    Return:
      - columns_map: dict[column_name -> list[str]]
      - summary_str: human-readable text summarizing key columns
        e.g. 'diagnosis: ["control", "patient"]; group: ["HC", "MDD"]'
    """
    try:
        df = pd.read_csv(path, sep='\t', dtype=str, na_values=['n/a', 'N/A', ''])

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

    except Exception as e:
        return {}, None


def parse_tasks(task_files: list[Path]) -> tuple[list[str], list[str], str]:
    """
    For each JSON:
      - load
      - extract fields like 'TaskName', 'TaskDescription', 'CognitiveDescription', 'Instructions'
    Collect:
      - task_names: list of task names
      - task_descriptions: list of short textual descriptions
    Build a short summary string joining them.
    """
    task_names = []
    task_descriptions = []
    summary_parts = []

    for task_file in task_files:
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract task name
            task_name = data.get("TaskName")
            if not task_name:
                # Try to infer from filename: task-<name>_bold.json -> <name>
                stem = task_file.stem
                if "task-" in stem:
                    task_name = stem.split("task-")[1].split("_")[0]

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


def parse_events(event_files: list[Path], max_unique: int = 40) -> tuple[list[str], str]:
    """
    Use pandas.read_csv for each file (sep='\t', low_memory=True, nrows maybe 2000 if large).
    Look for meaningful columns in priority:
       - 'trial_type', 'event_type', 'stim_type', 'condition'
    Aggregate unique values across all files for those columns.
    Deduplicate and sort, cap at max_unique.
    Return:
      - event_labels: list[str] of unique event type / condition values
      - summary_str: short text like 'trial_type: ["standard", "target"]; condition: ["rest", "task"]'
    """
    priority_columns = ['trial_type', 'event_type', 'stim_type', 'condition']
    all_values = {col: set() for col in priority_columns}

    for event_file in event_files:
        try:
            # Read limited rows to avoid memory issues
            df = pd.read_csv(event_file, sep='\t', dtype=str, nrows=2000,
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

def build_dataset_summary(root: Union[str, Path]) -> DatasetSummary:
    """
    Build a DatasetSummary from a local BIDS dataset repository.

    Args:
        root: Path to the local dataset root directory

    Returns:
        DatasetSummary object containing extracted metadata
    """
    root = Path(root).resolve()

    # dataset_id: basename of root folder (e.g. 'ds001785' or 'DS001785')
    dataset_id = root.name

    # dataset_description.json
    dd_path = find_first(root, ["dataset_description.json"])
    dd_info = parse_dataset_description(dd_path) if dd_path else {
        "title": None, "dataset_type": None, "modalities": [], "raw_json": {}
    }

    # README
    readme_path = find_first(root, ["README.md", "README.MD", "README.txt", "README"])
    readme_text = parse_readme(readme_path) if readme_path else None

    # participants
    participants_path = find_first(root, ["participants.tsv"])
    if participants_path:
        participants_columns, participants_summary = parse_participants(participants_path)
    else:
        participants_columns, participants_summary = {}, None

    # tasks
    task_files = glob_relative(root, "task-*.json")
    task_names, task_descriptions, tasks_summary = parse_tasks(task_files) if task_files else ([], [], None)

    # events
    event_files = glob_relative(root, "*events.tsv")
    event_types, events_summary = parse_events(event_files) if event_files else ([], None)

    return DatasetSummary(
        root=root,
        dataset_id=dataset_id,
        title=dd_info["title"],
        dataset_description=_shorten_json_description(dd_info["raw_json"]),
        modalities=dd_info["modalities"],
        dataset_type=dd_info["dataset_type"],
        readme_text=readme_text,
        participants_summary=participants_summary,
        tasks_summary=tasks_summary,
        events_summary=events_summary,
        task_names=task_names,
        task_descriptions=task_descriptions,
        participants_columns=participants_columns,
        event_types=event_types,
    )
