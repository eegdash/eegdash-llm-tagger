"""Offline parser contract tests on synthetic BIDS metadata."""
from pathlib import Path
import json

from eegdash_tagger.metadata import build_dataset_summary_from_path
from eegdash_tagger.tagging import OpenRouterTagger

FIXTURE = Path(__file__).parent / "fixtures" / "bids_small"


def test_local_bids_metadata_is_extracted_without_recordings():
    summary = build_dataset_summary_from_path(FIXTURE)
    payload = summary.to_llm_json()
    assert summary.title == "Synthetic visual task"
    assert summary.task_names == ["visual"]
    assert summary.event_types == ["face", "house"]
    assert summary.participants_columns["group"] == ["healthy"]
    assert summary.participants_columns["sex"] == ["F", "M"]
    assert "participant_id" not in summary.participants_columns
    assert "Healthy volunteers" in payload["readme"]
    assert "SamplingFrequency" in payload["json_metadata_summary"]
    assert "sub-01/eeg/sub-01_task-visual_eeg.json" in summary.json_metadata_files
    assert "json_metadata_files" not in payload
    transmitted = OpenRouterTagger(api_key="unused")._filter_metadata(payload)
    assert "dataset_id" not in transmitted
    assert "recording_modality" not in transmitted
    assert "json_metadata_summary" not in transmitted
    assert transmitted["tasks"] == ["visual"]
    assert transmitted["events"] == ["face", "house"]


def test_missing_optional_files_do_not_invent_metadata(tmp_path):
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "Minimal fixture"}))
    summary = build_dataset_summary_from_path(tmp_path)
    payload = summary.to_llm_json()
    assert summary.title == "Minimal fixture"
    assert summary.participants_columns == {}
    assert payload["tasks"] == []
    assert payload["events"] == []
    assert "participants_overview" not in payload
    assert "readme" not in payload


def test_bad_optional_json_does_not_hide_valid_task(tmp_path):
    (tmp_path / "dataset_description.json").write_text('{"Name":"mixed fixture"}')
    (tmp_path / "task-invalid_eeg.json").write_text('not JSON')
    (tmp_path / "task-valid_eeg.json").write_text('{"TaskName":"valid"}')
    summary = build_dataset_summary_from_path(tmp_path)
    assert summary.task_names == ["valid"]
    assert "task-invalid_eeg.json" not in summary.json_metadata_files


def test_local_payload_limits_are_applied_before_transmission(tmp_path):
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "X" * 6000}))
    (tmp_path / "README.md").write_text("fixture " * 2000)
    (tmp_path / "task-rest_events.tsv").write_text("onset\tduration\ttrial_type\n" + "".join(f"{i}\t1\tevent_{i:03d}\n" for i in range(65)))
    payload = build_dataset_summary_from_path(tmp_path).to_llm_json()
    assert len(payload["dataset_description"]) <= 1500
    assert len(payload["readme"]) <= 4000
    assert len(payload["events"]) == 40
    assert payload["events"] == [f"event_{i:03d}" for i in range(40)]
