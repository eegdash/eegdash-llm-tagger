"""Taxonomy v2: every label the tagger emits must be in the vocabulary."""

import json
from pathlib import Path

import pytest

from eegdash_tagger.tagging.taxonomy import LABELS, MAX_LABELS, normalize_labels

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("axis", sorted(LABELS))
def test_labels_never_contain_docs_separators(axis):
    # The docs table splits tag cells on these characters.
    for label in LABELS[axis]:
        assert not any(sep in label for sep in "/,;|"), label


def test_no_concept_on_two_axes():
    assert not set(LABELS["modality"]) & set(LABELS["type"]) - {"Other", "Unknown"}


@pytest.mark.parametrize(
    ("axis", "raw", "expected"),
    [
        ("pathology", ["Schizophrenia/Psychosis"], ["Schizophrenia spectrum"]),
        ("pathology", "Parkinson's Disease", ["Parkinson's"]),
        ("pathology", ["Other"], ["Other clinical"]),
        ("pathology", ["MCI"], ["Other clinical"]),
        ("pathology", ["Mild cognitive impairment"], ["Other clinical"]),
        ("pathology", ["Dementia"], ["Dementia"]),
        ("pathology", ["DBS"], ["Unknown"]),
        ("pathology", ["Development"], ["Unknown"]),  # retired: needs re-tagging
        ("modality", ["Resting State"], ["No stimulus"]),
        ("modality", ["visual", "AUDITORY"], ["Visual", "Auditory"]),
        ("type", ["Resting state"], ["Resting-state"]),
        ("type", ["Decision making"], ["Decision-making"]),
        ("type", ["made-up label"], ["Unknown"]),
        ("type", ["Unknown", "Memory"], ["Memory"]),
        ("type", None, ["Unknown"]),
    ],
)
def test_normalize_labels(axis, raw, expected):
    assert normalize_labels(axis, raw) == expected


def test_caps_label_count():
    assert len(normalize_labels("type", LABELS["type"])) == MAX_LABELS


def test_few_shot_examples_use_v2_labels():
    data = json.loads((ROOT / "data/processed/few_shot_examples.json").read_text())
    for ex in data["few_shot_examples"]:
        for axis in LABELS:
            assert normalize_labels(axis, ex[axis]) == ex[axis], (ex["dataset_id"], axis)


def test_prompt_lists_exactly_the_v2_labels():
    prompt = (ROOT / "prompt.md").read_text()
    for axis, labels in LABELS.items():
        assert json.dumps(labels, ensure_ascii=False) in prompt, axis


def test_tag_with_details_validates_labels(monkeypatch):
    from eegdash_tagger.tagging.llm_tagger import OpenRouterTagger

    tagger = OpenRouterTagger(api_key="test")
    raw = '{"pathology": ["Development"], "modality": ["Resting State"], "type": ["Clinical/Intervention", "Memory/Resting state"], "confidence": {"pathology":0.2,"modality":0.8,"type":0.8}}'
    monkeypatch.setattr(
        tagger, "_call_api", lambda *a: {"choices": [{"message": {"content": raw}}]}
    )
    out = tagger.tag_with_details({"title": "t"}, dataset_id="on000001")
    assert out["dataset_id"] == "on000001"
    assert out["pathology"] == ["Unknown"]
    assert out["modality"] == ["No stimulus"]
    assert out["type"] == ["Clinical"]


def test_response_preserves_primary_order_and_distinguishes_mci(monkeypatch):
    """The normalizer must not turn MCI into dementia or reorder two labels."""
    from eegdash_tagger.tagging.llm_tagger import OpenRouterTagger

    tagger = OpenRouterTagger(api_key="unused")
    raw = {"pathology": ["MCI"], "modality": ["Auditory", "Visual"],
           "type": ["Language", "Clinical"],
           "confidence": {"pathology": 0.8, "modality": 0.8, "type": 0.7}}
    monkeypatch.setattr(tagger, "_call_api", lambda *a: {
        "choices": [{"message": {"content": json.dumps(raw)}}]})
    output = tagger.tag_with_details({"title": "Synthetic MCI cohort"})
    assert output["pathology"] == ["Other clinical"]
    assert output["modality"] == ["Auditory", "Visual"]
    assert output["type"] == ["Language", "Clinical"]
