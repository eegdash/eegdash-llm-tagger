"""Failures must not become successful abstention records."""
import json
from pathlib import Path

import pytest
import requests

from eegdash_tagger.tagging import OpenRouterTagger, TaggingError


def record(**updates):
    result = {
        "pathology": ["Healthy"], "modality": ["Visual"], "type": ["Perception"],
        "confidence": {"pathology": 0.6, "modality": 0.8, "type": 0.8},
        "reasoning": {"metadata_analysis": "Synthetic evidence", "evidence_alignment_check": ["source"]},
    }
    result.update(updates)
    return result


def response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}]}


@pytest.fixture
def tagger():
    return OpenRouterTagger(api_key="unused")


@pytest.mark.parametrize("method", ["tag", "tag_with_details"])
@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.HTTPError("HTTP failure"), requests.ConnectionError("offline")])
def test_request_failures_raise_typed_error(tagger, monkeypatch, method, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(tagger, "_call_api", fail)
    with pytest.raises(TaggingError) as caught:
        getattr(tagger, method)({"title": "fixture"})
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("bad", [{}, {"choices": []}, {"choices": [{"message": {"content": "not json"}}]},
    response([]), response({}), response(record(pathology=4)), response(record(type=[None])),
    response(record(confidence={})), response(record(confidence={"pathology":float("nan"), "modality":0.2,"type":0.2})),
    response(record(confidence={"pathology":True, "modality":0.2,"type":0.2})),
    response(record(confidence={"pathology":1.1, "modality":0.2,"type":0.2})), response(record(reasoning=[]))])
@pytest.mark.parametrize("method", ["tag", "tag_with_details"])
def test_invalid_model_records_are_failures(tagger, monkeypatch, bad, method):
    monkeypatch.setattr(tagger, "_call_api", lambda *args: bad)
    with pytest.raises(TaggingError):
        getattr(tagger, method)({"title": "fixture"})


@pytest.mark.parametrize("method", ["tag", "tag_with_details"])
def test_explicit_unknown_is_success(tagger, monkeypatch, method):
    value = record(pathology=["Unknown"], modality=["Unknown"], type=["Unknown"],
        confidence={"pathology":0.0,"modality":0.0,"type":0.0})
    monkeypatch.setattr(tagger, "_call_api", lambda *args: response(value))
    result = getattr(tagger, method)({"title": "fixture"})
    assert (result["pathology"] if isinstance(result, dict) else result.pathology) == ["Unknown"]


def test_fences_and_structured_reasoning_are_preserved(tagger, monkeypatch):
    value = record()
    monkeypatch.setattr(tagger, "_call_api", lambda *args: {"choices": [{"message": {"content": "```json\n" + json.dumps(value) + "\n```"}}]})
    details = tagger.tag_with_details({"title": "fixture"})
    assert details["reasoning"] == value["reasoning"]
    assert tagger.tag({"title": "fixture"})["pathology"] == ["Healthy"]


def test_missing_prompt_does_not_silently_change_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        OpenRouterTagger(api_key="unused", prompt_path=tmp_path / "missing.md")


def test_endpoint_is_selected_when_instance_is_created(monkeypatch):
    monkeypatch.setenv("LLM_API_BASE", "http://localhost:9999/v1/")
    tagger = OpenRouterTagger(api_key="unused")
    assert tagger.endpoint == "http://localhost:9999/v1/chat/completions"


def test_default_resources_are_the_canonical_build_inputs(tagger):
    root = Path(__file__).resolve().parents[1]
    assert tagger._build_system_prompt() == (root / "prompt.md").read_text()
    examples = json.loads((root / "data/processed/few_shot_examples.json").read_text())["few_shot_examples"]
    assert len(tagger.few_shot_examples) == len(examples) == 15


def test_non_json_constant_in_reasoning_is_a_parse_failure(monkeypatch):
    tagger = OpenRouterTagger(api_key="unused")
    content = '{"pathology":["Unknown"],"modality":["Unknown"],"type":["Unknown"],"confidence":{"pathology":0,"modality":0,"type":0},"reasoning":{"unsupported":NaN}}'
    monkeypatch.setattr(tagger,"_call_api",lambda *a: {"choices":[{"message":{"content":content}}]})
    with pytest.raises(TaggingError,match="Non-JSON numeric"):
        tagger.tag_with_details({"title":"Synthetic"})
