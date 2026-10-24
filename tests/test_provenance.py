"""New writer provenance describes inputs, not generated reasoning."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import requests
from eegdash_tagger.tagging import OpenRouterTagger
from eegdash_tagger.tagging.provenance import canonical_hash, verified_result_provenance

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tagged(monkeypatch):
    tagger = OpenRouterTagger(api_key="unused", model="requested-model")
    output = {"pathology":["Healthy"],"modality":["Visual"],"type":["Perception"],
              "confidence":{"pathology":0.6,"modality":0.8,"type":0.8},"reasoning":{"metadata_analysis":"explanation"}}
    monkeypatch.setattr(tagger, "_call_api", lambda *args: {"model":"provider-returned-model", "id":"fixture-id",
        "choices":[{"message":{"content":json.dumps(output)}}]})
    return tagger.tag_with_details({"title":"Original metadata", "readme":"Actual input", "dataset_id":"excluded"}, dataset_id="fixture")


def test_input_identity_is_independent_of_generated_reasoning(tagged):
    provenance = verified_result_provenance(tagged)
    assert provenance["metadata_hash"] == canonical_hash({"title":"Original metadata", "readme":"Actual input"})
    changed = copy.deepcopy(tagged)
    changed["reasoning"] = {"metadata_analysis":"different generated explanation"}
    assert verified_result_provenance(changed) == provenance
    changed["input_metadata"]["readme"] = "different input"
    with pytest.raises(ValueError, match="Input metadata"):
        verified_result_provenance(changed)


def test_reported_model_and_version_are_not_inferred_from_writer_config(tagged):
    provenance = verified_result_provenance(tagged)
    assert provenance["model"] == "requested-model"
    assert provenance["provider_response_model"] == "provider-returned-model"
    assert provenance["vocabulary_version"] == "2"
    assert provenance["library_version"] == "0.1.1"
    assert all(len(provenance[key]) == 64 for key in ("prompt_hash", "examples_hash", "vocabulary_hash", "request_hash", "config_hash"))
    with pytest.raises(ValueError, match="Requested model"):
        verified_result_provenance(tagged, expected_model="different-model")


@pytest.mark.parametrize("name", ["push_tags_to_api", "push_tags_to_mongo"])
def test_both_writers_preserve_request_provenance_and_refuse_legacy(name, tagged, monkeypatch, tmp_path):
    writer = script(name)
    assert writer.build_tagger_meta(tagged) == tagged["tagger_meta"]
    with pytest.raises(ValueError, match="Missing request-time provenance"):
        writer.build_tagger_meta({"reasoning":{"metadata_analysis":"not input"}})
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"results":[{"dataset_id":"legacy","confidence":{"pathology":0.9}}]}))
    monkeypatch.setattr(sys, "argv", [name, "--llm-json", str(legacy), "--dry-run"])
    assert writer.main() == 1


@pytest.mark.parametrize("name", ["push_tags_to_api", "push_tags_to_mongo"])
def test_writers_validate_dry_run_without_database_client(name, tagged, monkeypatch, tmp_path):
    writer = script(name)
    path = tmp_path / "new.json"
    path.write_text(json.dumps({"results":[tagged]}))
    monkeypatch.setattr(sys, "argv", [name, "--llm-json", str(path), "--dry-run"])
    assert writer.main() == 0


def test_failed_batch_call_is_not_a_successful_unknown(monkeypatch):
    batch = script("tag_with_llm")
    tagger = OpenRouterTagger(api_key="unused")
    monkeypatch.setattr(tagger, "_call_api", lambda *args: (_ for _ in ()).throw(requests.Timeout()))
    result = batch.tag_single_dataset(tagger, {"dataset_id":"fixture", "metadata":{"title":"fixture"}}, 1, 1, False)
    assert result["status"] == "error"
    assert "pathology" not in result and "confidence" not in result


def test_provenance_uses_the_actual_transport_payload(monkeypatch):
    tagger = OpenRouterTagger(api_key="never-recorded", model="before-call")
    payload = {"pathology":["Unknown"],"modality":["Unknown"],"type":["Unknown"],
               "confidence":{"pathology":0,"modality":0,"type":0}}
    sent = {}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":json.dumps(payload)}}]}
    def fake_post(url, **kwargs):
        sent.update(url=url, payload=copy.deepcopy(kwargs["json"]))
        tagger.model = "changed-after-request"
        return Response()
    monkeypatch.setattr(requests, "post", fake_post)
    result = tagger.tag_with_details({"title":"fixture"})
    assert result["tagger_meta"]["request_hash"] == canonical_hash(sent["payload"])
    assert result["model"] == result["tagger_meta"]["model"] == "before-call"
    assert "never-recorded" not in json.dumps(result)


def test_mongo_configuration_has_no_embedded_credential(monkeypatch, tmp_path, tagged):
    writer = script("push_tags_to_mongo")
    path = tmp_path / "new.json"
    path.write_text(json.dumps({"results":[tagged]}))
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setattr(sys, "argv", ["writer", "--llm-json", str(path)])
    assert writer.main() == 1  # No configuration; never imports/connects a database client.
    for setting in ("environment", "argument"):
        monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27018")
        args = ["writer", "--llm-json", str(path), "--dry-run"]
        if setting == "argument": args += ["--mongo-uri", "mongodb://localhost:27019"]
        monkeypatch.setattr(sys, "argv", args)
        assert writer.main() == 0
    source = (ROOT / "scripts/push_tags_to_mongo.py").read_text()
    import re
    assert not re.search(r'mongodb(?:\+srv)?://[^\s"\']+:[^\s"\']+@', source)


class WriterResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")
    def json(self):
        return self.body


@pytest.mark.parametrize("body,expected", [
    ({"success":True,"updated_count":1,"inserted_count":0}, "ok"),
    ({"success":True,"updated_count":0,"inserted_count":0}, "ok"),
    ({"success":False,"updated_count":0,"inserted_count":0}, "error"),
    ({"success":True,"updated_count":0,"inserted_count":1}, "error"),
    ({"success":True,"updated_count":"1","inserted_count":0}, "error"),
    ({"success":True,"updated_count":True,"inserted_count":0}, "error"),
    ({"success":True,"updated_count":-1,"inserted_count":0}, "error"),
    ({"success":True,"updated_count":1}, "error"),
    ({"message":"rejected"}, "error"),
    ([], "error"),
])
def test_api_writer_uses_documented_http_protocol(body, expected, tagged, monkeypatch):
    writer = script("push_tags_to_api")
    calls = []
    def get(url, **kwargs):
        calls.append(("GET",url,kwargs))
        return WriterResponse({"success":True,"data":{"dataset_id":"fixture"}})
    def post(url, **kwargs):
        calls.append(("POST",url,kwargs))
        return WriterResponse(body)
    monkeypatch.setattr(writer.requests,"get",get)
    monkeypatch.setattr(writer.requests,"post",post)
    result = writer.push_one(tagged,api_url="https://fixture.invalid",database="test-db",token="test-token")
    assert result[0] == expected
    assert [c[1] for c in calls] == ["https://fixture.invalid/api/test-db/datasets/fixture", "https://fixture.invalid/admin/test-db/datasets"]
    assert calls[1][2]["headers"] == {"Authorization":"Bearer test-token"}
    assert calls[1][2]["json"] == {"dataset_id":"fixture", "tags":writer.build_tags(tagged), "tagger_meta":tagged["tagger_meta"]}


@pytest.mark.parametrize("body,status", [({"success":True,"data":None},200), ({"success":False},200), ({},404)])
def test_api_writer_does_not_write_when_existence_is_unconfirmed(body, status, tagged, monkeypatch):
    writer = script("push_tags_to_api")
    monkeypatch.setattr(writer.requests,"get",lambda *a, **kw: WriterResponse(body,status))
    monkeypatch.setattr(writer.requests,"post",lambda *a, **kw: pytest.fail("Unexpected write"))
    assert writer.push_one(tagged,api_url="https://fixture.invalid",database="test",token="test")[0] == "error"


def test_api_writer_reports_failed_write_exit_status(tagged, monkeypatch, tmp_path):
    writer = script("push_tags_to_api")
    path = tmp_path / "new.json"
    path.write_text(json.dumps({"results":[tagged]}))
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN","test-token")
    monkeypatch.setattr(sys,"argv",["writer","--llm-json",str(path),"--api-url","https://fixture.invalid"])
    monkeypatch.setattr(writer.requests,"get",lambda *a, **kw: WriterResponse({"success":True,"data":{"dataset_id":"fixture"}}))
    for success in (True,False):
        monkeypatch.setattr(writer.requests,"post",lambda *a, **kw: WriterResponse({"success":success,"updated_count":int(success),"inserted_count":0}))
        assert writer.main() == (0 if success else 1)


def test_api_writer_requires_explicit_token_before_requests(tagged, monkeypatch, tmp_path):
    writer = script("push_tags_to_api")
    path = tmp_path / "new.json"
    path.write_text(json.dumps({"results":[tagged]}))
    monkeypatch.delenv("EEGDASH_ADMIN_TOKEN",raising=False)
    monkeypatch.setattr(sys,"argv",["writer","--llm-json",str(path)])
    monkeypatch.setattr(writer.requests,"get",lambda *a, **kw: pytest.fail("Unexpected request"))
    assert writer.main() == 1


def test_request_provenance_does_not_disclose_endpoint_credentials(tagged, monkeypatch):
    tagger = OpenRouterTagger(api_key="synthetic-secret")
    tagger.endpoint = "https://fixture-user:fixture-password@provider.invalid:8443/v1/chat/completions?token=fixture-query"
    output = {"pathology":["Unknown"],"modality":["Unknown"],"type":["Unknown"],
              "confidence":{"pathology":0,"modality":0,"type":0}}
    monkeypatch.setattr(tagger,"_call_api",lambda *a: {"choices":[{"message":{"content":json.dumps(output)}}]})
    record = tagger.tag_with_details({"title":"Synthetic fixture"})
    assert record["tagger_meta"]["provider_origin"] == "https://provider.invalid:8443"
    assert record["tagger_meta"]["endpoint_hash"] == canonical_hash(tagger.endpoint)
    serialized = json.dumps(record)
    assert all(secret not in serialized for secret in ("synthetic-secret","fixture-user","fixture-password","fixture-query"))


@pytest.mark.parametrize("field,value", [("schema_version",2),("request_hash","not-a-digest"),("library_version",None)])
def test_writer_rejects_incomplete_or_invalid_provenance_identity(tagged, field, value):
    tagged["tagger_meta"][field] = value
    with pytest.raises(ValueError):
        verified_result_provenance(tagged)
