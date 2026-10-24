"""Request-time provenance for new tag records (not historical reconstruction)."""
from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.parse import urlsplit

from eegdash_tagger import __version__
from .taxonomy import LABELS


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_provenance(payload: dict, endpoint: str, response: dict) -> tuple[dict, dict]:
    """Hash the exact payload and its dataset portion at the time of generation.

    Only the provider origin is recorded in clear text; auth headers and endpoint
    userinfo/query strings are not copied into provenance. A provider-returned
    model string is reported as such, not treated as independent verification.
    """
    user = json.loads(payload["messages"][1]["content"])
    metadata = user["dataset"]
    prompt_hash = hashlib.sha256(payload["messages"][0]["content"].encode("utf-8")).hexdigest()
    examples_hash = canonical_hash(user["few_shot_examples"])
    vocabulary_hash = canonical_hash(LABELS)
    endpoint_hash = canonical_hash(endpoint)
    config = {key: value for key, value in payload.items() if key != "messages"}
    config.update(prompt_hash=prompt_hash, examples_hash=examples_hash,
                  vocabulary_hash=vocabulary_hash, endpoint_hash=endpoint_hash,
                  library_version=__version__)
    url = urlsplit(endpoint)
    origin = f"{url.scheme}://{url.hostname}" + (f":{url.port}" if url.port else "")
    provenance = {
        "schema_version": 1, "hash_algorithm": "sha256-canonical-json-v1",
        "annotation_origin": "machine_generated",
        "review_status_at_generation": "unreviewed",
        "metadata_hash": canonical_hash(metadata),
        "metadata_hash_scope": "filtered dataset field in the actual user message",
        "request_hash": canonical_hash(payload), "config_hash": canonical_hash(config),
        "prompt_hash": prompt_hash, "examples_hash": examples_hash,
        "vocabulary_hash": vocabulary_hash, "vocabulary_version": "2",
        "library_version": __version__, "model": payload["model"],
        "provider_origin": origin, "endpoint_hash": endpoint_hash,
        "tagged_at": datetime.now(timezone.utc).isoformat(),
    }
    for source, dest in (("model", "provider_response_model"), ("id", "provider_response_id")):
        if isinstance(response.get(source), str):
            provenance[dest] = response[source]
    return metadata, provenance


def verified_result_provenance(result: dict, expected_model: str | None = None) -> dict:
    """Require recorded inputs rather than deriving provenance from reasoning."""
    if result.get("status") == "error":
        raise ValueError("Failed model calls cannot be persisted as tag records")
    metadata, provenance = result.get("input_metadata"), result.get("tagger_meta")
    if not isinstance(metadata, dict) or not isinstance(provenance, dict):
        raise ValueError("Missing request-time provenance; legacy outputs must not be relabelled as reproducible new calls")
    hashes = ("metadata_hash", "request_hash", "config_hash", "prompt_hash", "examples_hash", "vocabulary_hash", "endpoint_hash")
    required = (*hashes, "model", "tagged_at", "library_version", "vocabulary_version", "metadata_hash_scope")
    if not all(provenance.get(key) for key in required):
        raise ValueError("Incomplete request-time provenance")
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported provenance schema version")
    if not all(isinstance(provenance[key], str) and re.fullmatch(r"[0-9a-f]{64}", provenance[key]) for key in hashes):
        raise ValueError("Invalid provenance digest")
    if provenance.get("hash_algorithm") != "sha256-canonical-json-v1":
        raise ValueError("Unsupported provenance hash algorithm")
    if canonical_hash(metadata) != provenance["metadata_hash"]:
        raise ValueError("Input metadata does not match its recorded request hash")
    if result.get("model") != provenance["model"] or (expected_model is not None and expected_model != provenance["model"]):
        raise ValueError("Requested model differs from recorded request provenance")
    return dict(provenance)
