#!/usr/bin/env python3
"""Run after a non-editable wheel install, using that environment's Python.

No model or dataset API calls. CLI write checks affect temporary fixtures only.
"""
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import eegdash_tagger
from eegdash_tagger.tagging import OpenRouterTagger, TaggingError


def main():
    module = Path(eegdash_tagger.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    assert prefix in module.parents and "site-packages" in module.parts, module
    tagger = OpenRouterTagger(api_key="unused")
    prompt = tagger.get_default_prompt_path().resolve()
    examples = tagger.get_default_few_shot_path().resolve()
    assert module.parent in prompt.parents and module.parent in examples.parents
    assert len(tagger.few_shot_examples) == 15
    assert "Unknown" in tagger._build_system_prompt()
    expected = {"pathology":["Unknown"],"modality":["Unknown"],"type":["Unknown"],
                "confidence":{"pathology":0.0,"modality":0.0,"type":0.0},"reasoning":{}}
    tagger._call_api = lambda *args: {"choices":[{"message":{"content":json.dumps(expected)}}]}
    assert tagger.tag_with_details({"title":"Synthetic fixture"})["pathology"] == ["Unknown"]
    tagger._call_api = lambda *args: {"choices":[]}
    try:
        tagger.tag_with_details({"title":"Synthetic fixture"})
    except TaggingError:
        pass
    else:
        raise AssertionError("Malformed response was accepted")
    env = {key:value for key,value in os.environ.items() if key not in ("PYTHONPATH", "LLM_API_KEY", "OPENROUTER_API_KEY")}
    cli = Path(sys.executable).parent
    with tempfile.TemporaryDirectory(prefix="tagger-wheel-smoke-") as folder:
        work = Path(folder)
        for name in ("eegdash-fetch-complete", "eegdash-fetch-incomplete", "eegdash-update-csv"):
            result = subprocess.run([str(cli/name), "--help"], cwd=work, env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0 and "usage:" in result.stdout.lower(), (name, result.stderr)
        csv_path = work / "labels.csv"
        csv_path.write_text("dataset,Type Subject,modality of exp,type of exp\nfixture,Healthy,,\n")
        original = csv_path.read_bytes()
        predictions = work / "results.json"
        predictions.write_text(json.dumps({"results":[{"dataset_id":"fixture", "pathology":["Epilepsy"],
            "modality":["Visual"],"type":["Memory"],"confidence":{"pathology":0.9,"modality":0.8,"type":0.2}}]}))
        command = [str(cli/"eegdash-update-csv"), "--csv", str(csv_path), "--llm-json", str(predictions)]
        dry = subprocess.run(command + ["--dry-run"], cwd=work, env=env, capture_output=True, text=True, timeout=30)
        assert dry.returncode == 0, dry.stderr
        assert csv_path.read_bytes() == original
        update = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=30)
        assert update.returncode == 0, update.stderr
        result = next(csv.DictReader(csv_path.open()))
        assert result == {"dataset":"fixture", "Type Subject":"Healthy", "modality of exp":"Visual", "type of exp":""}
    print(json.dumps({"passed":True,"version":version("eegdash-tagger"),"module":str(module),
        "prompt_sha256":hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "examples_sha256":hashlib.sha256(examples.read_bytes()).hexdigest(),
        "checks":["noneditable package location", "bundled constructor resources", "successful Unknown",
                  "malformed-output failure", "three CLI help commands", "CSV dry-run", "CSV updates preserve existing and low-confidence values"]},indent=2))


if __name__ == "__main__": main()
