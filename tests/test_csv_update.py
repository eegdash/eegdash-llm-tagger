"""CSV writes preserve textual cells and support empty columns on pandas 3."""
import csv
import json

from eegdash_tagger.utils.csv_updater import load_llm_predictions, update_csv_with_predictions


def test_empty_csv_columns_accept_labels_and_preserve_existing_values(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("dataset,Type Subject,modality of exp,type of exp,subject_count\nfixture,Healthy,,,0007\n")
    original = path.read_bytes()
    predictions = {"fixture":{"pathology":"Epilepsy","modality":"Visual","type":"Memory",
                               "confidence":{"pathology":0.9,"modality":0.8,"type":0.2}}}
    update_csv_with_predictions(path,predictions,dry_run=True)
    assert path.read_bytes() == original
    update_csv_with_predictions(path,predictions)
    with path.open() as handle:
        result = next(csv.DictReader(handle))
    assert result == {"dataset":"fixture","Type Subject":"Healthy","modality of exp":"Visual",
                      "type of exp":"","subject_count":"0007"}


def test_csv_loader_does_not_use_explicit_failure_records(tmp_path):
    path = tmp_path / "failed.json"
    path.write_text(json.dumps({"results":[{"dataset_id":"fixture","status":"error","pathology":["Unknown"],
                                           "confidence":{"pathology":1}}]}))
    assert load_llm_predictions(path) == {}
