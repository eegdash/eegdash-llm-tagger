"""Evaluate LLM tagging accuracy against ground truth."""

import json
import pandas as pd
from pathlib import Path


def load_ground_truth(csv_path: str) -> dict:
    """Load ground truth CSV and return dict keyed by dataset ID."""
    df = pd.read_csv(csv_path)
    gt = {}
    for _, row in df.iterrows():
        dataset_id = str(row["dataset"]).upper()
        gt[dataset_id] = {
            "pathology": str(row.get("Type Subject", "")).strip(),
            "modality": str(row.get("modality of exp", "")).strip(),
            "type": str(row.get("type of exp", "")).strip(),
        }
    return gt


def load_llm_output(json_path: str) -> dict:
    """Load LLM output JSON and return dict keyed by dataset ID."""
    with open(json_path) as f:
        data = json.load(f)

    output = {}
    for result in data.get("results", []):
        dataset_id = result["dataset_id"].upper()
        output[dataset_id] = {
            "pathology": result["pathology"][0] if result["pathology"] else "",
            "modality": result["modality"][0] if result["modality"] else "",
            "type": result["type"][0] if result["type"] else "",
        }
    return output


def normalize(value: str) -> str:
    """Normalize label for comparison."""
    return value.lower().strip().replace("-", " ").replace("_", " ")


def compare_labels(pred: str, truth: str) -> bool:
    """Compare predicted vs ground truth label."""
    if not truth or truth.lower() == "nan":
        return None  # Skip if no ground truth
    return normalize(pred) == normalize(truth)


def evaluate(gt_path: str, llm_path: str) -> dict:
    """Evaluate LLM output against ground truth."""
    gt = load_ground_truth(gt_path)
    llm = load_llm_output(llm_path)

    # Find matching datasets
    matched_ids = set(gt.keys()) & set(llm.keys())

    results = {
        "pathology": {"correct": 0, "total": 0},
        "modality": {"correct": 0, "total": 0},
        "type": {"correct": 0, "total": 0},
    }

    details = []

    for dataset_id in sorted(matched_ids):
        gt_labels = gt[dataset_id]
        llm_labels = llm[dataset_id]

        row = {"dataset_id": dataset_id}

        for field in ["pathology", "modality", "type"]:
            match = compare_labels(llm_labels[field], gt_labels[field])
            if match is not None:
                results[field]["total"] += 1
                if match:
                    results[field]["correct"] += 1
                row[f"{field}_pred"] = llm_labels[field]
                row[f"{field}_truth"] = gt_labels[field]
                row[f"{field}_match"] = match

        details.append(row)

    # Calculate accuracies
    accuracies = {}
    total_correct = 0
    total_count = 0

    for field in ["pathology", "modality", "type"]:
        if results[field]["total"] > 0:
            acc = results[field]["correct"] / results[field]["total"]
            accuracies[field] = acc
            total_correct += results[field]["correct"]
            total_count += results[field]["total"]

    accuracies["overall"] = total_correct / total_count if total_count > 0 else 0

    return {
        "accuracies": accuracies,
        "counts": results,
        "matched_datasets": len(matched_ids),
        "details": details,
    }


def main():
    base_path = Path(__file__).parent.parent.parent.parent
    gt_path = base_path / "ground-truth-data" / "dataset_summary.csv"
    llm_path = base_path / "data" / "processed" / "llm_output.json"

    results = evaluate(str(gt_path), str(llm_path))

    print(f"\nMatched datasets: {results['matched_datasets']}")
    print("\n--- Accuracy Results ---")
    print(f"Pathology: {results['accuracies'].get('pathology', 0):.1%} ({results['counts']['pathology']['correct']}/{results['counts']['pathology']['total']})")
    print(f"Modality:  {results['accuracies'].get('modality', 0):.1%} ({results['counts']['modality']['correct']}/{results['counts']['modality']['total']})")
    print(f"Type:      {results['accuracies'].get('type', 0):.1%} ({results['counts']['type']['correct']}/{results['counts']['type']['total']})")
    print(f"Overall:   {results['accuracies'].get('overall', 0):.1%}")

    print("\n--- Per-Dataset Details ---")
    for d in results["details"]:
        print(f"\n{d['dataset_id']}:")
        for field in ["pathology", "modality", "type"]:
            if f"{field}_match" in d:
                match_str = "✓" if d[f"{field}_match"] else "✗"
                print(f"  {field}: {d[f'{field}_pred']} vs {d[f'{field}_truth']} [{match_str}]")


if __name__ == "__main__":
    main()
