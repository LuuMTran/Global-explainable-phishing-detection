from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from artifact_4.experiment_helpers import SURROGATE_DIR, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect artifact_4 EBM/GAMI-Net reports into compact tables.")
    parser.add_argument("--variant-name", default="real_train_only_92602")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def row_from_payload(model_name: str, teacher_name: str, payload: dict) -> dict:
    error = payload["error_fidelity"]
    mis = error["misclassified_fidelity"]
    fp = error["teacher_false_positive_subset"]
    fn = error["teacher_false_negative_subset"]
    return {
        "surrogate": model_name,
        "teacher": teacher_name,
        "train_rows": payload["rows"]["teacher_dataset_total"],
        "holdout_rows": payload["rows"]["holdout"],
        "test_rows": payload["rows"]["real_test"],
        "holdout_fidelity_accuracy": payload["holdout_fidelity"]["accuracy"],
        "holdout_fidelity_f1": payload["holdout_fidelity"]["f1"],
        "overall_fidelity_accuracy": payload["real_test_fidelity_to_teacher"]["accuracy"],
        "overall_fidelity_f1": payload["real_test_fidelity_to_teacher"]["f1"],
        "surrogate_test_accuracy": payload["real_test_accuracy_to_true_label"]["accuracy"],
        "surrogate_test_f1": payload["real_test_accuracy_to_true_label"]["f1"],
        "teacher_error_count": error["teacher_misclassified_count"],
        "teacher_error_rate": error["teacher_misclassified_rate"],
        "error_fidelity_accuracy": mis["fidelity_to_reference"]["accuracy"],
        "error_fidelity_f1": mis["fidelity_to_reference"]["f1"],
        "error_match_truth_rate": mis["peer_matches_true_label_rate"],
        "teacher_fp_count": fp["row_count"],
        "fp_error_fidelity": fp["peer_matches_reference_rate"],
        "fp_match_truth_rate": fp["peer_matches_true_label_rate"],
        "teacher_fn_count": fn["row_count"],
        "fn_error_fidelity": fn["peer_matches_reference_rate"],
        "fn_match_truth_rate": fn["peer_matches_true_label_rate"],
    }


def main() -> None:
    args = parse_args()
    rows = []
    sources = {
        "ebm": SURROGATE_DIR / f"ebm_artifact4_{args.variant_name}_summary.json",
        "gaminet": SURROGATE_DIR / f"gaminet_artifact4_{args.variant_name}_summary.json",
    }
    for model_name, path in sources.items():
        if not path.exists():
            continue
        summary = load_json(path)
        for teacher_name, payload in summary["teachers"].items():
            rows.append(row_from_payload(model_name, teacher_name, payload))

    df = pd.DataFrame(rows)
    summary_csv = SURROGATE_DIR / f"artifact4_{args.variant_name}_fidelity_summary.csv"
    summary_json = SURROGATE_DIR / f"artifact4_{args.variant_name}_fidelity_summary.json"
    df.to_csv(summary_csv, index=False)
    save_json({"variant_name": args.variant_name, "rows": rows}, summary_json)
    print(df.to_string(index=False))
    print(f"Saved CSV: {summary_csv.resolve()}")
    print(f"Saved JSON: {summary_json.resolve()}")


if __name__ == "__main__":
    main()
