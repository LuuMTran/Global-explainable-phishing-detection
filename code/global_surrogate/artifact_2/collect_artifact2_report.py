from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from artifact_2.experiment_helpers import OUTPUT_ARTIFACT_DIR, SURROGATE_DIR, save_json


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(surrogate_name: str, teacher_name: str, payload: dict) -> dict:
    error = payload["error_fidelity"]
    mis = error["misclassified_fidelity"]
    fp = error["teacher_false_positive_subset"]
    fn = error["teacher_false_negative_subset"]
    return {
        "surrogate": surrogate_name,
        "teacher": teacher_name,
        "holdout_accuracy": payload["synthetic_holdout_fidelity"]["accuracy"],
        "holdout_precision": payload["synthetic_holdout_fidelity"]["precision"],
        "holdout_recall": payload["synthetic_holdout_fidelity"]["recall"],
        "holdout_f1": payload["synthetic_holdout_fidelity"]["f1"],
        "test_teacher_fidelity_accuracy": payload["real_test_fidelity_to_teacher"]["accuracy"],
        "test_teacher_fidelity_precision": payload["real_test_fidelity_to_teacher"]["precision"],
        "test_teacher_fidelity_recall": payload["real_test_fidelity_to_teacher"]["recall"],
        "test_teacher_fidelity_f1": payload["real_test_fidelity_to_teacher"]["f1"],
        "test_true_accuracy": payload["real_test_accuracy_to_true_label"]["accuracy"],
        "test_true_precision": payload["real_test_accuracy_to_true_label"]["precision"],
        "test_true_recall": payload["real_test_accuracy_to_true_label"]["recall"],
        "test_true_f1": payload["real_test_accuracy_to_true_label"]["f1"],
        "teacher_misclassified_count": error["teacher_misclassified_count"],
        "teacher_misclassified_rate": error["teacher_misclassified_rate"],
        "error_fidelity_accuracy": mis["fidelity_to_reference"]["accuracy"],
        "error_match_teacher_rate": mis["peer_matches_reference_rate"],
        "error_match_truth_rate": mis["peer_matches_true_label_rate"],
        "teacher_false_positive_count": fp["row_count"],
        "teacher_false_positive_fidelity": None if fp["fidelity_to_reference"] is None else fp["fidelity_to_reference"]["accuracy"],
        "teacher_false_positive_match_truth_rate": fp["peer_matches_true_label_rate"],
        "teacher_false_negative_count": fn["row_count"],
        "teacher_false_negative_fidelity": None if fn["fidelity_to_reference"] is None else fn["fidelity_to_reference"]["accuracy"],
        "teacher_false_negative_match_truth_rate": fn["peer_matches_true_label_rate"],
        "model_path": payload["artifacts"].get("model_path", payload["artifacts"].get("model_pickle")),
        "plot_dir": payload["artifacts"].get("plot_dir", payload["artifacts"].get("output_dir")),
    }


def main() -> None:
    ebm_summary = load_json(SURROGATE_DIR / "ebm_artifact2_summary.json")
    gaminet_summary = load_json(SURROGATE_DIR / "gaminet_artifact2_summary.json")

    report = {
        "experiment": "artifact_2",
        "config": {
            "ebm": ebm_summary["config"],
            "gaminet": gaminet_summary["config"],
            "data_mix": ebm_summary["data_mix"],
        },
        "surrogates": {
            "ebm": ebm_summary["teachers"],
            "gaminet": gaminet_summary["teachers"],
        },
    }

    rows = []
    for teacher_name, payload in ebm_summary["teachers"].items():
        rows.append(extract_row("ebm", teacher_name, payload))
    for teacher_name, payload in gaminet_summary["teachers"].items():
        rows.append(extract_row("gaminet", teacher_name, payload))

    summary_df = pd.DataFrame(rows).sort_values(["surrogate", "teacher"]).reset_index(drop=True)
    summary_csv_path = OUTPUT_ARTIFACT_DIR / "artifact2_experiment_report.csv"
    summary_json_path = OUTPUT_ARTIFACT_DIR / "artifact2_experiment_report.json"
    summary_df.to_csv(summary_csv_path, index=False)
    save_json(report, summary_json_path)

    print(f"Saved combined JSON report: {summary_json_path.resolve()}")
    print(f"Saved combined CSV report: {summary_csv_path.resolve()}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
