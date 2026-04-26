from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cloudpickle
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split

from artifact_4.experiment_helpers import (
    LOG_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SOURCE_ARTIFACT_DIR,
    SURROGATE_DIR,
    Logger,
    batched_predict,
    build_teacher_error_report,
    class_ratio,
    cleanup_memory,
    compute_metrics,
    ensure_output_dirs,
    fit_with_log_capture,
    load_or_create_teacher_test_cache,
    load_processed_splits,
    save_json,
)


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train artifact_4 EBM surrogates on real-train teacher labels.")
    parser.add_argument("--variant-name", default="real_train_only_92602")
    parser.add_argument("--teachers", nargs="+", default=list(SUPPORTED_TEACHERS), choices=list(SUPPORTED_TEACHERS))
    parser.add_argument("--processed-dataset", type=Path, default=SOURCE_ARTIFACT_DIR / "processed_dataset_with_split.csv")
    parser.add_argument("--ebm-interactions", type=int, default=25)
    parser.add_argument("--ebm-max-interaction-bins", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-n-interactions", type=int, default=5)
    return parser.parse_args()


def build_ebm(feature_names: list[str], interactions: int, max_interaction_bins: int) -> ExplainableBoostingClassifier:
    return ExplainableBoostingClassifier(
        feature_names=feature_names,
        interactions=interactions,
        validation_size=0.15,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.03,
        max_rounds=5000,
        early_stopping_rounds=100,
        max_bins=256,
        max_interaction_bins=max_interaction_bins,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        cloudpickle.dump(obj, handle)


def run_ebm_plot(model_path: Path, output_dir: Path, top_n: int, top_n_interactions: int, logger: Logger) -> None:
    cmd = [
        sys.executable,
        "plot_ebm_rf.py",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--top-n",
        str(top_n),
        "--top-n-interactions",
        str(top_n_interactions),
        "--preprocessor-path",
        str(PREPROCESSOR_PATH),
    ]
    logger.log(f"Running EBM plot command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = Logger(LOG_DIR / f"artifact4_ebm_{args.variant_name}_{timestamp}.log")
    log = logger.log

    log(f"Starting artifact_4 EBM surrogate variant: {args.variant_name}")
    train_df, test_df, feature_cols = load_processed_splits(args.processed_dataset)
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["label"].to_numpy(dtype=int, copy=True)
    log(f"Processed splits ready: train_rows={len(train_df)}, test_rows={len(X_test)}, features={len(feature_cols)}")

    summary_path = SURROGATE_DIR / f"ebm_artifact4_{args.variant_name}_summary.json"
    summary = {
        "experiment": f"artifact_4_ebm_{args.variant_name}",
        "config": {
            "variant_name": args.variant_name,
            "training_source": "processed_dataset train split pseudo-labeled by teacher",
            "random_state": RANDOM_STATE,
            "ebm_interactions": int(args.ebm_interactions),
            "ebm_max_interaction_bins": int(args.ebm_max_interaction_bins),
            "top_n": int(args.top_n),
            "top_n_interactions": int(args.top_n_interactions),
        },
        "data": {
            "processed_dataset": str(args.processed_dataset.resolve()),
            "real_train_rows": int(len(train_df)),
            "real_test_rows": int(len(test_df)),
            "feature_count": int(len(feature_cols)),
        },
        "teachers": {},
    }

    for teacher_name in args.teachers:
        teacher_dataset_path = SURROGATE_DIR / f"surrogate_training_{args.variant_name}_{teacher_name}.csv"
        if not teacher_dataset_path.exists():
            raise FileNotFoundError(
                f"Missing teacher pseudo-label dataset for {teacher_name}: {teacher_dataset_path}. "
                "Run artifact_4/prepare_real_train_teacher_datasets_artifact4.py first."
            )

        log(f"Loading teacher pseudo-label dataset: {teacher_dataset_path}")
        teacher_dataset = pd.read_csv(teacher_dataset_path).astype(np.float32)
        teacher_labels = teacher_dataset["label"].to_numpy(dtype=int, copy=True)
        teacher_ratios = class_ratio(teacher_labels)
        teacher_test_pred, teacher_prediction_source = load_or_create_teacher_test_cache(teacher_name, X_test, logger)

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            teacher_dataset[feature_cols],
            teacher_labels,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=teacher_labels,
        )
        log(
            f"Training EBM for {teacher_name}: train_rows={len(X_train)}, holdout_rows={len(X_holdout)}, "
            f"class_0_ratio={teacher_ratios['class_0_ratio']:.4f}, class_1_ratio={teacher_ratios['class_1_ratio']:.4f}"
        )
        model = build_ebm(feature_cols, args.ebm_interactions, args.ebm_max_interaction_bins)
        fit_with_log_capture(model, X_train, y_train, logger.log_path)

        holdout_pred = batched_predict(model, X_holdout)
        test_pred = batched_predict(model, X_test)
        model_path = SURROGATE_DIR / f"ebm_{args.variant_name}_{teacher_name}.pkl"
        plot_output_dir = SURROGATE_DIR / f"ebm_plots_{args.variant_name}_{teacher_name}"
        save_pickle(model, model_path)
        run_ebm_plot(model_path, plot_output_dir, args.top_n, args.top_n_interactions, logger)

        teacher_error_report = build_teacher_error_report(teacher_test_pred, test_pred, y_test)
        result = {
            "teacher": teacher_name,
            "teacher_dataset_path": str(teacher_dataset_path.resolve()),
            "teacher_prediction_source": teacher_prediction_source,
            "teacher_class_ratio": teacher_ratios,
            "rows": {
                "teacher_dataset_total": int(len(teacher_dataset)),
                "holdout": int(len(X_holdout)),
                "real_test": int(len(X_test)),
            },
            "holdout_fidelity": compute_metrics(y_holdout, holdout_pred),
            "real_test_fidelity_to_teacher": compute_metrics(teacher_test_pred, test_pred),
            "real_test_accuracy_to_true_label": compute_metrics(y_test, test_pred),
            "error_fidelity": teacher_error_report,
            "artifacts": {"model_path": str(model_path.resolve()), "plot_dir": str(plot_output_dir.resolve())},
        }
        summary["teachers"][teacher_name] = result
        save_json(summary, summary_path)
        log(
            f"Finished EBM for {teacher_name} | holdout_acc={result['holdout_fidelity']['accuracy']:.4f}, "
            f"test_teacher_fidelity_acc={result['real_test_fidelity_to_teacher']['accuracy']:.4f}, "
            f"test_true_acc={result['real_test_accuracy_to_true_label']['accuracy']:.4f}, "
            f"error_fidelity_acc={teacher_error_report['misclassified_fidelity']['fidelity_to_reference']['accuracy']:.4f}"
        )

        cleanup_memory(teacher_dataset, teacher_labels, X_train, X_holdout, y_train, y_holdout, holdout_pred, test_pred, model)

    save_json(summary, summary_path)
    log(f"artifact_4 EBM surrogate variant completed: {args.variant_name}")


if __name__ == "__main__":
    main()
