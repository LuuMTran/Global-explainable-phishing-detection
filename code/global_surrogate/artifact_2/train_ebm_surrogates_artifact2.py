from __future__ import annotations

import argparse
import pickle
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

from artifact_2.experiment_helpers import (
    LOG_DIR,
    PREDICT_BATCH_SIZE,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SOURCE_ARTIFACT_DIR,
    SURROGATE_DIR,
    VAE_SAMPLE_SIZE,
    Logger,
    batched_predict,
    build_teacher_error_report,
    class_ratio,
    cleanup_memory,
    compute_metrics,
    ensure_output_dirs,
    fit_with_log_capture,
    load_mixed_synthetic_dataset,
    load_processed_splits,
    load_teacher_model,
    load_or_create_teacher_test_cache,
    save_json,
)


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the artifact_2 EBM surrogate experiment.")
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=list(SUPPORTED_TEACHERS),
        choices=list(SUPPORTED_TEACHERS),
        help="Teacher models to process.",
    )
    parser.add_argument(
        "--processed-dataset",
        type=Path,
        default=SOURCE_ARTIFACT_DIR / "processed_dataset_with_split.csv",
        help="Processed dataset CSV containing feature columns plus label and split.",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=SOURCE_ARTIFACT_DIR / "synthetic_vae_ld20_warm10_temp0p85_filtered_300k.csv",
        help="Latest VAE synthetic dataset. A 100k sample is used to preserve the 3:1 mix.",
    )
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
    logger = Logger(LOG_DIR / f"artifact2_ebm_{timestamp}.log")
    log = logger.log

    log("Starting artifact_2 EBM surrogate experiment")
    log(f"Processed dataset: {args.processed_dataset.resolve()}")
    log(f"VAE source dataset: {args.vae_path.resolve()}")
    log(
        f"Mix config: local_permutation=300000, vae_sample={VAE_SAMPLE_SIZE}, "
        f"target_ratio=3:1, ebm_interactions={args.ebm_interactions}, "
        f"ebm_max_interaction_bins={args.ebm_max_interaction_bins}"
    )

    combined_df, feature_cols, mix_info = load_mixed_synthetic_dataset(args.vae_path)
    log(
        f"Mixed synthetic dataset ready: local={mix_info['local_permutation_rows']}, "
        f"vae_sample={mix_info['vae_sample_rows']}, total={mix_info['combined_rows']}, "
        f"features={mix_info['feature_count']}"
    )

    train_df, test_df, processed_feature_cols = load_processed_splits(args.processed_dataset)
    if feature_cols != processed_feature_cols:
        raise ValueError("Synthetic feature schema does not match the processed dataset feature schema.")

    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["label"].to_numpy(dtype=int, copy=True)
    log(f"Processed test split ready: rows={len(X_test)}")

    summary = {
        "experiment": "artifact_2_ebm",
        "config": {
            "random_state": RANDOM_STATE,
            "vae_sample_size": VAE_SAMPLE_SIZE,
            "ebm_interactions": int(args.ebm_interactions),
            "ebm_max_interaction_bins": int(args.ebm_max_interaction_bins),
            "top_n": int(args.top_n),
            "top_n_interactions": int(args.top_n_interactions),
        },
        "data_mix": mix_info,
        "teachers": {},
    }

    for teacher_name in args.teachers:
        log(f"Preparing pseudo-label dataset for teacher: {teacher_name}")
        teacher_dataset_path = SURROGATE_DIR / f"surrogate_training_{teacher_name}.csv"
        teacher_model = load_teacher_model(teacher_name)
        teacher_labels = batched_predict(teacher_model, combined_df, batch_size=PREDICT_BATCH_SIZE)
        teacher_dataset = combined_df.copy()
        teacher_dataset["label"] = teacher_labels
        teacher_dataset.to_csv(teacher_dataset_path, index=False, float_format="%.6g")
        teacher_test_pred, teacher_prediction_source = load_or_create_teacher_test_cache(teacher_name, X_test, logger)
        teacher_ratios = class_ratio(teacher_labels)

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            teacher_dataset[feature_cols],
            teacher_labels,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=teacher_labels,
        )
        log(
            f"Teacher dataset split for {teacher_name}: train={len(X_train)}, holdout={len(X_holdout)}, "
            f"class_0_ratio={teacher_ratios['class_0_ratio']:.4f}, class_1_ratio={teacher_ratios['class_1_ratio']:.4f}"
        )

        model = build_ebm(feature_cols, args.ebm_interactions, args.ebm_max_interaction_bins)
        log(f"Training EBM surrogate for teacher: {teacher_name}")
        fit_with_log_capture(model, X_train, y_train, logger.log_path)

        holdout_pred = batched_predict(model, X_holdout)
        test_pred = batched_predict(model, X_test)
        model_path = SURROGATE_DIR / f"ebm_{teacher_name}.pkl"
        plot_output_dir = SURROGATE_DIR / f"ebm_plots_{teacher_name}"
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
            "synthetic_holdout_fidelity": compute_metrics(y_holdout, holdout_pred),
            "real_test_fidelity_to_teacher": compute_metrics(teacher_test_pred, test_pred),
            "real_test_accuracy_to_true_label": compute_metrics(y_test, test_pred),
            "error_fidelity": teacher_error_report,
            "artifacts": {
                "model_path": str(model_path.resolve()),
                "plot_dir": str(plot_output_dir.resolve()),
            },
        }
        summary["teachers"][teacher_name] = result
        save_json(summary, SURROGATE_DIR / "ebm_artifact2_summary.json")
        log(
            f"Finished EBM for {teacher_name} | holdout_acc={result['synthetic_holdout_fidelity']['accuracy']:.4f}, "
            f"test_teacher_fidelity_acc={result['real_test_fidelity_to_teacher']['accuracy']:.4f}, "
            f"test_true_acc={result['real_test_accuracy_to_true_label']['accuracy']:.4f}, "
            f"error_fidelity_acc={teacher_error_report['misclassified_fidelity']['fidelity_to_reference']['accuracy']:.4f}"
        )

        cleanup_memory(
            teacher_model,
            teacher_dataset,
            teacher_labels,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
            holdout_pred,
            test_pred,
            model,
        )

    save_json(summary, SURROGATE_DIR / "ebm_artifact2_summary.json")
    log("artifact_2 EBM surrogate experiment completed")


if __name__ == "__main__":
    main()
