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

from artifact_2.experiment_helpers import (
    LOG_DIR,
    PREDICT_BATCH_SIZE,
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
    load_teacher_model,
    save_json,
)


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")
SUPPORTED_MODES = ("local_only", "local_plus_vae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run artifact_2 EBM surrogate variants.")
    parser.add_argument(
        "--variant-name",
        required=True,
        help="Output variant suffix, for example local_only or local_plus_small_vae.",
    )
    parser.add_argument(
        "--dataset-mode",
        required=True,
        choices=list(SUPPORTED_MODES),
        help="Synthetic source mode to use for pseudo-label training data.",
    )
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
        "--local-path",
        type=Path,
        default=SOURCE_ARTIFACT_DIR / "synthetic_local_permutation_300k.csv",
        help="Local permutation synthetic dataset.",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=SOURCE_ARTIFACT_DIR / "synthetic_vae_ld20_warm10_temp0p85_filtered_300k.csv",
        help="Optional VAE synthetic dataset used when dataset-mode=local_plus_vae.",
    )
    parser.add_argument(
        "--vae-sample-size",
        type=int,
        default=25_000,
        help="VAE sample size used when dataset-mode=local_plus_vae.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the synthetic training rows after the source mix is assembled.",
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


def load_synthetic_training_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict]:
    local_df = pd.read_csv(args.local_path).astype(np.float32)
    feature_cols = local_df.columns.tolist()
    if args.dataset_mode == "local_only":
        return local_df, feature_cols, {
            "dataset_mode": args.dataset_mode,
            "local_path": str(args.local_path.resolve()),
            "local_rows": int(len(local_df)),
            "vae_path": None,
            "vae_total_rows": 0,
            "vae_sample_rows": 0,
            "combined_rows": int(len(local_df)),
            "feature_count": int(len(feature_cols)),
        }

    vae_df = pd.read_csv(args.vae_path).astype(np.float32)
    vae_sample = vae_df.sample(n=args.vae_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    combined_df = pd.concat([local_df, vae_sample], axis=0, ignore_index=True)
    return combined_df, feature_cols, {
        "dataset_mode": args.dataset_mode,
        "local_path": str(args.local_path.resolve()),
        "local_rows": int(len(local_df)),
        "vae_path": str(args.vae_path.resolve()),
        "vae_total_rows": int(len(vae_df)),
        "vae_sample_rows": int(len(vae_sample)),
        "combined_rows": int(len(combined_df)),
        "feature_count": int(len(feature_cols)),
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"artifact2_ebm_{args.variant_name}_{timestamp}.log"
    logger = Logger(LOG_DIR / log_name)
    log = logger.log

    log(f"Starting artifact_2 EBM surrogate variant: {args.variant_name}")
    log(f"Processed dataset: {args.processed_dataset.resolve()}")
    log(f"Dataset mode: {args.dataset_mode}")
    log(f"Local dataset: {args.local_path.resolve()}")
    if args.dataset_mode == "local_plus_vae":
        log(f"VAE dataset: {args.vae_path.resolve()}")
        log(f"VAE sample size: {args.vae_sample_size}")
    if args.max_rows is not None:
        log(f"Max synthetic rows: {args.max_rows}")

    combined_df, feature_cols, mix_info = load_synthetic_training_data(args)
    if args.max_rows is not None and args.max_rows < len(combined_df):
        combined_df = combined_df.sample(n=args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        mix_info["sampled_combined_rows"] = int(len(combined_df))
    else:
        mix_info["sampled_combined_rows"] = int(len(combined_df))
    mix_info["requested_max_rows"] = None if args.max_rows is None else int(args.max_rows)
    log(
        f"Synthetic training data ready: total={mix_info['combined_rows']}, "
        f"sampled_total={mix_info['sampled_combined_rows']}, "
        f"local={mix_info['local_rows']}, vae_sample={mix_info['vae_sample_rows']}, "
        f"features={mix_info['feature_count']}"
    )

    _train_df, test_df, processed_feature_cols = load_processed_splits(args.processed_dataset)
    if feature_cols != processed_feature_cols:
        raise ValueError("Synthetic feature schema does not match the processed dataset feature schema.")

    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["label"].to_numpy(dtype=int, copy=True)
    log(f"Processed test split ready: rows={len(X_test)}")

    summary_path = SURROGATE_DIR / f"ebm_artifact2_{args.variant_name}_summary.json"
    summary = {
        "experiment": f"artifact_2_ebm_{args.variant_name}",
        "config": {
            "variant_name": args.variant_name,
            "dataset_mode": args.dataset_mode,
            "random_state": RANDOM_STATE,
            "ebm_interactions": int(args.ebm_interactions),
            "ebm_max_interaction_bins": int(args.ebm_max_interaction_bins),
            "top_n": int(args.top_n),
            "top_n_interactions": int(args.top_n_interactions),
            "vae_sample_size": int(args.vae_sample_size if args.dataset_mode == 'local_plus_vae' else 0),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
        },
        "data_mix": mix_info,
        "teachers": {},
    }

    for teacher_name in args.teachers:
        log(f"Preparing pseudo-label dataset for teacher: {teacher_name}")
        teacher_dataset_path = SURROGATE_DIR / f"surrogate_training_{args.variant_name}_{teacher_name}.csv"
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
        save_json(summary, summary_path)
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

    save_json(summary, summary_path)
    log(f"artifact_2 EBM surrogate variant completed: {args.variant_name}")


if __name__ == "__main__":
    main()
