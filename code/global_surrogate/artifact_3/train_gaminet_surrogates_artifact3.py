from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from artifact_3.experiment_helpers import (
    LOG_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SOURCE_ARTIFACT_DIR,
    SURROGATE_DIR,
    Logger,
    build_teacher_error_report,
    class_ratio,
    cleanup_memory,
    compute_metrics,
    ensure_output_dirs,
    load_or_create_teacher_test_cache,
    load_processed_splits,
    save_json,
)
from plot_gaminet import plot_gaminet_artifacts
from train_gaminet_deep_neural_net import build_meta_info_and_scale, ensure_gaminet_dependencies, gaminet_predict_labels


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run artifact_3 GAMI-Net surrogate variants.")
    parser.add_argument(
        "--variant-name",
        required=True,
        help="Variant suffix used to locate teacher pseudo-label CSVs and save outputs.",
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
    parser.add_argument("--interact-num", type=int, default=5)
    parser.add_argument("--main-effect-epochs", type=int, default=50)
    parser.add_argument("--interaction-epochs", type=int, default=50)
    parser.add_argument("--tuning-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-n-interactions", type=int, default=5)
    return parser.parse_args()


def build_gaminet(args: argparse.Namespace, meta_info: dict):
    tf, GAMINet, _, _ = ensure_gaminet_dependencies()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs visible: {len(gpu_devices)}", flush=True)

    return GAMINet(
        meta_info=meta_info,
        interact_num=args.interact_num,
        interact_arch=[40] * 5,
        subnet_arch=[40] * 5,
        batch_size=args.batch_size,
        task_type="Classification",
        activation_func=tf.nn.relu,
        main_effect_epochs=args.main_effect_epochs,
        interaction_epochs=args.interaction_epochs,
        tuning_epochs=args.tuning_epochs,
        lr_bp=[0.0001, 0.0001, 0.0001],
        early_stop_thres=[30, 30, 20],
        heredity=True,
        loss_threshold=0.0,
        reg_clarity=0.1,
        mono_increasing_list=[],
        mono_decreasing_list=[],
        verbose=True,
        val_ratio=0.15,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"artifact3_gaminet_{args.variant_name}_{timestamp}.log"
    logger = Logger(LOG_DIR / log_name)
    log = logger.log

    log(f"Starting artifact_3 GAMI-Net surrogate variant: {args.variant_name}")
    log(f"Processed dataset: {args.processed_dataset.resolve()}")
    log(
        f"Config: interact_num={args.interact_num}, batch_size={args.batch_size}, "
        f"main_effect_epochs={args.main_effect_epochs}, interaction_epochs={args.interaction_epochs}, "
        f"tuning_epochs={args.tuning_epochs}"
    )

    _train_df, test_df, feature_cols = load_processed_splits(args.processed_dataset)
    X_test_df = test_df[feature_cols].astype(np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32, copy=True)
    log(f"Processed test split ready: rows={len(X_test_df)}, features={len(feature_cols)}")

    summary_path = SURROGATE_DIR / f"gaminet_artifact3_{args.variant_name}_summary.json"
    summary = {
        "experiment": f"artifact_3_gaminet_{args.variant_name}",
        "config": {
            "variant_name": args.variant_name,
            "interact_num": int(args.interact_num),
            "batch_size": int(args.batch_size),
            "main_effect_epochs": int(args.main_effect_epochs),
            "interaction_epochs": int(args.interaction_epochs),
            "tuning_epochs": int(args.tuning_epochs),
            "top_n": int(args.top_n),
            "top_n_interactions": int(args.top_n_interactions),
            "random_state": RANDOM_STATE,
        },
        "teachers": {},
    }

    for teacher_name in args.teachers:
        teacher_dataset_path = SURROGATE_DIR / f"surrogate_training_{args.variant_name}_{teacher_name}.csv"
        if not teacher_dataset_path.exists():
            raise FileNotFoundError(
                f"Missing teacher pseudo-label dataset for {teacher_name}: {teacher_dataset_path}. "
                "Run the matching artifact_3 dataset-preparation step first."
            )

        log(f"Loading teacher pseudo-label dataset: {teacher_dataset_path}")
        teacher_dataset = pd.read_csv(teacher_dataset_path).astype(np.float32)
        teacher_labels = teacher_dataset["label"].to_numpy(dtype=np.int32, copy=True)
        teacher_ratios = class_ratio(teacher_labels)
        teacher_test_pred, teacher_prediction_source = load_or_create_teacher_test_cache(teacher_name, X_test_df, logger)

        X_train_df, X_holdout_df, y_train, y_holdout = train_test_split(
            teacher_dataset[feature_cols],
            teacher_labels,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=teacher_labels,
        )
        X_train_scaled, other_scaled, meta_info = build_meta_info_and_scale(X_train_df, [X_holdout_df, X_test_df])
        X_holdout_scaled, X_test_scaled = other_scaled
        y_train_gami = y_train.astype(np.float32).reshape(-1, 1)

        output_dir = SURROGATE_DIR / f"gaminet_{args.variant_name}_{teacher_name}_artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"gaminet_{args.variant_name}_{teacher_name}"

        log(
            f"Training GAMI-Net for {teacher_name}: train_rows={len(X_train_df)}, holdout_rows={len(X_holdout_df)}, "
            f"class_0_ratio={teacher_ratios['class_0_ratio']:.4f}, class_1_ratio={teacher_ratios['class_1_ratio']:.4f}"
        )
        model = build_gaminet(args, meta_info)
        fit_start = time.time()
        model.fit(X_train_scaled, y_train_gami)
        log(f"GAMI-Net training finished in {time.time() - fit_start:.1f}s")

        model.save(folder=str(output_dir) + "/", name=model_name)
        model_pickle_path = output_dir / f"{model_name}.pickle"
        plot_summary = plot_gaminet_artifacts(
            model_path=model_pickle_path,
            output_dir=output_dir,
            preprocessor_path=PREPROCESSOR_PATH,
            top_n=args.top_n,
            top_n_interactions=args.top_n_interactions,
        )

        holdout_pred = gaminet_predict_labels(model, X_holdout_scaled)
        test_pred = gaminet_predict_labels(model, X_test_scaled)
        teacher_error_report = build_teacher_error_report(teacher_test_pred, test_pred, y_test)
        result = {
            "teacher": teacher_name,
            "teacher_dataset_path": str(teacher_dataset_path.resolve()),
            "teacher_prediction_source": teacher_prediction_source,
            "teacher_class_ratio": teacher_ratios,
            "rows": {
                "teacher_dataset_total": int(len(teacher_dataset)),
                "holdout": int(len(X_holdout_df)),
                "real_test": int(len(X_test_df)),
            },
            "synthetic_holdout_fidelity": compute_metrics(y_holdout, holdout_pred),
            "real_test_fidelity_to_teacher": compute_metrics(teacher_test_pred, test_pred),
            "real_test_accuracy_to_true_label": compute_metrics(y_test, test_pred),
            "error_fidelity": teacher_error_report,
            "plot_summary": plot_summary,
            "artifacts": {
                "model_pickle": str(model_pickle_path.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
        }
        summary["teachers"][teacher_name] = result
        save_json(summary, summary_path)
        log(
            f"Finished GAMI-Net for {teacher_name} | holdout_acc={result['synthetic_holdout_fidelity']['accuracy']:.4f}, "
            f"test_teacher_fidelity_acc={result['real_test_fidelity_to_teacher']['accuracy']:.4f}, "
            f"test_true_acc={result['real_test_accuracy_to_true_label']['accuracy']:.4f}, "
            f"error_fidelity_acc={teacher_error_report['misclassified_fidelity']['fidelity_to_reference']['accuracy']:.4f}"
        )

        cleanup_memory(
            X_train_df,
            X_holdout_df,
            X_train_scaled,
            X_holdout_scaled,
            X_test_scaled,
            y_train_gami,
            y_train,
            y_holdout,
            holdout_pred,
            test_pred,
            model,
            teacher_dataset,
            teacher_labels,
        )

    save_json(summary, summary_path)
    log(f"artifact_3 GAMI-Net surrogate variant completed: {args.variant_name}")


if __name__ == "__main__":
    main()
