from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from artifact_4.experiment_helpers import (
    LOG_DIR,
    PREDICT_BATCH_SIZE,
    RANDOM_STATE,
    SOURCE_ARTIFACT_DIR,
    SURROGATE_DIR,
    Logger,
    batched_predict,
    class_ratio,
    cleanup_memory,
    ensure_output_dirs,
    load_processed_splits,
    load_teacher_model,
    save_json,
)


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare artifact_4 real-train teacher-labeled datasets.")
    parser.add_argument("--processed-dataset", type=Path, default=SOURCE_ARTIFACT_DIR / "processed_dataset_with_split.csv")
    parser.add_argument("--variant-name", default="real_train_only_92602")
    parser.add_argument("--teachers", nargs="+", default=list(SUPPORTED_TEACHERS), choices=list(SUPPORTED_TEACHERS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = Logger(LOG_DIR / f"artifact4_prepare_real_train_{timestamp}.log")
    log = logger.log

    train_df, test_df, feature_cols = load_processed_splits(args.processed_dataset)
    X_train = train_df[feature_cols].astype("float32")
    y_train_true = train_df["label"].to_numpy(dtype=int, copy=True)

    summary = {
        "experiment": "artifact_4_prepare_real_train_teacher_datasets",
        "config": {
            "variant_name": args.variant_name,
            "processed_dataset": str(args.processed_dataset.resolve()),
            "random_state": RANDOM_STATE,
        },
        "real_train_split": {
            "rows": int(len(X_train)),
            "feature_count": int(len(feature_cols)),
            "true_label_class_ratio": class_ratio(y_train_true),
        },
        "real_test_split": {"rows": int(len(test_df))},
        "teachers": {},
    }

    log(f"Loaded real train split: rows={len(X_train)}, features={len(feature_cols)}, test_rows={len(test_df)}")

    for teacher_name in args.teachers:
        log(f"Pseudo-labeling real train split with teacher: {teacher_name}")
        teacher_model = load_teacher_model(teacher_name)
        teacher_labels = batched_predict(teacher_model, X_train, batch_size=PREDICT_BATCH_SIZE)
        output_df = X_train.copy()
        output_df["label"] = teacher_labels
        output_path = SURROGATE_DIR / f"surrogate_training_{args.variant_name}_{teacher_name}.csv"
        output_df.to_csv(output_path, index=False, float_format="%.6g")
        ratios = class_ratio(teacher_labels)
        train_truth_agreement = float((teacher_labels == y_train_true).mean())
        summary["teachers"][teacher_name] = {
            "output_path": str(output_path.resolve()),
            "teacher_class_ratio": ratios,
            "agreement_with_true_train_label": train_truth_agreement,
        }
        log(
            f"Saved {teacher_name} pseudo-label data: {output_path} | "
            f"class_0_ratio={ratios['class_0_ratio']:.4f}, class_1_ratio={ratios['class_1_ratio']:.4f}, "
            f"train_truth_agreement={train_truth_agreement:.4f}"
        )
        cleanup_memory(teacher_model, teacher_labels, output_df)

    summary_path = SURROGATE_DIR / "artifact4_real_train_teacher_datasets_summary.json"
    save_json(summary, summary_path)
    log(f"Saved dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
