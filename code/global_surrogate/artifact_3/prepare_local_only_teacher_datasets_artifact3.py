from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from artifact_3.experiment_helpers import (
    LOG_DIR,
    PREDICT_BATCH_SIZE,
    RANDOM_STATE,
    SOURCE_ARTIFACT_DIR,
    SURROGATE_DIR,
    Logger,
    batched_predict,
    class_ratio,
    ensure_output_dirs,
    load_teacher_model,
    save_json,
)


SUPPORTED_TEACHERS = ("random_forest", "deep_neural_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare artifact_3 local-only teacher-labeled datasets.")
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=list(SUPPORTED_TEACHERS),
        choices=list(SUPPORTED_TEACHERS),
        help="Teacher models to process.",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=SOURCE_ARTIFACT_DIR / "synthetic_local_permutation_300k.csv",
        help="Local permutation synthetic dataset.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=92_602,
        help="Sample size for the capped local-only variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = Logger(LOG_DIR / f"artifact3_prepare_local_only_{timestamp}.log")
    log = logger.log

    local_df = pd.read_csv(args.local_path).astype("float32")
    sampled_df = local_df.sample(n=args.sample_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    variants = {
        "local_only_full_300k": local_df,
        "local_only_92602": sampled_df,
    }

    summary = {
        "experiment": "artifact_3_prepare_local_only_teacher_datasets",
        "config": {
            "random_state": RANDOM_STATE,
            "local_path": str(args.local_path.resolve()),
            "sample_rows": int(args.sample_rows),
        },
        "variants": {},
    }

    for variant_name, feature_df in variants.items():
        log(f"Preparing variant {variant_name}: rows={len(feature_df)}")
        summary["variants"][variant_name] = {
            "rows": int(len(feature_df)),
            "teachers": {},
        }
        for teacher_name in args.teachers:
            teacher_model = load_teacher_model(teacher_name)
            teacher_labels = batched_predict(teacher_model, feature_df, batch_size=PREDICT_BATCH_SIZE)
            output_df = feature_df.copy()
            output_df["label"] = teacher_labels
            output_path = SURROGATE_DIR / f"surrogate_training_{variant_name}_{teacher_name}.csv"
            output_df.to_csv(output_path, index=False, float_format="%.6g")
            ratios = class_ratio(teacher_labels)
            summary["variants"][variant_name]["teachers"][teacher_name] = {
                "output_path": str(output_path.resolve()),
                "class_ratio": ratios,
            }
            log(
                f"Saved {variant_name} pseudo-label dataset for {teacher_name}: {output_path} | "
                f"class_0_ratio={ratios['class_0_ratio']:.4f}, class_1_ratio={ratios['class_1_ratio']:.4f}"
            )

    summary_path = SURROGATE_DIR / "artifact3_local_only_teacher_datasets_summary.json"
    save_json(summary, summary_path)
    log(f"Saved dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
