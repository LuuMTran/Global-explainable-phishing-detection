from __future__ import annotations

import gc
import json
import pickle
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


SOURCE_ARTIFACT_DIR = Path("artifacts")
OUTPUT_ARTIFACT_DIR = Path("artifact_4")
SURROGATE_DIR = OUTPUT_ARTIFACT_DIR / "surrogates"
LOG_DIR = SURROGATE_DIR / "logs"
PREPROCESSOR_PATH = SOURCE_ARTIFACT_DIR / "preprocessor.pkl"

RANDOM_STATE = 42
PREDICT_BATCH_SIZE = 50_000


class Logger:
    def __init__(self, log_path: Path):
        self.start_time = time.time()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:8.1f}s] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def ensure_output_dirs() -> None:
    OUTPUT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SURROGATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cleanup_memory(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def class_ratio(labels: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    return {
        "class_0_count": int(counts[0]),
        "class_1_count": int(counts[1]),
        "class_0_ratio": float(counts[0] / total),
        "class_1_ratio": float(counts[1] / total),
    }


def batched_predict(model, X, batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
    preds = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = X.iloc[start:stop] if hasattr(X, "iloc") else X[start:stop]
        preds.append(np.asarray(model.predict(batch)).astype(int))
    return np.concatenate(preds, axis=0)


def fit_with_log_capture(model, X_train, y_train, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        tee_stdout = TeeStream(__import__("sys").stdout, handle)
        tee_stderr = TeeStream(__import__("sys").stderr, handle)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            model.fit(X_train, y_train)


def load_processed_splits(processed_dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    processed = pd.read_csv(processed_dataset)
    train_df = processed[processed["split"] == "train"].copy()
    test_df = processed[processed["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Expected both train and test splits in {processed_dataset}")
    feature_cols = [col for col in processed.columns if col not in {"label", "split"}]
    return train_df, test_df, feature_cols


def load_teacher_model(teacher_name: str):
    model_path = SOURCE_ARTIFACT_DIR / f"{teacher_name}.pkl"
    if not model_path.exists():
        model_path = Path(f"{teacher_name}.pkl")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def load_or_create_teacher_test_cache(
    teacher_name: str,
    X_test: pd.DataFrame,
    logger: Logger,
) -> tuple[np.ndarray, str]:
    cache_path = SURROGATE_DIR / f"teacher_real_test_predictions_{teacher_name}.npy"
    if cache_path.exists():
        logger.log(f"Loading cached real-test teacher predictions: {cache_path}")
        return np.load(cache_path), f"cache:{cache_path.resolve()}"

    for candidate in (
        Path("artifact_3") / "surrogates" / f"teacher_real_test_predictions_{teacher_name}.npy",
        Path("artifact_2") / "surrogates" / f"teacher_real_test_predictions_{teacher_name}.npy",
        Path("artifacts") / "surrogates" / f"teacher_real_test_predictions_{teacher_name}.npy",
        Path("model") / f"teacher_real_test_predictions_{teacher_name}.npy",
    ):
        if candidate.exists():
            logger.log(f"Reusing teacher prediction cache: {candidate}")
            preds = np.load(candidate)
            np.save(cache_path, preds)
            return preds, f"cache:{cache_path.resolve()}"

    logger.log(f"Computing real-test teacher predictions for {teacher_name}")
    teacher_model = load_teacher_model(teacher_name)
    teacher_pred = batched_predict(teacher_model, X_test)
    np.save(cache_path, teacher_pred)
    logger.log(f"Saved real-test teacher predictions: {cache_path}")
    cleanup_memory(teacher_model)
    return teacher_pred, f"cache:{cache_path.resolve()}"


def build_error_subset_report(
    subset_name: str,
    reference_pred: np.ndarray,
    peer_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {
            "subset_name": subset_name,
            "row_count": 0,
            "share_of_test_set": 0.0,
            "fidelity_to_reference": None,
            "peer_vs_true_label": None,
            "peer_matches_reference_count": 0,
            "peer_matches_reference_rate": None,
            "peer_matches_true_label_count": 0,
            "peer_matches_true_label_rate": None,
        }

    reference_subset = reference_pred[mask]
    peer_subset = peer_pred[mask]
    y_subset = y_true[mask]
    peer_matches_reference = peer_subset == reference_subset
    peer_matches_truth = peer_subset == y_subset

    return {
        "subset_name": subset_name,
        "row_count": count,
        "share_of_test_set": float(count / len(y_true)),
        "fidelity_to_reference": compute_metrics(reference_subset, peer_subset),
        "peer_vs_true_label": compute_metrics(y_subset, peer_subset),
        "peer_matches_reference_count": int(peer_matches_reference.sum()),
        "peer_matches_reference_rate": float(np.mean(peer_matches_reference)),
        "peer_matches_true_label_count": int(peer_matches_truth.sum()),
        "peer_matches_true_label_rate": float(np.mean(peer_matches_truth)),
    }


def build_teacher_error_report(teacher_pred: np.ndarray, peer_pred: np.ndarray, y_true: np.ndarray) -> dict:
    teacher_error_mask = teacher_pred != y_true
    teacher_fp_mask = (teacher_pred == 1) & (y_true == 0)
    teacher_fn_mask = (teacher_pred == 0) & (y_true == 1)
    return {
        "teacher_misclassified_count": int(teacher_error_mask.sum()),
        "teacher_misclassified_rate": float(np.mean(teacher_error_mask)),
        "misclassified_fidelity": build_error_subset_report(
            subset_name="teacher_misclassified",
            reference_pred=teacher_pred,
            peer_pred=peer_pred,
            y_true=y_true,
            mask=teacher_error_mask,
        ),
        "teacher_false_positive_subset": build_error_subset_report(
            subset_name="teacher_false_positives",
            reference_pred=teacher_pred,
            peer_pred=peer_pred,
            y_true=y_true,
            mask=teacher_fp_mask,
        ),
        "teacher_false_negative_subset": build_error_subset_report(
            subset_name="teacher_false_negatives",
            reference_pred=teacher_pred,
            peer_pred=peer_pred,
            y_true=y_true,
            mask=teacher_fn_mask,
        ),
    }
