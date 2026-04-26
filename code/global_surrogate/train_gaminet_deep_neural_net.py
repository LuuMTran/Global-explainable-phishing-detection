import argparse
import json
import pickle
import time
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from plot_gaminet import plot_gaminet_artifacts


ARTIFACT_DIR = Path("artifacts")
SURROGATE_DIR = ARTIFACT_DIR / "surrogates"
SURROGATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = SURROGATE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"

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
        if hasattr(X, "iloc"):
            batch = X.iloc[start:stop]
        else:
            batch = X[start:stop]
        preds.append(np.asarray(model.predict(batch)).astype(int))
    return np.concatenate(preds, axis=0)


def load_or_cache_real_teacher_test_labels(
    teacher_name: str,
    feature_cols: list[str],
    X_real_test_df: pd.DataFrame,
    logger: Logger,
) -> np.ndarray:
    cache_path = SURROGATE_DIR / f"teacher_real_test_predictions_{teacher_name}.npy"
    model_path = ARTIFACT_DIR / f"{teacher_name}.pkl"

    if cache_path.exists():
        logger.log(f"Loading cached real-test teacher predictions: {cache_path}")
        return np.load(cache_path)

    logger.log(f"Loading teacher model for real-test fidelity: {model_path}")
    with model_path.open("rb") as handle:
        teacher_model = pickle.load(handle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        teacher_real_test_labels = batched_predict(teacher_model, X_real_test_df)
    np.save(cache_path, teacher_real_test_labels)
    logger.log(f"Saved cached real-test teacher predictions: {cache_path}")
    cleanup_memory(teacher_model)
    return teacher_real_test_labels


def ensure_gaminet_dependencies() -> tuple[object, object, object, object]:
    try:
        import tensorflow as tf
        from gaminet import GAMINet
        from gaminet.utils import plot_regularization
        from gaminet.utils import plot_trajectory
    except ImportError as exc:
        raise RuntimeError(
            "GAMI-Net dependencies are missing in this environment. "
            "Install the official package stack first, e.g. `pip install gaminet tensorflow tensorflow-lattice`, "
            "then rerun this script."
        ) from exc

    return tf, GAMINet, plot_trajectory, plot_regularization


def build_meta_info_and_scale(
    X_train: pd.DataFrame,
    X_other_list: list[pd.DataFrame],
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    feature_names = X_train.columns.tolist()
    train_np = X_train.to_numpy(dtype=np.float32, copy=True)
    other_np_list = [frame.to_numpy(dtype=np.float32, copy=True) for frame in X_other_list]

    train_scaled = np.zeros_like(train_np, dtype=np.float32)
    other_scaled_list = [np.zeros_like(arr, dtype=np.float32) for arr in other_np_list]
    meta_info: dict[str, dict] = {}

    for idx, feature_name in enumerate(feature_names):
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        train_col = train_np[:, [idx]]
        scaler.fit(train_col)
        train_scaled[:, [idx]] = scaler.transform(train_col).astype(np.float32)
        for arr_idx, arr in enumerate(other_np_list):
            other_scaled_list[arr_idx][:, [idx]] = scaler.transform(arr[:, [idx]]).astype(np.float32)
        meta_info[feature_name] = {"type": "continuous", "scaler": scaler}

    meta_info["label"] = {"type": "target"}
    return train_scaled, other_scaled_list, meta_info


def gaminet_predict_labels(model, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    preds = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch_pred = np.asarray(model.predict(X[start:stop]), dtype=np.float32).reshape(-1)
        if np.nanmin(batch_pred) < 0.0 or np.nanmax(batch_pred) > 1.0:
            batch_pred = 1.0 / (1.0 + np.exp(-batch_pred))
        preds.append((batch_pred >= 0.5).astype(np.int32))
    return np.concatenate(preds, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GAMI-Net surrogate on a cached teacher-labeled dataset.")
    parser.add_argument(
        "--teacher",
        choices=["deep_neural_net", "random_forest"],
        default="deep_neural_net",
        help="Teacher model whose cached surrogate-training dataset should be used.",
    )
    parser.add_argument("--interact-num", type=int, default=5)
    parser.add_argument("--main-effect-epochs", type=int, default=50)
    parser.add_argument("--interaction-epochs", type=int, default=50)
    parser.add_argument("--tuning-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-n-interactions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_name = args.teacher

    tf, GAMINet, plot_trajectory, plot_regularization = ensure_gaminet_dependencies()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"gaminet_{teacher_name}_{timestamp}.log"
    logger = Logger(log_path)
    log = logger.log

    log(f"Starting GAMI-Net surrogate training for {teacher_name}")
    log(f"Artifacts directory: {ARTIFACT_DIR.resolve()}")

    gpu_devices = tf.config.list_physical_devices("GPU")
    log(f"TensorFlow version: {tf.__version__}")
    log(f"TensorFlow GPUs visible: {len(gpu_devices)}")

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    teacher_dataset_path = SURROGATE_DIR / f"surrogate_training_{teacher_name}.csv"
    if not teacher_dataset_path.exists():
        raise FileNotFoundError(
            f"Missing cached teacher dataset: {teacher_dataset_path}. "
            f"Generate the {teacher_name} pseudo-label CSV first."
        )

    log(f"Loading cached teacher dataset: {teacher_dataset_path}")
    teacher_dataset = pd.read_csv(teacher_dataset_path).astype(np.float32)
    feature_cols = [col for col in teacher_dataset.columns if col != "label"]
    teacher_labels = teacher_dataset["label"].to_numpy(dtype=np.int32, copy=True)
    teacher_ratios = class_ratio(teacher_labels)
    log(
        f"Pseudo-label ratio: class_0={teacher_ratios['class_0_count']} ({teacher_ratios['class_0_ratio']:.4f}), "
        f"class_1={teacher_ratios['class_1_count']} ({teacher_ratios['class_1_ratio']:.4f})"
    )

    X_train_df, X_holdout_df, y_train, y_holdout = train_test_split(
        teacher_dataset[feature_cols],
        teacher_labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=teacher_labels,
    )
    log(f"Teacher dataset split: train={len(X_train_df)}, holdout={len(X_holdout_df)}, features={len(feature_cols)}")

    log(f"Loading real processed test split and teacher predictions for {teacher_name}")
    processed = pd.read_csv(ARTIFACT_DIR / "processed_dataset_with_split.csv")
    real_test_df = processed[processed["split"] == "test"].copy()
    X_real_test_df = real_test_df[feature_cols].astype(np.float32)
    y_real_test = real_test_df["label"].to_numpy(dtype=np.int32, copy=True)
    cleanup_memory(processed, real_test_df)
    teacher_real_test_labels = load_or_cache_real_teacher_test_labels(teacher_name, feature_cols, X_real_test_df, logger)

    log("Scaling processed features to [0, 1] for GAMI-Net")
    X_train_scaled, other_scaled, meta_info = build_meta_info_and_scale(
        X_train_df,
        [X_holdout_df, X_real_test_df],
    )
    X_holdout_scaled, X_real_test_scaled = other_scaled
    y_train_gami = y_train.astype(np.float32).reshape(-1, 1)

    output_dir = SURROGATE_DIR / f"gaminet_{teacher_name}_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"gaminet_{teacher_name}"

    log("Building GAMI-Net model")
    model = GAMINet(
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

    log("Training GAMI-Net")
    fit_start = time.time()
    model.fit(X_train_scaled, y_train_gami)
    log(f"Training finished in {time.time() - fit_start:.1f}s")

    log("Evaluating GAMI-Net")
    holdout_pred = gaminet_predict_labels(model, X_holdout_scaled)
    real_pred = gaminet_predict_labels(model, X_real_test_scaled)
    metrics_holdout = compute_metrics(y_holdout, holdout_pred)
    metrics_real_teacher = compute_metrics(teacher_real_test_labels, real_pred)
    metrics_real_truth = compute_metrics(y_real_test, real_pred)

    log(
        f"GAMI-Net metrics | holdout_fidelity_acc={metrics_holdout['accuracy']:.4f}, "
        f"real_teacher_fidelity_acc={metrics_real_teacher['accuracy']:.4f}, "
        f"real_true_acc={metrics_real_truth['accuracy']:.4f}"
    )

    log("Saving GAMI-Net model and artifacts")
    model.save(folder=str(output_dir) + "/", name=model_name)

    logs_dict = model.summary_logs(save_dict=False)
    plot_trajectory(logs_dict, folder=str(output_dir) + "/", name="gaminet_training_trajectory", log_scale=True, save_png=True)
    plot_regularization(logs_dict, folder=str(output_dir) + "/", name="gaminet_regularization", log_scale=True, save_png=True)

    log("Generating GAMI-Net plots with the custom plotter")
    plot_summary = plot_gaminet_artifacts(
        model_path=output_dir / f"{model_name}.pickle",
        output_dir=output_dir,
        preprocessor_path=PREPROCESSOR_PATH,
        top_n=args.top_n,
        top_n_interactions=args.top_n_interactions,
    )

    summary = {
        "surrogate": "gaminet",
        "teacher": teacher_name,
        "random_state": RANDOM_STATE,
        "teacher_dataset_path": str(teacher_dataset_path.resolve()),
        "feature_count": int(len(feature_cols)),
        "rows": {
            "train": int(len(X_train_df)),
            "holdout": int(len(X_holdout_df)),
            "real_test": int(len(X_real_test_df)),
        },
        "class_ratio": teacher_ratios,
        "config": {
            "interact_num": args.interact_num,
            "interact_arch": [40, 40, 40, 40, 40],
            "subnet_arch": [40, 40, 40, 40, 40],
            "batch_size": args.batch_size,
            "main_effect_epochs": args.main_effect_epochs,
            "interaction_epochs": args.interaction_epochs,
            "tuning_epochs": args.tuning_epochs,
            "early_stop_thres": [30, 30, 20],
            "val_ratio": 0.15,
        },
        "synthetic_holdout_fidelity": metrics_holdout,
        "real_test_fidelity_to_teacher": metrics_real_teacher,
        "real_test_accuracy_to_true_label": metrics_real_truth,
        "plot_summary": plot_summary,
        "artifacts": {
            "model_pickle": str((output_dir / f"{model_name}.pickle").resolve()),
            "output_dir": str(output_dir.resolve()),
            "log_path": str(log_path.resolve()),
        },
    }

    summary_path = SURROGATE_DIR / f"gaminet_{teacher_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Saved summary JSON: {summary_path}")
    log("GAMI-Net surrogate training script completed")


if __name__ == "__main__":
    main()
