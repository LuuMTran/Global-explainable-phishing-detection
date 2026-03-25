import os
os.environ["MPLBACKEND"] = "Agg"

import time
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "processed_dataset_with_split.csv")
DNN_MODEL_PATH = os.path.join(BASE_DIR, "deep_neural_net.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "random_forest.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "xai_results")

DROP_COLUMNS = ["label", "split"]

# Practical sample sizes
DNN_SHAP_N_SAMPLES = 300
RF_SHAP_N_SAMPLES = 300
LIME_N_SAMPLES = 100


TOP_K_LOCAL = 10
LIME_NUM_FEATURES = 10
SPARSITY_THRESHOLD = 0.01
RANDOM_STATE = 42

# Keep False to avoid Tkinter/backend issues
SAVE_PLOTS = False


# ============================================================
# HELPERS
# ============================================================
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(data_path: str, drop_columns=None):
    df = pd.read_csv(data_path)

    if drop_columns is not None:
        cols_to_drop = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df


def safe_numeric_dataframe(X: pd.DataFrame):
    X_num = X.copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(0)
    return X_num


def sample_exact_or_all(X: pd.DataFrame, n_samples: int, random_state: int):
    if len(X) > n_samples:
        return X.sample(n=n_samples, random_state=random_state).copy()
    return X.copy()


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def compute_basic_stats(matrix: np.ndarray, prefix: str = ""):
    flat = matrix.flatten()
    return {
        f"{prefix}mean": float(np.mean(flat)),
        f"{prefix}std": float(np.std(flat)),
        f"{prefix}variance": float(np.var(flat)),
        f"{prefix}min": float(np.min(flat)),
        f"{prefix}max": float(np.max(flat)),
        f"{prefix}median": float(np.median(flat)),
        f"{prefix}q1": float(np.percentile(flat, 25)),
        f"{prefix}q3": float(np.percentile(flat, 75)),
    }


def compute_feature_level_stats(matrix: np.ndarray, feature_names):
    abs_matrix = np.abs(matrix)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean": np.mean(matrix, axis=0),
        "mean_abs": np.mean(abs_matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "variance": np.var(matrix, axis=0),
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "median": np.median(matrix, axis=0),
        "q1": np.percentile(matrix, 25, axis=0),
        "q3": np.percentile(matrix, 75, axis=0),
    }).sort_values("mean_abs", ascending=False)

    return df


def build_local_table_for_row(X_row: pd.Series, values_row: np.ndarray, value_name: str, top_k: int):
    df = pd.DataFrame({
        "feature": X_row.index,
        "feature_value": X_row.values,
        value_name: values_row
    })
    df[f"abs_{value_name}"] = df[value_name].abs()
    return df.sort_values(f"abs_{value_name}", ascending=False).head(top_k)


def get_model_prediction_fn(model):
    if hasattr(model, "predict_proba"):
        return model.predict_proba, "predict_proba"
    elif hasattr(model, "decision_function"):
        return model.decision_function, "decision_function"
    else:
        return model.predict, "predict"


def is_classifier_model(model):
    return hasattr(model, "predict_proba") or hasattr(model, "classes_")


def get_positive_class_predictions(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        return proba.reshape(-1)

    preds = model.predict(X)
    preds = np.asarray(preds)

    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1]
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0]
    return preds.reshape(-1)


def normalize_shap_matrix(shap_matrix: np.ndarray, n_features: int):
    """
    Convert SHAP output to shape (n_samples, n_features).

    Supports:
    - (n_samples, n_features)
    - (n_samples, n_features, n_outputs)
    - (n_samples, n_outputs, n_features)
    """
    shap_matrix = np.asarray(shap_matrix)

    if shap_matrix.ndim == 2:
        return shap_matrix

    if shap_matrix.ndim == 3:
        if shap_matrix.shape[1] == n_features:
            return shap_matrix[:, :, 0]

        if shap_matrix.shape[2] == n_features:
            return shap_matrix[:, 0, :]

    raise ValueError(
        f"Unsupported SHAP matrix shape: {shap_matrix.shape}. "
        f"Expected 2D or 3D with feature dimension = {n_features}."
    )


def get_top_k_feature_sets(matrix: np.ndarray, feature_names, k: int):
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    sets = []
    for row in matrix:
        idx = np.argsort(np.abs(row))[-k:]
        idx = np.asarray(idx).astype(int).flatten()
        sets.append(set(feature_names[i] for i in idx))
    return sets


def compute_feature_stability_score(matrix: np.ndarray, feature_names, top_k: int = 5):
    if len(matrix) < 2:
        return None

    top_sets = get_top_k_feature_sets(matrix, feature_names, top_k)
    overlaps = []

    for i in range(len(top_sets) - 1):
        a = top_sets[i]
        b = top_sets[i + 1]
        union = a | b
        inter = a & b
        score = len(inter) / len(union) if len(union) > 0 else 0.0
        overlaps.append(score)

    return float(np.mean(overlaps)) if overlaps else None


def compute_sparsity(matrix: np.ndarray, threshold: float):
    active_counts = np.sum(np.abs(matrix) > threshold, axis=1)
    return {
        "average_important_features_per_sample": float(np.mean(active_counts)),
        "median_important_features_per_sample": float(np.median(active_counts)),
        "max_important_features_per_sample": int(np.max(active_counts)),
        "min_important_features_per_sample": int(np.min(active_counts)),
    }


def parse_lime_feature_name(raw_name: str, feature_names):
    for f in feature_names:
        if f in raw_name:
            return f
    return raw_name


def try_build_shap_explainer(model, X_background: pd.DataFrame, model_name: str):
    lower_name = model_name.lower()

    if "forest" in lower_name or "tree" in lower_name or "rf" in lower_name:
        try:
            explainer = shap.TreeExplainer(model)
            return explainer, type(explainer).__name__, "tree"
        except Exception:
            pass

    try:
        explainer = shap.Explainer(model, X_background)
        return explainer, type(explainer).__name__, "generic"
    except Exception:
        pass

    try:
        background = shap.sample(X_background, min(100, len(X_background)), random_state=RANDOM_STATE)
        pred_fn, _ = get_model_prediction_fn(model)
        explainer = shap.KernelExplainer(pred_fn, background)
        return explainer, type(explainer).__name__, "kernel"
    except Exception as e:
        raise RuntimeError(f"Could not build SHAP explainer for {model_name}: {e}")


# ============================================================
# SHAP
# ============================================================
def run_shap(model, model_name: str, X_sample: pd.DataFrame, output_dir: str, save_plots: bool = True):
    print(f"Running SHAP for {model_name}...")
    explainer, explainer_type, mode = try_build_shap_explainer(model, X_sample, model_name)

    start = time.time()
    base_values = None

    chunk_size = 50 if model_name == "random_forest" else len(X_sample)
    all_shap_rows = []

    if mode == "generic":
        if chunk_size >= len(X_sample):
            shap_result = explainer(X_sample)
            shap_matrix = np.array(shap_result.values)
            shap_matrix = normalize_shap_matrix(shap_matrix, X_sample.shape[1])
            base_values = getattr(shap_result, "base_values", None)
        else:
            for start_idx in range(0, len(X_sample), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X_sample))
                print(f"SHAP {model_name}: processing rows {start_idx} to {end_idx} / {len(X_sample)}")
                chunk = X_sample.iloc[start_idx:end_idx]
                shap_result = explainer(chunk)
                chunk_matrix = np.array(shap_result.values)
                chunk_matrix = normalize_shap_matrix(chunk_matrix, X_sample.shape[1])
                all_shap_rows.append(chunk_matrix)

                if base_values is None:
                    base_values = getattr(shap_result, "base_values", None)

            shap_matrix = np.vstack(all_shap_rows)

    elif mode in ["tree", "kernel"]:
        if chunk_size >= len(X_sample):
            shap_result = explainer.shap_values(X_sample)

            if isinstance(shap_result, list):
                shap_matrix = np.array(shap_result[1] if len(shap_result) > 1 else shap_result[0])
            else:
                shap_matrix = np.array(shap_result)

            shap_matrix = normalize_shap_matrix(shap_matrix, X_sample.shape[1])

            try:
                base_values = explainer.expected_value
            except Exception:
                base_values = None
        else:
            for start_idx in range(0, len(X_sample), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X_sample))
                print(f"SHAP {model_name}: processing rows {start_idx} to {end_idx} / {len(X_sample)}")
                chunk = X_sample.iloc[start_idx:end_idx]

                shap_result = explainer.shap_values(chunk)

                if isinstance(shap_result, list):
                    chunk_matrix = np.array(shap_result[1] if len(shap_result) > 1 else shap_result[0])
                else:
                    chunk_matrix = np.array(shap_result)

                chunk_matrix = normalize_shap_matrix(chunk_matrix, X_sample.shape[1])
                all_shap_rows.append(chunk_matrix)

                if base_values is None:
                    try:
                        base_values = explainer.expected_value
                    except Exception:
                        base_values = None

            shap_matrix = np.vstack(all_shap_rows)

    else:
        raise ValueError(f"Unsupported SHAP mode: {mode}")

    elapsed = time.time() - start

    print(f"SHAP runtime ({model_name}): {elapsed:.4f} seconds")
    print(f"SHAP samples used ({model_name}): {len(X_sample)}")
    print(f"Normalized SHAP matrix shape ({model_name}): {shap_matrix.shape}")

    # Core metrics
    global_stats = compute_basic_stats(shap_matrix, prefix="shap_")
    global_stats["shap_runtime_seconds"] = elapsed
    global_stats["shap_runtime_per_sample_seconds"] = elapsed / len(X_sample)
    global_stats["shap_explainer_type"] = explainer_type
    global_stats["shap_samples_used"] = len(X_sample)

    # Stability
    shap_feature_stability = compute_feature_stability_score(
        shap_matrix,
        list(X_sample.columns),
        top_k=min(5, len(X_sample.columns))
    )
    global_stats["shap_feature_stability_score"] = shap_feature_stability

    # Per-feature stats
    feature_stats_df = compute_feature_level_stats(shap_matrix, X_sample.columns)
    feature_stats_path = os.path.join(output_dir, f"{model_name}_shap_feature_statistics.csv")
    feature_stats_df.to_csv(feature_stats_path, index=False)

    # Fidelity
    shap_reconstruction_error = None
    try:
        model_preds = get_positive_class_predictions(model, X_sample)

        bv = np.array(base_values) if base_values is not None else None
        approx_preds = None

        if bv is not None:
            if bv.ndim == 0:
                approx_preds = shap_matrix.sum(axis=1) + float(bv)
            elif bv.ndim == 1 and len(bv) == len(X_sample):
                approx_preds = shap_matrix.sum(axis=1) + bv
            elif bv.ndim == 1 and len(bv) >= 2:
                approx_preds = shap_matrix.sum(axis=1) + float(bv[1])
            elif bv.ndim == 2 and bv.shape[0] == len(X_sample):
                approx_preds = shap_matrix.sum(axis=1) + bv[:, 0]

        if approx_preds is not None:
            approx_preds = np.asarray(approx_preds).reshape(-1)
            model_preds = np.asarray(model_preds).reshape(-1)
            if len(approx_preds) == len(model_preds):
                shap_reconstruction_error = float(np.mean(np.abs(model_preds - approx_preds)))
    except Exception:
        shap_reconstruction_error = None

    global_stats["shap_reconstruction_error"] = shap_reconstruction_error

    # Interpretability
    sparsity = compute_sparsity(shap_matrix, SPARSITY_THRESHOLD)
    global_stats["shap_average_important_features_per_sample"] = sparsity["average_important_features_per_sample"]
    global_stats["shap_median_important_features_per_sample"] = sparsity["median_important_features_per_sample"]
    global_stats["shap_max_important_features_per_sample"] = sparsity["max_important_features_per_sample"]
    global_stats["shap_min_important_features_per_sample"] = sparsity["min_important_features_per_sample"]

    # Save full matrix
    full_shap_df = pd.DataFrame(shap_matrix, columns=X_sample.columns, index=X_sample.index)
    full_shap_path = os.path.join(output_dir, f"{model_name}_shap_values_full.csv")
    full_shap_df.to_csv(full_shap_path, index=True)

    # Local SHAP
    local_rows = []
    first_n = min(10, len(X_sample))
    for i, idx in enumerate(X_sample.index[:first_n]):
        local_df = build_local_table_for_row(
            X_sample.loc[idx],
            shap_matrix[i],
            "shap_value",
            TOP_K_LOCAL
        )
        local_df.insert(0, "sample_index", idx)
        local_rows.append(local_df)

    local_shap_df = pd.concat(local_rows, ignore_index=True)
    local_shap_path = os.path.join(output_dir, f"{model_name}_shap_local_top_features_first10.csv")
    local_shap_df.to_csv(local_shap_path, index=False)

    if save_plots:
        try:
            plt.figure()
            shap.summary_plot(shap_matrix, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{model_name}_shap_summary_beeswarm.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            print(f"Could not save SHAP beeswarm plot for {model_name}: {e}")

        try:
            plt.figure()
            shap.summary_plot(shap_matrix, X_sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{model_name}_shap_summary_bar.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            print(f"Could not save SHAP bar plot for {model_name}: {e}")

    meta = {
        "shape": list(shap_matrix.shape),
        "runtime_seconds": elapsed,
        "runtime_per_sample_seconds": elapsed / len(X_sample),
        "explainer_type": explainer_type,
        "mode": mode,
        "samples_used": len(X_sample),
        "feature_stats_file": feature_stats_path,
        "full_values_file": full_shap_path,
        "local_top_features_file": local_shap_path,
    }

    return {
        "matrix": shap_matrix,
        "stats": global_stats,
        "feature_stats_df": feature_stats_df,
        "local_df": local_shap_df,
        "meta": meta,
    }


# ============================================================
# LIME
# ============================================================
def run_lime(model, model_name: str, X_sample: pd.DataFrame, output_dir: str):
    print(f"Running LIME for {model_name}...")

    X_num = safe_numeric_dataframe(X_sample)
    feature_names = list(X_num.columns)

    predict_fn, pred_fn_name = get_model_prediction_fn(model)
    mode = "classification" if is_classifier_model(model) else "regression"

    explainer = LimeTabularExplainer(
        training_data=X_num.values,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    lime_rows = []
    lime_r2_scores = []

    start = time.time()

    for row_pos, (idx, row) in enumerate(X_num.iterrows()):
        exp = explainer.explain_instance(
            data_row=row.values,
            predict_fn=predict_fn,
            num_features=min(LIME_NUM_FEATURES, len(feature_names))
        )

        weights = {f: 0.0 for f in feature_names}
        for raw_feature, weight in exp.as_list():
            mapped_name = parse_lime_feature_name(raw_feature, feature_names)
            if mapped_name in weights:
                weights[mapped_name] = weight

        dense_row = [weights[f] for f in feature_names]
        lime_rows.append(dense_row)

        try:
            lime_r2_scores.append(float(exp.score))
        except Exception:
            pass

        if (row_pos + 1) % 50 == 0:
            print(f"LIME processed {row_pos + 1}/{len(X_num)} samples for {model_name}")

    elapsed = time.time() - start
    lime_matrix = np.array(lime_rows)

    print(f"LIME runtime ({model_name}): {elapsed:.4f} seconds")
    print(f"LIME samples used ({model_name}): {len(X_num)}")

    # Core metrics
    global_stats = compute_basic_stats(lime_matrix, prefix="lime_")
    global_stats["lime_runtime_seconds"] = elapsed
    global_stats["lime_runtime_per_sample_seconds"] = elapsed / len(X_num)
    global_stats["lime_prediction_function"] = pred_fn_name
    global_stats["lime_samples_used"] = len(X_num)

    # Stability
    lime_feature_stability = compute_feature_stability_score(
        lime_matrix,
        feature_names,
        top_k=min(5, len(feature_names))
    )
    global_stats["lime_feature_stability_score"] = lime_feature_stability

    # Fidelity
    global_stats["lime_r2_score_mean"] = float(np.mean(lime_r2_scores)) if lime_r2_scores else None
    global_stats["lime_r2_score_std"] = float(np.std(lime_r2_scores)) if lime_r2_scores else None
    global_stats["lime_r2_score_min"] = float(np.min(lime_r2_scores)) if lime_r2_scores else None
    global_stats["lime_r2_score_max"] = float(np.max(lime_r2_scores)) if lime_r2_scores else None

    # Interpretability
    sparsity = compute_sparsity(lime_matrix, SPARSITY_THRESHOLD)
    global_stats["lime_average_important_features_per_sample"] = sparsity["average_important_features_per_sample"]
    global_stats["lime_median_important_features_per_sample"] = sparsity["median_important_features_per_sample"]
    global_stats["lime_max_important_features_per_sample"] = sparsity["max_important_features_per_sample"]
    global_stats["lime_min_important_features_per_sample"] = sparsity["min_important_features_per_sample"]

    # Per-feature stats
    feature_stats_df = compute_feature_level_stats(lime_matrix, feature_names)
    feature_stats_path = os.path.join(output_dir, f"{model_name}_lime_feature_statistics.csv")
    feature_stats_df.to_csv(feature_stats_path, index=False)

    # Full matrix
    full_lime_df = pd.DataFrame(lime_matrix, columns=feature_names, index=X_num.index)
    full_lime_path = os.path.join(output_dir, f"{model_name}_lime_values_full.csv")
    full_lime_df.to_csv(full_lime_path, index=True)

    # Local LIME
    local_rows = []
    first_n = min(10, len(X_num))
    for i, idx in enumerate(X_num.index[:first_n]):
        local_df = build_local_table_for_row(
            X_num.loc[idx],
            lime_matrix[i],
            "lime_value",
            TOP_K_LOCAL
        )
        local_df.insert(0, "sample_index", idx)
        local_rows.append(local_df)

    local_lime_df = pd.concat(local_rows, ignore_index=True)
    local_lime_path = os.path.join(output_dir, f"{model_name}_lime_local_top_features_first10.csv")
    local_lime_df.to_csv(local_lime_path, index=False)

    meta = {
        "shape": list(lime_matrix.shape),
        "runtime_seconds": elapsed,
        "runtime_per_sample_seconds": elapsed / len(X_num),
        "prediction_function": pred_fn_name,
        "samples_used": len(X_num),
        "feature_stats_file": feature_stats_path,
        "full_values_file": full_lime_path,
        "local_top_features_file": local_lime_path,
    }

    return {
        "matrix": lime_matrix,
        "stats": global_stats,
        "feature_stats_df": feature_stats_df,
        "local_df": local_lime_df,
        "meta": meta,
    }


# ============================================================
# CONSISTENCY
# ============================================================
def compare_shap_lime(model_name: str, shap_matrix: np.ndarray, lime_matrix: np.ndarray, feature_names, output_dir: str):
    rows = min(len(shap_matrix), len(lime_matrix))
    shap_matrix = shap_matrix[:rows]
    lime_matrix = lime_matrix[:rows]

    comparison = {}

    try:
        flat_corr = np.corrcoef(shap_matrix.flatten(), lime_matrix.flatten())[0, 1]
    except Exception:
        flat_corr = np.nan
    comparison["global_flattened_correlation"] = float(flat_corr) if not np.isnan(flat_corr) else None

    shap_mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    lime_mean_abs = np.mean(np.abs(lime_matrix), axis=0)

    try:
        mean_abs_corr = np.corrcoef(shap_mean_abs, lime_mean_abs)[0, 1]
    except Exception:
        mean_abs_corr = np.nan
    comparison["mean_abs_feature_importance_correlation_pearson"] = float(mean_abs_corr) if not np.isnan(mean_abs_corr) else None

    try:
        sp_corr, _ = spearmanr(shap_mean_abs, lime_mean_abs)
    except Exception:
        sp_corr = np.nan
    comparison["mean_abs_feature_importance_correlation_spearman"] = float(sp_corr) if not np.isnan(sp_corr) else None

    comp_df = pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": shap_mean_abs,
        "lime_mean_abs": lime_mean_abs,
        "shap_variance": np.var(shap_matrix, axis=0),
        "lime_variance": np.var(lime_matrix, axis=0),
    }).sort_values("shap_mean_abs", ascending=False)

    comp_path = os.path.join(output_dir, f"{model_name}_shap_lime_feature_importance_comparison.csv")
    comp_df.to_csv(comp_path, index=False)

    comparison["feature_importance_comparison_file"] = comp_path
    return comparison


# ============================================================
# REPORTING
# ============================================================
def build_final_summary(model_name: str, shap_out, lime_out, comparison_out, output_dir: str):
    rows = []

    for k, v in shap_out["stats"].items():
        rows.append({"model": model_name, "method": "SHAP", "metric": k, "value": v})

    for k, v in lime_out["stats"].items():
        rows.append({"model": model_name, "method": "LIME", "metric": k, "value": v})

    for k, v in comparison_out.items():
        if not k.endswith("_file"):
            rows.append({"model": model_name, "method": "COMPARISON", "metric": k, "value": v})

    summary_df = pd.DataFrame(rows)
    summary_csv_path = os.path.join(output_dir, f"{model_name}_xai_summary_statistics.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    report_lines = [
        f"Explainability Evaluation Report - {model_name}",
        "=" * 50,
        "",
        f"SHAP samples used: {shap_out['meta']['samples_used']}",
        f"LIME samples used: {lime_out['meta']['samples_used']}",
        "",
        f"SHAP total runtime: {shap_out['meta']['runtime_seconds']:.4f} s",
        f"SHAP runtime per sample: {shap_out['meta']['runtime_per_sample_seconds']:.6f} s",
        f"LIME total runtime: {lime_out['meta']['runtime_seconds']:.4f} s",
        f"LIME runtime per sample: {lime_out['meta']['runtime_per_sample_seconds']:.6f} s",
        "",
        "Key metrics:",
        f"- SHAP feature stability score: {shap_out['stats'].get('shap_feature_stability_score')}",
        f"- LIME feature stability score: {lime_out['stats'].get('lime_feature_stability_score')}",
        f"- SHAP reconstruction error: {shap_out['stats'].get('shap_reconstruction_error')}",
        f"- LIME R² mean: {lime_out['stats'].get('lime_r2_score_mean')}",
        f"- SHAP average important features/sample: {shap_out['stats'].get('shap_average_important_features_per_sample')}",
        f"- LIME average important features/sample: {lime_out['stats'].get('lime_average_important_features_per_sample')}",
        f"- SHAP vs LIME Pearson correlation: {comparison_out.get('mean_abs_feature_importance_correlation_pearson')}",
        f"- SHAP vs LIME Spearman correlation: {comparison_out.get('mean_abs_feature_importance_correlation_spearman')}",
        "",
        f"Summary statistics CSV: {summary_csv_path}",
    ]

    report_path = os.path.join(output_dir, f"{model_name}_xai_report.txt")
    save_text(report_path, "\n".join(report_lines))
    return summary_csv_path, report_path


def evaluate_model(model, model_name: str, X: pd.DataFrame, base_output_dir: str):
    model_output_dir = os.path.join(base_output_dir, model_name)
    ensure_output_dir(model_output_dir)

    if model_name == "dnn":
        shap_n = DNN_SHAP_N_SAMPLES
        lime_n = LIME_N_SAMPLES
    else:
        shap_n = RF_SHAP_N_SAMPLES
        lime_n = LIME_N_SAMPLES

    X_shap = sample_exact_or_all(X, shap_n, RANDOM_STATE)
    X_lime = sample_exact_or_all(X, lime_n, RANDOM_STATE)

    print(f"{model_name} - SHAP dataset shape used: {X_shap.shape}")
    print(f"{model_name} - LIME dataset shape used: {X_lime.shape}")

    shap_out = run_shap(model, model_name, X_shap, model_output_dir, save_plots=SAVE_PLOTS)
    lime_out = run_lime(model, model_name, X_lime, model_output_dir)

    comparison_out = compare_shap_lime(
        model_name,
        shap_out["matrix"],
        lime_out["matrix"],
        list(X.columns),
        model_output_dir
    )

    meta = {
        "model_name": model_name,
        "model_type": str(type(model)),
        "original_dataset_shape": list(X.shape),
        "dropped_columns": DROP_COLUMNS,
        "shap_samples_used": len(X_shap),
        "lime_samples_used": len(X_lime),
        "threshold_for_sparsity": SPARSITY_THRESHOLD,
        "shap_meta": shap_out["meta"],
        "lime_meta": lime_out["meta"],
        "comparison_meta": comparison_out,
    }

    meta_path = os.path.join(model_output_dir, f"{model_name}_xai_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary_csv_path, report_path = build_final_summary(
        model_name,
        shap_out,
        lime_out,
        comparison_out,
        model_output_dir
    )

    print(f"\nDone for {model_name}.")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Report TXT: {report_path}")
    print(f"Metadata JSON: {meta_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_output_dir(OUTPUT_DIR)

    print("DATA_PATH =", DATA_PATH)
    print("DNN_MODEL_PATH =", DNN_MODEL_PATH)
    print("RF_MODEL_PATH =", RF_MODEL_PATH)
    print("Dataset exists?", os.path.exists(DATA_PATH))
    print("DNN model exists?", os.path.exists(DNN_MODEL_PATH))
    print("RF model exists?", os.path.exists(RF_MODEL_PATH))

    print("\nLoading dataset...")
    X = load_data(DATA_PATH, drop_columns=DROP_COLUMNS)
    X = safe_numeric_dataframe(X)
    print(f"Dataset shape after dropping columns: {X.shape}")
    print(f"Columns dropped if present: {DROP_COLUMNS}")

    print("\nLoading DNN model...")
    dnn_model = load_model(DNN_MODEL_PATH)
    print(f"DNN model type: {type(dnn_model)}")

    print("\nLoading Random Forest model...")
    rf_model = load_model(RF_MODEL_PATH)

    # reduce noisy threaded logging and potential instability
    if hasattr(rf_model, "verbose"):
        rf_model.verbose = 0
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1

    print(f"RF model type: {type(rf_model)}")

    print("\n========== Evaluating DNN ==========")
    evaluate_model(dnn_model, "dnn", X, OUTPUT_DIR)

    print("\n========== Evaluating Random Forest ==========")
    evaluate_model(rf_model, "random_forest", X, OUTPUT_DIR)


if __name__ == "__main__":
    main()