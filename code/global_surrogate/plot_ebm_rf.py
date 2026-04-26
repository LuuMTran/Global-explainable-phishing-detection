import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_surrogate_common import load_feature_mapper


def _is_onehot_feature(name: str) -> bool:
    return str(name).startswith("onehotcat__")


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return slug[:140] if slug else "feature"


def _to_float_array(values) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        out = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float)


def _split_term_name(term_name: str) -> list[str]:
    return [part.strip() for part in str(term_name).split(" & ")]


def _format_term_name(term_name: str, mapper) -> str:
    parts = _split_term_name(term_name)
    return " x ".join(mapper.display_name(part) for part in parts)


def _keep_term(term_name: str, keep_onehot: bool) -> bool:
    if keep_onehot:
        return True
    return not any(_is_onehot_feature(part) for part in _split_term_name(term_name))


def _pick_1d_axis(names, score_count: int):
    x = _to_float_array(names)
    if x.size == score_count + 1:
        return x[:-1]
    if x.size == score_count:
        return x
    return np.arange(score_count, dtype=float)


def _prepare_interaction_axis(names, target_size: int, feature_name: str, mapper):
    raw = _to_float_array(names)
    if raw.size == target_size + 1:
        values, is_numeric = mapper.transform_axis(feature_name, raw)
        return np.asarray(values), is_numeric, True
    if raw.size == target_size:
        values, is_numeric = mapper.transform_axis(feature_name, raw)
        return np.asarray(values), is_numeric, False
    fallback = np.arange(target_size, dtype=float)
    return fallback, True, False


def _write_importance_csv(rows: list[tuple[str, float]], output_path: Path, header_name: str) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header_name, "importance"])
        for name, value in rows:
            writer.writerow([name, f"{value:.12g}"])


def _plot_importance(rows: list[tuple[str, float]], output_path: Path, xlabel: str, title: str) -> None:
    if not rows:
        return

    names = [name for name, _ in rows][::-1]
    vals = [value for _, value in rows][::-1]

    fig_h = max(5, 0.28 * len(rows))
    plt.figure(figsize=(11, fig_h))
    plt.barh(names, vals)
    plt.xlabel(xlabel)
    plt.ylabel("Term")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close("all")


def _plot_univariate_term(term_name: str, term_data: dict, output_path: Path, mapper) -> None:
    scores = np.asarray(term_data.get("scores", []), dtype=float)
    if scores.size == 0:
        return

    x = _pick_1d_axis(term_data.get("names", []), scores.size)
    x_plot, is_numeric_x = mapper.transform_axis(term_name, x)
    display_name = mapper.display_name(term_name)

    plt.figure(figsize=(8.8, 4.4))
    if is_numeric_x and np.asarray(x_plot).size >= 8:
        x_numeric = np.asarray(x_plot, dtype=float)
        order = np.argsort(x_numeric)
        plt.plot(x_numeric[order], scores[order], linewidth=2.0)
        plt.xlabel(display_name)
    else:
        labels = [str(v) for v in np.asarray(x_plot)]
        if len(labels) == scores.size + 1:
            labels = labels[:-1]
        if len(labels) != scores.size:
            labels = [str(i) for i in range(scores.size)]
        pos = np.arange(scores.size)
        plt.bar(pos, scores)
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.xlabel(display_name)

    plt.ylabel("Term contribution")
    plt.title(f"EBM main effect: {display_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close("all")


def _plot_interaction_term(term_name: str, term_data: dict, output_path: Path, mapper) -> None:
    scores = np.asarray(term_data.get("scores", []), dtype=float)
    if scores.ndim != 2 or scores.size == 0:
        return

    left_term, right_term = _split_term_name(term_name)
    left_axis, left_numeric, left_has_edges = _prepare_interaction_axis(
        term_data.get("left_names", []), scores.shape[0], left_term, mapper
    )
    right_axis, right_numeric, right_has_edges = _prepare_interaction_axis(
        term_data.get("right_names", []), scores.shape[1], right_term, mapper
    )

    display_name = _format_term_name(term_name, mapper)
    vlim = float(np.nanmax(np.abs(scores))) if scores.size else 1.0
    norm = None if vlim == 0 else TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    plt.figure(figsize=(8.2, 6.2))
    if left_numeric and right_numeric:
        left_num = np.asarray(left_axis, dtype=float)
        right_num = np.asarray(right_axis, dtype=float)
        if left_has_edges and right_has_edges:
            mesh = plt.pcolormesh(right_num, left_num, scores, shading="auto", cmap="coolwarm", norm=norm)
        else:
            extent = [float(np.min(right_num)), float(np.max(right_num)), float(np.min(left_num)), float(np.max(left_num))]
            mesh = plt.imshow(scores, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", norm=norm)
        plt.xlabel(mapper.display_name(right_term))
        plt.ylabel(mapper.display_name(left_term))
    else:
        mesh = plt.imshow(scores, origin="lower", aspect="auto", cmap="coolwarm", norm=norm)
        left_labels = [str(v) for v in np.asarray(left_axis)]
        right_labels = [str(v) for v in np.asarray(right_axis)]
        if len(left_labels) == scores.shape[0] + 1:
            left_labels = left_labels[:-1]
        if len(right_labels) == scores.shape[1] + 1:
            right_labels = right_labels[:-1]
        y_pos = np.arange(scores.shape[0])
        x_pos = np.arange(scores.shape[1])
        if len(left_labels) == scores.shape[0]:
            plt.yticks(y_pos, left_labels)
        if len(right_labels) == scores.shape[1]:
            plt.xticks(x_pos, right_labels, rotation=35, ha="right")
        plt.xlabel(mapper.display_name(right_term))
        plt.ylabel(mapper.display_name(left_term))

    plt.colorbar(mesh, label="Interaction contribution")
    plt.title(f"EBM interaction: {display_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EBM explanations, including interaction terms.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/surrogates/ebm_random_forest.pkl"),
        help="Path to EBM pickle model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/surrogates/ebm_plots_random_forest"),
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N most important non-onehot main-effect terms to plot.",
    )
    parser.add_argument(
        "--top-n-interactions",
        type=int,
        default=5,
        help="Top N interaction terms to plot as 2D surfaces.",
    )
    parser.add_argument(
        "--keep-onehot",
        action="store_true",
        help="Keep one-hot terms and one-hot interactions (disabled by default).",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=Path("artifacts/preprocessor.pkl"),
        help="Path to the fitted preprocessor used for denormalizing feature axes.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_effect_dir = args.output_dir / "main_effect_terms"
    interaction_dir = args.output_dir / "interaction_terms"
    main_effect_dir.mkdir(parents=True, exist_ok=True)
    interaction_dir.mkdir(parents=True, exist_ok=True)

    with args.model_path.open("rb") as handle:
        model = cloudpickle.load(handle)
    mapper = load_feature_mapper(args.preprocessor_path)

    if not hasattr(model, "explain_global") or not hasattr(model, "term_importances"):
        raise AttributeError("Loaded model does not expose expected EBM APIs.")

    global_exp = model.explain_global()
    term_importances = np.asarray(model.term_importances(), dtype=float)
    term_names = list(model.term_names_)
    term_features = list(model.term_features_)

    main_rows: list[tuple[int, str, float]] = []
    interaction_rows: list[tuple[int, str, float]] = []
    overall_feature_scores: defaultdict[str, float] = defaultdict(float)

    for idx, (term_name, importance, feat_idx_tuple) in enumerate(zip(term_names, term_importances, term_features)):
        if not _keep_term(term_name, args.keep_onehot):
            continue

        parts = _split_term_name(term_name)
        display_parts = [mapper.display_name(part) for part in parts]
        if len(feat_idx_tuple) == 1:
            main_rows.append((idx, term_name, float(importance)))
            overall_feature_scores[display_parts[0]] += float(importance)
        elif len(feat_idx_tuple) == 2:
            interaction_rows.append((idx, term_name, float(importance)))
            credit = float(importance) / 2.0
            overall_feature_scores[display_parts[0]] += credit
            overall_feature_scores[display_parts[1]] += credit

    main_rows.sort(key=lambda item: item[2], reverse=True)
    interaction_rows.sort(key=lambda item: item[2], reverse=True)
    overall_rows = sorted(overall_feature_scores.items(), key=lambda item: item[1], reverse=True)
    main_effect_importance_rows = [
        (mapper.display_name(name), imp)
        for _, name, imp in main_rows
    ]
    interaction_importance_rows = [(_format_term_name(name, mapper), imp) for _, name, imp in interaction_rows]
    main_effect_importance_rows.sort(key=lambda item: item[1], reverse=True)

    top_n = max(1, int(args.top_n))
    top_n_interactions = max(1, int(args.top_n_interactions))

    saved_main_paths: list[Path] = []
    for rank, (term_idx, term_name, _importance) in enumerate(main_rows[:top_n], start=1):
        out = main_effect_dir / f"ebm_main_effect_top_{rank:02d}_{_safe_slug(mapper.display_name(term_name))}.png"
        _plot_univariate_term(term_name, global_exp.data(term_idx), out, mapper)
        if out.exists():
            saved_main_paths.append(out)

    saved_interaction_paths: list[Path] = []
    for rank, (term_idx, term_name, _importance) in enumerate(interaction_rows[:top_n_interactions], start=1):
        out = interaction_dir / f"ebm_interaction_top_{rank:02d}_{_safe_slug(_format_term_name(term_name, mapper))}.png"
        _plot_interaction_term(term_name, global_exp.data(term_idx), out, mapper)
        if out.exists():
            saved_interaction_paths.append(out)

    _plot_importance(
        main_effect_importance_rows,
        args.output_dir / "ebm_feature_importance.png",
        xlabel="EBM main effect importance",
        title="EBM Main Effect Importance",
    )
    _plot_importance(
        interaction_importance_rows,
        args.output_dir / "ebm_interaction_importance.png",
        xlabel="EBM interaction term importance",
        title="EBM Interaction Importance",
    )
    _plot_importance(
        overall_rows,
        args.output_dir / "ebm_overall_feature_importance.png",
        xlabel="Overall importance = main + 0.5 * interaction contributions",
        title="EBM Overall Feature Importance",
    )

    _write_importance_csv(main_effect_importance_rows, args.output_dir / "ebm_feature_importance.csv", "feature")
    _write_importance_csv(interaction_importance_rows, args.output_dir / "ebm_interaction_importance.csv", "interaction_term")
    _write_importance_csv(overall_rows, args.output_dir / "ebm_overall_feature_importance.csv", "feature")

    print(f"Loaded model: {args.model_path}")
    print(f"Total terms: {len(term_names)}")
    print(f"Main-effect terms kept: {len(main_rows)}")
    print(f"Interaction terms kept: {len(interaction_rows)}")
    print("Top main-effect features:")
    for rank, (feature_name, score) in enumerate(main_effect_importance_rows[:top_n], start=1):
        print(f"  {rank:02d}. {feature_name} (main_effect_importance={score:.6f})")
    print("Top overall aggregated features:")
    for rank, (feature_name, score) in enumerate(overall_rows[:top_n], start=1):
        print(f"  {rank:02d}. {feature_name} (overall_importance={score:.6f})")
    print(f"Saved main-effect importance: {args.output_dir / 'ebm_feature_importance.png'}")
    print(f"Saved interaction-only importance: {args.output_dir / 'ebm_interaction_importance.png'}")
    print(f"Saved overall aggregated importance: {args.output_dir / 'ebm_overall_feature_importance.png'}")

    print("Saved top main-effect plots:")
    for path in saved_main_paths:
        print(f"  - {path}")
    print("Saved top interaction plots:")
    for path in saved_interaction_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
