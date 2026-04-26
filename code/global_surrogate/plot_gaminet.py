import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_surrogate_common import load_feature_mapper


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return slug[:140] if slug else "feature"


def _split_interaction_name(term_name: str) -> list[str]:
    return [part.strip() for part in str(term_name).split(" vs. ")]


def _format_term_name(term_name: str, mapper) -> str:
    return " x ".join(mapper.display_name(part) for part in _split_interaction_name(term_name))


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


def _plot_main_effect(term_name: str, payload: dict, output_path: Path, mapper) -> None:
    x = np.asarray(payload["inputs"], dtype=float)
    y = np.asarray(payload["outputs"], dtype=float)
    x_plot, is_numeric_x = mapper.transform_axis(term_name, x)
    display_name = mapper.display_name(term_name)

    plt.figure(figsize=(8.8, 4.4))
    if is_numeric_x and np.asarray(x_plot).size >= 8:
        x_numeric = np.asarray(x_plot, dtype=float)
        order = np.argsort(x_numeric)
        plt.plot(x_numeric[order], y[order], linewidth=2.0)
        plt.xlabel(display_name)
    else:
        labels = [str(v) for v in np.asarray(x_plot)]
        pos = np.arange(len(labels))
        plt.bar(pos, y)
        plt.xticks(pos, labels, rotation=35, ha="right")
        plt.xlabel(display_name)

    plt.ylabel("Main effect contribution")
    plt.title(f"GAMI-Net main effect: {display_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close("all")


def _plot_interaction(term_name: str, payload: dict, output_path: Path, mapper) -> None:
    left_term, right_term = _split_interaction_name(term_name)
    left_display = mapper.display_name(left_term)
    right_display = mapper.display_name(right_term)

    x1 = np.asarray(payload["input1"], dtype=float)
    x2 = np.asarray(payload["input2"], dtype=float)
    scores = np.asarray(payload["outputs"], dtype=float)

    x1_plot, x1_numeric = mapper.transform_axis(left_term, x1)
    x2_plot, x2_numeric = mapper.transform_axis(right_term, x2)

    vlim = float(np.nanmax(np.abs(scores))) if scores.size else 1.0
    norm = None if vlim == 0 else TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    plt.figure(figsize=(8.2, 6.2))
    if x1_numeric and x2_numeric:
        mesh = plt.imshow(
            scores,
            origin="lower",
            aspect="auto",
            extent=[
                float(np.min(np.asarray(x2_plot, dtype=float))),
                float(np.max(np.asarray(x2_plot, dtype=float))),
                float(np.min(np.asarray(x1_plot, dtype=float))),
                float(np.max(np.asarray(x1_plot, dtype=float))),
            ],
            cmap="coolwarm",
            norm=norm,
        )
        plt.xlabel(right_display)
        plt.ylabel(left_display)
    else:
        mesh = plt.imshow(scores, origin="lower", aspect="auto", cmap="coolwarm", norm=norm)
        x_labels = [str(v) for v in np.asarray(x2_plot)]
        y_labels = [str(v) for v in np.asarray(x1_plot)]
        x_ticks = np.linspace(0, len(x_labels) - 1, num=min(10, len(x_labels)), dtype=int)
        y_ticks = np.linspace(0, len(y_labels) - 1, num=min(10, len(y_labels)), dtype=int)
        plt.xticks(x_ticks, [x_labels[i] for i in x_ticks], rotation=35, ha="right")
        plt.yticks(y_ticks, [y_labels[i] for i in y_ticks])
        plt.xlabel(right_display)
        plt.ylabel(left_display)

    plt.colorbar(mesh, label="Interaction contribution")
    plt.title(f"GAMI-Net interaction: {left_display} x {right_display}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close("all")


def load_gaminet_model(model_path: Path):
    from gaminet import GAMINet

    with model_path.open("rb") as handle:
        model_dict = pickle.load(handle)

    model = GAMINet(
        meta_info=model_dict["meta_info"],
        subnet_arch=model_dict["subnet_arch"],
        interact_arch=model_dict["interact_arch"],
        lr_bp=model_dict["lr_bp"],
        batch_size=model_dict["batch_size"],
        task_type=model_dict["task_type"],
        activation_func=model_dict["activation_func"],
        tuning_epochs=model_dict["tuning_epochs"],
        main_effect_epochs=model_dict["main_effect_epochs"],
        interaction_epochs=model_dict["interaction_epochs"],
        early_stop_thres=model_dict["early_stop_thres"],
        heredity=model_dict["heredity"],
        reg_clarity=model_dict["reg_clarity"],
        loss_threshold=model_dict["loss_threshold"],
        mono_increasing_list=model_dict["mono_increasing_list"],
        mono_decreasing_list=model_dict["mono_decreasing_list"],
        lattice_size=model_dict["lattice_size"],
        verbose=False,
        val_ratio=model_dict["val_ratio"],
        random_state=model_dict["random_state"],
        interact_num=model_dict["interact_num"],
    )
    model.load(folder=str(model_path.parent) + "/", name=model_path.stem)
    return model


def plot_gaminet_artifacts(model_path: Path, output_dir: Path, preprocessor_path: Path, top_n: int = 10, top_n_interactions: int = 5) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_effect_dir = output_dir / "main_effect_terms"
    interaction_dir = output_dir / "interaction_terms"
    main_effect_dir.mkdir(parents=True, exist_ok=True)
    interaction_dir.mkdir(parents=True, exist_ok=True)

    mapper = load_feature_mapper(preprocessor_path)
    model = load_gaminet_model(model_path)
    global_dict = model.global_explain(save_dict=False)

    main_rows: list[tuple[str, float]] = []
    interaction_rows: list[tuple[str, float]] = []
    overall_scores: defaultdict[str, float] = defaultdict(float)

    for term_name, payload in global_dict.items():
        term_type = str(payload.get("type"))
        importance = float(payload.get("importance", 0.0))
        if term_type == "continuous":
            display = mapper.display_name(term_name)
            main_rows.append((str(term_name), importance))
            overall_scores[display] += importance
        elif term_type == "pairwise":
            left_term, right_term = _split_interaction_name(term_name)
            interaction_rows.append((str(term_name), importance))
            overall_scores[mapper.display_name(left_term)] += importance / 2.0
            overall_scores[mapper.display_name(right_term)] += importance / 2.0

    main_rows.sort(key=lambda item: item[1], reverse=True)
    interaction_rows.sort(key=lambda item: item[1], reverse=True)
    main_effect_importance_rows = [(mapper.display_name(name), imp) for name, imp in main_rows]
    interaction_importance_rows = [(_format_term_name(name, mapper), imp) for name, imp in interaction_rows]
    overall_rows = sorted(overall_scores.items(), key=lambda item: item[1], reverse=True)

    saved_main_paths = []
    for rank, (term_name, _importance) in enumerate(main_rows[:top_n], start=1):
        out = main_effect_dir / f"gaminet_main_effect_top_{rank:02d}_{_safe_slug(mapper.display_name(term_name))}.png"
        _plot_main_effect(term_name, global_dict[term_name], out, mapper)
        saved_main_paths.append(out)

    saved_interaction_paths = []
    for rank, (term_name, _importance) in enumerate(interaction_rows[:top_n_interactions], start=1):
        out = interaction_dir / f"gaminet_interaction_top_{rank:02d}_{_safe_slug(_format_term_name(term_name, mapper))}.png"
        _plot_interaction(term_name, global_dict[term_name], out, mapper)
        saved_interaction_paths.append(out)

    _plot_importance(main_effect_importance_rows, output_dir / "gaminet_feature_importance.png", "GAMI-Net main effect importance", "GAMI-Net Main Effect Importance")
    _plot_importance(interaction_importance_rows, output_dir / "gaminet_interaction_importance.png", "GAMI-Net interaction term importance", "GAMI-Net Interaction Importance")
    _plot_importance(overall_rows, output_dir / "gaminet_overall_feature_importance.png", "Overall importance = main + 0.5 * interaction contributions", "GAMI-Net Overall Feature Importance")

    _write_importance_csv(main_effect_importance_rows, output_dir / "gaminet_feature_importance.csv", "feature")
    _write_importance_csv(interaction_importance_rows, output_dir / "gaminet_interaction_importance.csv", "interaction_term")
    _write_importance_csv(overall_rows, output_dir / "gaminet_overall_feature_importance.csv", "feature")

    return {
        "main_effect_count": len(main_rows),
        "interaction_count": len(interaction_rows),
        "top_main": main_effect_importance_rows[:top_n],
        "top_interactions": interaction_importance_rows[:top_n_interactions],
        "saved_main_paths": [str(path.resolve()) for path in saved_main_paths],
        "saved_interaction_paths": [str(path.resolve()) for path in saved_interaction_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved GAMI-Net surrogate artifacts.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved GAMI-Net .pickle file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated plots.")
    parser.add_argument("--preprocessor-path", type=Path, default=Path("artifacts/preprocessor.pkl"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-n-interactions", type=int, default=5)
    args = parser.parse_args()

    result = plot_gaminet_artifacts(
        model_path=args.model_path,
        output_dir=args.output_dir,
        preprocessor_path=args.preprocessor_path,
        top_n=args.top_n,
        top_n_interactions=args.top_n_interactions,
    )

    print(f"Loaded model: {args.model_path}")
    print(f"Main-effect terms: {result['main_effect_count']}")
    print(f"Interaction terms: {result['interaction_count']}")
    print("Top main effects:")
    for idx, (name, score) in enumerate(result["top_main"], start=1):
        print(f"  {idx:02d}. {name} (importance={score:.6f})")
    print("Top interactions:")
    for idx, (name, score) in enumerate(result["top_interactions"], start=1):
        print(f"  {idx:02d}. {name} (importance={score:.6f})")


if __name__ == "__main__":
    main()
