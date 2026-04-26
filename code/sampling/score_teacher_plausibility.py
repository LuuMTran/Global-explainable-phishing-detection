import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score synthetic samples with teacher models and flag high-plausibility rows.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Candidate synthetic samples with feature columns only.")
    parser.add_argument("--output-npz", type=Path, required=True, help="Path to save teacher scores and plausibility masks.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=0.80, help="Minimum per-teacher max probability required.")
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        help="If set, only keep rows where both teachers predict the same class.",
    )
    return parser.parse_args()


def batched_predict_proba(model, X: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    parts = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        parts.append(np.asarray(model.predict_proba(X[start:stop]), dtype=np.float32))
    return np.vstack(parts)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    X = df.to_numpy(dtype=np.float32, copy=True)

    model_paths = {
        "random_forest": args.artifact_dir / "random_forest.pkl",
        "deep_neural_net": args.artifact_dir / "deep_neural_net.pkl",
    }

    probas = {}
    preds = {}
    for name, path in model_paths.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with path.open("rb") as handle:
                model = pickle.load(handle)
            model_proba = batched_predict_proba(model, X)
        probas[name] = model_proba
        preds[name] = model_proba.argmax(axis=1).astype(np.int32)

    rf_conf = probas["random_forest"].max(axis=1)
    dnn_conf = probas["deep_neural_net"].max(axis=1)
    agreement_mask = preds["random_forest"] == preds["deep_neural_net"]
    confidence_mask = (rf_conf >= args.threshold) & (dnn_conf >= args.threshold)
    if args.require_agreement:
        plausible_mask = agreement_mask & confidence_mask
    else:
        plausible_mask = confidence_mask

    out_dir = args.output_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        plausible_mask=plausible_mask.astype(np.uint8),
        agreement_mask=agreement_mask.astype(np.uint8),
        confidence_mask=confidence_mask.astype(np.uint8),
        random_forest_pred=preds["random_forest"],
        deep_neural_net_pred=preds["deep_neural_net"],
        random_forest_prob_1=probas["random_forest"][:, 1].astype(np.float32),
        deep_neural_net_prob_1=probas["deep_neural_net"][:, 1].astype(np.float32),
        random_forest_confidence=rf_conf.astype(np.float32),
        deep_neural_net_confidence=dnn_conf.astype(np.float32),
    )

    kept = int(plausible_mask.sum())
    total = int(len(plausible_mask))
    print(
        f"Saved plausibility scores to {args.output_npz} | kept={kept}/{total} "
        f"({(kept / max(total, 1)):.4f}) | agreement={(agreement_mask.mean()):.4f} | "
        f"confidence={(confidence_mask.mean()):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
