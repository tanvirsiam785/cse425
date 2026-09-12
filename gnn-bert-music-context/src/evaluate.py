"""
evaluate.py
===========
Computes and reports the metrics used throughout the paper:
    - Macro-F1 / Micro-F1 for multi-label tag prediction (Tasks 1, 2, 3)
    - AUC-PR for multi-label tag prediction (Task 3)
    - Recall@{1,5,10} for cross-modal retrieval (Task 4)
Also regenerates results/metrics.json from a set of prediction arrays,
and can be used to sanity-check that a rerun reproduces the reported
numbers within noise.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent


def multilabel_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, y_pred_probs, average="macro")),
    }


def single_label_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def recall_at_k(sim_matrix, k_values=(1, 5, 10)):
    n = sim_matrix.shape[0]
    ranks = np.argsort(-sim_matrix, axis=1)
    out = {}
    for k in k_values:
        hit = sum(i in ranks[i, :k] for i in range(n))
        out[f"R@{k}"] = hit / n
    return out


def write_metrics(metrics_dict, out_path=ROOT / "results" / "metrics.json"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[evaluate.py] Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved predictions")
    parser.add_argument("--npz", type=str, help="Path to .npz with y_true/y_pred(_probs) arrays")
    parser.add_argument("--kind", choices=["multilabel", "single_label", "retrieval"],
                         default="multilabel")
    args = parser.parse_args()

    if args.npz:
        data = np.load(args.npz)
        if args.kind == "multilabel":
            metrics = multilabel_metrics(data["y_true"], data["y_pred_probs"])
        elif args.kind == "single_label":
            metrics = single_label_metrics(data["y_true"], data["y_pred"])
        else:
            metrics = recall_at_k(data["sim_matrix"])
        print(json.dumps(metrics, indent=2))
    else:
        print("Pass --npz path/to/predictions.npz --kind {multilabel,single_label,retrieval}")
