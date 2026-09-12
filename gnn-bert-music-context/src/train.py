"""
train.py
========
Unified training entrypoint for all four tasks.

Usage:
    python src/train.py --task 1   # BERT baseline
    python src/train.py --task 2   # GraphSAGE + CNN/majority baselines
    python src/train.py --task 3 --variant cross_attention
    python src/train.py --task 4   # contrastive dual-encoder

Notes
-----
Tasks 1-4 were originally developed and run as standalone Kaggle
notebook cells (see src/tasks/*.py — each file is copy-paste-into-Kaggle
ready and contains the exact code used to produce the numbers in the
report). This script is a thin, config-driven wrapper around the
reusable pieces in audio_features.py / graph_builder.py / bert_encoder.py
/ gnn_model.py / fusion_model.py / contrastive.py, intended for running
the pipeline outside of Kaggle (e.g. a local GPU box or a cluster job),
reading paths and hyperparameters from config.yaml.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASK_SCRIPTS = {
    1: ROOT / "src" / "tasks" / "task1_bert_baseline.py",
    2: ROOT / "src" / "tasks" / "task2_gnn_baseline.py",
    3: ROOT / "src" / "tasks" / "task3_fusion.py",
    4: ROOT / "src" / "tasks" / "task4_contrastive_retrieval.py",
}


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train GNN-BERT music-context models")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--variant", type=str, default="cross_attention",
                         choices=["bert_only", "gnn_only", "early_concat", "cross_attention"],
                         help="Only used for --task 3 (fusion ablation, Table 3).")
    parser.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[train.py] Loaded config: {args.config}")
    print(f"[train.py] Task {args.task} config: {cfg.get('tasks', {}).get(str(args.task), {})}")

    script = TASK_SCRIPTS[args.task]
    print(f"[train.py] Running {script.relative_to(ROOT)}")
    print("[train.py] NOTE: task scripts were authored for Kaggle notebook cells "
          "('# %% CELL' blocks) and expect /kaggle/input dataset mounts. "
          "Adjust the INPUT_ROOT / dataset paths in the script (or set the "
          "equivalent local paths in config.yaml -> data.raw_dir) before running "
          "off-platform.")
    subprocess.run([sys.executable, str(script)] + (["--variant", args.variant] if args.task == 3 else []),
                    check=True)


if __name__ == "__main__":
    main()
