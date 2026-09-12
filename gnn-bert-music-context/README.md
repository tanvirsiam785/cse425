# GNN-BERT: A Hybrid Graph Neural Network and Language Model Approach for Understanding Musical Context

Tanvir Mohammad Siam — Department of Computer Science and Engineering, BRAC University

This repository contains the code, results, and report for a four-stage study building toward a
hybrid Graph Neural Network (GNN) + BERT system for multi-label music tag prediction and
cross-modal audio-text retrieval. Full write-up: [`report/final_report.pdf`](report/final_report.pdf).

## Summary of results

| Task | Best model | Headline metric |
|---|---|---|
| 1 — BERT tag classifier (MusicCaps) | BERT (fine-tuned, masked captions) | Macro-F1 **0.627** |
| 2 — GNN on FMA-small (official split) | CNN (mel-spectrogram) baseline beats chroma-only GNN | Macro-F1 0.441 (CNN) vs. 0.305 (GNN) |
| 3 — GNN-BERT fusion ablation | Cross-attention fusion | Macro-F1 **0.611** |
| 4 — Contrastive dual-encoder | — | Caption→Audio R@10 = 0.049; zero-shot Macro-F1 = 0.108 |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                 # FMA, MusicCaps downloads (not committed — see Setup)
│   ├── processed/           # cached graphs, mel-spectrograms, BERT tokenizations
│   └── splits/              # train/val/test JSON manifests
├── notebooks/
│   ├── eda.ipynb                       # dataset exploration / sanity checks
│   ├── demo_context.ipynb              # end-to-end demo: caption -> tags -> retrieval
│   ├── tasks1-3_full_pipeline.ipynb    # original Kaggle notebook, Tasks 1-3
│   └── task4_contrastive_retrieval.ipynb  # original Kaggle notebook, Task 4
├── src/
│   ├── audio_features.py    # windowing, chroma, mel-spectrogram extraction
│   ├── graph_builder.py     # chroma segment-graph construction (nodes/edges)
│   ├── bert_encoder.py      # BERT tower + aspect-phrase masking (leakage guard)
│   ├── gnn_model.py         # GraphSAGE encoder + CNN(mel) baseline
│   ├── fusion_model.py      # Task 3: cross-attention / early-concat / ablations
│   ├── contrastive.py       # Task 4: dual-encoder + InfoNCE + recall@k
│   ├── train.py             # unified CLI entrypoint (python src/train.py --task N)
│   ├── evaluate.py          # metrics computation (Macro/Micro-F1, AUC-PR, R@k)
│   └── tasks/                # original, self-contained Kaggle-notebook scripts
│       ├── task1_bert_baseline.py
│       ├── task2_gnn_baseline.py
│       ├── task3_fusion.py
│       └── task4_contrastive_retrieval.py
├── results/
│   ├── metrics.json         # all tables from the report, machine-readable
│   ├── plots/                # F1-vs-epoch curves, t-SNE embedding plots
│   └── retrieval_examples/   # qualitative caption->audio retrieval cases
└── report/
    └── final_report.pdf
```

## Datasets

- **MusicCaps** — official `musiccaps-public.csv` (5,521 ten-second clips, free-text captions +
  musician-written `aspect_list`). Audio via the community re-upload *MusicCapsAudio5308*
  (5,308/5,521 clips), matched to captions by YouTube ID.
- **FMA-small** — 8,000 tracks, 8 genres, 30s clips. Uses FMA's **official** train/val/test split
  (6,400 / 800 / 800) to avoid artist-level leakage, per assignment spec.

Place raw downloads under `data/raw/` matching the paths in `config.yaml` (`data.musiccaps_csv`,
`data.musiccaps_audio_dir`, `data.fma_audio_dir`, `data.fma_tracks_csv`) before running the
pipeline outside Kaggle. The scripts in `src/tasks/` were originally written as Kaggle notebook
cells and auto-detect dataset paths under `/kaggle/input`; adjust `INPUT_ROOT` at the top of each
script (or point `config.yaml` at your local copies) if running elsewhere.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

Each task can be run as a standalone script (fastest path — these are the exact scripts used to
produce the report's numbers, designed to be copy-pasted into Kaggle notebook cells one `# %%
CELL` block at a time, or executed directly with `python`):

```bash
python src/tasks/task1_bert_baseline.py
python src/tasks/task2_gnn_baseline.py
python src/tasks/task3_fusion.py
python src/tasks/task4_contrastive_retrieval.py
```

Or via the unified CLI, which reads hyperparameters from `config.yaml`:

```bash
python src/train.py --task 1
python src/train.py --task 2
python src/train.py --task 3 --variant cross_attention   # or bert_only / gnn_only / early_concat
python src/train.py --task 4
```

Evaluate saved predictions:

```bash
python src/evaluate.py --npz path/to/predictions.npz --kind multilabel
```

## Key methodology notes

- **Label-leakage prevention**: matched aspect/tag phrases are masked (`[MASK]`) out of MusicCaps
  captions before tokenization (Tasks 1, 3, 4), forcing BERT to infer tags from context rather than
  literal keyword overlap.
- **No artist leakage**: Task 2 uses FMA's official split column, not a random split.
- **Graphs**: nodes = fixed-length audio windows (5s FMA / 2s MusicCaps); edges = temporal
  adjacency + chroma cosine similarity > 0.85.
- **Reproducibility**: fixed random seed (42) throughout; all cached graphs and checkpoints are
  intended to live under `data/processed/` and are not committed to version control by default.

## Citation

If you use this code, please cite the accompanying report (`report/final_report.pdf`) and the
underlying datasets/methods:

- Agostinelli et al., *MusicLM: Generating Music From Text*, arXiv:2301.11325, 2023 (MusicCaps).
- Defferrard et al., *FMA: A Dataset For Music Analysis*, ISMIR, 2017.
- Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS, 2017 (GraphSAGE).
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL-HLT, 2019.
