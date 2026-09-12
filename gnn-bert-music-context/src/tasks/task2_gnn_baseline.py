"""
TASK 2: GNN on Music Structure Graphs (FMA-small) — FINAL COMPLETE VERSION
==============================================================================
Run this in a Kaggle Notebook.

BEFORE RUNNING:
1. Add Input: "FMA - Free Music Archive - Small & Medium" (imsparsh/...)
2. Settings -> Accelerator -> GPU T4 x2

Dataset: FMA-small (same as before) — 8,000 tracks, 8 genres.

Guideline compliance included in this version:
- OFFICIAL FMA split (not random) — the assignment requires "Use
  official FMA / MagnaTagATune splits" and "no artist leakage across
  train/test"; FMA's own ('set','split') column satisfies both.
- B1 baseline: majority-class predictor (required minimum baseline).
- B2 baseline: CNN on mel-spectrogram (required deliverable: "Comparison
  vs. CNN baseline on mel-spectrogram").
- GNN (GraphSAGE) on chroma segment-graphs — the main model.
- Accuracy/F1 curve plot vs. training epoch for both GNN and CNN.
- Final 3-way comparison table (B1 vs CNN vs GNN).

Copy-paste each "# %% CELL" block into a separate Kaggle notebook cell.
This is long — expect the full run (preprocessing + GNN + CNN training)
to take a while; preprocessing is CPU-only, so if you want to save GPU
quota, you can run CELL 1-8 (graph preprocessing) with the accelerator
set to None, then switch to GPU before CELL 9 onward.
"""

# %% CELL 1 — Install dependencies
!pip install torch-geometric -q

# %% CELL 2 — Imports & Auto-detect dataset paths
import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import classification_report, f1_score, accuracy_score

INPUT_ROOT = "/kaggle/input"

def find_path(base_dir, target_name, is_file=False):
    for root, dirs, files_in_dir in os.walk(base_dir):
        if is_file and target_name in files_in_dir:
            return os.path.join(root, target_name)
        if not is_file and target_name in dirs:
            return os.path.join(root, target_name)
    return None

def find_audio_root(base_dir):
    """Finds the folder directly containing genre subfolders (000, 001, ...)
    with .mp3 files, handling any nested 'fma_small/fma_small/...' structure."""
    for root, dirs, files_in_dir in os.walk(base_dir):
        if any(f.endswith('.mp3') for f in files_in_dir):
            return os.path.dirname(root)
    return None

FMA_SMALL_CANDIDATE = find_path(INPUT_ROOT, "fma_small")
FMA_AUDIO_DIR = find_audio_root(FMA_SMALL_CANDIDATE) if FMA_SMALL_CANDIDATE else find_audio_root(INPUT_ROOT)
TRACKS_CSV = find_path(INPUT_ROOT, "tracks.csv", is_file=True)

print(f"FMA_AUDIO_DIR: {FMA_AUDIO_DIR}")
print(f"TRACKS_CSV: {TRACKS_CSV}")
assert FMA_AUDIO_DIR is not None, "fma_small mp3 files not found — did you add the dataset via 'Add Input'?"
assert TRACKS_CSV is not None, "tracks.csv not found — did you add the dataset via 'Add Input'?"

WORKING_DIR = "/kaggle/working"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SEGMENT_DURATION = 5.0
SAMPLE_RATE = 22050
N_CHROMA = 12
SIMILARITY_THRESHOLD = 0.85
N_MELS = 128
FIXED_FRAMES = 640   # ~15s worth of frames at hop_length=512, fixed size for CNN input

# %% CELL 3 — Load metadata (genre labels + OFFICIAL FMA split)
tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
small_subset = tracks[tracks[('set', 'subset')] == 'small']
genre_labels = small_subset[('track', 'genre_top')].dropna()
official_split = small_subset[('set', 'split')]   # 'training' / 'validation' / 'test'

unique_genres = sorted(genre_labels.unique())
genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
NUM_CLASSES = len(unique_genres)
print(f"Total genres: {NUM_CLASSES}")
print(unique_genres)
print(f"\nOfficial split distribution:\n{official_split.loc[genre_labels.index].value_counts()}")

def track_id_to_path(track_id):
    tid_str = f"{track_id:06d}"
    return os.path.join(FMA_AUDIO_DIR, tid_str[:3], f"{tid_str}.mp3")

# ============================================================
# PART A: GNN on Chroma Segment-Graphs (main model)
# ============================================================

# %% CELL 4 — Audio to Segment Graph builder
def build_segment_graph(audio_path, segment_duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    """
    Builds a segment-graph from one audio file:
    - Node = chroma feature (mean-pooled, 12-dim) for each 5-second window
    - Edge = temporal adjacency + cosine similarity above threshold
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=30)
    except Exception:
        return None
    if len(y) < sr:
        return None

    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    frames_per_segment = int(segment_duration * sr / hop_length)
    n_segments = max(1, chroma.shape[1] // frames_per_segment)

    node_features = []
    for i in range(n_segments):
        start = i * frames_per_segment
        end = min(start + frames_per_segment, chroma.shape[1])
        node_features.append(chroma[:, start:end].mean(axis=1))
    if len(node_features) < 2:
        return None
    node_features = np.stack(node_features)

    edges = []
    for i in range(len(node_features)):
        if i + 1 < len(node_features):
            edges.append([i, i + 1]); edges.append([i + 1, i])
        for j in range(i + 1, len(node_features)):
            sim = np.dot(node_features[i], node_features[j]) / (
                np.linalg.norm(node_features[i]) * np.linalg.norm(node_features[j]) + 1e-8)
            if sim > SIMILARITY_THRESHOLD:
                edges.append([i, j]); edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    return x, edge_index

# %% CELL 5 — Preprocess all tracks into cached graph files (reuse if available)
GRAPH_CACHE_DIR = os.path.join(WORKING_DIR, "fma_graphs_cache")
EXISTING_GRAPH_CACHE = find_path(INPUT_ROOT, "fma_graphs_cache")
if EXISTING_GRAPH_CACHE is not None and len(os.listdir(EXISTING_GRAPH_CACHE)) > 0:
    GRAPH_CACHE_DIR = EXISTING_GRAPH_CACHE
    print(f"Reusing existing graph cache: {GRAPH_CACHE_DIR} ({len(os.listdir(GRAPH_CACHE_DIR))} files)")
else:
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

def preprocess_all_tracks():
    processed, skipped = 0, 0
    for track_id, genre in genre_labels.items():
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{track_id}.pt")
        if os.path.exists(cache_path):
            continue
        audio_path = track_id_to_path(track_id)
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        result = build_segment_graph(audio_path)
        if result is None:
            skipped += 1
            continue
        x, edge_index = result
        y = torch.tensor([genre_to_idx[genre]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        torch.save(data, cache_path)
        processed += 1
        if processed % 200 == 0:
            print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}")

if EXISTING_GRAPH_CACHE is None:
    preprocess_all_tracks()

# %% CELL 6 — Dataset class + OFFICIAL split (not random)
class FMAGraphDataset(PyGDataset):
    def __init__(self, cache_dir):
        super().__init__()
        self.files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')]
        self.track_ids = [int(os.path.basename(f).replace('.pt', '')) for f in self.files]

    def len(self):
        return len(self.files)

    def get(self, idx):
        return torch.load(self.files[idx], weights_only=False)

full_dataset = FMAGraphDataset(GRAPH_CACHE_DIR)

train_indices, val_indices, test_indices = [], [], []
for i, tid in enumerate(full_dataset.track_ids):
    split_label = official_split.get(tid, "training")
    if split_label == "training":
        train_indices.append(i)
    elif split_label == "validation":
        val_indices.append(i)
    else:
        test_indices.append(i)

train_set = torch.utils.data.Subset(full_dataset, train_indices)
val_set = torch.utils.data.Subset(full_dataset, val_indices)
test_set = torch.utils.data.Subset(full_dataset, test_indices)

train_loader = PyGDataLoader(train_set, batch_size=32, shuffle=True)
val_loader = PyGDataLoader(val_set, batch_size=32)
test_loader = PyGDataLoader(test_set, batch_size=32)
print(f"GNN — Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)} (official FMA split)")

# %% CELL 7 — B1 BASELINE: Majority-class predictor
majority_genre = genre_labels.loc[[full_dataset.track_ids[i] for i in train_indices]].value_counts().idxmax()
test_true_genres = [genre_labels[full_dataset.track_ids[i]] for i in test_indices]
b1_preds = [majority_genre] * len(test_true_genres)

b1_acc = accuracy_score(test_true_genres, b1_preds)
b1_macro_f1 = f1_score(test_true_genres, b1_preds, average='macro', zero_division=0)
print(f"\nB1 Majority-Class Baseline — Test Accuracy: {b1_acc:.4f}, Macro-F1: {b1_macro_f1:.4f}")
print(f"(Majority genre: {majority_genre})")

# %% CELL 8 — GraphSAGE Model
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels=N_CHROMA, hidden_channels=64, num_classes=NUM_CLASSES, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_mean_pool(x, batch)
        return self.classifier(g)

gnn_model = GraphSAGEClassifier().to(device)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001, weight_decay=5e-4)
gnn_criterion = nn.CrossEntropyLoss()
GNN_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "task2_gnn_checkpoint.pt")

def gnn_train_epoch():
    gnn_model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        gnn_optimizer.zero_grad()
        out = gnn_model(batch.x, batch.edge_index, batch.batch)
        loss = gnn_criterion(out, batch.y)
        loss.backward()
        gnn_optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def gnn_evaluate(loader):
    gnn_model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = gnn_model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total

# %% CELL 9 — GNN Training loop (with history for the required curve plot)
gnn_history = {"epoch": [], "train_loss": [], "val_acc": []}
start_epoch = 0
if os.path.exists(GNN_CHECKPOINT_PATH):
    ckpt = torch.load(GNN_CHECKPOINT_PATH, weights_only=False, map_location=device)
    gnn_model.load_state_dict(ckpt['model_state'])
    gnn_optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    gnn_history = ckpt.get('history', gnn_history)
    print(f"Resumed from epoch {start_epoch}")

NUM_EPOCHS_GNN = 50
for epoch in range(start_epoch, NUM_EPOCHS_GNN):
    train_loss = gnn_train_epoch()
    val_acc = gnn_evaluate(val_loader)
    print(f"[GNN] Epoch {epoch+1}/{NUM_EPOCHS_GNN} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    gnn_history["epoch"].append(epoch + 1)
    gnn_history["train_loss"].append(train_loss)
    gnn_history["val_acc"].append(val_acc)

    torch.save({
        'epoch': epoch, 'model_state': gnn_model.state_dict(),
        'optimizer_state': gnn_optimizer.state_dict(), 'history': gnn_history,
    }, GNN_CHECKPOINT_PATH)

# %% CELL 10 — GNN Accuracy curve plot (required visualization)
plt.figure(figsize=(8, 5))
plt.plot(gnn_history["epoch"], gnn_history["val_acc"], marker='o', color='tab:blue')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Task 2 (GNN): Validation Accuracy vs. Training Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, "task2_gnn_accuracy_curve.png"), dpi=150)
plt.show()

# %% CELL 11 — GNN Final Test Evaluation
@torch.no_grad()
def gnn_get_predictions(loader):
    gnn_model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = gnn_model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    return all_preds, all_labels

gnn_test_acc = gnn_evaluate(test_loader)
gnn_preds, gnn_labels = gnn_get_predictions(test_loader)
gnn_macro_f1 = f1_score(gnn_labels, gnn_preds, average='macro')
gnn_micro_f1 = f1_score(gnn_labels, gnn_preds, average='micro')

print(f"\nGNN Final Test Accuracy: {gnn_test_acc:.4f}")
print(f"GNN Macro-F1: {gnn_macro_f1:.4f}")
print(f"GNN Micro-F1: {gnn_micro_f1:.4f}")
print(classification_report(gnn_labels, gnn_preds, target_names=unique_genres))

# ============================================================
# PART B: CNN Baseline (B2) on Mel-Spectrogram
# ============================================================

# %% CELL 12 — Mel-spectrogram extraction (reuse if cached)
MEL_CACHE_DIR = os.path.join(WORKING_DIR, "mel_cache")
EXISTING_MEL_CACHE = find_path(INPUT_ROOT, "mel_cache")
if EXISTING_MEL_CACHE is not None and len(os.listdir(EXISTING_MEL_CACHE)) > 0:
    MEL_CACHE_DIR = EXISTING_MEL_CACHE
    print(f"Reusing existing mel cache: {MEL_CACHE_DIR} ({len(os.listdir(MEL_CACHE_DIR))} files)")
else:
    os.makedirs(MEL_CACHE_DIR, exist_ok=True)

def extract_mel(audio_path):
    try:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=15)
    except Exception:
        return None
    if len(y) < SAMPLE_RATE:
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]
    return mel_db.astype(np.float32)

def preprocess_mel_all():
    processed, skipped = 0, 0
    for track_id, genre in genre_labels.items():
        cache_path = os.path.join(MEL_CACHE_DIR, f"{track_id}.npy")
        if os.path.exists(cache_path):
            continue
        audio_path = track_id_to_path(track_id)
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        mel = extract_mel(audio_path)
        if mel is None:
            skipped += 1
            continue
        np.save(cache_path, mel)
        processed += 1
        if processed % 500 == 0:
            print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}")

if EXISTING_MEL_CACHE is None:
    preprocess_mel_all()

# %% CELL 13 — CNN Dataset class using the SAME official split
valid_mel_ids = [tid for tid in genre_labels.index if os.path.exists(os.path.join(MEL_CACHE_DIR, f"{tid}.npy"))]
print(f"Usable tracks for CNN: {len(valid_mel_ids)}")

mel_train_ids = [tid for tid in valid_mel_ids if official_split.get(tid) == "training"]
mel_val_ids = [tid for tid in valid_mel_ids if official_split.get(tid) == "validation"]
mel_test_ids = [tid for tid in valid_mel_ids if official_split.get(tid) == "test"]
print(f"CNN — Train: {len(mel_train_ids)}, Val: {len(mel_val_ids)}, Test: {len(mel_test_ids)} (official FMA split)")

class MelDataset(TorchDataset):
    def __init__(self, track_ids):
        self.track_ids = track_ids

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, i):
        tid = self.track_ids[i]
        mel = np.load(os.path.join(MEL_CACHE_DIR, f"{tid}.npy"))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        x = torch.tensor(mel, dtype=torch.float).unsqueeze(0)
        y = torch.tensor(genre_to_idx[genre_labels[tid]], dtype=torch.long)
        return x, y

cnn_train_loader = TorchDataLoader(MelDataset(mel_train_ids), batch_size=32, shuffle=True)
cnn_val_loader = TorchDataLoader(MelDataset(mel_val_ids), batch_size=32)
cnn_test_loader = TorchDataLoader(MelDataset(mel_test_ids), batch_size=32)

# %% CELL 14 — CNN Model + Training
class MelCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv(x))

cnn_model = MelCNN().to(device)
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_criterion = nn.CrossEntropyLoss()
CNN_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "task2_cnn_checkpoint.pt")

def cnn_train_epoch():
    cnn_model.train()
    total_loss = 0
    for x, y in cnn_train_loader:
        x, y = x.to(device), y.to(device)
        cnn_optimizer.zero_grad()
        out = cnn_model(x)
        loss = cnn_criterion(out, y)
        loss.backward()
        cnn_optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(cnn_train_loader.dataset)

@torch.no_grad()
def cnn_evaluate(loader):
    cnn_model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = cnn_model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

cnn_history = {"epoch": [], "train_loss": [], "val_acc": []}
start_epoch_cnn = 0
if os.path.exists(CNN_CHECKPOINT_PATH):
    ckpt = torch.load(CNN_CHECKPOINT_PATH, weights_only=False, map_location=device)
    cnn_model.load_state_dict(ckpt['model_state'])
    cnn_optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch_cnn = ckpt['epoch'] + 1
    cnn_history = ckpt.get('history', cnn_history)
    print(f"Resumed CNN from epoch {start_epoch_cnn}")

NUM_EPOCHS_CNN = 30
for epoch in range(start_epoch_cnn, NUM_EPOCHS_CNN):
    train_loss = cnn_train_epoch()
    val_acc = cnn_evaluate(cnn_val_loader)
    print(f"[CNN] Epoch {epoch+1}/{NUM_EPOCHS_CNN} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    cnn_history["epoch"].append(epoch + 1)
    cnn_history["train_loss"].append(train_loss)
    cnn_history["val_acc"].append(val_acc)

    torch.save({
        'epoch': epoch, 'model_state': cnn_model.state_dict(),
        'optimizer_state': cnn_optimizer.state_dict(), 'history': cnn_history,
    }, CNN_CHECKPOINT_PATH)

# %% CELL 15 — CNN Accuracy curve + Final Evaluation
plt.figure(figsize=(8, 5))
plt.plot(cnn_history["epoch"], cnn_history["val_acc"], marker='s', color='tab:orange')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Task 2 (CNN Baseline): Validation Accuracy vs. Training Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, "task2_cnn_accuracy_curve.png"), dpi=150)
plt.show()

@torch.no_grad()
def cnn_get_predictions(loader):
    cnn_model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        pred = cnn_model(x).argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.numpy())
    return all_preds, all_labels

cnn_test_acc = cnn_evaluate(cnn_test_loader)
cnn_preds, cnn_labels = cnn_get_predictions(cnn_test_loader)
cnn_macro_f1 = f1_score(cnn_labels, cnn_preds, average='macro')
cnn_micro_f1 = f1_score(cnn_labels, cnn_preds, average='micro')

print(f"\nCNN Baseline Final Test Accuracy: {cnn_test_acc:.4f}")
print(f"CNN Baseline Macro-F1: {cnn_macro_f1:.4f}")
print(f"CNN Baseline Micro-F1: {cnn_micro_f1:.4f}")
print(classification_report(cnn_labels, cnn_preds, target_names=unique_genres))

# %% CELL 16 — FINAL 3-WAY COMPARISON TABLE + Save all results
print("\n" + "=" * 60)
print("TASK 2 — FINAL BASELINE COMPARISON (official FMA split)")
print("=" * 60)
print(f"{'Model':<25}{'Accuracy':<12}{'Macro-F1':<12}")
print(f"{'B1 (Majority-class)':<25}{b1_acc:<12.4f}{b1_macro_f1:<12.4f}")
print(f"{'B2 (CNN mel-spectrogram)':<25}{cnn_test_acc:<12.4f}{cnn_macro_f1:<12.4f}")
print(f"{'GNN (GraphSAGE, ours)':<25}{gnn_test_acc:<12.4f}{gnn_macro_f1:<12.4f}")

results = {
    "b1_baseline": {"accuracy": float(b1_acc), "macro_f1": float(b1_macro_f1)},
    "b2_cnn_baseline": {"accuracy": float(cnn_test_acc), "macro_f1": float(cnn_macro_f1), "micro_f1": float(cnn_micro_f1)},
    "gnn_model": {"accuracy": float(gnn_test_acc), "macro_f1": float(gnn_macro_f1), "micro_f1": float(gnn_micro_f1)},
    "genre_names": unique_genres,
    "gnn_history": gnn_history,
    "cnn_history": cnn_history,
    "split_type": "official_fma_split",
}
RESULTS_PATH = os.path.join(WORKING_DIR, "task2_results.json")
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved at: {RESULTS_PATH}")
print("\nIMPORTANT: click 'Save Version' (top right) now, or these files")
print("will be lost when this session ends.")