"""
TASK 4: Contrastive GNN-BERT for MusicCaps Cross-Modal Retrieval — FIXED / SPEC-COMPLIANT
============================================================================================
Run this in a Kaggle Notebook (can be the same notebook as Task 3, or a
separate one — this script is written to run as a SEPARATE notebook that
pulls Task 3's outputs in via Kaggle Input, since that's how you're set up).

BEFORE RUNNING:
1. Add Input: "MusicCaps" (googleai/musiccaps)
2. Add Input: "MusicCapsAudio5308" (ayushsharma4045/musiccapsaudio5308)
3. Add Input: your Task 3 notebook's saved output (the committed version that
   has task3_checkpoint.pt, task3_graphs_cache/, task3_results.json in
   /kaggle/working). Kaggle -> "+ Add Input" -> "Your Work" -> select that
   notebook's latest version. This is what lets this script SKIP graph
   preprocessing and reuse Task 3's trained GNN + BERT weights instead of
   recomputing anything.
4. Settings -> Accelerator -> GPU T4 x2

WHAT CHANGED vs. the previous version, and why
------------------------------------------------
Spec Section 4.4 lists four deliverables for Task 4. The previous script
only satisfied three of them:

  1. Dual-encoder GNN-BERT with contrastive training   -> unchanged (Cell 10-13)
  2. Retrieval evaluation table (R@1/5/10, both dirs)   -> unchanged, now
                                                            printed as a clean
                                                            table (Cell 14)
  3. 10 qualitative retrieval examples                  -> unchanged (Cell 15)
  4. Zero-shot tag prediction from captions vs.          -> NEW (Cell 16):
     Task 3 supervised model                                this was missing
                                                              entirely before

Also added: a human-evaluation template (Cell 17), because spec Section 6
("Human evaluation (Task 4). Minimum 5 listeners rate whether retrieved
clip matches caption on scale [1,5]") can't be automated — this exports a
CSV with the queries + top-3 matches + blank rating columns for 5 listeners
to fill in by hand, so you have something to attach to the report.

Copy-paste each "# %% CELL" block into a separate Kaggle notebook cell.
"""

# %% CELL 1 — Install dependencies
!pip install torch-geometric transformers scikit-learn -q

# %% CELL 2 — Auto-detect dataset + Task 3 output paths
import os

INPUT_ROOT = "/kaggle/input"

def find_path(base_dir, target_name, is_file=False):
    for root, dirs, files_in_dir in os.walk(base_dir):
        if is_file and target_name in files_in_dir:
            return os.path.join(root, target_name)
        if not is_file and target_name in dirs:
            return os.path.join(root, target_name)
    return None

def find_wav_root(base_dir):
    for root, dirs, files_in_dir in os.walk(base_dir):
        if any(f.endswith('.wav') for f in files_in_dir):
            return root
    return None

CSV_PATH = find_path(INPUT_ROOT, "musiccaps-public.csv", is_file=True)
AUDIO_FILES_CANDIDATE = find_path(INPUT_ROOT, "audioFiles")
AUDIO_DIR = find_wav_root(AUDIO_FILES_CANDIDATE) if AUDIO_FILES_CANDIDATE else find_wav_root(INPUT_ROOT)

print(f"CSV_PATH: {CSV_PATH}")
print(f"AUDIO_DIR: {AUDIO_DIR}")
assert CSV_PATH is not None, "musiccaps-public.csv not found — add the MusicCaps dataset"
assert AUDIO_DIR is not None, "No .wav files found — add the MusicCapsAudio5308 dataset"

WORKING_DIR = "/kaggle/working"

# Task 3 artifacts — all pulled from Input, none of this gets recomputed
TASK3_CHECKPOINT = find_path(INPUT_ROOT, "task3_checkpoint.pt", is_file=True)
TASK3_GRAPH_CACHE = find_path(INPUT_ROOT, "task3_graphs_cache")
TASK3_RESULTS = find_path(INPUT_ROOT, "task3_results.json", is_file=True)
print(f"Task 3 checkpoint (warm-start GNN+BERT): {TASK3_CHECKPOINT}")
print(f"Task 3 graph cache (skip preprocessing): {TASK3_GRAPH_CACHE}")
print(f"Task 3 results.json (tag vocab + supervised scores): {TASK3_RESULTS}")

# %% CELL 3 — Imports & Config
import re
import ast
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, average_precision_score
from collections import Counter

MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 2e-5
SEGMENT_DURATION = 2.0
SAMPLE_RATE = 22050
N_CHROMA = 12
SIMILARITY_THRESHOLD = 0.85
TEMPERATURE = 0.07
EMBED_DIM = 256
TOP_N_TAGS = 50  # only used as a fallback if task3_results.json isn't found

CHECKPOINT_PATH = os.path.join(WORKING_DIR, "task4_checkpoint.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% CELL 4 — Load captions + tag vocabulary (reuse Task 3's exact vocabulary)
df = pd.read_csv(CSV_PATH)
df['aspects_parsed'] = df['aspect_list'].apply(ast.literal_eval)
df['aspects_parsed'] = df['aspects_parsed'].apply(lambda lst: [a.strip().lower() for a in lst])

if TASK3_RESULTS is not None:
    with open(TASK3_RESULTS) as f:
        task3_results = json.load(f)
    TAG_NAMES = task3_results['tag_names']
    print(f"Loaded {len(TAG_NAMES)} tags from Task 3's results.json (same vocabulary — required "
          f"for the zero-shot-vs-supervised comparison in Cell 16 to be apples-to-apples).")
else:
    # Fallback: recompute the same way Task 3 did. Only used if task3_results.json
    # wasn't added as an Input — the zero-shot-vs-Task-3 comparison in Cell 16 will
    # be skipped in that case since there's no supervised score to compare against.
    print("WARNING: task3_results.json not found — recomputing tag vocabulary from "
          "scratch. This should match Task 3's vocabulary exactly since it's the same "
          "deterministic top-N-by-frequency logic, but the Cell 16 comparison needs "
          "Task 3's actual saved scores, so add that file as an Input if you want it.")
    all_aspects = []
    for lst in df['aspects_parsed']:
        all_aspects.extend(lst)
    counter = Counter(all_aspects)
    TAG_NAMES = [tag for tag, _ in counter.most_common(TOP_N_TAGS)]
    task3_results = None

NUM_TAGS = len(TAG_NAMES)

def build_labels(row):
    aspects_present = set(row['aspects_parsed'])
    labels = np.zeros(NUM_TAGS, dtype=np.float32)
    for i, tag in enumerate(TAG_NAMES):
        if tag in aspects_present:
            labels[i] = 1.0
    return labels

df['labels'] = df.apply(build_labels, axis=1)

# %% CELL 5 — Match ytid to audio files (same robust logic as Task 3)
def normalize_id(s):
    s = s[:-4] if s.endswith('.wav') else s
    if s.startswith('-') or s.startswith('_'):
        s = s[1:]
    return s

wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
normalized_lookup = {normalize_id(f): f for f in wav_files}

def find_audio_file(ytid):
    key = normalize_id(ytid)
    return os.path.join(AUDIO_DIR, normalized_lookup[key]) if key in normalized_lookup else None

df['audio_path'] = df['ytid'].apply(find_audio_file)
df_matched = df[df['audio_path'].notna()].reset_index(drop=True)
print(f"Matched {len(df_matched)} / {len(df)} captions to audio files")
assert len(df_matched) > 0, "No matches found"

# %% CELL 6 — Audio to Segment Graph builder (identical to Task 3, for cache compatibility)
# Only used if the cache is missing — with Task 3's cache mounted via Input, this
# function is defined but preprocess_all() below should find everything cached
# and never actually call it.
def build_segment_graph(audio_path, segment_duration=SEGMENT_DURATION, sr=SAMPLE_RATE):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=10)
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

# %% CELL 7 — Locate the graph cache (reuse Task 3's — no preprocessing here)
GRAPH_CACHE_DIR = TASK3_GRAPH_CACHE if TASK3_GRAPH_CACHE and len(os.listdir(TASK3_GRAPH_CACHE)) > 0 else None

if GRAPH_CACHE_DIR is None:
    print("Task 3 graph cache not found via Input — building it now (this is the "
          "slow path you're trying to avoid; add Task 3's output as an Input to skip this).")
    GRAPH_CACHE_DIR = os.path.join(WORKING_DIR, "task4_graphs_cache")
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    processed, skipped = 0, 0
    for idx, row in df_matched.iterrows():
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{idx}.pt")
        if os.path.exists(cache_path):
            continue
        result = build_segment_graph(row['audio_path'])
        if result is None:
            skipped += 1
            continue
        x, edge_index = result
        torch.save({'x': x, 'edge_index': edge_index}, cache_path)
        processed += 1
    print(f"DONE. Processed: {processed}, Skipped: {skipped}")
else:
    print(f"Reusing Task 3's graph cache: {GRAPH_CACHE_DIR} — no preprocessing needed.")

valid_indices = [i for i in range(len(df_matched)) if os.path.exists(os.path.join(GRAPH_CACHE_DIR, f"{i}.pt"))]
df_final = df_matched.iloc[valid_indices].reset_index(drop=True)
graph_files = [os.path.join(GRAPH_CACHE_DIR, f"{i}.pt") for i in valid_indices]
print(f"Final usable samples: {len(df_final)}")

# %% CELL 8 — Train/Val/Test split (same seed as Task 3 -> same split on the same data)
from sklearn.model_selection import train_test_split

indices = list(range(len(df_final)))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# %% CELL 9 — Dataset class (now also carries labels, needed for Cell 16's zero-shot eval)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class RetrievalDataset(TorchDataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = df_final.iloc[idx]
        graph_data = torch.load(graph_files[idx], weights_only=False, map_location='cpu')
        encoding = tokenizer(
            row['caption'], truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        return {
            'x': graph_data['x'],
            'edge_index': graph_data['edge_index'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['labels'], dtype=torch.float),
        }

def collate_fn(batch):
    x_list, edge_index_list, batch_idx = [], [], []
    node_offset = 0
    for i, item in enumerate(batch):
        x_list.append(item['x'])
        edge_index_list.append(item['edge_index'] + node_offset)
        batch_idx.extend([i] * item['x'].shape[0])
        node_offset += item['x'].shape[0]
    return {
        'x': torch.cat(x_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'batch': torch.tensor(batch_idx, dtype=torch.long),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }

# shuffle=False for val/test so embeddings/labels align with df_final order for retrieval eval
train_loader = DataLoader(RetrievalDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(RetrievalDataset(val_idx), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(RetrievalDataset(test_idx), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# %% CELL 10 — Dual Encoder Model
class GNNEncoder(nn.Module):
    def __init__(self, in_channels=N_CHROMA, hidden_channels=64, num_layers=3, out_dim=EMBED_DIM):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(hidden_channels, out_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_mean_pool(x, batch)
        return F.normalize(self.proj(g), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.proj(cls), dim=-1)


gnn_encoder = GNNEncoder().to(device)
text_encoder = TextEncoder().to(device)

if TASK3_CHECKPOINT is not None:
    task3_ckpt = torch.load(TASK3_CHECKPOINT, weights_only=False, map_location=device)
    gnn_state = {k.replace('gnn.', ''): v for k, v in task3_ckpt['model_state'].items() if k.startswith('gnn.')}
    bert_state = {k.replace('bert.', ''): v for k, v in task3_ckpt['model_state'].items() if k.startswith('bert.')}
    gnn_encoder.load_state_dict(gnn_state, strict=False)
    text_encoder.bert.load_state_dict(bert_state, strict=False)
    print("Warm-started GNN and BERT encoders from Task 3 checkpoint.")
else:
    print("No Task 3 checkpoint found — training both encoders from scratch/pretrained BERT.")

# %% CELL 11 — InfoNCE Contrastive Loss + Training setup
optimizer = torch.optim.AdamW(
    list(gnn_encoder.parameters()) + list(text_encoder.parameters()), lr=LR
)

def info_nce_loss(audio_emb, text_emb, temperature=TEMPERATURE):
    logits = audio_emb @ text_emb.t() / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.t(), labels)
    return (loss_a2t + loss_t2a) / 2

def train_epoch():
    gnn_encoder.train()
    text_encoder.train()
    total_loss = 0
    for batch in train_loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        b_idx = batch['batch'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        audio_emb = gnn_encoder(x, edge_index, b_idx)
        text_emb = text_encoder(input_ids, attention_mask)
        loss = info_nce_loss(audio_emb, text_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# %% CELL 12 — Retrieval evaluation (Recall@1/5/10, both directions) + embedding/label extraction
@torch.no_grad()
def get_all_embeddings_and_labels(loader):
    gnn_encoder.eval()
    text_encoder.eval()
    audio_embs, text_embs, labels_list = [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        b_idx = batch['batch'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        audio_embs.append(gnn_encoder(x, edge_index, b_idx).cpu())
        text_embs.append(text_encoder(input_ids, attention_mask).cpu())
        labels_list.append(batch['labels'])
    return torch.cat(audio_embs), torch.cat(text_embs), torch.cat(labels_list).numpy()

def compute_recall_at_k(audio_embs, text_embs, k_values=(1, 5, 10)):
    sim = text_embs @ audio_embs.t()
    n = sim.shape[0]
    ranks_t2a = []
    for i in range(n):
        order = torch.argsort(sim[i], descending=True)
        rank = (order == i).nonzero(as_tuple=True)[0].item()
        ranks_t2a.append(rank)
    ranks_t2a = np.array(ranks_t2a)

    sim_a2t = audio_embs @ text_embs.t()
    ranks_a2t = []
    for i in range(n):
        order = torch.argsort(sim_a2t[i], descending=True)
        rank = (order == i).nonzero(as_tuple=True)[0].item()
        ranks_a2t.append(rank)
    ranks_a2t = np.array(ranks_a2t)

    results = {}
    for k in k_values:
        results[f"caption_to_audio_R@{k}"] = float((ranks_t2a < k).mean())
        results[f"audio_to_caption_R@{k}"] = float((ranks_a2t < k).mean())
    return results

# %% CELL 13 — Resume + Training loop
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
    gnn_encoder.load_state_dict(ckpt['gnn_state'])
    text_encoder.load_state_dict(ckpt['text_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss = train_epoch()
    val_audio_embs, val_text_embs, _ = get_all_embeddings_and_labels(val_loader)
    val_recall = compute_recall_at_k(val_audio_embs, val_text_embs)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | "
          f"Val C2A R@10: {val_recall['caption_to_audio_R@10']:.4f} | "
          f"Val A2C R@10: {val_recall['audio_to_caption_R@10']:.4f}")

    torch.save({
        'epoch': epoch,
        'gnn_state': gnn_encoder.state_dict(),
        'text_state': text_encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, CHECKPOINT_PATH)

# %% CELL 14 — Final Test Evaluation (spec deliverable #2: retrieval table)
test_audio_embs, test_text_embs, test_labels = get_all_embeddings_and_labels(test_loader)
test_recall = compute_recall_at_k(test_audio_embs, test_text_embs)

print(f"\n{'='*55}\nTASK 4 — RETRIEVAL EVALUATION TABLE (MusicCaps test split)\n{'='*55}")
print(f"{'Metric':<25}{'Value':<10}")
for k, v in test_recall.items():
    print(f"{k:<25}{v:<10.4f}")

# %% CELL 15 — 10 qualitative retrieval examples (spec deliverable #3)
test_df_final = df_final.iloc[test_idx].reset_index(drop=True)
sim = test_text_embs @ test_audio_embs.t()

qualitative_examples = []
print("\n--- Qualitative Retrieval Examples (caption -> top-3 audio matches) ---")
np.random.seed(7)
sample_query_indices = np.random.choice(len(test_idx), min(10, len(test_idx)), replace=False)
for qi in sample_query_indices:
    top3 = torch.argsort(sim[qi], descending=True)[:3].tolist()
    query_caption = test_df_final.iloc[qi]['caption']
    print(f"\nQuery caption: {query_caption[:100]}...")
    matches = []
    for rank, ai in enumerate(top3):
        is_correct = (ai == qi)
        fname = os.path.basename(test_df_final.iloc[ai]['audio_path'])
        marker = " <-- CORRECT MATCH" if is_correct else ""
        print(f"  Top-{rank+1}: {fname}{marker}")
        matches.append({"rank": rank + 1, "audio_file": fname, "is_correct_match": bool(is_correct)})
    qualitative_examples.append({"query_caption": query_caption, "top3_matches": matches})

# %% CELL 16 — NEW: Zero-shot tag prediction from captions vs. Task 3 supervised model
# spec deliverable #4. Uses the CONTRASTIVELY-TRAINED text encoder to embed each of
# the 50 tag names, then classifies each test clip's audio embedding by cosine
# similarity to those tag embeddings — no labeled fine-tuning for this task, hence
# "zero-shot". The threshold is tuned on the val split (to pick a fair operating
# point) and then applied once to test. Compared directly against Task 3's
# supervised cross-attention model's saved scores.
tag_texts = [f"the music has the following quality: {tag}" for tag in TAG_NAMES]
tag_encodings = tokenizer(tag_texts, truncation=True, padding='max_length',
                           max_length=32, return_tensors='pt')
with torch.no_grad():
    text_encoder.eval()
    tag_embeddings = text_encoder(
        tag_encodings['input_ids'].to(device), tag_encodings['attention_mask'].to(device)
    )  # (NUM_TAGS, EMBED_DIM), L2-normalized

@torch.no_grad()
def zero_shot_scores(audio_embs):
    # audio_embs: (N, EMBED_DIM), L2-normalized. tag_embeddings: (NUM_TAGS, EMBED_DIM).
    return (audio_embs.to(device) @ tag_embeddings.t()).cpu().numpy()  # (N, NUM_TAGS) cosine sim

val_audio_embs, _, val_labels = get_all_embeddings_and_labels(val_loader)
val_scores = zero_shot_scores(val_audio_embs)
test_scores = zero_shot_scores(test_audio_embs)

# Tune a single global threshold on val to maximize macro-F1
best_threshold, best_val_f1 = 0.0, -1.0
for t in np.linspace(val_scores.min(), val_scores.max(), 50):
    preds = (val_scores > t).astype(np.float32)
    f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
    if f1 > best_val_f1:
        best_val_f1, best_threshold = f1, t
print(f"\nZero-shot threshold tuned on val: {best_threshold:.4f} (val Macro-F1: {best_val_f1:.4f})")

test_preds = (test_scores > best_threshold).astype(np.float32)
zs_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
zs_micro_f1 = f1_score(test_labels, test_preds, average='micro', zero_division=0)
valid_tags = [k for k in range(test_labels.shape[1]) if test_labels[:, k].sum() > 0]
zs_auc_pr = (average_precision_score(test_labels[:, valid_tags], test_scores[:, valid_tags], average='macro')
             if valid_tags else float('nan'))

zero_shot_results = {"macro_f1": zs_macro_f1, "micro_f1": zs_micro_f1, "auc_pr": zs_auc_pr,
                      "threshold": float(best_threshold)}

print(f"\n{'='*65}\nTASK 4 ZERO-SHOT vs. TASK 3 SUPERVISED (spec deliverable #4)\n{'='*65}")
print(f"{'Model':<35}{'Macro-F1':<12}{'Micro-F1':<12}{'AUC-PR':<12}")
print(f"{'Task 4: Zero-shot (contrastive)':<35}{zs_macro_f1:<12.4f}{zs_micro_f1:<12.4f}{zs_auc_pr:<12.4f}")
if task3_results is not None and 'cross_attention' in task3_results.get('ablation_results', {}):
    t3 = task3_results['ablation_results']['cross_attention']
    print(f"{'Task 3: Supervised (cross-attn)':<35}{t3['macro_f1']:<12.4f}{t3['micro_f1']:<12.4f}{t3['auc_pr']:<12.4f}")
else:
    print("Task 3 supervised scores not available (task3_results.json not found or "
          "missing 'cross_attention' entry — add it as an Input to get this comparison).")

# %% CELL 17 — NEW: Human evaluation template (spec Section 6 requirement)
# Exports the 10 qualitative examples above as a CSV with blank rating columns
# for 5 listeners to fill in by hand (scale 1-5: does the retrieved clip match
# the caption?). This can't be automated, so this cell just prepares the sheet.
human_eval_rows = []
for ex in qualitative_examples:
    for m in ex['top3_matches']:
        human_eval_rows.append({
            "query_caption": ex['query_caption'],
            "rank": m['rank'],
            "audio_file": m['audio_file'],
            "listener_1_rating": "", "listener_2_rating": "", "listener_3_rating": "",
            "listener_4_rating": "", "listener_5_rating": "",
        })
human_eval_df = pd.DataFrame(human_eval_rows)
HUMAN_EVAL_PATH = os.path.join(WORKING_DIR, "task4_human_eval_template.csv")
human_eval_df.to_csv(HUMAN_EVAL_PATH, index=False)
print(f"Human evaluation template saved to: {HUMAN_EVAL_PATH}")
print("Have 5 listeners rate each retrieved clip 1-5 for caption match, then average per query.")

# %% CELL 18 — Save results
RESULTS_PATH = os.path.join(WORKING_DIR, "task4_results.json")
results = {
    "test_recall": test_recall,
    "zero_shot_vs_task3": {
        "task4_zero_shot": zero_shot_results,
        "task3_supervised": task3_results['ablation_results']['cross_attention'] if task3_results else None,
    },
    "qualitative_examples": qualitative_examples,
    "num_samples": len(df_final),
    "embed_dim": EMBED_DIM,
    "temperature": TEMPERATURE,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Checkpoint saved at: {CHECKPOINT_PATH}")
print(f"Results saved at: {RESULTS_PATH}")
print(f"Human eval template saved at: {HUMAN_EVAL_PATH}")
print("\nIMPORTANT: click 'Save Version' (top right) now, or these files")
print("will be lost when this session ends.")