"""
TASK 3: GNN-BERT Fusion for Multi-Context Understanding — FIXED / SPEC-COMPLIANT
==================================================================================
Run this in a Kaggle Notebook.

BEFORE RUNNING:
1. Add Input: "MusicCaps" (googleai/musiccaps) — for aspect_list/caption
2. Add Input: "MusicCapsAudio5308" (ayushsharma4045/musiccapsaudio5308) — for audio .wav files
3. (Optional) Add your Task 1 notebook's output as Input, for the BERT warm-start
4. Settings -> Accelerator -> GPU T4 x2

WHAT CHANGED vs. the previous version, and why
------------------------------------------------
The assignment spec (Section 4.3) lists five deliverables for Task 3. The
previous script only satisfied two of them. This version satisfies all five:

  1. End-to-end GNN-BERT fusion model              -> unchanged (Cell 10, 'cross_attention')
  2. Ablation: BERT-only / GNN-only / early-concat  -> NEW (Cell 12): all four
     / cross-attention                                 variants trained & compared
  3. Results (Macro-F1, AUC-PR)                     -> NEW: AUC-PR was missing
                                                        entirely; evaluate() now
                                                        returns it (Cell 11)
  4. t-SNE of z colored by genre AND mood           -> FIXED: previous version
                                                        colored by "top tag" only,
                                                        not genre/mood specifically
                                                        (Cell 14)
  5. 3 case studies showing graph paths +           -> FIXED: previous version
     caption/lyric alignment                           printed caption + audio
                                                        path only, no graph
                                                        structure or predictions
                                                        (Cell 15)

DATASET NOTE: the spec's suggested pairing for Task 3 is FMA-medium or
MagnaTagATune. This script keeps MusicCaps instead, because Task 1's BERT
checkpoint, the cached segment graphs, and Task 4's contrastive retrieval
are all already built on MusicCaps — switching now would break checkpoint
and cache reuse across the whole pipeline. If your instructor requires the
literal FMA-medium/MagnaTagATune pairing, that needs a separate data-loading
cell (different CSV, different label schema) — flag it and I'll write that
version.

MusicCaps has no explicit genre/mood columns, so "genre" and "mood" for the
t-SNE plots are derived from the tag vocabulary itself via keyword rules
(Cell 4b) — e.g. "rock", "pop", "instrumental" -> genre bucket;
"energetic", "happy", "romantic" -> mood bucket. This is a documented
approximation, not ground-truth genre/mood labels.

Copy-paste each "# %% CELL" block into a separate Kaggle notebook cell.
"""

# %% CELL 1 — Install dependencies
!pip install torch-geometric transformers scikit-learn matplotlib -q

# %% CELL 2 — Auto-detect dataset paths
import os

INPUT_ROOT = "/kaggle/input"

def find_path(base_dir, target_name, is_file=False):
    for root, dirs, files_in_dir in os.walk(base_dir):
        if is_file and target_name in files_in_dir:
            return os.path.join(root, target_name)
        if not is_file and target_name in dirs:
            return os.path.join(root, target_name)
    return None

CSV_PATH = find_path(INPUT_ROOT, "musiccaps-public.csv", is_file=True)

def find_wav_root(base_dir):
    for root, dirs, files_in_dir in os.walk(base_dir):
        if any(f.endswith('.wav') for f in files_in_dir):
            return root
    return None

AUDIO_FILES_CANDIDATE = find_path(INPUT_ROOT, "audioFiles")
AUDIO_DIR = find_wav_root(AUDIO_FILES_CANDIDATE) if AUDIO_FILES_CANDIDATE else find_wav_root(INPUT_ROOT)

print(f"CSV_PATH: {CSV_PATH}")
print(f"AUDIO_DIR: {AUDIO_DIR}")
assert CSV_PATH is not None, "musiccaps-public.csv not found — add the MusicCaps dataset"
assert AUDIO_DIR is not None, "No .wav files found — add the MusicCapsAudio5308 dataset"

WORKING_DIR = "/kaggle/working"

TASK1_CHECKPOINT = find_path(INPUT_ROOT, "task1_checkpoint.pt", is_file=True)
print(f"Task 1 checkpoint (optional warm-start): {TASK1_CHECKPOINT}")

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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, average_precision_score, classification_report
from collections import Counter

MAX_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 20          # epochs for the primary (cross-attention) model
ABLATION_EPOCHS = 10     # shorter budget for the 3 comparison variants
TOP_N_TAGS = 50
SEGMENT_DURATION = 2.0
SAMPLE_RATE = 22050
N_CHROMA = 12
SIMILARITY_THRESHOLD = 0.85

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% CELL 4 — Load captions, parse aspect_list, build tag vocabulary
df = pd.read_csv(CSV_PATH)
df['aspects_parsed'] = df['aspect_list'].apply(ast.literal_eval)
df['aspects_parsed'] = df['aspects_parsed'].apply(lambda lst: [a.strip().lower() for a in lst])

all_aspects = []
for lst in df['aspects_parsed']:
    all_aspects.extend(lst)
counter = Counter(all_aspects)
TAG_NAMES = [tag for tag, _ in counter.most_common(TOP_N_TAGS)]
NUM_TAGS = len(TAG_NAMES)
print(f"Total tags: {NUM_TAGS}")

def build_labels_and_masked_caption(row):
    aspects_present = set(row['aspects_parsed'])
    labels = np.zeros(NUM_TAGS, dtype=np.float32)
    masked = row['caption']
    for i, tag in enumerate(TAG_NAMES):
        if tag in aspects_present:
            labels[i] = 1.0
            masked = re.sub(re.escape(tag), "[MASK]", masked, flags=re.IGNORECASE)
    return labels, masked

results = df.apply(build_labels_and_masked_caption, axis=1)
df['labels'] = results.apply(lambda r: r[0])
df['masked_caption'] = results.apply(lambda r: r[1])

# %% CELL 4b — NEW: derive pseudo genre/mood buckets from the tag vocabulary
# MusicCaps has no explicit genre/mood columns. We bucket each of the top-50
# tags into a category via keyword rules, so the t-SNE plots required by the
# spec (colored by genre AND mood) have something meaningful to color by.
MOOD_KEYWORDS = ['emotional', 'passionate', 'energetic', 'groovy', 'happy', 'spirited',
                  'romantic', 'exciting', 'upbeat', 'mellow', 'fun', 'cheerful',
                  'easygoing', 'youthful', 'sad', 'calm', 'relaxing', 'dark', 'intense']
GENRE_KEYWORDS = ['rock', 'pop', 'jazz', 'hip-hop', 'hip hop', 'electro', 'electronic',
                   'folk', 'classical', 'country', 'reggae', 'blues', 'metal', 'punk',
                   'funk', 'disco', 'afrobeat', 'latin', 'instrumental']
TEMPO_KEYWORDS = ['tempo', 'uptempo']
VOCAL_KEYWORDS = ['vocal', 'voice', 'singer', 'singing']
INSTRUMENT_KEYWORDS = ['guitar', 'drum', 'bass', 'piano', 'kick', 'snare', 'hat',
                        'percussion', 'keyboard', 'synth', 'strings', 'brass', 'violin']
PRODUCTION_KEYWORDS = ['quality', 'noisy', 'amateur', 'live performance', 'mono', 'loud', 'studio']

def categorize_tag(tag):
    t = tag.lower()
    if any(k in t for k in MOOD_KEYWORDS):
        return 'mood'
    if any(k in t for k in TEMPO_KEYWORDS):
        return 'tempo'
    if any(k in t for k in VOCAL_KEYWORDS):
        return 'vocal'
    if any(k in t for k in INSTRUMENT_KEYWORDS):
        return 'instrument'
    if any(k in t for k in PRODUCTION_KEYWORDS):
        return 'production'
    if any(k in t for k in GENRE_KEYWORDS):
        return 'genre'
    return 'other'

TAG_CATEGORY = {tag: categorize_tag(tag) for tag in TAG_NAMES}
MOOD_TAG_IDX = [i for i, t in enumerate(TAG_NAMES) if TAG_CATEGORY[t] == 'mood']
GENRE_TAG_IDX = [i for i, t in enumerate(TAG_NAMES) if TAG_CATEGORY[t] == 'genre']
print(f"Mood-bucket tags ({len(MOOD_TAG_IDX)}): {[TAG_NAMES[i] for i in MOOD_TAG_IDX]}")
print(f"Genre-bucket tags ({len(GENRE_TAG_IDX)}): {[TAG_NAMES[i] for i in GENRE_TAG_IDX]}")

def dominant_label(label_vec, idx_list, fallback='none'):
    present = [TAG_NAMES[i] for i in idx_list if label_vec[i] == 1.0]
    return present[0] if present else fallback

# %% CELL 5 — Match ytid to actual audio files (robust matching)
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
assert len(df_matched) > 0, "No matches found — check AUDIO_DIR contents and ytid format"

# %% CELL 6 — Audio to Segment Graph builder (same approach as Task 2, chroma-only)
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

# %% CELL 7 — Preprocess into graph cache (skip if already cached from a previous session)
GRAPH_CACHE_DIR = os.path.join(WORKING_DIR, "task3_graphs_cache")
EXISTING_CACHE = find_path(INPUT_ROOT, "task3_graphs_cache")
if EXISTING_CACHE is not None and len(os.listdir(EXISTING_CACHE)) > 0:
    GRAPH_CACHE_DIR = EXISTING_CACHE
    print(f"Reusing existing graph cache: {GRAPH_CACHE_DIR}")
else:
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

def preprocess_all():
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
        if processed % 500 == 0:
            print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}")

if EXISTING_CACHE is None:
    preprocess_all()

valid_indices = [i for i in range(len(df_matched)) if os.path.exists(os.path.join(GRAPH_CACHE_DIR, f"{i}.pt"))]
df_final = df_matched.iloc[valid_indices].reset_index(drop=True)
graph_files = [os.path.join(GRAPH_CACHE_DIR, f"{i}.pt") for i in valid_indices]
print(f"Final usable samples: {len(df_final)}")

# %% CELL 8 — Train/Val/Test split
from sklearn.model_selection import train_test_split

indices = list(range(len(df_final)))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# %% CELL 9 — Dataset class (graph + tokenized masked caption + labels)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FusionDataset(TorchDataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = df_final.iloc[idx]
        graph_data = torch.load(graph_files[idx], weights_only=False)
        encoding = tokenizer(
            row['masked_caption'], truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        return {
            'x': graph_data['x'],
            'edge_index': graph_data['edge_index'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['labels'], dtype=torch.float)
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

train_loader = DataLoader(FusionDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(FusionDataset(val_idx), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(FusionDataset(test_idx), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# %% CELL 10 — NEW: Unified model supporting all 4 ablation variants
class GNNEncoder(nn.Module):
    """Same architecture as Task 2's GraphSAGE, outputs a graph embedding."""
    def __init__(self, in_channels=N_CHROMA, hidden_channels=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return global_mean_pool(x, batch)


class UnifiedFusionModel(nn.Module):
    """
    fusion_type in {'bert_only', 'gnn_only', 'early_concat', 'cross_attention'}.
    Required by spec Section 4.3 deliverable #2: ablation across all four.
    """
    def __init__(self, fusion_type, num_tags=NUM_TAGS, gnn_hidden=64, bert_hidden=768):
        super().__init__()
        assert fusion_type in ('bert_only', 'gnn_only', 'early_concat', 'cross_attention')
        self.fusion_type = fusion_type

        if fusion_type != 'bert_only':
            self.gnn = GNNEncoder(hidden_channels=gnn_hidden)
        if fusion_type != 'gnn_only':
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if fusion_type == 'cross_attention':
            self.query_proj = nn.Linear(gnn_hidden, bert_hidden)
            self.attn = nn.MultiheadAttention(embed_dim=bert_hidden, num_heads=8, batch_first=True)
            classifier_in = gnn_hidden + bert_hidden
        elif fusion_type == 'early_concat':
            classifier_in = gnn_hidden + bert_hidden
        elif fusion_type == 'gnn_only':
            classifier_in = gnn_hidden
        else:  # bert_only
            classifier_in = bert_hidden

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tags)
        )

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        if self.fusion_type == 'bert_only':
            t = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            return self.classifier(t)

        if self.fusion_type == 'gnn_only':
            g = self.gnn(x, edge_index, batch)
            return self.classifier(g)

        g = self.gnn(x, edge_index, batch)
        Htext = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if self.fusion_type == 'early_concat':
            t_cls = Htext[:, 0, :]
            z = torch.cat([g, t_cls], dim=1)
            return self.classifier(z)

        # cross_attention
        q = self.query_proj(g).unsqueeze(1)
        attended, _ = self.attn(q, Htext, Htext, key_padding_mask=(attention_mask == 0))
        z = torch.cat([g, attended.squeeze(1)], dim=1)
        return self.classifier(z)

    def get_fused_embedding(self, x, edge_index, batch, input_ids, attention_mask):
        """Only defined for cross_attention — used for the t-SNE cell."""
        assert self.fusion_type == 'cross_attention'
        g = self.gnn(x, edge_index, batch)
        Htext = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        q = self.query_proj(g).unsqueeze(1)
        attended, _ = self.attn(q, Htext, Htext, key_padding_mask=(attention_mask == 0))
        return torch.cat([g, attended.squeeze(1)], dim=1)


def maybe_warm_start_bert(model):
    if TASK1_CHECKPOINT is not None and hasattr(model, 'bert'):
        ckpt = torch.load(TASK1_CHECKPOINT, weights_only=False)
        bert_state = {k.replace('bert.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('bert.')}
        model.bert.load_state_dict(bert_state, strict=False)
        return True
    return False

# %% CELL 11 — NEW: training/eval helpers, now with AUC-PR
criterion = nn.BCEWithLogitsLoss()

def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        b_idx = batch['batch'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(x, edge_index, b_idx, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        b_idx = batch['batch'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(x, edge_index, b_idx, input_ids, attention_mask)
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    # AUC-PR (spec Section 6): mean average precision over tags that have at
    # least one positive example in this split (undefined otherwise).
    valid_tags = [k for k in range(all_labels.shape[1]) if all_labels[:, k].sum() > 0]
    if valid_tags:
        auc_pr = average_precision_score(all_labels[:, valid_tags], all_probs[:, valid_tags], average='macro')
    else:
        auc_pr = float('nan')

    return {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'auc_pr': auc_pr}, all_probs, all_preds, all_labels

# %% CELL 12 — NEW: Ablation study (spec deliverable #2)
# Trains all four fusion variants and compares them. The primary
# 'cross_attention' model gets the full NUM_EPOCHS budget with
# checkpoint/resume support; the other three get ABLATION_EPOCHS since
# they exist purely for comparison, not as the final deliverable model.
FUSION_TYPES = ['bert_only', 'gnn_only', 'early_concat', 'cross_attention']
ablation_results = {}
models = {}

for fusion_type in FUSION_TYPES:
    print(f"\n{'='*60}\nTraining variant: {fusion_type}\n{'='*60}")
    model = UnifiedFusionModel(fusion_type).to(device)
    warm_started = maybe_warm_start_bert(model)
    if warm_started:
        print("Warm-started BERT weights from Task 1 checkpoint.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = NUM_EPOCHS if fusion_type == 'cross_attention' else ABLATION_EPOCHS
    ckpt_path = os.path.join(WORKING_DIR, f"task3_{fusion_type}_checkpoint.pt")

    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed {fusion_type} from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, optimizer, train_loader)
        val_metrics, _, _, _ = evaluate(model, val_loader)
        print(f"[{fusion_type}] Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val AUC-PR: {val_metrics['auc_pr']:.4f}")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()}, ckpt_path)

    test_metrics, _, _, _ = evaluate(model, test_loader)
    ablation_results[fusion_type] = test_metrics
    models[fusion_type] = model
    print(f"[{fusion_type}] TEST — Macro-F1: {test_metrics['macro_f1']:.4f} | "
          f"Micro-F1: {test_metrics['micro_f1']:.4f} | AUC-PR: {test_metrics['auc_pr']:.4f}")

main_model = models['cross_attention']  # the primary deliverable model

# %% CELL 13 — NEW: Ablation comparison table (spec Table 3 style)
print(f"\n{'='*70}\nTASK 3 — ABLATION COMPARISON (spec Section 4.3 / Table 3)\n{'='*70}")
print(f"{'Model':<20}{'Macro-F1':<12}{'Micro-F1':<12}{'AUC-PR':<12}")
label_map = {'bert_only': 'BERT-only', 'gnn_only': 'GNN-only',
             'early_concat': 'Early-concat', 'cross_attention': 'Cross-attention (ours)'}
for ft in FUSION_TYPES:
    m = ablation_results[ft]
    print(f"{label_map[ft]:<20}{m['macro_f1']:<12.4f}{m['micro_f1']:<12.4f}{m['auc_pr']:<12.4f}")

# %% CELL 14 — FIXED: t-SNE colored by genre AND by mood (spec deliverable #4)
from sklearn.manifold import TSNE

@torch.no_grad()
def get_fused_embeddings_and_labels(model, loader):
    model.eval()
    embeddings, mood_labels, genre_labels = [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        b_idx = batch['batch'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].numpy()

        z = model.get_fused_embedding(x, edge_index, b_idx, input_ids, attention_mask)
        embeddings.append(z.cpu().numpy())
        for lbl in labels:
            mood_labels.append(dominant_label(lbl, MOOD_TAG_IDX))
            genre_labels.append(dominant_label(lbl, GENRE_TAG_IDX))
    return np.concatenate(embeddings), mood_labels, genre_labels

embeddings, mood_labels, genre_labels = get_fused_embeddings_and_labels(main_model, test_loader)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
proj = tsne.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, group_labels, title in [(axes[0], genre_labels, "colored by genre"),
                                  (axes[1], mood_labels, "colored by mood")]:
    unique_vals = list(set(group_labels))[:15]  # cap legend at 15 for readability
    for val in unique_vals:
        mask = [g == val for g in group_labels]
        ax.scatter(proj[mask, 0], proj[mask, 1], label=val, alpha=0.6, s=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.set_title(f"t-SNE of Fused GNN-BERT Embeddings, {title}")
plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, "task3_tsne_genre_mood.png"), dpi=150)
plt.show()

# %% CELL 15 — FIXED: case studies with graph paths + caption/lyric alignment
# spec deliverable #5 explicitly asks for "graph paths + caption/lyric
# alignment" — the previous version only printed the audio path and caption.
# This version prints: segment-graph structure (nodes + adjacency), which
# words were masked out for label-leakage prevention (the alignment between
# graph segments and text is implicit in "same clip, same tags"), and the
# model's actual top-5 predicted tags next to the ground truth so the
# alignment between graph-driven prediction and text is visible.
main_model.eval()
np.random.seed(7)
sample_indices = np.random.choice(len(test_idx), min(3, len(test_idx)), replace=False)

for i in sample_indices:
    idx = test_idx[i]
    row = df_final.iloc[idx]
    graph_data = torch.load(graph_files[idx], weights_only=False)
    x, edge_index = graph_data['x'], graph_data['edge_index']

    adjacency = {}
    for e in range(edge_index.shape[1]):
        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
        adjacency.setdefault(src, set()).add(dst)
    adjacency = {k: sorted(v) for k, v in adjacency.items()}

    encoding = tokenizer(row['masked_caption'], truncation=True, padding='max_length',
                          max_length=MAX_LEN, return_tensors='pt')
    with torch.no_grad():
        logits = main_model(
            x.to(device), edge_index.to(device),
            torch.zeros(x.shape[0], dtype=torch.long).to(device),
            encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    top5_idx = np.argsort(probs)[::-1][:5]
    predicted = [(TAG_NAMES[j], round(float(probs[j]), 3)) for j in top5_idx]
    actual = [TAG_NAMES[j] for j, v in enumerate(row['labels']) if v == 1.0]

    print(f"\n--- Case Study (test idx {idx}) ---")
    print(f"Audio file: {row['audio_path']}")
    print(f"Segment graph: {x.shape[0]} nodes (~{SEGMENT_DURATION}s each), "
          f"{edge_index.shape[1]} directed edges")
    print(f"Graph adjacency (segment -> connected segments): {adjacency}")
    print(f"Original caption: {row['caption'][:150]}...")
    print(f"Masked caption (aspect keywords -> [MASK] to prevent leakage): {row['masked_caption'][:150]}...")
    print(f"Actual tags: {actual}")
    print(f"Model top-5 predicted tags (tag, probability): {predicted}")

# %% CELL 16 — Save results
RESULTS_PATH = os.path.join(WORKING_DIR, "task3_results.json")
results = {
    "ablation_results": ablation_results,
    "primary_model": "cross_attention",
    "tag_names": TAG_NAMES,
    "tag_categories": TAG_CATEGORY,
    "num_samples": len(df_final),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

torch.save({'model_state': main_model.state_dict()}, os.path.join(WORKING_DIR, "task3_checkpoint.pt"))

print(f"Results saved at: {RESULTS_PATH}")
print("\nIMPORTANT: click 'Save Version' (top right) now, or these files")
print("will be lost when this session ends.")