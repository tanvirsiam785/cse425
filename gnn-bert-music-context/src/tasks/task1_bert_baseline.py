"""
TASK 1: BERT Baseline for Music Tag Understanding — FINAL COMPLETE VERSION
==============================================================================
Run this in a Kaggle Notebook.

BEFORE RUNNING:
1. Add Input: "MusicCaps" (googleai/musiccaps) via the "+" button
2. Settings -> Accelerator -> GPU T4 x2

Dataset: official MusicCaps (musiccaps-public.csv), using the real
"aspect_list" column (musician-written ground-truth tags). Top-50 most
frequent aspects are used as the tag vocabulary.

Guideline compliance included in this version:
- Label leakage prevention: matched aspect phrases are masked out of
  the caption ([MASK]) before being given to BERT as input.
- Macro-F1 / Micro-F1 CURVES vs. training epoch (actual plot, not just
  printed numbers) — required deliverable.
- 5 example predictions — required deliverable.
- B1 majority-class baseline comparison — required minimum baseline.

Copy-paste each "# %% CELL" block into a separate Kaggle notebook cell.
"""

# %% CELL 1 — Install dependencies
!pip install transformers scikit-learn matplotlib -q

# %% CELL 2 — Auto-detect the MusicCaps CSV path
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
print(f"CSV_PATH: {CSV_PATH}")
assert CSV_PATH is not None, "musiccaps-public.csv not found — did you add the MusicCaps dataset via the '+' button?"

WORKING_DIR = "/kaggle/working"

# %% CELL 3 — Imports & Config
import re
import ast
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

CHECKPOINT_PATH = os.path.join(WORKING_DIR, "task1_checkpoint.pt")
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 2e-5
TOP_N_TAGS = 50   # how many most-frequent aspects to use as the tag vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% CELL 4 — Load captions and parse aspect_list
df = pd.read_csv(CSV_PATH)
df['aspects_parsed'] = df['aspect_list'].apply(ast.literal_eval)
df['aspects_parsed'] = df['aspects_parsed'].apply(lambda lst: [a.strip().lower() for a in lst])

print(df[['ytid', 'caption', 'aspects_parsed']].head())
print(f"Total captions: {len(df)}")

# %% CELL 5 — Build tag vocabulary from top-N most frequent real aspects
all_aspects = []
for lst in df['aspects_parsed']:
    all_aspects.extend(lst)

counter = Counter(all_aspects)
TAG_NAMES = [tag for tag, _ in counter.most_common(TOP_N_TAGS)]
NUM_TAGS = len(TAG_NAMES)
print(f"Total proxy tags (top {TOP_N_TAGS} real aspects): {NUM_TAGS}")
print(TAG_NAMES)

def build_labels_and_masked_caption(row):
    """
    Build a multi-hot label vector from real aspect_list tags (only the
    ones in our top-N vocabulary), and mask those matched phrases out
    of the caption text to prevent label leakage — without this, BERT
    could just detect the literal keyword instead of learning genuine
    context, inflating F1 to near-100% with no real learning.
    """
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

no_tag_count = sum(1 for l in df['labels'] if l.sum() == 0)
print(f"\nNo-tag captions (none of the top-{TOP_N_TAGS} aspects matched): {no_tag_count} / {len(df)}")
print("\nExample of masking:")
print(f"Original: {df['caption'].iloc[0]}")
print(f"Masked:   {df['masked_caption'].iloc[0]}")

# %% CELL 6 — Train/Val/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# %% CELL 7 — Dataset class
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CaptionTagDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN):
        self.captions = dataframe['masked_caption'].tolist()   # masked, not raw caption
        self.labels = np.stack(dataframe['labels'].tolist())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.captions[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = CaptionTagDataset(train_df, tokenizer)
val_dataset = CaptionTagDataset(val_df, tokenizer)
test_dataset = CaptionTagDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# %% CELL 8 — BERT Multi-label Classifier Model
class BERTTagClassifier(nn.Module):
    def __init__(self, num_tags=NUM_TAGS):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # CLS token
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)   # sigmoid is applied inside BCEWithLogitsLoss

# %% CELL 9 — Training setup
model = BERTTagClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        preds = (torch.sigmoid(outputs) > threshold).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    return macro_f1, micro_f1, all_preds, all_labels

# %% CELL 10 — Resume from checkpoint (if it exists) + Training loop
# Track Macro-F1 / Micro-F1 per epoch for the required curve plot (CELL 11)
history = {"epoch": [], "train_loss": [], "val_macro_f1": [], "val_micro_f1": []}

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    history = ckpt.get('history', history)
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss = train_epoch()
    val_macro_f1, val_micro_f1, _, _ = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Micro-F1: {val_micro_f1:.4f}")

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_macro_f1"].append(val_macro_f1)
    history["val_micro_f1"].append(val_micro_f1)

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
    }, CHECKPOINT_PATH)

# %% CELL 11 — REQUIRED DELIVERABLE: Macro-F1 / Micro-F1 curves vs. training epoch
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["val_macro_f1"], marker='o', label="Val Macro-F1")
plt.plot(history["epoch"], history["val_micro_f1"], marker='s', label="Val Micro-F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Task 1: Macro-F1 / Micro-F1 vs. Training Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, "task1_f1_curves.png"), dpi=150)
plt.show()
print("F1 curve plot saved to task1_f1_curves.png")

# %% CELL 12 — Final Test Evaluation
test_macro_f1, test_micro_f1, test_preds, test_labels = evaluate(test_loader)
print(f"\nFinal Test Macro-F1: {test_macro_f1:.4f}")
print(f"Final Test Micro-F1: {test_micro_f1:.4f}")
print("\nPer-tag report:")
print(classification_report(test_labels, test_preds, target_names=TAG_NAMES, zero_division=0))

# %% CELL 13 — REQUIRED DELIVERABLE: 5 example predictions
model.eval()
sample_df = test_df.sample(5, random_state=1)
example_predictions = []
for _, row in sample_df.iterrows():
    encoding = tokenizer(row['masked_caption'], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(output).cpu().numpy()[0]
    predicted_tags = [TAG_NAMES[i] for i, p in enumerate(probs) if p > 0.5]
    actual_tags = [TAG_NAMES[i] for i, v in enumerate(row['labels']) if v == 1.0]
    print(f"\nOriginal caption: {row['caption'][:100]}...")
    print(f"Model saw (masked): {row['masked_caption'][:100]}...")
    print(f"Actual tags: {actual_tags}")
    print(f"Predicted tags: {predicted_tags}")
    example_predictions.append({
        "caption": row['caption'], "actual_tags": actual_tags, "predicted_tags": predicted_tags
    })

# %% CELL 14 — REQUIRED BASELINE: B1 Majority-Class comparison
# Predicts each tag independently at its overall training-set frequency
# (base rate), thresholded at 0.5 — the simplest required baseline (B1).
train_labels_arr = np.stack(train_df['labels'].tolist())
tag_base_rates = train_labels_arr.mean(axis=0)
b1_pred_vector = (tag_base_rates > 0.5).astype(np.float32)   # same prediction for every sample
b1_preds_repeated = np.tile(b1_pred_vector, (len(test_labels), 1))

b1_macro_f1 = f1_score(test_labels, b1_preds_repeated, average='macro', zero_division=0)
b1_micro_f1 = f1_score(test_labels, b1_preds_repeated, average='micro', zero_division=0)

print(f"\n--- B1 Majority-Class Baseline vs. BERT ---")
print(f"B1 Baseline: Macro-F1 {b1_macro_f1:.4f}, Micro-F1 {b1_micro_f1:.4f}")
print(f"BERT (this model): Macro-F1 {test_macro_f1:.4f}, Micro-F1 {test_micro_f1:.4f}")

# %% CELL 15 — Save all results
results = {
    "test_macro_f1": float(test_macro_f1),
    "test_micro_f1": float(test_micro_f1),
    "b1_baseline_macro_f1": float(b1_macro_f1),
    "b1_baseline_micro_f1": float(b1_micro_f1),
    "tag_names": TAG_NAMES,
    "history": history,
    "example_predictions": example_predictions,
}
RESULTS_PATH = os.path.join(WORKING_DIR, "task1_results.json")
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCheckpoint saved at: {CHECKPOINT_PATH}")
print(f"Results saved at: {RESULTS_PATH}")
print(f"F1 curve plot saved at: {os.path.join(WORKING_DIR, 'task1_f1_curves.png')}")
print("\nIMPORTANT: click 'Save Version' (top right) now, or these files")
print("will be lost when this session ends.")