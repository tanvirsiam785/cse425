"""
contrastive.py
===============
Task 4: contrastive dual-encoder (GraphSAGE audio tower + BERT text tower,
both projected to a shared 256-D space) trained with symmetric InfoNCE,
used for caption<->audio retrieval and zero-shot tag prediction
(Table 4, Table 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTagEncoder
from gnn_model import GraphSAGEEncoder

PROJECTION_DIM = 256
TEMPERATURE = 0.07


class DualEncoder(nn.Module):
    def __init__(self, projection_dim=PROJECTION_DIM, graph_dim=128, text_dim=768):
        super().__init__()
        self.bert = BertTagEncoder(num_labels=None)
        self.gnn = GraphSAGEEncoder(in_channels=12, out_channels=graph_dim,
                                     num_classes=None)
        self.text_proj = nn.Linear(text_dim, projection_dim)
        self.audio_proj = nn.Linear(graph_dim, projection_dim)

    def encode_text(self, input_ids, attention_mask):
        pooled, _ = self.bert.encode(input_ids, attention_mask)
        return F.normalize(self.text_proj(pooled), dim=-1)

    def encode_audio(self, graph_x, graph_edge_index, graph_batch):
        graph_embedding = self.gnn.embed(graph_x, graph_edge_index, graph_batch)
        return F.normalize(self.audio_proj(graph_embedding), dim=-1)

    def forward(self, input_ids, attention_mask, graph_x, graph_edge_index, graph_batch):
        text_emb = self.encode_text(input_ids, attention_mask)
        audio_emb = self.encode_audio(graph_x, graph_edge_index, graph_batch)
        return text_emb, audio_emb


def symmetric_info_nce_loss(text_emb, audio_emb, temperature=TEMPERATURE):
    """Symmetric InfoNCE loss over an in-batch similarity matrix
    (batch_size=32 in the paper's Task 4 run)."""
    logits = text_emb @ audio_emb.t() / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t2a = F.cross_entropy(logits, labels)
    loss_a2t = F.cross_entropy(logits.t(), labels)
    return (loss_t2a + loss_a2t) / 2


def recall_at_k(sim_matrix, k_values=(1, 5, 10)):
    """Caption->Audio (or Audio->Caption, symmetric) recall@k given a full
    (N, N) similarity matrix where the ground-truth match for row i is
    column i."""
    n = sim_matrix.size(0)
    ranks = sim_matrix.argsort(dim=1, descending=True)
    results = {}
    for k in k_values:
        hit = sum(i in ranks[i, :k] for i in range(n))
        results[f"R@{k}"] = hit / n
    return results
