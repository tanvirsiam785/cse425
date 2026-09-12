"""
fusion_model.py
================
Task 3: GNN-BERT fusion via cross-attention, plus the three ablation
variants reported in the paper (Table 3): BERT-only, GNN-only,
early-concatenation, and cross-attention (best model, Macro-F1 0.611).

The graph embedding queries the sequence of BERT token embeddings;
the attended vector is concatenated with the graph embedding before the
final multi-label classification head.
"""

import torch
import torch.nn as nn

from bert_encoder import BertTagEncoder
from gnn_model import GraphSAGEEncoder


class CrossAttentionFusion(nn.Module):
    """Graph embedding attends over BERT token embeddings."""

    def __init__(self, graph_dim=128, text_dim=768, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(graph_dim, text_dim)
        self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads,
                                           batch_first=True)

    def forward(self, graph_embedding, token_embeddings, attention_mask):
        # graph_embedding: (B, graph_dim) -> query: (B, 1, text_dim)
        query = self.query_proj(graph_embedding).unsqueeze(1)
        key_padding_mask = attention_mask == 0  # True = ignore
        attended, _ = self.attn(query, token_embeddings, token_embeddings,
                                 key_padding_mask=key_padding_mask)
        return attended.squeeze(1)  # (B, text_dim)


class GNNBertFusionModel(nn.Module):
    """variant in {'bert_only', 'gnn_only', 'early_concat', 'cross_attention'}"""

    def __init__(self, num_labels, variant="cross_attention",
                 graph_dim=128, text_dim=768):
        super().__init__()
        assert variant in {"bert_only", "gnn_only", "early_concat", "cross_attention"}
        self.variant = variant

        self.bert = BertTagEncoder(num_labels=None)          # no head; raw pooled + tokens
        self.gnn = GraphSAGEEncoder(in_channels=12, out_channels=graph_dim,
                                     num_classes=None)

        if variant == "bert_only":
            head_in = text_dim
        elif variant == "gnn_only":
            head_in = graph_dim
        elif variant == "early_concat":
            head_in = text_dim + graph_dim
            self.cross_attn = None
        else:  # cross_attention
            self.cross_attn = CrossAttentionFusion(graph_dim, text_dim)
            head_in = text_dim + graph_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, graph_x, graph_edge_index, graph_batch):
        pooled_text, token_embeddings = self.bert.encode(input_ids, attention_mask)
        graph_embedding = self.gnn.embed(graph_x, graph_edge_index, graph_batch)

        if self.variant == "bert_only":
            fused = pooled_text
        elif self.variant == "gnn_only":
            fused = graph_embedding
        elif self.variant == "early_concat":
            fused = torch.cat([pooled_text, graph_embedding], dim=-1)
        else:  # cross_attention (best model, Table 3)
            attended_text = self.cross_attn(graph_embedding, token_embeddings, attention_mask)
            fused = torch.cat([attended_text, graph_embedding], dim=-1)

        return self.classifier(fused)
