"""
bert_encoder.py
================
BERT text encoder shared by Task 1 (standalone classifier), Task 3
(cross-attention fusion) and Task 4 (contrastive text tower).

Includes the label-leakage masking step described in the paper (Section 2):
matched aspect/tag phrases are replaced with [MASK] in the caption before
tokenization, so the model must infer tags from context rather than from
verbatim keyword overlap with the aspect_list.
"""

import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

BERT_NAME = "bert-base-uncased"


def mask_aspect_phrases(caption, aspects):
    """Replace every matched aspect phrase in `caption` with [MASK].

    Args:
        caption: raw MusicCaps caption string.
        aspects: iterable of aspect/tag strings for this clip
            (from the musician-written aspect_list column).
    Returns:
        masked caption string.
    """
    masked = caption
    for phrase in sorted(aspects, key=len, reverse=True):
        pattern = re.escape(phrase.strip())
        if not pattern:
            continue
        masked = re.sub(pattern, "[MASK]", masked, flags=re.IGNORECASE)
    return masked


class BertTagEncoder(nn.Module):
    """BERT encoder producing a pooled [CLS] embedding, optionally with a
    linear multi-label classification head (Task 1) or raw token
    embeddings for cross-attention (Task 3)."""

    def __init__(self, num_labels=None, bert_name=BERT_NAME, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels) if num_labels else None

    def encode(self, input_ids, attention_mask):
        """Returns (pooled_cls_embedding, token_embeddings)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = out.last_hidden_state          # (B, L, H) — used by cross-attention (Task 3/4)
        pooled = self.dropout(out.last_hidden_state[:, 0])  # [CLS], (B, H)
        return pooled, token_embeddings

    def forward(self, input_ids, attention_mask):
        pooled, token_embeddings = self.encode(input_ids, attention_mask)
        if self.classifier is not None:
            logits = self.classifier(pooled)
            return logits
        return pooled, token_embeddings
