"""
gnn_model.py
============
3-layer GraphSAGE encoder over chroma segment-graphs (Task 2), reused as
the audio tower in Task 3 (fusion) and Task 4 (contrastive dual-encoder).

Also includes the CNN-on-mel-spectrogram baseline (B2) used for comparison
in Task 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """3-layer GraphSAGE with mean-pooling readout.

    Used standalone with a linear genre head in Task 2, and as the audio
    tower (embedding only, head stripped) in Tasks 3 and 4.
    """

    def __init__(self, in_channels=12, hidden_channels=64, out_channels=128,
                 num_classes=None, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(out_channels, num_classes) if num_classes else None

    def embed(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index)
        graph_embedding = global_mean_pool(h, batch)  # (num_graphs, out_channels)
        return graph_embedding

    def forward(self, x, edge_index, batch):
        graph_embedding = self.embed(x, edge_index, batch)
        if self.classifier is not None:
            return self.classifier(graph_embedding)
        return graph_embedding


class MelCNNBaseline(nn.Module):
    """B2 baseline: small CNN on 128-band log-mel spectrograms
    (Task 2, Table 2)."""

    def __init__(self, num_classes=8, n_mels=128, fixed_frames=640):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, mel):
        # mel: (B, 1, n_mels, fixed_frames)
        h = self.conv(mel)
        return self.fc(h)
