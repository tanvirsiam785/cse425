"""
graph_builder.py
=================
Builds per-track segment graphs from chroma features (Tasks 2-4).

Nodes  = fixed-length audio windows (see audio_features.chroma_sequence).
Edges  = (a) temporally adjacent windows, and
         (b) any pair of windows whose chroma cosine similarity > 0.85
             (captures repeated/harmonically-related sections, e.g. a
             recurring chorus or chord progression).

Returns a torch_geometric.data.Data object per track, ready for the
GraphSAGE encoder in gnn_model.py.
"""

import numpy as np
import torch
from torch_geometric.data import Data

CHROMA_SIM_THRESHOLD = 0.85


def _cosine_sim_matrix(X):
    """Pairwise cosine similarity, X: (num_nodes, n_chroma)."""
    norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def build_segment_graph(chroma_seq, label=None, sim_threshold=CHROMA_SIM_THRESHOLD):
    """Construct a single track's segment graph.

    Args:
        chroma_seq: (num_windows, n_chroma) array from
            audio_features.chroma_sequence(...)
        label: optional integer class label (genre id or multi-hot tag
            vector) attached to the graph for supervised training.
        sim_threshold: cosine-similarity cutoff for non-adjacent edges.

    Returns:
        torch_geometric.data.Data with x, edge_index, (y)
    """
    num_nodes = chroma_seq.shape[0]
    x = torch.tensor(chroma_seq, dtype=torch.float32)

    edges = set()
    # (a) temporal adjacency
    for i in range(num_nodes - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))

    # (b) chroma-similarity edges
    sim = _cosine_sim_matrix(chroma_seq)
    idx_i, idx_j = np.where(sim > sim_threshold)
    for i, j in zip(idx_i, idx_j):
        if i != j:
            edges.add((int(i), int(j)))

    if len(edges) == 0:
        # Degenerate single-node track: self-loop so message passing is defined.
        edges.add((0, 0))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        if np.isscalar(label):
            data.y = torch.tensor([label], dtype=torch.long)
        else:
            data.y = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    return data
