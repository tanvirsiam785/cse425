"""
audio_features.py
==================
Audio feature extraction shared across Tasks 2-4: fixed-length windowing,
12-D chroma vectors (librosa), and 128-band log-mel spectrograms for the
Task 2 CNN baseline.

This module documents and centralizes the feature-extraction logic that
originally lived inline inside the Kaggle notebook cells for each task
(see src/tasks/task2_gnn_baseline.py, task3_fusion.py and
task4_contrastive_retrieval.py, which remain the runnable, self-contained
scripts). Import from here when wiring a non-notebook pipeline.
"""

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Config (matches the paper / notebooks)
# ---------------------------------------------------------------------------
SR = 22050
FMA_SEGMENT_DURATION = 5.0        # seconds, FMA-small (30s clips)
MUSICCAPS_SEGMENT_DURATION = 2.0  # seconds, MusicCaps (10s clips)
N_CHROMA = 12
N_MELS = 128
MEL_FIXED_FRAMES = 640            # pad/truncate width for CNN baseline


def load_audio(path, sr=SR, duration=None):
    """Load an audio file as a mono waveform at the target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y


def segment_waveform(y, sr=SR, segment_duration=FMA_SEGMENT_DURATION):
    """Split a waveform into fixed-length, non-overlapping windows.

    Returns a list of 1-D numpy arrays. The final partial window (if any)
    is zero-padded to full length so every window contributes one graph node.
    """
    seg_len = int(segment_duration * sr)
    if seg_len <= 0:
        raise ValueError("segment_duration must be > 0")

    segments = []
    for start in range(0, len(y), seg_len):
        chunk = y[start:start + seg_len]
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))
        segments.append(chunk)
    return segments


def chroma_vector(segment, sr=SR, n_chroma=N_CHROMA):
    """Compute a single 12-D chroma feature vector for one audio window
    (mean chroma_cqt over time within the window)."""
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, n_chroma=n_chroma)
    return chroma.mean(axis=1)  # (n_chroma,)


def chroma_sequence(y, sr=SR, segment_duration=FMA_SEGMENT_DURATION,
                     n_chroma=N_CHROMA):
    """Segment a waveform and return one chroma vector per window,
    stacked as (num_windows, n_chroma). This is the node-feature matrix
    used to build the graph in graph_builder.py."""
    segments = segment_waveform(y, sr=sr, segment_duration=segment_duration)
    return np.stack([chroma_vector(seg, sr=sr, n_chroma=n_chroma)
                      for seg in segments])


def log_mel_spectrogram(y, sr=SR, n_mels=N_MELS, fixed_frames=MEL_FIXED_FRAMES):
    """128-band log-mel spectrogram, padded/truncated to a fixed frame
    width for the Task 2 CNN baseline (B2)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < fixed_frames:
        pad_width = fixed_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant",
                          constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :fixed_frames]
    return log_mel  # (n_mels, fixed_frames)
