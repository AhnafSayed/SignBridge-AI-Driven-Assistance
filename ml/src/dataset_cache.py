"""Shared on-disk cache for loaded (X, y, participants, label_names)
datasets, keyed by whatever arguments determined how they were built.
Used by train.py and train_asl_combined.py so a full ASL reload (~15
minutes for 94,477 parquet files) is only ever paid once per distinct
data selection, not once per training experiment.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")

# Bump whenever a dataset loader's *extraction contract* changes (landmark
# shape, normalization, which frames get kept) -- baked into every cache key
# so a stale cache from before the change can never be silently reloaded as
# if it still matched what the current code produces. v1: single-hand
# (21, 3). v2: two-hand (42, 3), geometric x-sorted slots (see landmarks.py).
# v3: reinstated the per-hand >50%-missing quality filter v2 had dropped --
# v2 alone measured *worse* than v1 (52.0% -> 47.2% on the same benchmark)
# from admitting too much unreliably-tracked data.
# v4: appended the in-frame position anchor row, (T, 42, 3) -> (T, 43, 3)
# (see landmarks.frame_anchor). Exactly the kind of extraction-contract
# change this constant exists for: a v3 cache is still a perfectly valid
# .npz and would reload without error, just with the wrong final dimension.
LANDMARK_VERSION = "v4_twohand_posanchor"


def cache_path(prefix: str, *key_parts) -> str:
    key = "|".join(str(p) for p in (LANDMARK_VERSION, *key_parts))
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{digest}.npz")


def load(path: str):
    with np.load(path, allow_pickle=True) as data:
        return data["X"], data["y"], data["participants"], [str(s) for s in data["label_names"]]


def save(path: str, X: np.ndarray, y: np.ndarray, participants: np.ndarray, label_names) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, X=X, y=y, participants=participants, label_names=label_names)
