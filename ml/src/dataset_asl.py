"""Loader for Kaggle's "Google - Isolated Sign Language Recognition"
dataset (competition slug: asl-signs).

Expected layout after `kaggle competitions download -c asl-signs` and
unzipping into `data_dir`:

    data_dir/
      train.csv                      # sequence_id, path, participant_id, sign
      train_landmark_files/<pid>/<sequence_id>.parquet

Each parquet file holds one sequence: one row per (frame, landmark),
columns [frame, row_id, type, landmark_index, x, y, z], where `type` is
one of 'face', 'pose', 'left_hand', 'right_hand'. Pose/face are still
discarded (see landmarks.py's module docstring on why face specifically
doesn't help this project's word/letter classification task), but *both*
hands are now kept -- discarding whichever hand tracked worse, as an
earlier version of this loader did, threw away real signal for this
dataset's many two-handed signs.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from landmarks import DEFAULT_SEQ_LEN, NUM_LANDMARKS, combine_two_hands_by_x, prepare_two_hand_sequence

HAND_TYPES = ("right_hand", "left_hand")


@dataclass
class AslSignsIndex:
    data_dir: str
    train_df: pd.DataFrame

    @property
    def signs(self) -> list[str]:
        return sorted(self.train_df["sign"].unique().tolist())


def load_index(data_dir: str, csv_name: str = "train.csv") -> AslSignsIndex:
    train_df = pd.read_csv(os.path.join(data_dir, csv_name))
    return AslSignsIndex(data_dir=data_dir, train_df=train_df)


def core_vocabulary(index: AslSignsIndex, size: int = 50) -> list[str]:
    """The `size` most frequent signs -- a reproducible stand-in for the
    plan's hand-curated "most useful, most separable" core vocabulary.
    Frequent signs have more training examples, which is a reasonable
    correlate for "learns reliably" even if not a perfect one; swap this
    out for a hand-picked list once the team has one.
    """
    counts = index.train_df["sign"].value_counts()
    return counts.head(size).index.tolist()


def _extract_hand_aligned(landmark_df: pd.DataFrame, hand_type: str, all_frames: np.ndarray) -> np.ndarray:
    """Returns (len(all_frames), 21, 3) for one semantic MediaPipe hand
    type, aligned onto a *shared* frame axis (so it can be combined
    frame-by-frame with the other hand type below). Frames with no row at
    all for this hand type stay genuinely zero -- a real "this hand isn't
    in frame" signal, not a gap to interpolate over. Short in-frame
    tracking dropouts *within* this hand's own presence are filled by
    interpolation, same reasoning as the single-hand loader this replaces.

    Quality gate carried over from that single-hand loader: if this hand
    type is tracked so unreliably overall (>50% of its own landmarks NaN
    on average, across the frames it has any row for at all) that it's
    more noise than signal, treat it as fully absent rather than salvage
    scraps of it. An earlier version of this function dropped this check
    entirely -- more inclusive (every one of the 94,477 sequences produced
    *some* data, versus 69% before), but measurably worse: held-out
    accuracy on the same 250-word benchmark fell from 52.0% (single-hand,
    with this filter) to 47.2% (two-hand, without it). Letting in that
    much low-quality tracking hurt more than the extra two-hand signal
    helped.
    """
    hand = landmark_df[landmark_df["type"] == hand_type]
    seq = np.zeros((len(all_frames), NUM_LANDMARKS, 3), dtype=np.float32)
    if hand.empty:
        return seq

    frame_missing = hand["x"].isna().groupby(hand["frame"]).mean()
    if frame_missing.mean() > 0.5:
        return seq

    frame_col = hand["frame"].to_numpy()
    pos = np.searchsorted(all_frames, frame_col)
    landmark_idx = hand["landmark_index"].to_numpy()
    valid = landmark_idx < NUM_LANDMARKS

    sub = np.full((len(all_frames), NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    sub[pos[valid], landmark_idx[valid], 0] = hand["x"].to_numpy()[valid]
    sub[pos[valid], landmark_idx[valid], 1] = hand["y"].to_numpy()[valid]
    sub[pos[valid], landmark_idx[valid], 2] = hand["z"].to_numpy()[valid]

    frame_has_hand = ~np.all(np.isnan(sub), axis=(1, 2))
    if not frame_has_hand.any():
        return seq

    present_idx = np.flatnonzero(frame_has_hand)
    flat = sub[present_idx].reshape(len(present_idx), -1)
    filled = pd.DataFrame(flat).interpolate(limit_direction="both").to_numpy()
    filled = np.nan_to_num(filled, nan=0.0).reshape(len(present_idx), NUM_LANDMARKS, 3)
    seq[present_idx] = filled
    return seq


def load_sequence(data_dir: str, relative_path: str) -> np.ndarray | None:
    """Loads one sample's *two*-hand landmarks as (T, 42, 3): both hands
    kept, geometrically sorted into slots by wrist x-coordinate each frame
    (see landmarks.py's module docstring for why geometric, not the
    dataset's own left_hand/right_hand semantic labels). A frame with only
    one hand present fills slot 0 and zero-fills slot 1; a frame with
    neither is genuinely all-zero.
    """
    df = pd.read_parquet(os.path.join(data_dir, relative_path))
    hands_df = df[df["type"].isin(HAND_TYPES)]
    if hands_df.empty:
        return None
    all_frames = np.unique(hands_df["frame"].to_numpy())
    if len(all_frames) == 0:
        return None

    left = _extract_hand_aligned(df, "left_hand", all_frames)
    right = _extract_hand_aligned(df, "right_hand", all_frames)
    if not (np.any(left != 0) or np.any(right != 0)):
        return None
    return combine_two_hands_by_x(left, right)


def _load_and_prepare(args: tuple[str, str, int]) -> np.ndarray | None:
    data_dir, path, seq_len = args
    seq = load_sequence(data_dir, path)
    if seq is None or seq.shape[0] == 0:
        return None
    return prepare_two_hand_sequence(seq, seq_len)


def build_dataset(
    index: AslSignsIndex,
    vocabulary: list[str],
    seq_len: int = DEFAULT_SEQ_LEN,
    max_samples_per_class: int | None = None,
    workers: int | None = None,
):
    """Returns (X, y, participant_ids, label_names).

    X: float32 array (N, seq_len, 42, 3) -- both hands, already resampled +
    normalized (see landmarks.py's prepare_two_hand_sequence).
    y: int array (N,) indexing into label_names.
    participant_ids: array (N,) -- for a participant-based train/val split.

    Loads sequences across a thread pool -- at full dataset scale (up to
    94,477 parquet files) a serial loop here is the dominant cost of the
    whole training run, and each file's work (parse parquet, extract a
    hand, resample+normalize) is independent of every other file. This is
    deliberately a *thread* pool, not a process pool: pandas/pyarrow's
    parquet parsing and numpy's array ops both release the GIL for their
    C-level work, so threads still parallelize the I/O- and CPU-bound parts
    that matter here -- without paying for a process pool's cost on
    Windows, where each spawned worker process re-imports this whole
    process's modules (including TensorFlow) from scratch. Multiplying a
    multi-GB TensorFlow import across a dozen-plus worker processes is what
    exhausted the system paging file the first time this ran.
    """
    vocab_set = set(vocabulary)
    df = index.train_df[index.train_df["sign"].isin(vocab_set)]
    if max_samples_per_class is not None:
        df = df.groupby("sign", group_keys=False).head(max_samples_per_class)

    label_names = sorted(vocab_set)
    label_to_idx = {label: i for i, label in enumerate(label_names)}

    rows = list(df.itertuples())
    workers = workers or min(32, (os.cpu_count() or 4) * 2)

    X, y, participants = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_load_and_prepare, (index.data_dir, row.path, seq_len)): row
            for row in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="loading sequences"):
            row = futures[future]
            seq = future.result()
            if seq is None:
                continue
            X.append(seq)
            y.append(label_to_idx[row.sign])
            participants.append(row.participant_id)

    return (
        np.stack(X).astype(np.float32),
        np.array(y, dtype=np.int64),
        np.array(participants),
        label_names,
    )
