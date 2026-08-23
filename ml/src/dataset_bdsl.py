"""Loader for the BdSLW60 landmark dataset (Kaggle:
kaziredwanislam/bdslw60-landmark-data-trials).

Layout confirmed by inspection:

    data_dir/
      Landmark_Data_Trials/Word_<word_id>/U<signer_id>W<word_id>F_T<trial>.npy

Each .npy is a (num_frames, 126) float64 array: 126 = 2 hands x 21
landmarks x 3 coords, packed as [left_hand(63), right_hand(63)] per
frame. Frames where a hand wasn't detected are zero-padded -- unlike
asl-signs' parquet (which simply omits undetected frames), but the same
real signal either way: this hand isn't in view this frame.

Both hands are kept (an earlier version of this loader picked whichever
one tracked more completely and discarded the other, throwing away real
signal for two-handed signs) and combined by wrist x-coordinate, not the
file's own left_hand/right_hand column split -- see landmarks.py's module
docstring for why geometric slotting is used everywhere in this project
instead of semantic left/right identity.

Word IDs -> English gloss, transcribed from the dataset's own
BdSLW60.pdf (ml/data/bdslw60_docs/BdSLW60.pdf).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from landmarks import NUM_LANDMARKS, combine_two_hands_by_x, prepare_two_hand_sequence

WORD_LABELS = {
    1: "father", 2: "relatives", 3: "brother", 4: "sister", 5: "wife",
    6: "paternal_uncle", 7: "paternal_aunt", 8: "grandfather", 9: "grandmother",
    10: "responsibility", 11: "brother_in_law", 12: "sisters_husband",
    19: "daughter", 20: "mother", 37: "mango", 38: "potato", 39: "pineapple",
    40: "grapes", 41: "apple", 42: "biscuits", 43: "jujube", 44: "cake",
    45: "tea", 46: "rice", 47: "sugar", 48: "chips", 49: "chocolate",
    50: "lentils", 91: "button", 92: "cap", 93: "shawl", 94: "comb",
    95: "spectacles", 96: "bangles", 97: "clip", 98: "cream", 99: "data",
    100: "indebted", 111: "twin_baby", 112: "shoe", 211: "toothpaste",
    212: "tshirt", 213: "tubelight", 214: "television", 215: "air_conditioner",
    216: "apartment", 217: "audio_cassette", 218: "mirror", 219: "water_bucket",
    220: "sand", 351: "aids", 352: "arthritis", 353: "bandage", 354: "capsule",
    355: "treatment", 356: "conjunctivitis", 357: "dengue", 358: "doctor",
    359: "bite", 360: "weak",
}

_FILENAME_RE = re.compile(r"U(\d+)W(\d+)F?_T(\d+)")


@dataclass
class BdslIndex:
    data_dir: str
    samples: list  # list of (relative_path, word_id, signer_id, label)

    @property
    def signs(self) -> list[str]:
        return sorted({s[3] for s in self.samples})


def load_index(data_dir: str) -> BdslIndex:
    root = os.path.join(data_dir, "Landmark_Data_Trials")
    samples = []
    for word_dir in sorted(os.listdir(root)):
        m = re.match(r"Word_(\d+)$", word_dir)
        if not m:
            continue
        word_id = int(m.group(1))
        label = WORD_LABELS.get(word_id)
        if label is None:
            continue  # folder outside the documented 60-word table
        word_path = os.path.join(root, word_dir)
        for fname in os.listdir(word_path):
            fm = _FILENAME_RE.match(fname)
            if not fm:
                continue
            signer_id = int(fm.group(1))
            rel_path = os.path.join("Landmark_Data_Trials", word_dir, fname)
            samples.append((rel_path, word_id, signer_id, label))
    return BdslIndex(data_dir=data_dir, samples=samples)


def core_vocabulary(index: BdslIndex, size: int = 60) -> list[str]:
    """BdSLW60 IS a 60-word core vocabulary -- nothing to curate down from."""
    return index.signs[:size]


def load_sequence(data_dir: str, relative_path: str) -> np.ndarray | None:
    arr = np.load(os.path.join(data_dir, relative_path))
    left = arr[:, :63].reshape(-1, NUM_LANDMARKS, 3).astype(np.float32)
    right = arr[:, 63:].reshape(-1, NUM_LANDMARKS, 3).astype(np.float32)

    any_present = (np.any(left != 0, axis=(1, 2))) | (np.any(right != 0, axis=(1, 2)))
    if not any_present.any():
        return None
    return combine_two_hands_by_x(left[any_present], right[any_present])


def build_dataset(
    index: BdslIndex,
    vocabulary: list[str],
    seq_len: int = 40,
    max_samples_per_class: int | None = None,
):
    """Same contract as dataset_asl.build_dataset: (X, y, signer_ids, label_names)."""
    vocab_set = set(vocabulary)
    samples = [s for s in index.samples if s[3] in vocab_set]

    if max_samples_per_class is not None:
        by_label: dict[str, list] = {}
        for s in samples:
            by_label.setdefault(s[3], []).append(s)
        samples = [s for group in by_label.values() for s in group[:max_samples_per_class]]

    label_names = sorted(vocab_set)
    label_to_idx = {label: i for i, label in enumerate(label_names)}

    X, y, signers = [], [], []
    for rel_path, word_id, signer_id, label in tqdm(samples, desc="loading sequences"):
        seq = load_sequence(index.data_dir, rel_path)
        if seq is None or seq.shape[0] == 0:
            continue
        X.append(prepare_two_hand_sequence(seq, seq_len))
        y.append(label_to_idx[label])
        signers.append(signer_id)

    return (
        np.stack(X).astype(np.float32),
        np.array(y, dtype=np.int64),
        np.array(signers),
        label_names,
    )
