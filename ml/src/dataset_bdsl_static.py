"""Loader for the BdSL digit and letter datasets from
rayeed045/bangla-sign-language-dataset (Kaggle).

Layout (shared between "Sign Digits - CSV Files" and "Sign Letters -
CSV Files"):

    <root>/User - <Name> (<Gender>, <Age>)/<file>.csv

Each CSV is one (signer, class) pair: ~100 rows, each an
*independently photographed* single-frame hand pose (verified via the
embedded "Single Image Frame" path column, e.g. ".../Sign 05 - Aye/Sign
05 - Sample (1).jpg") -- not a temporal sequence like BdSLW60's video
clips. Columns are [Single Image Frame, x00,y00,z00, ..., x20,y20,z20,
Label], where Label is the verified numeric class id (confirmed by
cross-checking against the embedded file paths, e.g. Sign_5.csv has
Label==5 and paths under ".../Digits/Sign 5/...").
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd


def _signer_name(folder_name: str) -> str:
    m = re.match(r"User - (\w+)", folder_name)
    return m.group(1) if m else folder_name


def load_static_dataset(
    root: str,
    max_samples_per_class: int | None = None,
    seed: int = 42,
):
    """Returns (X, y, signers) -- X is (N, 21, 3) float32 single frames,
    y is the raw numeric Label per row, signers is the signer name per row
    (for a participant-based train/val split).
    """
    rows_by_class: dict[int, list] = {}
    for user_dir in sorted(glob.glob(os.path.join(root, "*"))):
        signer = _signer_name(os.path.basename(user_dir))
        for csv_path in sorted(glob.glob(os.path.join(user_dir, "*.csv"))):
            df = pd.read_csv(csv_path)
            coords = df.iloc[:, 1:64].to_numpy(dtype=np.float32).reshape(-1, 21, 3)
            labels = df["Label"].to_numpy()
            for frame, lbl in zip(coords, labels):
                rows_by_class.setdefault(int(lbl), []).append((frame, signer))

    rng = np.random.default_rng(seed)
    X, y, signers = [], [], []
    for lbl, items in sorted(rows_by_class.items()):
        if max_samples_per_class is not None and len(items) > max_samples_per_class:
            idx = rng.choice(len(items), size=max_samples_per_class, replace=False)
            items = [items[i] for i in idx]
        for frame, signer in items:
            X.append(frame)
            y.append(lbl)
            signers.append(signer)

    return np.stack(X), np.array(y), np.array(signers)


# Romanized names for the 37 "Sign Letters" classes, transcribed from the
# dataset's own embedded file paths (e.g. "Sign 05 - Aye"). Every name maps
# to a real Bangla glyph in app/lib/services/bdsl_glossary.dart: the names
# follow the standard Bangla alphabet order exactly (vowels in vowel order,
# then consonants in consonant order), with a doubled-letter romanization
# (Tta/Dda/Ddha) marking the retroflex row apart from the single-letter
# dental row right after it (Taa/Daa/Dha) -- that positional match is what
# fixes each name to one glyph. The dataset consolidates a few
# near-homophone letter pairs into a single class (e.g. one "Naa" class
# covers both retroflex ণ and dental ন), so a handful of standard letters
# (dontyo ন, ভ, য, র, ষ, স) have no separate class here -- a modeling
# choice made upstream in the dataset, not a gap in this mapping.
LETTER_NAMES = [
    "Awe", "Aaa", "Eee", "Uuu", "Ree", "Aye", "Oii", "Ooo", "Oww",  # vowels 0-8
    "Kaa", "Kha", "Gaa", "Gha", "Uyo", "Cha", "Chha", "Jaa", "Jha", "Eyo",
    "Tta", "Ttha", "Dda", "Ddha", "Naa", "Taa", "Tha", "Daa", "Dha",
    "Paa", "Pha", "Baa", "Maa", "Laa", "Haa", "Sha", "Anushar", "Chandra",
]
