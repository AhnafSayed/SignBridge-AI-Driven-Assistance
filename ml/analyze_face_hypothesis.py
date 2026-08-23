"""Does discarding face landmarks measurably hurt specific ASL word classes?

Motivation: the project's per-class metrics (eval_full_metrics.py) showed
the ten worst-recognized ASL words are dominated by signs whose citation
form is articulated *at a specific place on the face* -- chin, lips,
tongue, cheek. That is exactly the information a hands-only pipeline
throws away: this project extracts only hand landmarks, and normalizes
each frame so the hand's own wrist is the origin (ml/src/landmarks.py),
which by construction deletes where the hand sits relative to the face.

This script tests that observation rather than trusting it. The
face-located word list below was written from each gloss's meaning BEFORE
looking at any per-class score, so the grouping is not fitted to the
result it is being used to test. It then compares per-class recall for
face-located versus other words with a Mann-Whitney U test (rank-based,
no normality assumption -- appropriate for bounded, skewed per-class
recall values).

Usage:
    python analyze_face_hypothesis.py --data-dir /path/to/asl-signs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from eval_confidence import category_of, run_tflite  # noqa: E402

# Pre-committed: ASL glosses whose citation form is articulated at a
# specific location on the face or head, so that *where the hand is
# relative to the face* -- not just its shape or motion -- carries the
# meaning. Written from the vocabulary list by meaning, before any
# per-class score was inspected.
FACE_LOCATED = {
    "chin", "lips", "tongue", "cheek", "ear", "eye", "nose", "face", "mouth", "tooth",
    "hair", "head", "awake", "hungry", "thirsty", "taste", "smile", "cry", "listen",
    "hear", "think", "yucky", "drink", "morning", "sleepy", "sick", "mad", "sad",
    "happy", "toothbrush", "shhh", "quiet", "kiss", "talk", "say", "hello", "nap",
    "sleep", "dad", "mom", "grandma", "grandpa", "boy", "girl", "man", "cute", "pretty",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--alphabet-path", default=None)
    p.add_argument("--digit-path", default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()

    # Only the *word* portion's landmark tensor is ever needed here, but the
    # train/val split must still be the exact one the model was trained with
    # -- which is computed over the full words+letters+digits group list.
    # So: load the words' X (the only heavy array), and load just the LABELS
    # of the letter/digit sources to rebuild the same combined group vector.
    # Materializing the full combined X instead would peak ~3GB and has
    # actually OOM'd on this machine; this keeps it to the ~1.3GB the words
    # alone require.
    import train_asl_combined as t

    class A:
        words_dir = args.data_dir or "data/asl-signs"
        alphabet_path = args.alphabet_path or "data/asl_alphabet_landmarks.npz"
        digit_path = args.digit_path or "data/asl_digit_landmarks.npz"
        seq_len, no_cache, workers, csv_name = 40, False, None, "train.csv"

    a = A()
    print("Loading ASL words...")
    X_w, y_w, s_w = t.load_words(a)
    with np.load(a.alphabet_path, allow_pickle=True) as d:
        y_a = [f"letter_{v}" for v in d["y"]]
    with np.load(a.digit_path, allow_pickle=True) as d:
        y_d = [f"digit_{v}" for v in d["y"]]
    s_a = [f"letter_{i}" for i in range(len(y_a))]
    s_d = [f"digit_{i}" for i in range(len(y_d))]

    all_labels = list(y_w) + y_a + y_d
    label_names = sorted(set(all_labels))
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in all_labels], dtype=np.int64)
    signers = np.array(list(s_w) + s_a + s_d)
    n_words = len(y_w)

    model_path = args.model or os.path.join(
        os.path.dirname(__file__), "..", "app", "assets", "models", "asl", "model.tflite"
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(splitter.split(np.zeros((len(y), 1), dtype=np.int8), y, groups=signers))
    word_val_idx = val_idx[val_idx < n_words]  # words occupy the first n_words rows
    print(f"Val words: {len(word_val_idx)} of {len(val_idx)} total val samples")

    X_val, y_val = X_w[word_val_idx], y[word_val_idx]
    del X_w
    scores = run_tflite(model_path, X_val)
    preds = scores.argmax(axis=1)

    face_recalls, other_recalls = [], []
    face_names, other_names = [], []
    for idx in np.unique(y_val):
        label = label_names[idx]
        if category_of(label) != "word":
            continue  # letters/digits are a different evaluation regime entirely
        m = y_val == idx
        recall = float(np.mean(preds[m] == idx))
        if label in FACE_LOCATED:
            face_recalls.append(recall)
            face_names.append((label, recall))
        else:
            other_recalls.append(recall)
            other_names.append((label, recall))

    face_recalls = np.array(face_recalls)
    other_recalls = np.array(other_recalls)
    u, pval = mannwhitneyu(face_recalls, other_recalls, alternative="less")

    print(f"Face-located words : n={len(face_recalls):3d}  mean recall={face_recalls.mean():.4f}  median={np.median(face_recalls):.4f}")
    print(f"Other words        : n={len(other_recalls):3d}  mean recall={other_recalls.mean():.4f}  median={np.median(other_recalls):.4f}")
    print(f"\nGap in mean recall : {other_recalls.mean() - face_recalls.mean():+.4f}")
    print(f"Mann-Whitney U     : U={u:.1f}, one-sided p={pval:.3e}")
    print("  (H1: face-located words are recognized WORSE than other words)")

    matched = [n for n in face_names if n[0] in FACE_LOCATED]
    matched.sort(key=lambda kv: kv[1])
    print("\nLowest-recall face-located words:")
    for label, r in matched[:12]:
        print(f"  {label:16s} {r:.3f}")

    out = {
        "face_located": {
            "n": int(len(face_recalls)),
            "mean_recall": float(face_recalls.mean()),
            "median_recall": float(np.median(face_recalls)),
        },
        "other_words": {
            "n": int(len(other_recalls)),
            "mean_recall": float(other_recalls.mean()),
            "median_recall": float(np.median(other_recalls)),
        },
        "mannwhitney_u": float(u),
        "p_one_sided": float(pval),
        "face_located_words_evaluated": sorted(n for n, _ in face_names),
    }
    path = os.path.join(os.path.dirname(__file__), "face_hypothesis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
