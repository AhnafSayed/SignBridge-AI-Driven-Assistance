"""Renders our ASL alphabet + digit landmark corpora into the same
skeleton-image representation as Devansh-47's AtoZ_3.1 corpus, so the two
can be merged into one training set (see src/skeleton_render.py for the
geometry-matching rationale and validation).

Per real sample, N_AUGMENT_COPIES additional renders are produced with
scale/rotation jitter -- both to act as ordinary augmentation, and
specifically to spread our hand-size distribution across theirs (see
SCALE_JITTER's docstring) so a classifier cannot learn "which corpus" as
a shortcut for "which letter".

Every copy, including the first, is independently mirrored with
probability MIRROR_PROB. Both source corpora (grassknoted/asl-alphabet
here, AtoZ_3.1 for the other side of the merge) were collected from
whichever hand each contributor happened to sign with, with no attempt to
balance left- and right-handed samples -- a fingerspelling letter's label
does not depend on which hand performs it, so a corpus skewed toward one
hand is a data gap, not a real class property, and mirroring closes it
the same way it already does for the shipped landmark-sequence model
(src/augment.py's mirror_prob). Left unaddressed, a signer of the hand
underrepresented in training would face a quietly worse model with no
indication why.

Output layout mirrors AtoZ_3.1 exactly (one directory per class letter,
sequentially numbered JPEGs), so the two trees can be point at the same
loader with zero special-casing:

    data/asl_letters_skeleton/<LETTER>/<n>.jpg      -- from our landmarks
    data/asl_digits_skeleton/<DIGIT>/<n>.jpg        -- from our landmarks
    data/asl_letters_skeleton/manifest.csv          -- see below
    data/asl_digits_skeleton/manifest.csv

The manifest exists for one reason: to make a leakage-safe split
possible. 1 + N_AUGMENT_COPIES files are rendered from every real sample,
so a plain random split over image files would happily put three
augmented copies of the *same* base landmark sample in training and the
fourth in validation -- near-duplicates on both sides, inflating held-out
accuracy without the model having generalized to anything. The manifest's
`base_index` column records which real sample each file came from, so the
training script can group-split on it (all copies of one base sample
always land on the same side), exactly as this project's other held-out
evaluations group by signer rather than by row.

Usage:
    python render_skeleton_dataset.py
"""

from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from skeleton_render import SCALE_JITTER, render_from_landmarks  # noqa: E402

N_AUGMENT_COPIES = 3  # extra rendered variants per real sample
ROTATION_RANGE_DEG = 12.0
MIRROR_PROB = 0.5


def render_corpus(npz_path: str, out_dir: str, rng: np.random.Generator) -> int:
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    counters: dict[str, int] = {}
    manifest_rows: list[dict] = []
    mirrored_count = 0
    written = 0
    for i in range(len(X)):
        label = str(y[i])
        cls_dir = os.path.join(out_dir, label)
        os.makedirs(cls_dir, exist_ok=True)
        n = counters.get(label, 0)

        for copy_idx in range(1 + N_AUGMENT_COPIES):
            mirror = bool(rng.random() < MIRROR_PROB)
            mirrored_count += mirror
            if copy_idx == 0:
                # The first copy of each sample is the real render: no
                # scale/rotation jitter, only the (independent) mirror
                # coin-flip, so at least one un-augmented rendering of
                # every real sample survives in the training set.
                im = render_from_landmarks(X[i], mirror=mirror)
            else:
                scale = float(rng.uniform(*SCALE_JITTER))
                angle = float(rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG))
                im = render_from_landmarks(X[i], scale=scale, rotation_deg=angle, mirror=mirror)
            filename = f"{n}.jpg"
            cv2.imwrite(os.path.join(cls_dir, filename), im)
            manifest_rows.append({
                "relpath": os.path.join(label, filename),
                "label": label,
                "base_index": i,  # groups all copies of one real sample
                "copy_idx": copy_idx,
                "mirrored": int(mirror),
            })
            written += 1
            n += 1

        counters[label] = n

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relpath", "label", "base_index", "copy_idx", "mirrored"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  ({mirrored_count}/{written} = {mirrored_count / written:.1%} mirrored)")
    return written


def main():
    rng = np.random.default_rng(42)
    here = os.path.dirname(__file__)

    jobs = [
        ("data/asl_alphabet_landmarks.npz", "data/asl_letters_skeleton"),
        ("data/asl_digit_landmarks.npz", "data/asl_digits_skeleton"),
    ]
    for npz_name, out_name in jobs:
        npz_path = os.path.join(here, npz_name)
        out_dir = os.path.join(here, out_name)
        print(f"Rendering {npz_path} -> {out_dir} ...")
        n = render_corpus(npz_path, out_dir, rng)
        print(f"  wrote {n} images ({1 + N_AUGMENT_COPIES} per real sample)")


if __name__ == "__main__":
    main()
