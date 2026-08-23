"""One-time offline extraction: runs the real MediaPipe HandLandmarker
(the same Tasks-API model family the Flutter app's hand_landmarker plugin
uses, `hand_landmarker.task`) over the grassknoted/asl-alphabet image
dataset to produce real (21, 3) hand landmarks per image.

Why this exists: grassknoted/asl-alphabet ships raw JPEGs, not landmarks --
unlike BdSL's digit/letter dataset, there is no pre-extracted landmark CSV
to trust or verify here. Rather than guess, this script generates the
landmarks itself with the actual detector, so there's nothing to verify
after the fact: a folder named "A" produces landmarks for whatever hand
shape MediaPipe found in that folder's images, full stop.

Labels: the 26 letter folders (A-Z) only. The dataset's `space`, `del`,
and `nothing` folders are utility signs for continuous fingerspelling
input, not part of the alphabet, and are skipped.

No participant/signer metadata exists for this dataset (unlike asl-signs
or BdSLW60), so the resulting train/val split (done later, in
train_asl_combined.py) for this portion is a plain random split, not a
held-out-signer split -- documented there, not hidden.

Usage:
    python extract_asl_alphabet_landmarks.py \
        --images-dir data/asl-alphabet/asl_alphabet_train/asl_alphabet_train \
        --max-per-class 400
"""

from __future__ import annotations

import argparse
import glob
import os
import string
import sys

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), ".models", "hand_landmarker.task")
LETTERS = list(string.ascii_uppercase)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--max-per-class", type=int, default=400)
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "data", "asl_alphabet_landmarks.npz"),
    )
    return p.parse_args()


def build_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_one(landmarker, path: str) -> np.ndarray | None:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def main():
    args = parse_args()
    landmarker = build_landmarker()

    X, y = [], []
    skipped = 0
    for letter in LETTERS:
        folder = os.path.join(args.images_dir, letter)
        paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))[: args.max_per_class]
        found = 0
        for path in paths:
            landmarks = extract_one(landmarker, path)
            if landmarks is None:
                skipped += 1
                continue
            X.append(landmarks)
            y.append(letter)
            found += 1
        print(f"{letter}: {found}/{len(paths)} images had a detectable hand", flush=True)

    landmarker.close()

    if not X:
        print("No hands detected in any image -- aborting.", file=sys.stderr)
        sys.exit(1)

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, y=y)
    print(f"\nExtracted {len(X)} landmark samples ({skipped} images skipped, no hand found)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
