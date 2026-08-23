"""One-time offline extraction: runs the real MediaPipe HandLandmarker
(the same Tasks-API model family the Flutter app's hand_landmarker plugin
uses, `hand_landmarker.task`) over rayeed045/american-sign-language-digit-
dataset to produce real (21, 3) hand landmarks per image -- same approach,
same reasoning, as extract_asl_alphabet_landmarks.py for the ASL alphabet:
this source ships raw JPEGs, not landmarks, so there is nothing to trust
or verify from a third party's export. A folder for digit "3" produces
landmarks for whatever hand shape MediaPipe actually found in that
folder's images, full stop.

This is the digit counterpart requested after the alphabet was added --
ASL previously had no digit vocabulary at all (BdSL has one; ASL didn't).
Source folder layout (Kaggle, CC0-1.0):
    American Sign Language Digits Dataset/<digit>/Input Images - Sign <digit>/*.jpeg

No participant/signer metadata exists for this dataset (the images derive
from the Massey University 2D static ASL gesture dataset, not a
multi-signer video corpus), so -- like the alphabet portion -- its share
of the train/val split in train_asl_combined.py is a plain random split,
not a held-out-signer split. Documented here and in the app docs, not
hidden.

Usage:
    python extract_asl_digit_landmarks.py \
        --images-dir data/asl-digits/"American Sign Language Digits Dataset" \
        --max-per-class 400
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), ".models", "hand_landmarker.task")
DIGITS = [str(d) for d in range(10)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--max-per-class", type=int, default=400)
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "data", "asl_digit_landmarks.npz"),
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
    for digit in DIGITS:
        folder = os.path.join(args.images_dir, digit, f"Input Images - Sign {digit}")
        paths = sorted(
            glob.glob(os.path.join(folder, "*.jpeg")) + glob.glob(os.path.join(folder, "*.jpg"))
        )[: args.max_per_class]
        found = 0
        for path in paths:
            landmarks = extract_one(landmarker, path)
            if landmarks is None:
                skipped += 1
                continue
            X.append(landmarks)
            y.append(digit)
            found += 1
        print(f"{digit}: {found}/{len(paths)} images had a detectable hand", flush=True)

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
