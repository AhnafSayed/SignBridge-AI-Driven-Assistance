"""Decisive old-vs-new comparison for the ASL letters/digits swap made this
session: does the rotation-augmented direct static classifier
(models/asl_static_direct/) actually beat reverting to the pre-existing
sequence model (models/asl_combined_cnn_rot15_baseline/, what
app/assets/models/asl/model.tflite already is) on real held-out rotation
robustness -- not assumed, measured on the identical rotated letter set for
both, following the exact methodology eval_rotation_robustness.py already
established for this project.

Usage:
    python compare_asl_letter_rotation.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from landmarks import (  # noqa: E402
    NUM_LANDMARKS,
    TWO_HAND_LANDMARKS,
    embed_single_hand,
    normalize_hand_frame,
    prepare_two_hand_sequence,
)

TEST_ANGLES_DEG = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
CHECK_LETTERS = ("V", "W", "P", "M")
SEQ_LEN = 40

OLD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "asl_combined_cnn_rot15_baseline")
NEW_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "asl_static_direct")


def rotate_land_block(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    out = seq.copy()
    has_anchor = out.shape[-2] > TWO_HAND_LANDMARKS
    n_land = TWO_HAND_LANDMARKS if has_anchor else out.shape[-2]
    land = out[..., :n_land, :]
    a = np.deg2rad(angle_deg)
    cos, sin = np.cos(a), np.sin(a)
    x, y = land[..., 0].copy(), land[..., 1].copy()
    land[..., 0] = x * cos - y * sin
    land[..., 1] = x * sin + y * cos
    return out


def rotate_hand_frame(X: np.ndarray, angle_deg: float) -> np.ndarray:
    out = X.copy()
    a = np.deg2rad(angle_deg)
    cos, sin = np.cos(a), np.sin(a)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = x * cos - y * sin
    out[..., 1] = x * sin + y * cos
    return out


def run_tflite(model_path: str, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_detail, out_detail = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.resize_tensor_input(in_detail["index"], [batch_size] + list(X.shape[1:]))
    interp.allocate_tensors()
    out_all = []
    for start in range(0, len(X), batch_size):
        chunk = X[start : start + batch_size]
        n = len(chunk)
        if n < batch_size:
            pad = np.zeros((batch_size - n,) + X.shape[1:], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        interp.set_tensor(in_detail["index"], chunk)
        interp.invoke()
        out_all.append(interp.get_tensor(out_detail["index"])[:n])
    return np.concatenate(out_all, axis=0)


def main():
    npz = np.load(os.path.join(os.path.dirname(__file__), "data/asl_alphabet_landmarks.npz"), allow_pickle=True)
    X_raw, y_raw = npz["X"], npz["y"]
    print(f"Loaded {len(y_raw)} real ASL alphabet static letter poses (canonical, ~0 deg)")

    with open(os.path.join(OLD_MODEL_DIR, "labels.json")) as f:
        old_labels = json.load(f)
    with open(os.path.join(NEW_MODEL_DIR, "labels.json")) as f:
        new_labels = json.load(f)

    old_to_idx = {name: i for i, name in enumerate(old_labels)}
    new_to_idx = {name: i for i, name in enumerate(new_labels)}
    old_letter_idx = np.array([i for i, n in enumerate(old_labels) if n.startswith("letter_")])
    new_letter_idx = np.array([i for i, n in enumerate(new_labels) if n.startswith("letter_")])
    y_true_old = np.array([old_to_idx[f"letter_{c}"] for c in y_raw])
    y_true_new = np.array([new_to_idx[f"letter_{c}"] for c in y_raw])

    # Old model input: (N, 40, 43, 3) two-hand sequence contract.
    X_seq = np.stack([
        prepare_two_hand_sequence(embed_single_hand(frame)[np.newaxis, ...], SEQ_LEN, neutral_anchor=True)
        for frame in X_raw
    ]).astype(np.float32)

    # New model input: (N, 21, 3) normalized single-hand frame, flattened to 63.
    X_frame = np.stack([normalize_hand_frame(f) for f in X_raw]).astype(np.float32)

    old_path = os.path.join(OLD_MODEL_DIR, "model.tflite")
    new_path = os.path.join(NEW_MODEL_DIR, "model.tflite")

    print(f"\n{'angle':>6s} {'OLD (shipped pre-today)':>26s} {'NEW (rotation-fixed direct)':>30s}")
    results = {}
    for angle in TEST_ANGLES_DEG:
        X_seq_rot = rotate_land_block(X_seq, angle)
        scores_old = run_tflite(old_path, X_seq_rot)
        masked_old = np.full_like(scores_old, -np.inf)
        masked_old[:, old_letter_idx] = scores_old[:, old_letter_idx]
        pred_old = masked_old.argmax(axis=1)
        acc_old = float(np.mean(pred_old == y_true_old))

        X_frame_rot = rotate_hand_frame(X_frame, angle).reshape(len(X_frame), 63)
        scores_new = run_tflite(new_path, X_frame_rot)
        masked_new = np.full_like(scores_new, -np.inf)
        masked_new[:, new_letter_idx] = scores_new[:, new_letter_idx]
        pred_new = masked_new.argmax(axis=1)
        acc_new = float(np.mean(pred_new == y_true_new))

        results[angle] = (acc_old, acc_new)
        print(f"{angle:>+5d}° {acc_old:>25.4f} {acc_new:>29.4f}   (delta {acc_new - acc_old:+.4f})")

        for letter in CHECK_LETTERS:
            li_old, li_new = old_to_idx.get(f"letter_{letter}"), new_to_idx.get(f"letter_{letter}")
            if li_old is None or li_new is None:
                continue
            m_old, m_new = y_true_old == li_old, y_true_new == li_new
            r_old = float(np.mean(pred_old[m_old] == li_old)) if m_old.any() else float("nan")
            r_new = float(np.mean(pred_new[m_new] == li_new)) if m_new.any() else float("nan")
            print(f"        {letter} recall: old={r_old:.2f}  new={r_new:.2f}")

    with open("asl_letter_rotation_comparison.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\nWrote asl_letter_rotation_comparison.json")


if __name__ == "__main__":
    main()
