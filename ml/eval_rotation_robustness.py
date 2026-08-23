"""Does wider rotation augmentation actually help with real hand rotation?

Real-device testing found ASL "V" and "W" reading as "Q"/"R" specifically
when the signer's hand was rotated relative to the camera, and found it
went away when the hand was held more upright. The standard held-out
benchmark (eval_full_metrics.py) cannot see this at all -- it evaluates on
the *un-rotated* test split, which contains none of the rotation that
caused the problem in the first place. Comparing two models only on that
benchmark would tell us nothing about which one is actually more robust to
rotation; it would just tell us neither model's un-rotated accuracy
regressed.

So this constructs the test the bug actually calls for: take real ASL
alphabet static poses, apply the same 2D in-plane rotation transform
training's own augmentation uses (src/augment.py, applied to the
hand-relative normalized landmark block, never to the frame-absolute
anchor row), across a range of angles, and measure each model's accuracy
at each angle. This directly answers "does the wider-rotation model hold
up better as rotation increases", which is the actual question, rather
than treating "the aggregate benchmark didn't regress" as if it were an
answer to a question it cannot see.

Usage:
    python eval_rotation_robustness.py \\
        --model-a models/asl_combined_cnn_rot15_baseline/model.tflite \\
        --model-b models/asl_combined_cnn/model.tflite
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from landmarks import TWO_HAND_LANDMARKS, embed_single_hand, prepare_two_hand_sequence  # noqa: E402

TEST_ANGLES_DEG = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

# V/W: the originally-reported confusion. P/M: a second, live-tested report
# on the rotation-fixed model that was never isolated from a specific
# controlled test -- added here so it gets exactly the same measurement V/W
# did, rather than being accepted or dismissed on an informal impression.
CHECK_LETTERS = ("V", "W", "P", "M")


def rotate_land_block(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    """Applies the exact rotation augment.py uses -- 2D rotation of the
    normalized, hand-relative landmark block's x/y, leaving the trailing
    frame-absolute anchor row (if present) untouched -- to a *fixed* angle
    rather than a random jittered one, since this is a controlled sweep,
    not training-time augmentation."""
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


def load_letter_test_set(seq_len: int):
    npz = np.load(os.path.join(os.path.dirname(__file__), "data/asl_alphabet_landmarks.npz"),
                  allow_pickle=True)
    X_raw, y_raw = npz["X"], npz["y"]
    prepared = np.stack([
        prepare_two_hand_sequence(embed_single_hand(frame)[np.newaxis, ...], seq_len, neutral_anchor=True)
        for frame in X_raw
    ]).astype(np.float32)
    return prepared, y_raw


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


def evaluate_model(model_path: str, X_by_angle: dict, y_raw, label_names, letter_label_idx):
    print(f"\n=== {model_path} ===")
    to_idx = {name: i for i, name in enumerate(label_names)}
    y_true_idx = np.array([to_idx[f"letter_{c}"] for c in y_raw])
    results = {}
    v_w_recall = {}
    confusions = {}
    for angle, X in X_by_angle.items():
        scores = run_tflite(model_path, X)
        masked = np.full_like(scores, -np.inf)
        masked[:, letter_label_idx] = scores[:, letter_label_idx]
        pred_idx = masked.argmax(axis=1)
        acc = float(np.mean(pred_idx == y_true_idx))
        results[angle] = acc
        row = []
        for letter in CHECK_LETTERS:
            li = to_idx[f"letter_{letter}"]
            m = y_true_idx == li
            recall = float(np.mean(pred_idx[m] == li)) if m.any() else float("nan")
            row.append(f"{letter}={recall:.2f}")
            # Not just recall -- what it gets confused *as*, when wrong.
            # Recall alone would hide "P misread as M" behind "P recall
            # dropped"; this names the actual confusion, the same way a
            # confusion-matrix cell does.
            wrong = m & (pred_idx != li)
            if wrong.any():
                mistaken_idx, counts = np.unique(pred_idx[wrong], return_counts=True)
                top = mistaken_idx[np.argmax(counts)]
                top_label = label_names[top].removeprefix("letter_")
                top_frac = counts.max() / wrong.sum()
                confusions.setdefault(angle, {})[letter] = f"-> {top_label} ({top_frac:.0%} of errors)"
        v_w_recall[angle] = ", ".join(row)
        print(f"  rotation {angle:+3d} deg -> letter accuracy {acc:.4f}   ({v_w_recall[angle]})")
        if angle in confusions:
            for letter, desc in confusions[angle].items():
                print(f"      {letter} errors mostly {desc}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", required=True, help="Baseline (e.g. rotation_deg=15 training)")
    p.add_argument("--model-b", required=True, help="Candidate (e.g. rotation_deg=30 training)")
    args = p.parse_args()

    with open(os.path.join(os.path.dirname(args.model_a), "labels.json")) as f:
        label_names = json.load(f)
    letter_label_idx = np.array([i for i, n in enumerate(label_names) if n.startswith("letter_")])

    seq_len = 40
    X_prepared, y_raw = load_letter_test_set(seq_len)
    print(f"Loaded {len(y_raw)} real ASL alphabet static poses (canonical, ~0 deg orientation)")

    X_by_angle = {angle: rotate_land_block(X_prepared, angle) for angle in TEST_ANGLES_DEG}

    results_a = evaluate_model(args.model_a, X_by_angle, y_raw, label_names, letter_label_idx)
    results_b = evaluate_model(args.model_b, X_by_angle, y_raw, label_names, letter_label_idx)

    print("\n=== Summary: letter accuracy by rotation angle ===")
    print(f"{'angle':>8s} {'model_a (15deg)':>16s} {'model_b (30deg)':>16s} {'delta':>8s}")
    for angle in TEST_ANGLES_DEG:
        a, b = results_a[angle], results_b[angle]
        print(f"{angle:>+7d}  {a:16.4f} {b:16.4f} {b - a:+8.4f}")

    out = {"model_a": results_a, "model_b": results_b}
    with open("rotation_robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote rotation_robustness_results.json")


if __name__ == "__main__":
    main()
