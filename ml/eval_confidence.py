"""Diagnostic: what does the shipped .tflite model's confidence actually
look like on real held-out data, broken down by category (word/digit/
letter)?

Written to answer a concrete real-device complaint: the app gates a
recognition behind two hardcoded confidence thresholds
(HomeScreenState._minCommitConfidence = 0.55, _controlOverrideConfidence =
0.6 in app/lib/screens/home_screen.dart) that were picked by feel, not
measured against this model's actual output distribution. If most correct
predictions -- especially for fingerspelling letters, which is where the
complaint concentrated -- never reach 0.55 to begin with, the gate silently
throws away almost everything regardless of whether recognition is working.

Reproduces the exact same train/val split train_asl_combined.py and
train_bdsl_combined.py use (same cache keys, same GroupShuffleSplit
random_state=42), so "val" here is genuinely never-seen-signer data, not
train data being peeked at.

Usage:
    python eval_confidence.py --language asl
    python eval_confidence.py --language bdsl
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def category_of(label: str) -> str:
    if label.startswith("digit_"):
        return "digit"
    if label.startswith("letter_"):
        return "letter"
    return "word"


def load_asl(args):
    import train_asl_combined as t

    class A:
        words_dir = args.data_dir or "data/asl-signs"
        alphabet_path = args.alphabet_path or "data/asl_alphabet_landmarks.npz"
        digit_path = args.digit_path or "data/asl_digit_landmarks.npz"
        seq_len = 40
        no_cache = False
        workers = None
        csv_name = "train.csv"

    a = A()
    print("Loading ASL words...")
    X_w, y_w, s_w = t.load_words(a)
    print("Loading ASL alphabet...")
    X_a, y_a, s_a = t.load_alphabet(a.alphabet_path, a.seq_len)
    print("Loading ASL digits...")
    X_d, y_d, s_d = t.load_digits(a.digit_path, a.seq_len)
    all_labels = list(y_w) + list(y_a) + list(y_d)
    label_names = sorted(set(all_labels))
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    X = np.concatenate([X_w, X_a, X_d], axis=0).astype(np.float32)
    y = np.array([label_to_idx[label] for label in all_labels], dtype=np.int64)
    signers = np.array(list(s_w) + list(s_a) + list(s_d))
    return X, y, signers, label_names


def load_bdsl(args):
    import train_bdsl_combined as t
    import dataset_bdsl_static as static_ds

    class A:
        words_dir = args.data_dir or "data/bdslw60"
        digits_dir = args.digits_dir or "data/bdsl_digits/CSV Files/Sign Digits - CSV Files"
        letters_dir = args.letters_dir or "data/bdsl_digits/CSV Files/Sign Letters - CSV Files"
        seq_len = 40
        max_static_samples_per_class = 100

    a = A()
    print("Loading BdSL words...")
    X_w, y_w, s_w = t.load_words(a)
    print("Loading BdSL digits...")
    X_d, y_d, s_d = t.load_static(
        a.digits_dir, {i: str(i) for i in range(10)}, "digit", a.seq_len, a.max_static_samples_per_class
    )
    print("Loading BdSL letters...")
    X_l, y_l, s_l = t.load_static(
        a.letters_dir, dict(enumerate(static_ds.LETTER_NAMES)), "letter", a.seq_len,
        a.max_static_samples_per_class,
    )
    all_labels = list(y_w) + list(y_d) + list(y_l)
    label_names = sorted(set(all_labels))
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    X = np.concatenate([X_w, X_d, X_l], axis=0).astype(np.float32)
    y = np.array([label_to_idx[label] for label in all_labels], dtype=np.int64)
    signers = np.array(list(s_w) + list(s_d) + list(s_l))
    return X, y, signers, label_names


def run_tflite(model_path, X, batch_size=256):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    # Real .tflite models from this pipeline have a fixed batch dim baked
    # in at conversion time -- resize the input tensor to the batch size we
    # actually want to feed instead of guessing what it was exported with.
    interpreter.resize_tensor_input(input_details["index"], [batch_size] + list(input_details["shape"][1:]))
    interpreter.allocate_tensors()

    n = len(X)
    out = np.zeros((n, output_details["shape"][-1]), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]
        if len(chunk) < batch_size:
            pad = np.zeros((batch_size - len(chunk),) + chunk.shape[1:], dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        interpreter.set_tensor(input_details["index"], chunk.astype(np.float32))
        interpreter.invoke()
        result = interpreter.get_tensor(output_details["index"])
        out[start:end] = result[: end - start]
    return out


def report(label_names, y_val, scores):
    preds = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    correct = preds == y_val
    categories = np.array([category_of(label_names[i]) for i in y_val])

    print(f"\nOverall: {correct.mean():.3f} accuracy over {len(y_val)} val samples\n")
    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]

    print("-- recall: of predictions that were actually CORRECT, fraction clearing each bar --")
    print("(low = the gate is silently discarding real, correct recognitions)")
    header = "category    n    acc   " + "  ".join(f">={t:.2f}" for t in thresholds)
    print(header)
    for cat in ("word", "digit", "letter"):
        mask = categories == cat
        if mask.sum() == 0:
            continue
        cat_correct = correct[mask]
        cat_conf = confidences[mask]
        acc = cat_correct.mean()
        correct_conf = cat_conf[cat_correct]
        fracs = [
            (correct_conf >= t).mean() if len(correct_conf) else float("nan") for t in thresholds
        ]
        frac_str = "  ".join(f"{f:5.2f}" for f in fracs)
        print(f"{cat:8s} {mask.sum():5d} {acc:.3f}   {frac_str}")

    print("\n-- precision: of predictions clearing each bar (right or wrong), fraction actually CORRECT --")
    print("(low = lowering the bar lets a lot of wrong guesses get committed as if real)")
    print(header)
    for cat in ("word", "digit", "letter"):
        mask = categories == cat
        if mask.sum() == 0:
            continue
        cat_correct = correct[mask]
        cat_conf = confidences[mask]
        acc = cat_correct.mean()
        precisions = []
        for t in thresholds:
            above = cat_conf >= t
            precisions.append(cat_correct[above].mean() if above.any() else float("nan"))
        prec_str = "  ".join(f"{p:5.2f}" for p in precisions)
        print(f"{cat:8s} {mask.sum():5d} {acc:.3f}   {prec_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["asl", "bdsl"], required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--alphabet-path", default=None)
    p.add_argument("--digit-path", default=None, help="ASL digit landmarks npz")
    p.add_argument("--digits-dir", default=None, help="BdSL digit CSV dir")
    p.add_argument("--letters-dir", default=None)
    p.add_argument(
        "--model",
        default=None,
        help="Path to model.tflite; defaults to the shipped app asset for this language",
    )
    args = p.parse_args()

    if args.language == "asl":
        X, y, signers, label_names = load_asl(args)
        model_path = args.model or os.path.join(
            os.path.dirname(__file__), "..", "app", "assets", "models", "asl", "model.tflite"
        )
    else:
        X, y, signers, label_names = load_bdsl(args)
        model_path = args.model or os.path.join(
            os.path.dirname(__file__), "..", "app", "assets", "models", "bdsl", "model.tflite"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(splitter.split(X, y, groups=signers))
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Val set: {len(val_idx)} samples from {len(set(signers[val_idx]))} unseen groups")

    print(f"Running {model_path} ...")
    scores = run_tflite(model_path, X_val)
    report(label_names, y_val, scores)


if __name__ == "__main__":
    main()
