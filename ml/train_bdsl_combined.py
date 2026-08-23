"""Trains one combined BdSL model over three sources: words (BdSLW60,
real temporal sequences, both hands), digits, and alphabet letters (both
from rayeed045/bangla-sign-language-dataset, single-frame static poses,
one hand only).

Digit/letter samples are single photographed frames, not videos, and only
ever show one hand (fingerspelling and counting are one-handed gestures in
BdSL) -- each is embedded into slot 0 of the two-hand contract
(landmarks.embed_single_hand) and expanded to the fixed-length sequence
shape the word model uses (landmarks.prepare_two_hand_sequence, which per
its own resample logic just repeats a length-1 input across the full
window). This lets one architecture and one inference path handle all
three sources: it's still exactly one .tflite model consuming a
(seq_len, 42, 3) window, just with more classes in its output.

Usage:
    python train_bdsl_combined.py \
        --words-dir data/bdslw60 \
        --digits-dir "data/bdsl_digits/CSV Files/Sign Digits - CSV Files" \
        --letters-dir "data/bdsl_digits/CSV Files/Sign Letters - CSV Files" \
        --epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import dataset_bdsl as words_ds  # noqa: E402
import dataset_bdsl_static as static_ds  # noqa: E402
from augment import build_augmented_trainset  # noqa: E402
from landmarks import DEFAULT_SEQ_LEN, embed_single_hand, prepare_two_hand_sequence  # noqa: E402
from model import build_1dcnn, build_transformer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--words-dir", default="data/bdslw60")
    p.add_argument("--digits-dir", default="data/bdsl_digits/CSV Files/Sign Digits - CSV Files")
    p.add_argument("--letters-dir", default="data/bdsl_digits/CSV Files/Sign Letters - CSV Files")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--arch", choices=["cnn", "transformer"], default="cnn")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-static-samples-per-class", type=int, default=100)
    p.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Extra augmented copies of the training split (mirror/rotate/scale/jitter), "
        "each the same size as the real training set -- 0 disables augmentation",
    )
    p.add_argument(
        "--class-weight",
        action="store_true",
        help="Weight each class inversely to its frequency. Targets the measured "
        "zero-recall failure on rare alphabet letters (see the comment at the fit() "
        "call); trades a little overall accuracy for coverage of rare classes.",
    )
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def load_words(args):
    index = words_ds.load_index(args.words_dir)
    vocab = index.signs  # all 60
    X, y, signers, label_names = words_ds.build_dataset(index, vocab, seq_len=args.seq_len)
    labels = [label_names[i] for i in y]
    signers = [f"word_{s}" for s in signers]
    return X, labels, signers


def load_static(root, names_by_index, prefix, seq_len, max_samples_per_class):
    X_raw, y_raw, signers_raw = static_ds.load_static_dataset(
        root, max_samples_per_class=max_samples_per_class
    )
    X = np.stack(
        [prepare_two_hand_sequence(embed_single_hand(frame)[np.newaxis, ...], seq_len, neutral_anchor=True) for frame in X_raw]
    )
    labels = [f"{prefix}_{names_by_index[i]}" for i in y_raw]
    signers = [f"{prefix}_{s}" for s in signers_raw]
    return X, labels, signers


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "models", f"bdsl_combined_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading BdSLW60 words...")
    X_w, y_w, s_w = load_words(args)
    print(f"  {len(y_w)} word sequences, {len(set(y_w))} classes")

    print("Loading BdSL digits...")
    X_d, y_d, s_d = load_static(
        args.digits_dir, {i: str(i) for i in range(10)}, "digit", args.seq_len,
        args.max_static_samples_per_class,
    )
    print(f"  {len(y_d)} digit samples, {len(set(y_d))} classes")

    print("Loading BdSL letters...")
    X_l, y_l, s_l = load_static(
        args.letters_dir, dict(enumerate(static_ds.LETTER_NAMES)), "letter", args.seq_len,
        args.max_static_samples_per_class,
    )
    print(f"  {len(y_l)} letter samples, {len(set(y_l))} classes")

    all_labels = list(y_w) + list(y_d) + list(y_l)
    label_names = sorted(set(all_labels))
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    num_classes = len(label_names)
    print(f"\nCombined vocabulary: {num_classes} classes "
          f"({len(set(y_w))} words + {len(set(y_d))} digits + {len(set(y_l))} letters)")

    X = np.concatenate([X_w, X_d, X_l], axis=0).astype(np.float32)
    y = np.array([label_to_idx[label] for label in all_labels], dtype=np.int64)
    signers = np.array(list(s_w) + list(s_d) + list(s_l))
    print(f"Total sequences: {len(y)}, total signers: {len(set(signers))}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=signers))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(
        f"Train: {len(train_idx)} from {len(set(signers[train_idx]))} signers | "
        f"Val: {len(val_idx)} from {len(set(signers[val_idx]))} unseen signers"
    )
    seq_len, num_landmarks = X.shape[1], X.shape[2]
    del X  # dead weight from here on -- see train.py's matching comment.

    if args.augment_copies > 0:
        rng = np.random.default_rng(42)
        # Extra *training* data only -- X_val/y_val above stay real,
        # unaugmented, never-seen-signer data, so held-out accuracy still
        # measures true generalization.
        X_train, y_train = build_augmented_trainset(X_train, y_train, args.augment_copies, rng)
        print(f"Augmented train set: {len(X_train)} sequences ({args.augment_copies} extra copies)")

    build_fn = build_1dcnn if args.arch == "cnn" else build_transformer
    model = build_fn(seq_len=seq_len, num_classes=num_classes, num_landmarks=num_landmarks)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5),
    ]

    # Class weighting exists to address a specific measured failure: on the
    # unweighted model, 7 of the 37 BdSL alphabet letters scored *zero*
    # recall on held-out signers -- the model never identified them
    # correctly even once (see docs/PROJECT_REPORT.md §5.1). The cause is
    # imbalance rather than difficulty alone: the 60 word classes supply
    # far more training sequences than the 37 letter classes, so plain
    # cross-entropy can drive overall accuracy up by collapsing rare
    # letters into their more frequent neighbours and barely be penalized
    # for it. Inverse-frequency weights make each class contribute equally
    # to the loss regardless of how many samples it has.
    class_weight = None
    if args.class_weight:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        # Guard against divide-by-zero for any class absent from this split.
        safe = np.where(counts > 0, counts, 1.0)
        weights = len(y_train) / (num_classes * safe)
        weights[counts == 0] = 0.0
        class_weight = {i: float(w) for i, w in enumerate(weights)}
        present = counts > 0
        print(
            f"Class weighting on: weight range "
            f"{weights[present].min():.2f}-{weights[present].max():.2f} "
            f"across {int(present.sum())} classes present in train"
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal held-out (unseen-signer) accuracy: {val_acc:.3f}")

    saved_model_dir = os.path.join(out_dir, "saved_model")
    model.export(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(label_names, f, indent=2)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()}
            | {"val_accuracy_final": float(val_acc), "val_loss_final": float(val_loss)},
            f,
            indent=2,
        )

    print(f"\nWrote model.tflite + labels.json ({num_classes} classes) to {out_dir}")


if __name__ == "__main__":
    main()
