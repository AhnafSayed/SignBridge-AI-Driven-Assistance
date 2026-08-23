"""Trains one combined ASL model over three sources: words (asl-signs,
real temporal sequences, 250 classes), the fingerspelling alphabet (A-Z,
single-frame static poses extracted from grassknoted/asl-alphabet by
extract_asl_alphabet_landmarks.py -- run that first), and digits (0-9,
single-frame static poses extracted from rayeed045/american-sign-
language-digit-dataset by extract_asl_digit_landmarks.py -- run that
first too). ASL previously had no digit vocabulary at all; this adds one,
matching what BdSL already had.

Alphabet and digit samples are single photographed frames, not videos,
and only ever show one hand (fingerspelling and counting are one-handed
gestures) -- each is embedded into slot 0 of the two-hand contract
(landmarks.embed_single_hand) and expanded to the same fixed-length
sequence shape the word model uses via landmarks.prepare_two_hand_sequence,
exactly like the BdSL digit/letter classes in train_bdsl_combined.py. One
architecture, one on-device inference path, just a bigger labels.json.

Neither the alphabet nor the digit source has participant/signer metadata
(both are independent photos, not a multi-signer video corpus), so their
portion of the train/val split is an ordinary random split, not a
held-out-*signer* split like the word portion and like every other model
this project ships. This is called out here and in the app docs rather
than glossed over.

Usage:
    python extract_asl_alphabet_landmarks.py \
        --images-dir data/asl-alphabet/asl_alphabet_train/asl_alphabet_train
    python extract_asl_digit_landmarks.py \
        --images-dir "data/asl-digits/American Sign Language Digits Dataset"
    python train_asl_combined.py \
        --words-dir data/asl-signs \
        --alphabet-path data/asl_alphabet_landmarks.npz \
        --digit-path data/asl_digit_landmarks.npz \
        --epochs 40 --augment-copies 2
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

import dataset_asl as words_ds  # noqa: E402
import dataset_cache  # noqa: E402
from augment import build_augmented_trainset  # noqa: E402
from landmarks import DEFAULT_SEQ_LEN, embed_single_hand, prepare_two_hand_sequence  # noqa: E402
from model import build_1dcnn, build_transformer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--words-dir", default="data/asl-signs")
    p.add_argument("--alphabet-path", default="data/asl_alphabet_landmarks.npz")
    p.add_argument("--digit-path", default="data/asl_digit_landmarks.npz")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--arch", choices=["cnn", "transformer"], default="cnn")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--augment-copies", type=int, default=0)
    p.add_argument(
        "--rotation-deg",
        type=float,
        default=15.0,
        help="Max +/- in-plane rotation jitter for STATIC letter/digit samples only -- word "
        "sequences always keep augment.py's own default (15) regardless of this flag, since a "
        "single-variable test found widening it for words too cost ~0.9 points of word accuracy "
        "(see eval_rotation_robustness.py). Raised in response to real-device testing: users "
        "signing one-handed with a phone naturally roll the signing hand more than a posed "
        "training photo does, and V/W were observed misreading as Q/R specifically when "
        "rotated -- a gap the clean, un-rotated offline benchmark can't see, since it contains "
        "no rotation in the first place.",
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def load_words(args):
    """Same cache key train.py's `--language asl --mode full` run uses --
    reuses that already-warm cache instead of re-parsing 94,477 parquet
    files."""
    cache_path = dataset_cache.cache_path(
        "asl_full",
        "asl",
        os.path.abspath(args.words_dir),
        "full",
        50,  # core_size -- unused in full mode, but part of train.py's cache key
        args.seq_len,
        None,  # max_samples_per_class
        "train.csv",
    )
    if not args.no_cache and os.path.exists(cache_path):
        print(f"Loading cached word dataset from {cache_path} ...")
        X, y, participants, label_names = dataset_cache.load(cache_path)
    else:
        index = words_ds.load_index(args.words_dir)
        X, y, participants, label_names = words_ds.build_dataset(
            index, index.signs, seq_len=args.seq_len, workers=args.workers
        )
        if not args.no_cache:
            dataset_cache.save(cache_path, X, y, participants, label_names)

    labels = [label_names[i] for i in y]
    signers = [f"word_{p}" for p in participants]
    return X, labels, signers


def load_alphabet(path, seq_len):
    with np.load(path, allow_pickle=True) as data:
        X_raw, y_raw = data["X"], data["y"]
    X = np.stack(
        [prepare_two_hand_sequence(embed_single_hand(frame)[np.newaxis, ...], seq_len, neutral_anchor=True) for frame in X_raw]
    )
    labels = [f"letter_{letter}" for letter in y_raw]
    # No real signer identity for this source (see module docstring) -- a
    # unique id per sample makes GroupShuffleSplit fall back to an ordinary
    # random split for this portion, which is what actually happened here,
    # rather than falsely implying a held-out-signer test.
    signers = [f"letter_{i}" for i in range(len(labels))]
    return X, labels, signers


def load_digits(path, seq_len):
    """Same shape as load_alphabet, above -- see that function's comments
    for why embed_single_hand + a synthetic per-sample signer id are used.
    """
    with np.load(path, allow_pickle=True) as data:
        X_raw, y_raw = data["X"], data["y"]
    X = np.stack(
        [prepare_two_hand_sequence(embed_single_hand(frame)[np.newaxis, ...], seq_len, neutral_anchor=True) for frame in X_raw]
    )
    labels = [f"digit_{digit}" for digit in y_raw]
    signers = [f"digit_{i}" for i in range(len(labels))]
    return X, labels, signers


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "models", f"asl_combined_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ASL words (asl-signs)...")
    X_w, y_w, s_w = load_words(args)
    print(f"  {len(y_w)} word sequences, {len(set(y_w))} classes")

    print("Loading ASL alphabet...")
    X_a, y_a, s_a = load_alphabet(args.alphabet_path, args.seq_len)
    print(f"  {len(y_a)} alphabet samples, {len(set(y_a))} classes")

    print("Loading ASL digits...")
    X_d, y_d, s_d = load_digits(args.digit_path, args.seq_len)
    print(f"  {len(y_d)} digit samples, {len(set(y_d))} classes")

    all_labels = list(y_w) + list(y_a) + list(y_d)
    label_names = sorted(set(all_labels))
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    num_classes = len(label_names)
    print(
        f"\nCombined vocabulary: {num_classes} classes "
        f"({len(set(y_w))} words + {len(set(y_a))} letters + {len(set(y_d))} digits)"
    )

    X = np.concatenate([X_w, X_a, X_d], axis=0).astype(np.float32)
    y = np.array([label_to_idx[label] for label in all_labels], dtype=np.int64)
    signers = np.array(list(s_w) + list(s_a) + list(s_d))
    # Tracked alongside X/y/signers through the same concatenation and
    # split so the augmentation step below can give static letter/digit
    # samples a different rotation range than word samples -- see
    # --rotation-deg's help text and augment.py's per-sample rotation_deg
    # docstring for why the two categories want different amounts.
    is_static = np.array([False] * len(y_w) + [True] * len(y_a) + [True] * len(y_d))
    print(f"Total sequences: {len(y)}, total signer groups: {len(set(signers))}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=signers))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    is_static_train = is_static[train_idx]
    print(
        f"Train: {len(train_idx)} from {len(set(signers[train_idx]))} groups | "
        f"Val: {len(val_idx)} from {len(set(signers[val_idx]))} unseen groups"
    )
    seq_len, num_landmarks = X.shape[1], X.shape[2]
    del X  # dead weight from here on -- see train.py's matching comment.

    if args.augment_copies > 0:
        rng = np.random.default_rng(42)
        # Per-sample rotation range, not one flat number: real-device
        # testing found static letters ("V" especially) misreading badly
        # once the signer's hand was rotated more than +/-15 degrees, and
        # eval_rotation_robustness.py confirmed a wider range fixes it --
        # but applying that same wider range to word *sequences* cost
        # ~0.9 points of word accuracy in a single-variable test, so words
        # keep the original range while only static letters/digits widen.
        rotation_deg_per_sample = np.where(is_static_train, args.rotation_deg, 15.0).astype(np.float32)
        X_train, y_train = build_augmented_trainset(
            X_train, y_train, args.augment_copies, rng, rotation_deg=rotation_deg_per_sample
        )
        print(
            f"Augmented train set: {len(X_train)} sequences ({args.augment_copies} extra copies, "
            f"words +/-15 deg, static letters/digits +/-{args.rotation_deg:.0f} deg)"
        )

    build_fn = build_1dcnn if args.arch == "cnn" else build_transformer
    model = build_fn(seq_len=seq_len, num_classes=num_classes, num_landmarks=num_landmarks)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal held-out accuracy: {val_acc:.3f}")

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
