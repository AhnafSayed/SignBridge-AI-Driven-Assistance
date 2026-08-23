"""Trains a direct single-frame classifier for static signs (digits and
fingerspelling letters) only -- words stay on the sequence model.

Why this exists. A third-party project (git-sajib/ai-hearing-speech-
assistant) reported "99.9% accuracy" for its ASL letter/digit model and
was proposed as a reason to replace the letters/digits half of this
project's pipeline. Two things were checked before touching anything:

1. The claim. Its training script (`train_model.py`) does
   `train_test_split(X, y, test_size=0.15, stratify=y)` -- a plain random
   split, no signer/subject grouping. That is the *same* methodology this
   project already uses for its own ASL statics (documented as a known
   limitation, since neither source dataset carries signer identity), on
   the *same* underlying dataset (grassknoted/asl-alphabet). Their 99.9%
   and this project's already-measured 99.08% are not different rigor --
   they're the same rigor, and a 0.9-point gap between two near-ceiling
   MLPs measured the same way is noise, not evidence of a better model.

2. The plausible real reason a live demo might still *feel* better,
   separate from the headline number: their model is architecturally a
   direct single-frame classifier (63 raw landmark values in, one letter
   out). This project's shipped pipeline instead forces every static pose
   through the *sequence* contract built for dynamic word signs --
   repeating one frame 40 times and running it through the motion-energy
   gesture segmenter (landmarks.py, gesture_segmenter.dart) meant to find
   where a *moving* sign starts and ends. That is real, unnecessary
   overhead for something that never moves, and is worth testing on its
   own merits -- not by trusting their number, but by measuring a
   comparable direct classifier against this project's own held-out data.

Architecture mirrors theirs (a fair basis for comparison, not an
endorsement of their claim): Dense(256)+BN+Dropout(0.3) ->
Dense(128)+BN+Dropout(0.2) -> Dense(64)+BN -> Dense(num_classes, softmax).
Input is `normalize_hand_frame`'s existing single-hand normalization
(wrist-centered, scaled by wrist-to-middle-MCP), flattened to 63 features
-- the same normalization convention used everywhere else in this
project, not a new one invented for this comparison.

Usage:
    python train_static_direct.py --language asl
    python train_static_direct.py --language bdsl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from augment import build_augmented_trainset  # noqa: E402
from landmarks import normalize_hand_frame  # noqa: E402

# Real-device report: ASL letters/digits misread depending on how the
# signer's hand (or, equivalently, the phone) is angled toward the
# camera. This project already measured and root-caused the identical
# symptom once before, on the old sequence model (see ml/README.md,
# "Rotation robustness: ASL V/W reading as Q/R") -- a static pose trained
# with no rotation jitter, or only a narrow +/-15 degree jitter, degrades
# sharply past ~30 degrees of real hand rotation, because the offline
# benchmark's photos are all shot at ~0 degrees and never tests otherwise.
# `train_static_direct.py`'s first version (see this file's own docstring)
# trained on *zero* rotation augmentation -- strictly worse than even that
# +/-15 baseline -- which is why it regressed on exactly this axis. +/-30
# is this project's own already-measured fix for static categories
# specifically (word accuracy was the thing that value protects *against*,
# and there are no words here), so it's the default rather than a new
# guess.
ROTATION_DEG_STATIC_DEFAULT = 30.0
ROTATION_EVAL_ANGLES = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
ROTATION_CHECK_LETTERS = ("V", "W", "P", "M")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=["asl", "bdsl"], required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--rotation-deg", type=float, default=ROTATION_DEG_STATIC_DEFAULT,
        help="Training-time rotation jitter in degrees (0 disables it, reproducing "
             "this script's original behavior). Applied via src/augment.py, the same "
             "transform and convention the sequence model's category-specific rotation "
             "fix uses.",
    )
    p.add_argument(
        "--augment-copies", type=int, default=4,
        help="Extra augmented copies of the training split (mirror/rotate/scale/jitter) "
             "appended alongside the original, unaugmented copy.",
    )
    return p.parse_args()


def rotate_hand_frame(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """Fixed-angle 2D rotation of a batch of normalized (21, 3) hand frames'
    x/y -- the same transform `augment.py` applies at random training-time
    angles and `eval_rotation_robustness.py` applies at a fixed sweep, used
    here to measure (not train) rotation robustness on held-out data."""
    out = X.copy()
    a = np.deg2rad(angle_deg)
    cos, sin = np.cos(a), np.sin(a)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = x * cos - y * sin
    out[..., 1] = x * sin + y * cos
    return out


def load_asl():
    import numpy as _np

    letters = _np.load("data/asl_alphabet_landmarks.npz", allow_pickle=True)
    digits = _np.load("data/asl_digit_landmarks.npz", allow_pickle=True)
    X = _np.concatenate([letters["X"], digits["X"]], axis=0)
    y = _np.concatenate(
        [[f"letter_{c}" for c in letters["y"]], [f"digit_{c}" for c in digits["y"]]]
    )
    # Neither source carries signer identity (independent photos) -- a
    # unique id per row is the honest way to say "no real grouping exists"
    # rather than implying a held-out-signer test that isn't possible here.
    signers = _np.array([f"row_{i}" for i in range(len(y))])
    return X, y, signers


def load_bdsl():
    import dataset_bdsl_static as static_ds

    X_d, y_d, s_d = static_ds.load_static_dataset(
        "data/bdsl_digits/CSV Files/Sign Digits - CSV Files"
    )
    X_l, y_l, s_l = static_ds.load_static_dataset(
        "data/bdsl_digits/CSV Files/Sign Letters - CSV Files"
    )
    X = np.concatenate([X_d, X_l], axis=0)
    y = np.concatenate(
        [[f"digit_{c}" for c in y_d], [f"letter_{static_ds.LETTER_NAMES[c]}" for c in y_l]]
    )
    signers = np.concatenate([[f"digit_{s}" for s in s_d], [f"letter_{s}" for s in s_l]])
    return X, y, signers


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(63,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "models", f"{args.language}_static_direct")
    os.makedirs(out_dir, exist_ok=True)

    if args.language == "asl":
        X_raw, y_raw, signers = load_asl()
        split_kind = "random (no signer identity available -- same as this project's existing static evaluation)"
    else:
        X_raw, y_raw, signers = load_bdsl()
        split_kind = "held-out signer (BdSL statics DO carry signer identity, unlike ASL's)"

    label_names = sorted(set(y_raw.tolist()))
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_to_idx[label] for label in y_raw], dtype=np.int64)
    # Kept as (N, 21, 3) here, not flattened yet -- rotation augmentation
    # below needs the x/y/z structure to rotate x/y in place. Flattened to
    # 63 features right before each split goes into the model.
    X = np.stack([normalize_hand_frame(f) for f in X_raw]).astype(np.float32)

    print(f"{args.language}: {len(y)} samples, {len(label_names)} classes, split = {split_kind}")

    if args.language == "bdsl":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y, groups=signers))
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(f"Train: {len(train_idx)} from {len(set(signers[train_idx]))} signers | "
              f"Val: {len(val_idx)} from {len(set(signers[val_idx]))} unseen signers")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)} | Val (random 15%): {len(X_val)}")

    if args.rotation_deg > 0 and args.augment_copies > 0:
        rng = np.random.default_rng(42)
        # build_augmented_trainset expects a (N, seq_len, landmarks, 3)
        # array -- these are single static frames, so seq_len=1 makes each
        # one a length-1 "sequence"; augment.py's per-sample transforms
        # (mirror/rotate/scale/noise) apply identically either way since
        # they broadcast across whatever seq_len is present.
        X_train_aug, y_train = build_augmented_trainset(
            X_train[:, np.newaxis, :, :], y_train, args.augment_copies, rng,
            rotation_deg=args.rotation_deg,
        )
        X_train = X_train_aug[:, 0, :, :]
        print(f"Augmented training set: {len(X_train)} samples "
              f"({args.augment_copies} extra copies, +/-{args.rotation_deg:.0f} deg rotation)")
    else:
        print("Rotation augmentation disabled (--rotation-deg 0)")

    # Flatten to the model's actual (63,) input shape now that any
    # augmentation (which needs the x/y/z structure) is done. X_val stays
    # un-augmented -- held-out accuracy must reflect real, canonical poses,
    # exactly as this project's existing evaluations do; the rotation
    # sweep below is the deliberate, separate check for the rotated case.
    X_train = X_train.reshape(len(X_train), 63)
    X_val_frames = X_val  # kept as (N, 21, 3) for the rotation sweep below
    X_val = X_val.reshape(len(X_val), 63)

    model = build_model(len(label_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal held-out accuracy (0 deg, canonical): {val_acc:.4f}")

    # Per-category breakdown, matching eval_full_metrics.py's convention,
    # since a combined digit+letter number would hide exactly the kind of
    # disparity this project has repeatedly found mattered (§5.1.1).
    preds = model.predict(X_val, verbose=0).argmax(axis=1)
    for cat in ("digit", "letter"):
        mask = np.array([label_names[i].startswith(cat) for i in y_val])
        if mask.any():
            acc = float(np.mean(preds[mask] == y_val[mask]))
            print(f"  {cat:8s} n={mask.sum():4d}  acc={acc:.4f}")

    # Rotation-robustness sweep on the held-out split -- the check this
    # project's own prior V/W-reading-as-Q/R investigation established is
    # the one that actually matters, since the standard 0-degree benchmark
    # above cannot see a rotation-sensitivity regression at all.
    if args.language == "asl":
        letter_mask = np.array([label_names[i].startswith("letter") for i in y_val])
        if letter_mask.any():
            print("\n=== Rotation robustness (held-out letters only) ===")
            to_idx = {name: i for i, name in enumerate(label_names)}
            check_idx = {
                letter: to_idx[f"letter_{letter}"]
                for letter in ROTATION_CHECK_LETTERS
                if f"letter_{letter}" in to_idx
            }
            X_val_letters = X_val_frames[letter_mask]
            y_val_letters = y_val[letter_mask]
            for angle in ROTATION_EVAL_ANGLES:
                X_rot = rotate_hand_frame(X_val_letters, angle).reshape(len(X_val_letters), 63)
                pred = model.predict(X_rot, verbose=0).argmax(axis=1)
                acc = float(np.mean(pred == y_val_letters))
                recalls = []
                for letter, li in check_idx.items():
                    m = y_val_letters == li
                    r = float(np.mean(pred[m] == li)) if m.any() else float("nan")
                    recalls.append(f"{letter}={r:.2f}")
                print(f"  rotation {angle:+3d} deg -> letter accuracy {acc:.4f}   ({', '.join(recalls)})")

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
            f, indent=2,
        )
    print(f"\nWrote model.tflite + labels.json ({len(label_names)} classes) to {out_dir}")


if __name__ == "__main__":
    main()
