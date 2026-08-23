"""Train a landmark-sequence sign classifier and export it to .tflite.

Usage (once the ASL dataset has been downloaded via Kaggle -- see
../docs or the top-level README for how):

    python train.py --language asl --data-dir data/asl-signs \
        --mode core --arch cnn --epochs 40

Outputs, under models/<language>_<mode>_<arch>/:
    model.tflite      -- quantized, ready to bundle into the Flutter app
    labels.json        -- ["sign_0", "sign_1", ...] index -> label name
    history.json        -- per-epoch train/val accuracy + loss
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

import dataset_cache  # noqa: E402
from augment import build_augmented_trainset  # noqa: E402
from landmarks import DEFAULT_SEQ_LEN  # noqa: E402
from model import build_1dcnn, build_transformer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=["asl", "bdsl"], required=True)
    p.add_argument("--data-dir", required=True, help="Path to the unzipped dataset")
    p.add_argument("--mode", choices=["core", "full"], default="core")
    p.add_argument("--core-size", type=int, default=50, help="Vocabulary size for --mode core")
    p.add_argument("--arch", choices=["cnn", "transformer"], default="cnn")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples-per-class", type=int, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes for loading ASL parquet files (default: cpu_count - 1)",
    )
    p.add_argument(
        "--csv-name",
        default="train.csv",
        help="Index file inside --data-dir, e.g. train_subset.csv from download_asl_subset.py",
    )
    p.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="Extra augmented copies of the training split (mirror/rotate/scale/jitter), "
        "each the same size as the real training set -- 0 disables augmentation",
    )
    p.add_argument("--no-cache", action="store_true", help="Skip the on-disk dataset cache")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _cache_path(args) -> str:
    return dataset_cache.cache_path(
        f"{args.language}_{args.mode}",
        args.language,
        os.path.abspath(args.data_dir),
        args.mode,
        args.core_size,
        args.seq_len,
        args.max_samples_per_class,
        args.csv_name,
    )


def load_dataset(args):
    """Loads (X, y, participants, label_names), transparently caching the
    result to disk. At full ASL scale, parsing 94,477 parquet files takes
    ~15 minutes -- a real tax to pay again for every architecture or
    hyperparameter experiment on the exact same data selection.
    """
    cache_path = _cache_path(args)
    if not args.no_cache and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path} ...")
        return dataset_cache.load(cache_path)

    if args.language == "asl":
        import dataset_asl as ds

        index = ds.load_index(args.data_dir, csv_name=args.csv_name)
        vocab = (
            ds.core_vocabulary(index, size=args.core_size)
            if args.mode == "core"
            else index.signs
        )
        result = ds.build_dataset(
            index,
            vocab,
            seq_len=args.seq_len,
            max_samples_per_class=args.max_samples_per_class,
            workers=args.workers,
        )
    else:
        import dataset_bdsl as ds

        index = ds.load_index(args.data_dir)
        vocab = (
            ds.core_vocabulary(index, size=args.core_size)
            if args.mode == "core"
            else index.signs
        )
        result = ds.build_dataset(
            index, vocab, seq_len=args.seq_len, max_samples_per_class=args.max_samples_per_class
        )

    if not args.no_cache:
        X, y, participants, label_names = result
        dataset_cache.save(cache_path, X, y, participants, label_names)
        print(f"Cached dataset to {cache_path}")
    return result


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "models", f"{args.language}_{args.mode}_{args.arch}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.language} dataset from {args.data_dir} ...")
    X, y, participants, label_names = load_dataset(args)
    num_classes = len(label_names)
    seq_len, num_landmarks = X.shape[1], X.shape[2]
    print(f"{X.shape[0]} sequences, {num_classes} classes, seq_len={seq_len}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=participants))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(
        f"Train: {len(train_idx)} sequences from "
        f"{len(set(participants[train_idx]))} signers | "
        f"Val: {len(val_idx)} sequences from {len(set(participants[val_idx]))} unseen signers"
    )
    # X[train_idx]/X[val_idx] above are fancy-indexing copies, not views --
    # the full-size X is dead weight from here on, and at ASL's full
    # two-hand scale (~2GB) holding onto it while augmentation allocates
    # its own several-GB worth of temporaries is what pushed this past
    # available RAM the first time this ran.
    del X

    if args.augment_copies > 0:
        rng = np.random.default_rng(42)
        # Augmented copies are additional *training* data only -- X_val/y_val
        # above are untouched, so held-out accuracy still measures
        # generalization to real, unaugmented, never-seen signers.
        X_train, y_train = build_augmented_trainset(X_train, y_train, args.augment_copies, rng)
        print(f"Augmented train set: {len(X_train)} sequences ({args.augment_copies} extra copies)")

    build_fn = build_1dcnn if args.arch == "cnn" else build_transformer
    model = build_fn(seq_len=seq_len, num_classes=num_classes, num_landmarks=num_landmarks)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal held-out (unseen-signer) accuracy: {val_acc:.3f}")

    # --- export ---
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

    print(f"\nWrote model.tflite + labels.json to {out_dir}")


if __name__ == "__main__":
    main()
