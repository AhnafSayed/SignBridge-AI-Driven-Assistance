"""Trains a small CNN directly on skeleton-image renderings of ASL
fingerspelling letters, to test a specific question: does merging in
Devansh-47's independently-captured AtoZ_3.1 corpus improve on this
project's existing shipped letter recognizer?

Two corpora, one representation (see src/skeleton_render.py for how the
representations were made to match):

  - "ours": grassknoted/asl-alphabet, extracted to MediaPipe landmarks by
    this project, then rendered to skeleton images by
    render_skeleton_dataset.py. Same underlying source the shipped
    landmark-sequence model already trains on -- rendering it to an image
    is a strictly *lossier* re-encoding of the same information (z is
    dropped, coordinates are quantized to pixels), so "ours" alone is not
    expected to beat the shipped model. It exists here as the controlled
    baseline the merge is measured against.
  - "theirs": AtoZ_3.1, ~180 real photographed skeleton drawings per
    letter from an independent capture session -- different people,
    camera, lighting than grassknoted's photos. This is the one thing
    that could plausibly add real value: not a better representation, but
    genuine additional real-world diversity.

Because the question is specifically "does merging help", three variants
are trained and each is evaluated against BOTH corpora's own held-out
split, not just its own: ours_only, theirs_only, merged -- each scored on
theirs_held_out and ours_held_out.

If merging genuinely buys robustness, the merged model should hold up on
theirs-held-out at least as well as theirs_only does, while not regressing
badly on ours-held-out relative to ours_only. If it doesn't, that's a real
negative result and is reported as one (see this project's own track
record of reporting negative results in ml/README.md's class-weighting
section) rather than shipped anyway.

The bar that actually matters for a ship decision is not any of these six
numbers alone: it's the shipped landmark-sequence model's own in-app
letter accuracy (99.08%, ml/eval_full_metrics.py), because that is what
would be replaced.

Splitting, carefully:
  - "ours": grouped by manifest.csv's base_index, so every augmented copy
    of one real landmark sample stays on the same side of the split --
    otherwise near-duplicate renders would leak between train and val and
    inflate held-out accuracy without real generalization.
  - "theirs": AtoZ_3.1 ships no participant or session metadata at all.
    Consecutively-numbered files within a class are plausibly frames from
    the same short capture burst, which carries the same leakage risk as
    above but with no ID to group on. As a conservative proxy, each
    class's files are split into contiguous BLOCKS by filename order
    (first 80% of the sorted sequence -> train, last 20% -> val) rather
    than shuffled, so a train/val boundary can only fall between blocks,
    not inside a probable burst. This is not as rigorous as this project's
    signer-held-out protocol elsewhere, and is stated as a limitation
    rather than presented as equivalent to it.

Memory: images are decoded lazily through a tf.data pipeline (tf.io.
read_file + tf.image.decode_jpeg), never materialized as one big numpy
array. An earlier version of this script loaded the full "ours" corpus
(37,456 images) into a single (N, 128, 128, 3) float32 array up front --
6.86 GiB in one allocation -- and crashed with ArrayMemoryError. Only
file-path lists and string labels are held in Python memory; decoding
happens per-batch during training/evaluation.

Usage:
    python train_asl_letters_skeleton.py
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = 128  # downscaled from the 400x400 render/capture size -- the
# drawn skeleton is a few thick lines and dots on a blank field, not fine
# texture, so it survives downscaling; keeping the model small is what
# makes it deployable at all.
BATCH_SIZE = 32
HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "models", "asl_letters_skeleton")
os.makedirs(OUT_DIR, exist_ok=True)


def _decode(path, label_idx):
    raw = tf.io.read_file(path)
    im = tf.image.decode_jpeg(raw, channels=3)
    im = tf.image.resize(im, (IMG_SIZE, IMG_SIZE), method="area")
    im = tf.cast(im, tf.float32) / 255.0
    return im, label_idx


def make_dataset(paths, label_indices, shuffle: bool, batch_size: int = BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(label_indices, dtype=np.int64)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def list_ours(root: str):
    """Returns (paths, labels, groups) for our rendered corpus, using
    manifest.csv's base_index as the leakage-safe split group. Only paths
    and label strings are held in memory -- no image decoding here."""
    manifest_path = os.path.join(root, "manifest.csv")
    paths, labels, groups = [], [], []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(os.path.join(root, row["relpath"]))
            labels.append(row["label"])
            groups.append(f"ours_{row['label']}_{row['base_index']}")
    return np.array(paths), np.array(labels), np.array(groups)


def list_theirs_block_split(root: str, val_fraction: float = 0.2):
    """Returns (paths_train, labels_train, paths_val, labels_val) for
    AtoZ_3.1, splitting each class's files into contiguous filename-ordered
    blocks rather than shuffling (see module docstring for why)."""
    paths_train, labels_train, paths_val, labels_val = [], [], [], []
    for cls_dir in sorted(glob.glob(os.path.join(root, "*"))):
        label = os.path.basename(cls_dir)
        if len(label) != 1 or not label.isalpha():
            continue
        files = sorted(
            glob.glob(os.path.join(cls_dir, "*.jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        n_val = max(1, int(len(files) * val_fraction))
        train_files, val_files = files[:-n_val], files[-n_val:]
        paths_train.extend(train_files)
        labels_train.extend([label] * len(train_files))
        paths_val.extend(val_files)
        labels_val.extend([label] * len(val_files))
    return (np.array(paths_train), np.array(labels_train), np.array(paths_val), np.array(labels_val))


def group_split_paths(paths, labels, groups, val_fraction: float = 0.2, seed: int = 42):
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(paths, labels, groups=groups))
    return paths[train_idx], labels[train_idx], paths[val_idx], labels[val_idx]


def build_model(num_classes: int) -> tf.keras.Model:
    # No BatchNormalization -- see diagnose_lr.py for the full investigation.
    # A first attempt at Adam's default learning rate (1e-3), no clipping,
    # was numerically unstable: val_loss repeatedly spiked into the tens or
    # hundreds (peak 148) while training accuracy climbed smoothly and
    # val_accuracy gyrated near chance level rather than trending -- a
    # healthy-train/exploding-val pattern. Lowering the learning rate to
    # 1e-4 removed the spikes but converged far too slowly to be usable.
    # Gradient clipping at the original learning rate reduced but did not
    # eliminate the spikes (148 -> 15-20), and lowering BatchNorm's
    # momentum to adapt its running statistics faster made it worse (spike
    # to 42.7), which together rule out both gradient explosion and
    # momentum as the primary cause. Removing BatchNorm entirely fixed it
    # outright: max val_loss 3.23 (never exceeding the random-chance
    # starting point) and val_accuracy climbing smoothly to 55.6% in the
    # same 8 epochs the earlier configs spent oscillating near chance.
    # The likely mechanism: with only ~118 steps per epoch on the smaller
    # corpus, BatchNorm's running mean/variance (an exponential moving
    # average) hadn't converged to the true batch statistics, so
    # train-mode and eval-mode behavior diverged sharply -- a known
    # failure mode for BatchNorm on few-step regimes, and this network
    # (three small conv blocks) doesn't need it for stability once removed.
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    # Gradient clipping kept as cheap, harmless insurance even though the
    # diagnostic showed BatchNorm was the real cause, not gradient scale.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def encode_labels(label_names, *label_arrays):
    to_idx = {name: i for i, name in enumerate(label_names)}
    return [np.array([to_idx[label] for label in arr]) for arr in label_arrays]


def train_and_eval(name, paths_train, y_train_idx, label_names, eval_path_sets, epochs=25):
    print(f"\n=== Training {name}: {len(paths_train)} samples, {len(label_names)} classes ===")
    model = build_model(len(label_names))

    # A held-out slice of the TRAINING data itself (not either eval set)
    # for early stopping, so stopping time is never chosen by looking at
    # either of the two evaluation splits.
    n_val = max(1, int(len(paths_train) * 0.1))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(paths_train))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    fit_ds = make_dataset(paths_train[fit_idx], y_train_idx[fit_idx], shuffle=True)
    dev_ds = make_dataset(paths_train[val_idx], y_train_idx[val_idx], shuffle=False)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                                   restore_best_weights=True)]
    model.fit(fit_ds, validation_data=dev_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    results = {}
    for eval_name, (eval_paths, eval_y_idx) in eval_path_sets.items():
        eval_ds = make_dataset(eval_paths, eval_y_idx, shuffle=False)
        loss, acc = model.evaluate(eval_ds, verbose=0)
        results[eval_name] = float(acc)
        print(f"  {name} -> eval on {eval_name}: {acc:.4f}")
    return model, results


def main():
    print("Listing AtoZ_3.1 (theirs)...")
    atoz_root = (
        r"C:/Users/Ahnaf Sayed/Downloads/"
        r"Sign-Language-To-Text-and-Speech-Conversion-master/"
        r"Sign-Language-To-Text-and-Speech-Conversion-master/AtoZ_3.1"
    )
    pt_tr, yt_tr, pt_va, yt_va = list_theirs_block_split(atoz_root)
    print(f"  theirs: {len(yt_tr)} train, {len(yt_va)} val")

    print("Listing our rendered corpus...")
    po, yo, go = list_ours(os.path.join(HERE, "data", "asl_letters_skeleton"))
    po_tr, yo_tr, po_va, yo_va = group_split_paths(po, yo, go)
    print(f"  ours: {len(yo_tr)} train, {len(yo_va)} val")

    label_names = sorted(set(yt_tr.tolist()) | set(yo_tr.tolist()))
    assert label_names == [chr(ord("A") + i) for i in range(26)], \
        f"expected exactly A-Z, got {label_names}"

    (yt_tr_i, yt_va_i, yo_tr_i, yo_va_i) = encode_labels(
        label_names, yt_tr, yt_va, yo_tr, yo_va
    )
    eval_path_sets = {
        "theirs_held_out": (pt_va, yt_va_i),
        "ours_held_out": (po_va, yo_va_i),
    }

    all_results = {}
    _, all_results["theirs_only"] = train_and_eval(
        "theirs_only", pt_tr, yt_tr_i, label_names, eval_path_sets
    )
    _, all_results["ours_only"] = train_and_eval(
        "ours_only", po_tr, yo_tr_i, label_names, eval_path_sets
    )
    p_merged = np.concatenate([pt_tr, po_tr])
    y_merged_i = np.concatenate([yt_tr_i, yo_tr_i])
    merged_model, all_results["merged"] = train_and_eval(
        "merged", p_merged, y_merged_i, label_names, eval_path_sets
    )

    print("\n=== Summary (accuracy) ===")
    print(f"{'variant':14s} {'theirs_held_out':>16s} {'ours_held_out':>14s}")
    for variant, res in all_results.items():
        print(f"{variant:14s} {res['theirs_held_out']:16.4f} {res['ours_held_out']:14.4f}")

    with open(os.path.join(OUT_DIR, "cross_source_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Only the merged model is saved -- it's the one under consideration
    # for shipping, and is compared against the shipped model's 99.08%
    # in-app letter accuracy before any ship decision is made.
    saved_model_dir = os.path.join(OUT_DIR, "saved_model")
    merged_model.export(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(OUT_DIR, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"\nWrote merged model.tflite ({os.path.getsize(tflite_path) / 1024:.0f} KB) to {OUT_DIR}")


if __name__ == "__main__":
    main()
