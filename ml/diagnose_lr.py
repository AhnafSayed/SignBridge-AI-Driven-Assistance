"""Fast diagnostic: what actually stabilizes training, run on the smallest
corpus (theirs, ~3.7k samples) for a handful of epochs each -- cheap
enough to compare several configs before committing an hour to the real
3-variant training run.

Two hypotheses tested, in order:

1. Gradient explosion. Adam at its default learning rate (1e-3), no
   clipping, produced val_loss spikes into the tens/hundreds (peak 148).
   Lowering the learning rate to 1e-4 (with clipnorm=1.0) removed the
   spikes but converged far too slowly (11% train accuracy by epoch 2) to
   reach a competitive result in a reasonable epoch budget. Clipping alone
   at the original 1e-3, tested next, reduced the spikes (148 -> 15-20)
   but did not eliminate them, while val_accuracy stayed pinned near
   random chance (3-4% for 26 classes) even as train accuracy climbed
   past 80% -- ruling gradient explosion out as the *primary* cause, since
   clipping directly bounds it and the instability persisted anyway.

2. BatchNorm running-statistics mismatch. The pattern above -- healthy,
   smoothly-improving train accuracy alongside wildly unstable val
   loss/accuracy that doesn't trend -- is the textbook signature of
   BatchNorm's inference-mode statistics (an exponential moving average,
   default momentum 0.99) not having converged: with only ~118 steps per
   epoch on this corpus, the running mean/variance can still be far from
   the true batch statistics used during training, so train-mode and
   eval-mode behavior diverge sharply early on. A genuinely hard-but-
   real val distribution would produce poor *and stable* numbers, not
   loss swinging 7x between consecutive epochs while train loss falls
   smoothly -- that swing is what points at a mode mismatch rather than
   task difficulty. This script tests two fixes: a faster-adapting
   momentum (0.9), and removing BatchNorm entirely from a network this
   small, where it may be unnecessary complexity in the first place.

Usage:
    python diagnose_lr.py
"""

from __future__ import annotations

import os

import tensorflow as tf

from train_asl_letters_skeleton import (
    IMG_SIZE,
    encode_labels,
    list_theirs_block_split,
    make_dataset,
)

HERE = os.path.dirname(__file__)
ATOZ_ROOT = (
    r"C:/Users/Ahnaf Sayed/Downloads/"
    r"Sign-Language-To-Text-and-Speech-Conversion-master/"
    r"Sign-Language-To-Text-and-Speech-Conversion-master/AtoZ_3.1"
)


def build_model(num_classes, lr, clipnorm, bn_momentum=0.99, use_bn=True):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    opt_kwargs = {"learning_rate": lr}
    if clipnorm is not None:
        opt_kwargs["clipnorm"] = clipnorm
    model.compile(optimizer=tf.keras.optimizers.Adam(**opt_kwargs),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    pt_tr, yt_tr, pt_va, yt_va = list_theirs_block_split(ATOZ_ROOT)
    label_names = sorted(set(yt_tr.tolist()))
    (yt_tr_i, yt_va_i) = encode_labels(label_names, yt_tr, yt_va)
    train_ds = make_dataset(pt_tr, yt_tr_i, shuffle=True)
    val_ds = make_dataset(pt_va, yt_va_i, shuffle=False)

    configs = [
        ("lr=1e-3, clip=1.0, bn_momentum=0.9", dict(lr=1e-3, clipnorm=1.0, bn_momentum=0.9)),
        ("lr=1e-3, clip=1.0, no BatchNorm", dict(lr=1e-3, clipnorm=1.0, use_bn=False)),
    ]
    for name, kwargs in configs:
        print(f"\n=== {name} ===")
        tf.keras.utils.set_random_seed(42)
        model = build_model(len(label_names), **kwargs)
        history = model.fit(train_ds, validation_data=val_ds, epochs=8, verbose=2)
        max_val_loss = max(history.history["val_loss"])
        final_val_acc = history.history["val_accuracy"][-1]
        best_val_acc = max(history.history["val_accuracy"])
        print(f"  -> max val_loss over run: {max_val_loss:.2f} "
              f"(>10 signals instability), final val_acc: {final_val_acc:.3f}, "
              f"best val_acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
