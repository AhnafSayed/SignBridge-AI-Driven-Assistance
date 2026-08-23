"""Model architectures over (seq_len, 21, 3) hand-landmark sequences.

Both models flatten the per-frame (21, 3) landmarks to a 63-dim vector and
treat the sequence as a 1D signal over time -- this is the "landmark
sequence classifier", not a raw-image CNN (see the project's locked scope
decision on this).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _flatten_input(seq_len: int, num_landmarks: int, num_coords: int):
    inputs = tf.keras.Input(shape=(seq_len, num_landmarks, num_coords))
    x = layers.Reshape((seq_len, num_landmarks * num_coords))(inputs)
    return inputs, x


def build_1dcnn(
    seq_len: int,
    num_classes: int,
    num_landmarks: int = 21,
    num_coords: int = 3,
) -> tf.keras.Model:
    """Baseline: three 1D-conv blocks over the time axis + a dense head."""
    inputs, x = _flatten_input(seq_len, num_landmarks, num_coords)

    for filters in (64, 128, 128):
        x = layers.Conv1D(filters, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="signbridge_1dcnn")


def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model)
    )
    angles = position * div_term
    pe = tf.zeros((seq_len, d_model))
    pe = tf.tensor_scatter_nd_update(
        pe,
        indices=[[i, j] for i in range(seq_len) for j in range(0, d_model, 2)],
        updates=tf.reshape(tf.sin(angles), [-1]),
    )
    return pe[tf.newaxis, ...]


def build_transformer(
    seq_len: int,
    num_classes: int,
    num_landmarks: int = 21,
    num_coords: int = 3,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
) -> tf.keras.Model:
    """A small Transformer encoder -- the plan's step up from the 1D-CNN
    baseline once that's evaluated and a sequence model is worth the extra
    latency budget (see O3's <300ms target).
    """
    inputs, x = _flatten_input(seq_len, num_landmarks, num_coords)
    x = layers.Dense(d_model)(x)
    x = x + _positional_encoding(seq_len, d_model)

    for _ in range(num_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
            x, x
        )
        x = layers.LayerNormalization()(x + attn_out)
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        x = layers.LayerNormalization()(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="signbridge_transformer")
