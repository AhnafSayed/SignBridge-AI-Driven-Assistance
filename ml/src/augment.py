"""Landmark-sequence augmentation for training only.

The gap this targets: on the full ASL run, train accuracy reached ~76%
while held-out-*signer* accuracy plateaued around 52% -- a classic sign of
overfitting to the exact hand geometry/speed of the 16-17 training signers
rather than learning the sign itself. With only ~21 signers total, there's
no way to buy more signer diversity; augmenting the geometry synthetically
is the standard substitute. Every transform here is one a real signer's
hand could plausibly produce (mirrored hand, slightly different camera
angle/distance, MediaPipe's own landmark jitter) -- nothing that would
change *which sign it is*.
"""

from __future__ import annotations

import numpy as np

from landmarks import NUM_LANDMARKS, TWO_HAND_LANDMARKS, TWO_HAND_LANDMARKS_WITH_ANCHOR


def augment_sequences(
    X: np.ndarray,
    rng: np.random.Generator,
    rotation_deg: float | np.ndarray = 15.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    noise_std: float = 0.02,
    mirror_prob: float = 0.5,
) -> np.ndarray:
    """Returns one augmented copy of X, same shape as X -- (N, seq_len, 21, 3)
    for single-hand data, (N, seq_len, 42, 3) for two-hand.

    Applied per-sequence (one random transform per sample, held constant
    across that sample's frames) since a single sign is one continuous
    shot, not frames from different cameras.
    """
    out = np.empty_like(X, dtype=np.float32)
    augment_sequences_into(out, X, rng, rotation_deg, scale_range, noise_std, mirror_prob)
    return out


def augment_sequences_into(
    out: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
    rotation_deg: float | np.ndarray = 15.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    noise_std: float = 0.02,
    mirror_prob: float = 0.5,
) -> None:
    """Same transform as `augment_sequences`, but writes into a caller-
    supplied `out` buffer instead of allocating its own -- lets
    `build_augmented_trainset` below fill many augmented copies directly
    into one pre-allocated destination array, without a `X.copy()` here
    *and* a same-size slot in that destination existing at once. At ASL's
    full two-hand scale (~2GB base array), that redundant copy was enough
    to push a training run past this machine's available RAM.

    `rotation_deg` may be a per-sample array (shape (N,)) instead of one
    scalar for the whole batch. This exists because rotation robustness
    and word accuracy pull in different directions: real-device testing
    found static letters (ASL "V" especially) misreading badly once the
    signer's hand was rotated more than the training augmentation's range,
    and widening that range measurably fixed it (see
    eval_rotation_robustness.py) -- but applying the same wider range to
    *word* sequences cost about 0.9 points of word accuracy, plausibly
    because a dynamic sign's motion pattern is a harder thing to hold
    steady under added spatial jitter than a static pose is. Passing a
    per-sample array lets the caller give static letter/digit samples a
    wider range than word samples, in one training run, rather than
    picking one compromise value for every class.
    """
    n = X.shape[0]
    out[...] = X

    # The optional trailing anchor row (landmarks.frame_anchor) is NOT a
    # landmark: it holds (origin_x, origin_y, scale) in *raw frame*
    # coordinates, not hand-relative ones. Every transform below therefore
    # has to be applied to it separately or not at all -- sweeping it along
    # with the landmark block would corrupt it, in the same way that
    # sweeping single-hand frames through an unconditional slot swap
    # corrupted them (see the long comment further down). `land` and
    # `anchor` are basic-slice views, so in-place writes reach `out`.
    has_anchor = out.shape[-2] == TWO_HAND_LANDMARKS_WITH_ANCHOR
    n_land = TWO_HAND_LANDMARKS if has_anchor else out.shape[-2]
    land = out[:, :, :n_land, :]
    anchor = out[:, :, n_land:, :] if has_anchor else None

    mirrored = rng.random(n) < mirror_prob
    land[mirrored, ..., 0] *= -1.0
    if has_anchor:
        # Landmark x is hand-relative and centred on 0, so mirroring is a
        # sign flip. Anchor x is an absolute position in a [0, 1] frame,
        # where the mirror of x is 1 - x. Applying the landmark rule here
        # would place the hand outside the frame entirely.
        anchor[mirrored, ..., 0] = 1.0 - anchor[mirrored, ..., 0]

    # Two-hand data slots by wrist x-coordinate (see landmarks.py), so
    # mirroring -- which flips which hand has the smaller x -- must also
    # swap the two hand slots, or "slot 0 = smaller x" silently breaks.
    # Critically, this must be per-*frame*, not per-sample: a sample is
    # "mirrored" as a whole, but most individual frames (even in a
    # genuinely two-handed sign, and effectively all of them for ASL,
    # where only ~0.1% of frames have a real second hand -- see
    # ml/README.md) only have ONE hand present, and a lone hand always
    # belongs in slot 0 (see landmarks.py's sort_hands_by_x /
    # combine_two_hands_by_x and the app's matching Dart logic) regardless
    # of mirroring. Swapping unconditionally for every frame of a mirrored
    # *sample*, as an earlier version of this function did, moved a lone
    # hand into slot 1 for roughly half of all mirrored single-hand
    # frames -- a real train/inference mismatch for exactly the case that
    # dominates this data, and the direct cause of a training run scoring
    # *below* the single-hand baseline (46.5% vs. 52.0%) before this fix.
    # Note this tests `n_land`, the landmark-block width, not out.shape[-2]
    # -- with an anchor row appended the latter is 43, so testing it
    # directly would silently skip the swap entirely and reintroduce the
    # exact slot-convention break this block exists to prevent.
    if n_land == TWO_HAND_LANDMARKS:
        slot1_present = np.any(land[:, :, NUM_LANDMARKS:, :] != 0, axis=(2, 3))
        swap = mirrored[:, None] & slot1_present
        left_half = land[:, :, :NUM_LANDMARKS, :].copy()
        right_half = land[:, :, NUM_LANDMARKS:, :].copy()
        swap_3d = swap[:, :, None, None]
        land[:, :, :NUM_LANDMARKS, :] = np.where(swap_3d, right_half, left_half)
        land[:, :, NUM_LANDMARKS:, :] = np.where(swap_3d, left_half, right_half)

    # Rotation and scale jitter apply to the hand-relative landmark block
    # only. Rotating the anchor would spin an absolute frame position about
    # the frame's top-left corner, which corresponds to no real camera
    # motion; scaling it would drag the hand across the frame rather than
    # changing its apparent size.
    angles = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, size=n)).astype(np.float32)
    cos, sin = np.cos(angles), np.sin(angles)
    x = land[..., 0].copy()
    y = land[..., 1].copy()
    land[..., 0] = x * cos[:, None, None] - y * sin[:, None, None]
    land[..., 1] = x * sin[:, None, None] + y * cos[:, None, None]

    scales = rng.uniform(scale_range[0], scale_range[1], size=n).astype(np.float32)
    land *= scales[:, None, None, None]
    if has_anchor:
        # The anchor's third component *is* an apparent-size measure, so
        # the same jitter is exactly right for it -- it models the signer
        # standing nearer to or further from the camera. Its x/y position
        # is deliberately left alone.
        anchor[..., 2] *= scales[:, None, None]

    # rng.normal() has no dtype parameter and always returns float64 --
    # silently *doubling* memory for this one temporary array versus the
    # float32 everything else here uses (2.7GB instead of 1.35GB at ASL's
    # full two-hand scale), which is what was actually exhausting memory
    # even after every other optimization in this module.
    # standard_normal does support dtype=float32 directly.
    noise = rng.standard_normal(size=out.shape, dtype=np.float32)
    noise *= noise_std
    out += noise


def build_augmented_trainset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    copies: int,
    rng: np.random.Generator,
    **augment_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the augmented training set by pre-allocating the final
    (copies + 1) * N destination array once and filling each slice
    directly, instead of building a Python list of same-size arrays and
    `np.concatenate`-ing them at the end -- the list + fresh-concat-result
    pattern briefly holds both at once, doubling peak memory right when
    it's already highest. Still needs the final array (~(copies+1)x the
    input) plus the original X_train alive as a read source; if that's
    still too much, reduce --augment-copies rather than fight this
    function further.
    """
    n = len(X_train)
    total = n * (1 + copies)
    X_out = np.empty((total,) + X_train.shape[1:], dtype=np.float32)
    y_out = np.empty((total,), dtype=y_train.dtype)
    X_out[:n] = X_train
    y_out[:n] = y_train
    for i in range(copies):
        start = (i + 1) * n
        augment_sequences_into(X_out[start : start + n], X_train, rng, **augment_kwargs)
        y_out[start : start + n] = y_train
    return X_out, y_out
