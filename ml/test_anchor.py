"""Verification for the in-frame position anchor and its augmentation.

Exists because the last change to this augmentation path shipped a defect
that corrupted roughly half of all mirrored single-hand training frames
and drove held-out accuracy five points BELOW baseline, while the code
read as correct (see ml/README.md, "Two hands, not one"). The anchor row
touches the same function, so it gets checked rather than assumed.

Run:  python test_anchor.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from augment import augment_sequences  # noqa: E402
from landmarks import (  # noqa: E402
    NUM_LANDMARKS,
    TWO_HAND_LANDMARKS,
    TWO_HAND_LANDMARKS_WITH_ANCHOR,
    embed_single_hand,
    frame_anchor,
    prepare_two_hand_sequence,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def one_hand_frame(origin_xy=(0.30, 0.20), spread=0.05):
    """A (42,3) frame with a plausible hand in slot 0 and slot 1 empty."""
    f = np.zeros((TWO_HAND_LANDMARKS, 3), dtype=np.float32)
    pts = np.linspace(0, spread, NUM_LANDMARKS, dtype=np.float32)
    f[:NUM_LANDMARKS, 0] = origin_xy[0] + pts
    f[:NUM_LANDMARKS, 1] = origin_xy[1] + pts
    return f


def two_hand_frame():
    f = one_hand_frame((0.30, 0.20))
    pts = np.linspace(0, 0.05, NUM_LANDMARKS, dtype=np.float32)
    f[NUM_LANDMARKS:, 0] = 0.70 + pts
    f[NUM_LANDMARKS:, 1] = 0.25 + pts
    return f


print("1. Anchor extraction")
f = one_hand_frame(origin_xy=(0.30, 0.20))
a = frame_anchor(f)
check("anchor x == raw wrist x", abs(a[0] - 0.30) < 1e-5, f"got {a[0]}")
check("anchor y == raw wrist y", abs(a[1] - 0.20) < 1e-5, f"got {a[1]}")
check("anchor scale > 0", a[2] > 0, f"got {a[2]}")

empty = np.zeros((TWO_HAND_LANDMARKS, 3), dtype=np.float32)
check("empty frame -> zero anchor", np.allclose(frame_anchor(empty), 0))

print("\n2. Sequence shape and position sensitivity")
seq_lo = np.stack([one_hand_frame((0.30, 0.20))] * 5)
seq_hi = np.stack([one_hand_frame((0.30, 0.75))] * 5)
p_lo = prepare_two_hand_sequence(seq_lo, target_len=8)
p_hi = prepare_two_hand_sequence(seq_hi, target_len=8)
check("shape is (T, 43, 3)", p_lo.shape == (8, TWO_HAND_LANDMARKS_WITH_ANCHOR, 3), f"got {p_lo.shape}")
check(
    "landmark block identical for both heights (normalization still invariant)",
    np.allclose(p_lo[:, :TWO_HAND_LANDMARKS], p_hi[:, :TWO_HAND_LANDMARKS], atol=1e-5),
)
check(
    "anchor DOES differ -- the whole point: position is no longer discarded",
    not np.allclose(p_lo[:, TWO_HAND_LANDMARKS], p_hi[:, TWO_HAND_LANDMARKS], atol=1e-3),
    f"lo={p_lo[0, TWO_HAND_LANDMARKS]} hi={p_hi[0, TWO_HAND_LANDMARKS]}",
)

print("\n3. Backward compatibility")
p_old = prepare_two_hand_sequence(seq_lo, target_len=8, with_anchor=False)
check("with_anchor=False gives (T, 42, 3)", p_old.shape == (8, TWO_HAND_LANDMARKS, 3), f"got {p_old.shape}")
check("landmark block unchanged vs anchored", np.allclose(p_old, p_lo[:, :TWO_HAND_LANDMARKS], atol=1e-6))

print("\n4. Neutral anchor for static posed photos")
static = prepare_two_hand_sequence(
    embed_single_hand(one_hand_frame()[:NUM_LANDMARKS])[np.newaxis, ...],
    target_len=4,
    neutral_anchor=True,
)
check("static anchor is the fixed neutral value", np.allclose(static[:, TWO_HAND_LANDMARKS, :2], 0.5))
check("static anchor constant across frames", np.allclose(static[0, TWO_HAND_LANDMARKS], static[-1, TWO_HAND_LANDMARKS]))

print("\n5. Augmentation -- the regression-prone part")
rng = np.random.default_rng(0)
# All-single-hand batch: slot 1 must stay empty no matter how many are mirrored.
X_single = np.stack([prepare_two_hand_sequence(seq_lo, target_len=8) for _ in range(200)])
aug = augment_sequences(X_single, rng, noise_std=0.0)
slot1 = aug[:, :, NUM_LANDMARKS:TWO_HAND_LANDMARKS, :]
check(
    "single-hand: slot 1 stays empty after mirroring (Defect 2 regression guard)",
    np.allclose(slot1, 0, atol=1e-6),
    f"max abs {np.abs(slot1).max()}",
)

# Anchor x must land in [0,1] -- the sign-flip rule would send it negative.
anchor_x = aug[:, :, TWO_HAND_LANDMARKS, 0]
check(
    "anchor x stays within the [0,1] frame after mirroring",
    anchor_x.min() >= -1e-5 and anchor_x.max() <= 1.0 + 1e-5,
    f"range [{anchor_x.min():.3f}, {anchor_x.max():.3f}]",
)
# With origin_x = 0.30, every sample is either 0.30 or its mirror 0.70.
uniq = np.unique(anchor_x)
# float32 round-trips don't land on exact decimal literals, so compare
# with a tolerance rather than by set equality.
check(
    "anchor x takes exactly the two values {0.30, 0.70} -- mirrored as 1-x, not -x",
    len(uniq) == 2 and np.isclose(uniq.min(), 0.30, atol=1e-5) and np.isclose(uniq.max(), 0.70, atol=1e-5),
    f"got {uniq.tolist()}",
)

# Two-hand batch: the slot swap must still fire.
X_two = np.stack([prepare_two_hand_sequence(np.stack([two_hand_frame()] * 5), target_len=8) for _ in range(200)])
aug2 = augment_sequences(X_two, rng, noise_std=0.0)
occupied = np.any(aug2[:, :, NUM_LANDMARKS:TWO_HAND_LANDMARKS, :] != 0, axis=(2, 3))
check("two-hand: both slots remain occupied after augmentation", occupied.all())

# Anchor scale must be jittered (it models camera distance), never zero/negative.
anchor_scale = aug2[:, :, TWO_HAND_LANDMARKS, 2]
check("anchor scale stays positive", anchor_scale.min() > 0, f"min {anchor_scale.min()}")
check("anchor scale is jittered across samples", np.std(anchor_scale) > 1e-6)

print("\n6. Single-hand (21-landmark) legacy path still works")
X_legacy = np.random.default_rng(1).normal(size=(10, 8, NUM_LANDMARKS, 3)).astype(np.float32)
aug_legacy = augment_sequences(X_legacy, rng, noise_std=0.0)
check("legacy 21-landmark input unchanged in shape", aug_legacy.shape == X_legacy.shape)

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("All anchor checks passed.")
