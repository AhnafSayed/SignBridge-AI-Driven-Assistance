"""Verification for src/skeleton_render.py.

Exists because a real defect was found and fixed while building it:
render_two_hand_from_landmarks transformed landmark points (mirror/rotate/
scale) but never converted them from normalized [0,1] landmark space to
pixel space before drawing -- every point collapsed to canvas centre after
int truncation, rendering a single dot instead of two hands. It was caught
by looking at the actual output image, not by reasoning about the code,
which is exactly the kind of defect a numeric-only check can miss -- so
several of the checks below assert on pixel spread (bounding-box size,
non-white pixel count), not just "it ran without an exception".

Run:  python test_skeleton_render.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skeleton_render import (  # noqa: E402
    CANVAS,
    measure_drawn_hand_size,
    render_from_landmarks,
    render_two_hand_from_landmarks,
    render_two_hand_skeleton,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def one_hand(origin_xy=(0.4, 0.4), spread=0.15):
    """A plausible-looking (21, 3) single hand, not degenerate points."""
    pts = np.linspace(0, spread, 21, dtype=np.float32)
    f = np.zeros((21, 3), dtype=np.float32)
    f[:, 0] = origin_xy[0] + pts
    f[:, 1] = origin_xy[1] + pts * 1.3
    return f


def two_hand_frame(sep=0.3):
    left = one_hand((0.2, 0.4))
    right = one_hand((0.2 + sep, 0.35))
    return np.concatenate([left, right], axis=0)


print("1. Single-hand rendering produces a real, non-degenerate skeleton")
im = render_from_landmarks(one_hand())
nonwhite = int(np.sum(np.any(im != 255, axis=2)))
check("single-hand render has a substantial non-white pixel count", nonwhite > 200,
      f"got {nonwhite}")
w, h = measure_drawn_hand_size(im)
check("single-hand render has a real bounding box, not a point", w > 20 and h > 20,
      f"got ({w}, {h})")

print("\n2. Two-hand rendering -- the exact bug this file guards against")
frame = two_hand_frame()
im2 = render_two_hand_from_landmarks(frame)
nonwhite2 = int(np.sum(np.any(im2 != 255, axis=2)))
# The collapsed-to-a-dot bug produced on the order of 10-15 non-white
# pixels (a 2px joint marker or two); a real two-hand render produces
# thousands, so this threshold is comfortably clear of the bug's signature
# without being so tight it starts failing on legitimate small hands.
check("two-hand render has a substantial non-white pixel count", nonwhite2 > 1000,
      f"got {nonwhite2} -- this is the collapsed-to-a-point regression if small")

print("\n3. Two hands render at their real separation, not on top of each other")
# Split the canvas in half and confirm bone pixels exist on both sides --
# a collapsed render puts everything in one tiny cluster near the centre.
left_half = im2[:, : CANVAS // 2]
right_half = im2[:, CANVAS // 2 :]
left_nonwhite = int(np.sum(np.any(left_half != 255, axis=2)))
right_nonwhite = int(np.sum(np.any(right_half != 255, axis=2)))
check("both hands have visible pixels on their own side of the canvas",
      left_nonwhite > 100 and right_nonwhite > 100,
      f"got left={left_nonwhite} right={right_nonwhite}")

print("\n4. Mirroring a two-hand frame flips it left-right without collapsing it")
im2_mirror = render_two_hand_from_landmarks(frame, mirror=True)
nonwhite2_mirror = int(np.sum(np.any(im2_mirror != 255, axis=2)))
check("mirrored two-hand render is not degenerate either", nonwhite2_mirror > 1000,
      f"got {nonwhite2_mirror}")
# Mirroring reflects the point cloud about its own centroid *before* the
# same centring math re-centres the (same-size) bbox onto the canvas, so
# the finished image should closely match a plain horizontal flip of the
# unmirrored render -- a stronger, more direct check than comparing L/R
# pixel totals, which a coincidentally near-symmetric test frame can pass
# or fail without actually exercising the mirror at all.
flipped_original = cv2.flip(im2, 1)
matching_pixels = np.mean(np.all(im2_mirror == flipped_original, axis=2))
check("mirrored render matches a horizontal flip of the original (>=98% of pixels)",
      matching_pixels >= 0.98, f"got {matching_pixels:.1%} matching")

print("\n5. Absent-hand handling")
one_hand_only_frame = np.concatenate([one_hand(), np.zeros((21, 3), dtype=np.float32)], axis=0)
im_single_via_two = render_two_hand_from_landmarks(one_hand_only_frame)
nonwhite_single = int(np.sum(np.any(im_single_via_two != 255, axis=2)))
check("a single present hand through the two-hand path still renders substantially",
      nonwhite_single > 200, f"got {nonwhite_single}")

blank = render_two_hand_skeleton(None, None)
check("no hands present -> a blank (all-white) canvas, not a crash",
      bool(np.all(blank == 255)))

zero_hand = np.zeros((21, 2), dtype=np.int32)
blank2 = render_two_hand_skeleton(zero_hand, None)
check("an all-zero (absent) hand array is treated as absent, not drawn",
      bool(np.all(blank2 == 255)))

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("All skeleton_render checks passed.")
