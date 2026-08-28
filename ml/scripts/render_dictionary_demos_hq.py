"""Regenerates the in-app dictionary's demo images at higher visual
quality, for four of the six categories under
`app/assets/dictionary_demos/`:

    bdsl_word_<gloss>.png     (60 files -- two-hand BdSLW60 signs)
    bdsl_letter_<Name>.png    (37 files -- one-hand static letter poses)
    bdsl_digit_<0-9>.png      (10 files -- one-hand static digit poses)
    asl_word_<gloss>.png      (14 files -- the only ASL word glosses whose
                               parquet landmarks are present on this
                               machine; see ASL_WORD_GLOSSES below)

`asl_letter_*.png` and `asl_digit_*.png` are deliberately NOT touched --
another process is replacing those with real photographs, not skeleton
renders.

Why a separate script instead of reusing render_skeleton/render_two_hand_
skeleton from ml/src/skeleton_render.py directly: that module's whole job
is pixel-for-pixel compatibility with an external corpus's rendering
convention (see its module docstring) -- fixed 400x400 canvas, 3px bones,
2px joints, non-anti-aliased, hand drawn at native/unscaled size. That
convention is exactly why the demo images look sparse in the app: the
hand can occupy a small corner of the frame depending on how the original
recording happened to be scaled. None of that is a bug to fix in
skeleton_render.py -- ml training code elsewhere depends on its exact
output -- so this script duplicates just the drawing geometry (finger
chains, palm edges) with different, opt-in-only rendering choices:

    - a larger canvas (CANVAS below) instead of 400
    - a per-image scale factor computed from the rendered content's own
      bounding box, so the hand fills a healthy majority of the frame
      instead of whatever size the source recording happened to imply
    - thicker, anti-aliased (cv2.LINE_AA) strokes proportional to the
      bigger canvas

Source data (same corpora the original demo generation drew from):
    BdSL words:          ml/data/bdslw60 (ml/src/dataset_bdsl.py)
    BdSL letters/digits: ml/data/bdsl_digits/CSV Files/... (dataset_bdsl_static.py)
    ASL words:           ml/data/asl-signs (ml/src/dataset_asl.py) -- only
                          the 14 glosses in ASL_WORD_GLOSSES have any
                          parquet files actually present on disk (the rest
                          of the 250-word vocabulary's landmark files were
                          never downloaded to this machine).

For the 3-panel word filmstrips (bdsl_word_*, asl_word_*), each panel
gets its own scale factor + centring point from its own hand(s)'
bounding box -- same as the single-panel letter/digit renders, and the
same thing render_two_hand_skeleton already did for centring (just
without ever scaling). Independently filling every panel to the target
is what actually fixes the "sparse" complaint in all three frames, not
just whichever frame happens to have the largest excursion.

Usage:
    python render_dictionary_demos_hq.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ML_ROOT)
sys.path.insert(0, os.path.join(ML_ROOT, "src"))

import dataset_asl as asl_ds  # noqa: E402
import dataset_bdsl as bdsl_word_ds  # noqa: E402
from dataset_bdsl_static import LETTER_NAMES, load_static_dataset  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "app", "assets", "dictionary_demos")
CAPTIONS_PATH = os.path.join(OUT_DIR, "captions.json")

BDSL_DIGITS_ROOT = os.path.join(ML_ROOT, "data", "bdsl_digits", "CSV Files", "Sign Digits - CSV Files")
BDSL_LETTERS_ROOT = os.path.join(ML_ROOT, "data", "bdsl_digits", "CSV Files", "Sign Letters - CSV Files")
BDSLW60_ROOT = os.path.join(ML_ROOT, "data", "bdslw60")
ASL_SIGNS_ROOT = os.path.join(ML_ROOT, "data", "asl-signs")

# The only 14 ASL word glosses whose train_landmark_files parquets are
# actually present on this machine (see the module docstring above).
ASL_WORD_GLOSSES = [
    "bird", "duck", "brother", "bee", "brown", "drink", "awake", "bye",
    "farm", "donkey", "cat", "cow", "find", "doll",
]

# --- HQ rendering geometry -------------------------------------------------
# Deliberately separate from skeleton_render.py's CANVAS/BONE_THICKNESS/
# JOINT_RADIUS constants -- see module docstring.
CANVAS = 720
PANEL_GAP = 18
TARGET_FILL = 0.70  # rendered content's larger bbox dimension / canvas

BONE_COLOR = (0, 255, 0)   # BGR, same convention as skeleton_render.py
JOINT_COLOR = (0, 0, 255)
BONE_THICKNESS = 6
JOINT_RADIUS = 5

FINGER_CHAINS = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 20))
PALM_EDGES = ((5, 9), (9, 13), (13, 17), (0, 5), (0, 17))


def _draw_hand_hq(img: np.ndarray, pts_px: np.ndarray) -> None:
    """Draws one hand's bones + joints with thick, anti-aliased strokes.
    `pts_px` is (21, 2) integer pixel coordinates."""
    for start, end in FINGER_CHAINS:
        for t in range(start, end):
            cv2.line(img, tuple(pts_px[t]), tuple(pts_px[t + 1]), BONE_COLOR,
                      BONE_THICKNESS, lineType=cv2.LINE_AA)
    for a, b in PALM_EDGES:
        cv2.line(img, tuple(pts_px[a]), tuple(pts_px[b]), BONE_COLOR,
                  BONE_THICKNESS, lineType=cv2.LINE_AA)
    for i in range(len(pts_px)):
        cv2.circle(img, tuple(pts_px[i]), JOINT_RADIUS, JOINT_COLOR, -1,
                    lineType=cv2.LINE_AA)


def _present_hands(hands_xy: list[np.ndarray | None]) -> list[np.ndarray]:
    return [h for h in hands_xy if h is not None and np.any(h)]


def render_filmstrip(frames_hands: list[list[np.ndarray | None]],
                      canvas: int = CANVAS, gap: int = PANEL_GAP,
                      target_fill: float = TARGET_FILL) -> np.ndarray:
    """Renders 1-3 panels side by side. `frames_hands` is a list of
    per-panel hand lists (each item 0-2 (21, 2) normalized-xy arrays, or
    None/all-zero for an absent hand). Each panel is independently
    centred and scaled from its own present hand(s)' bounding box, so
    every panel -- not just whichever frame has the largest hand --
    fills a healthy majority of its own frame.
    """
    panels = []
    for frame in frames_hands:
        present = _present_hands(frame)
        img = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255
        if present:
            all_pts = np.concatenate(present, axis=0)
            x0, x1 = all_pts[:, 0].min(), all_pts[:, 0].max()
            y0, y1 = all_pts[:, 1].min(), all_pts[:, 1].max()
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            content_scale = (target_fill * canvas) / max(w, h)
            for hand in present:
                px = (hand[:, 0] - cx) * content_scale + canvas / 2.0
                py = (hand[:, 1] - cy) * content_scale + canvas / 2.0
                pts_px = np.round(np.stack([px, py], axis=1)).astype(np.int32)
                np.clip(pts_px, 0, canvas - 1, out=pts_px)
                _draw_hand_hq(img, pts_px)
        panels.append(img)

    n = len(panels)
    total_w = canvas * n + gap * (n - 1)
    strip = np.ones((canvas, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels:
        strip[:, x:x + canvas] = p
        x += canvas + gap
    return strip


def _two_hand_frame_to_hands(frame42: np.ndarray) -> list[np.ndarray | None]:
    """(42, 3) two-hand-contract frame -> [slot0_xy_or_None, slot1_xy_or_None]."""
    slot0, slot1 = frame42[:21, :2], frame42[21:42, :2]
    return [
        slot0 if np.any(slot0) else None,
        slot1 if np.any(slot1) else None,
    ]


def _pick_three_indices(t: int) -> tuple[int, int, int]:
    if t <= 3:
        # pad by repeating the last index rather than crashing on very
        # short sequences.
        idxs = list(range(t)) + [t - 1] * (3 - t)
        return tuple(idxs[:3])
    return 0, t // 2, t - 1


# --- BdSL words (60, two hands, temporal) ----------------------------------

def render_bdsl_words() -> dict[str, np.ndarray]:
    index = bdsl_word_ds.load_index(BDSLW60_ROOT)
    by_label: dict[str, list[str]] = {}
    for rel_path, word_id, signer_id, label in index.samples:
        by_label.setdefault(label, []).append(rel_path)

    out = {}
    missing = []
    for label, rel_paths in sorted(by_label.items()):
        best = None  # (num_frames, seq)
        for rel_path in rel_paths:
            seq = bdsl_word_ds.load_sequence(index.data_dir, rel_path)
            if seq is None or seq.shape[0] == 0:
                continue
            if best is None or seq.shape[0] > best[0]:
                best = (seq.shape[0], seq)
        if best is None:
            missing.append(label)
            continue
        t, seq = best
        i0, i1, i2 = _pick_three_indices(t)
        frames_hands = [_two_hand_frame_to_hands(seq[i]) for i in (i0, i1, i2)]
        out[f"bdsl_word_{label}"] = render_filmstrip(frames_hands)
    if missing:
        print(f"  [bdsl_word] no usable sequence for: {missing}")
    return out


# --- BdSL letters / digits (static single-hand poses) ----------------------

def _pick_representative(frames: list[np.ndarray]) -> np.ndarray:
    """Picks the sample whose hand bbox size is closest to the class's
    median, so the rendered demo is a typical pose rather than an
    outlier crop."""
    sizes = []
    for f in frames:
        xy = f[:, :2]
        sizes.append(max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()))
    median = float(np.median(sizes))
    idx = int(np.argmin(np.abs(np.array(sizes) - median)))
    return frames[idx]


def render_bdsl_static(root: str, label_names: list[str], key_prefix: str) -> dict[str, np.ndarray]:
    X, y, _signers = load_static_dataset(root)
    by_label: dict[int, list[np.ndarray]] = {}
    for frame, lbl in zip(X, y):
        by_label.setdefault(int(lbl), []).append(frame)

    out = {}
    missing = []
    for i, name in enumerate(label_names):
        frames = by_label.get(i)
        if not frames:
            missing.append(name)
            continue
        frame = _pick_representative(frames)
        hands = [frame[:, :2], None]
        out[f"{key_prefix}_{name}"] = render_filmstrip([hands])
    if missing:
        print(f"  [{key_prefix}] no usable sample for: {missing}")
    return out


# --- ASL words (14, two hands, temporal) -----------------------------------

def render_asl_words() -> dict[str, np.ndarray]:
    index = asl_ds.load_index(ASL_SIGNS_ROOT)
    df = index.train_df
    out = {}
    missing = []
    for gloss in ASL_WORD_GLOSSES:
        sub = df[df["sign"] == gloss]
        exists_mask = sub["path"].apply(lambda p: os.path.exists(os.path.join(index.data_dir, p)))
        candidates = sub[exists_mask]["path"].tolist()
        best = None  # (num_frames, seq)
        for path in candidates:
            seq = asl_ds.load_sequence(index.data_dir, path)
            if seq is None or seq.shape[0] == 0:
                continue
            if best is None or seq.shape[0] > best[0]:
                best = (seq.shape[0], seq)
            # Scanning every locally-present file per gloss (~40) is cheap
            # enough (small parquet files) to just find the true best.
        if best is None:
            missing.append(gloss)
            continue
        t, seq = best
        i0, i1, i2 = _pick_three_indices(t)
        frames_hands = [_two_hand_frame_to_hands(seq[i]) for i in (i0, i1, i2)]
        out[f"asl_word_{gloss}"] = render_filmstrip(frames_hands)
    if missing:
        print(f"  [asl_word] no usable sequence for: {missing}")
    return out


def main():
    with open(CAPTIONS_PATH, encoding="utf-8") as f:
        captions = json.load(f)

    print("Loading BdSL word sequences (BdSLW60)...")
    bdsl_words = render_bdsl_words()
    print(f"  rendered {len(bdsl_words)}/60")

    print("Loading BdSL letter poses...")
    bdsl_letters = render_bdsl_static(BDSL_LETTERS_ROOT, LETTER_NAMES, "bdsl_letter")
    print(f"  rendered {len(bdsl_letters)}/{len(LETTER_NAMES)}")

    print("Loading BdSL digit poses...")
    digit_names = [str(d) for d in range(10)]
    bdsl_digits = render_bdsl_static(BDSL_DIGITS_ROOT, digit_names, "bdsl_digit")
    print(f"  rendered {len(bdsl_digits)}/10")

    print("Loading ASL word sequences (asl-signs)...")
    asl_words = render_asl_words()
    print(f"  rendered {len(asl_words)}/{len(ASL_WORD_GLOSSES)}")

    all_images = {**bdsl_words, **bdsl_letters, **bdsl_digits, **asl_words}

    expected_keys = {
        k for k in captions
        if k.startswith("bdsl_word_") or k.startswith("bdsl_letter_")
        or k.startswith("bdsl_digit_") or (k.startswith("asl_word_")
        and k[len("asl_word_"):] in ASL_WORD_GLOSSES)
    }
    unexpected = set(all_images) - expected_keys
    not_rendered = expected_keys - set(all_images)
    if unexpected:
        print(f"WARNING: rendered keys not in captions.json / not expected: {sorted(unexpected)}")
    if not_rendered:
        print(f"WARNING: expected keys with no rendered image (source data missing): {sorted(not_rendered)}")

    written = []
    for key, img in sorted(all_images.items()):
        out_path = os.path.join(OUT_DIR, f"{key}.png")
        ok = cv2.imwrite(out_path, img)
        if not ok:
            print(f"  FAILED to write {out_path}")
            continue
        written.append(key)

    print(f"\nWrote {len(written)} images to {OUT_DIR} at canvas={CANVAS} "
          f"(filmstrip gap={PANEL_GAP}, target_fill={TARGET_FILL}).")
    skipped = expected_keys - set(written)
    if skipped:
        print(f"Skipped (no usable source landmark data found): {sorted(skipped)}")


if __name__ == "__main__":
    main()
