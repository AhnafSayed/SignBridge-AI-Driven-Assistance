"""Renders 21 hand landmarks as a skeleton image on a white canvas.

This reproduces the drawing convention used by Devansh-47's
Sign-Language-To-Text-and-Speech-Conversion (`data_collection_final.py`),
so that its AtoZ_3.1 corpus and our own landmark corpora become the *same
representation* and can be trained on together.

Why bother, when we already have the landmarks? Because their images are
JPEG-compressed line drawings: the 21 red joint dots are 2px circles drawn
over 3px green bones, and compression destroys most of them (a spot check
recovered 9 of 21). Their landmarks are therefore not recoverable from
their images -- but ours *are* renderable into their space. Meeting in
image space is the only representation both corpora can occupy.

The convention, matched exactly (see their lines 40-60):

    canvas   400x400, white
    bones    green (0,255,0), thickness 3
             thumb 0-1-2-3-4, index 5-6-7-8, middle 9-10-11-12,
             ring 13-14-15-16, pinky 17-18-19-20
    palm     5-9, 9-13, 13-17, 0-5, 0-17
    joints   red (0,0,255), radius 2, thickness 1, all 21 points
    centring hand bbox centred on the canvas, drawn at native pixel scale

That last point matters and is easy to get wrong. They do *not* rescale
the hand to fill the canvas -- they translate it to the centre and leave
its pixel size alone, so apparent hand size varies with how far the signer
sat from the camera. Rescaling our samples to a fixed size would put them
in a visibly different distribution from theirs and the merge would teach
the model to separate the two sources rather than the letters.
"""

from __future__ import annotations

import cv2
import numpy as np

CANVAS = 400
BONE_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 0, 255)
BONE_THICKNESS = 3
JOINT_RADIUS = 2
JOINT_THICKNESS = 1

# Our landmarks are stored frame-normalized in [0, 1], so mapping them
# back to pixels needs a frame size. It must be SQUARE: the alphabet and
# digit corpora are square images, so their normalized coordinates are
# relative to equal width and height. Using a 640x480 frame (the obvious
# guess, matching a webcam) stretches x by 4/3 and measurably flattened
# the rendered hands -- aspect 1.10 against their 1.79. With a square
# frame the aspect is a stable 1.47 at every scale, which is what says the
# mapping is right; the remaining 1.47-vs-1.79 difference is a real
# difference between the two corpora's signers and crops, not a bug, and
# is what SCALE_JITTER below exists to blur.
#
# 545 is chosen so our median rendered hand height (236 px) matches
# theirs (240 px), putting both corpora at the same apparent distance
# from the camera.
VIRTUAL_FRAME = 545
VIRTUAL_FRAME_W = VIRTUAL_FRAME
VIRTUAL_FRAME_H = VIRTUAL_FRAME

# Their hand sizes span p10-p90 = 195-312 px tall against our median of
# 236, so jittering our renders over this range covers their spread. This
# is doing double duty: it is ordinary scale augmentation, and it stops
# the network from telling the two sources apart by hand size alone --
# which it would otherwise be free to do, learning "source" instead of
# "letter".
SCALE_JITTER = (0.80, 1.30)

FINGER_CHAINS = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 20))
PALM_EDGES = ((5, 9), (9, 13), (13, 17), (0, 5), (0, 17))


def landmarks_to_pixels(landmarks: np.ndarray,
                        frame_w: int = VIRTUAL_FRAME_W,
                        frame_h: int = VIRTUAL_FRAME_H) -> np.ndarray:
    """Frame-normalized (x, y) in [0,1] -> integer pixel coordinates.

    Only x and y are used; z is dropped, because the target representation
    is a flat drawing and their pipeline never had a z either. That is a
    real loss of information relative to our landmark model, and is one of
    the things the evaluation has to weigh.
    """
    pts = np.empty((len(landmarks), 2), dtype=np.int32)
    pts[:, 0] = np.round(landmarks[:, 0] * frame_w).astype(np.int32)
    pts[:, 1] = np.round(landmarks[:, 1] * frame_h).astype(np.int32)
    return pts


def _draw_hand(canvas_img: np.ndarray, p: np.ndarray) -> None:
    """Draws one hand's bones + joints onto an existing canvas at already
    positioned pixel coordinates `p` ((21, 2), int). Shared by the
    single-hand and two-hand renderers so the two never drift apart."""
    for start, end in FINGER_CHAINS:
        for t in range(start, end):
            cv2.line(canvas_img, tuple(p[t]), tuple(p[t + 1]), BONE_COLOR, BONE_THICKNESS)
    for a, b in PALM_EDGES:
        cv2.line(canvas_img, tuple(p[a]), tuple(p[b]), BONE_COLOR, BONE_THICKNESS)
    for i in range(len(p)):
        cv2.circle(canvas_img, tuple(p[i]), JOINT_RADIUS, JOINT_COLOR, JOINT_THICKNESS)


def render_skeleton(pts: np.ndarray, canvas: int = CANVAS) -> np.ndarray:
    """Draws centred bones + joints for one hand. `pts` is (21, 2) pixels."""
    white = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255

    # Centre exactly as they do: offset by half the leftover space after
    # the hand's own bounding box, minus their fixed 15px crop margin.
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    w = pts[:, 0].max() - x0
    h = pts[:, 1].max() - y0
    ox = ((canvas - w) // 2) - 15 - x0
    oy = ((canvas - h) // 2) - 15 - y0
    p = pts + np.array([ox, oy], dtype=np.int32)
    np.clip(p, 0, canvas - 1, out=p)
    _draw_hand(white, p)
    return white


def render_two_hand_skeleton(hand0: np.ndarray | None, hand1: np.ndarray | None,
                             canvas: int = CANVAS) -> np.ndarray:
    """Draws up to two hands from already-pixel-mapped (21, 2) arrays,
    for BdSL signs -- unlike ASL fingerspelling, ~43% of BdSLW60's frames
    genuinely use both hands (see ml/README.md's "Two hands, not one").

    Either argument may be `None`, or an all-zero array (the two-hand
    contract's zero-fill for an absent hand, see ml/src/landmarks.py), to
    mean that hand isn't present in this frame.

    Centring is done on the *combined* bounding box of whichever hands are
    present, not per-hand. That distinction matters here in a way it
    doesn't for a lone hand: independently centering each hand would erase
    how far apart they are and where one sits relative to the other, and
    that arrangement is part of what a two-handed BdSL sign actually
    means -- the same reasoning `normalize_two_hand_frame` already applies
    when it picks one shared origin/scale for both hands rather than
    normalizing each in isolation (see landmarks.py's module docstring).
    """
    present = [h[:, :2] for h in (hand0, hand1) if h is not None and np.any(h)]
    white = np.ones((canvas, canvas, 3), dtype=np.uint8) * 255
    if not present:
        return white

    all_pts = np.concatenate(present, axis=0)
    x0, y0 = all_pts[:, 0].min(), all_pts[:, 1].min()
    w = all_pts[:, 0].max() - x0
    h = all_pts[:, 1].max() - y0
    ox = ((canvas - w) // 2) - 15 - x0
    oy = ((canvas - h) // 2) - 15 - y0
    offset = np.array([ox, oy], dtype=np.int32)

    for hpts in present:
        p = (hpts + offset).astype(np.int32)
        np.clip(p, 0, canvas - 1, out=p)
        _draw_hand(white, p)
    return white


def render_from_landmarks(landmarks: np.ndarray, canvas: int = CANVAS,
                          scale: float = 1.0, rotation_deg: float = 0.0,
                          mirror: bool = False) -> np.ndarray:
    """Convenience: frame-normalized (21, 3) landmarks -> skeleton image.

    `scale` and `rotation_deg` are applied in landmark space rather than
    to the finished bitmap. Rotating a drawn image would resample the
    3px bones into soft grey edges that look nothing like either corpus;
    rotating the points first re-draws crisp bones at the new angle, so
    augmented samples stay in the same visual distribution as real ones.

    `mirror` flips the hand left-right about its own centroid before
    drawing. This is the same operation, and the same justification, as
    the landmark-sequence model's `mirror_prob` augmentation
    (src/augment.py): every ASL fingerspelling letter performed by a right
    hand is the mirror image of the same letter performed by a left hand,
    with no change of label, so mirroring is label-preserving and is how
    a corpus collected from mostly one-handed signers is made to cover
    both. It needs to happen here rather than as a raster flip on the
    caller's side only because callers may also want it combined with
    in-plane scale/rotation jitter in one landmark-space pass.
    """
    pts = _transform_points(landmarks[:, :2], mirror, rotation_deg, scale)
    padded = np.concatenate([pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1)
    return render_skeleton(landmarks_to_pixels(padded), canvas)


def _transform_points(pts: np.ndarray, mirror: bool, rotation_deg: float,
                      scale: float) -> np.ndarray:
    """Applies mirror/rotate/scale to a (N, 2) point cloud about its own
    centroid, in that order. Factored out of render_from_landmarks so the
    two-hand path can apply the same transform to *both* hands' points
    jointly (one shared centroid) rather than each hand independently,
    which would scramble their relative position -- see
    render_two_hand_from_landmarks's docstring.
    """
    pts = pts.astype(np.float32).copy()
    if mirror:
        centre_x = pts[:, 0].mean()
        pts[:, 0] = 2 * centre_x - pts[:, 0]
    if rotation_deg:
        centre = pts.mean(axis=0)
        a = np.deg2rad(rotation_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
        pts = (pts - centre) @ rot.T + centre
    if scale != 1.0:
        centre = pts.mean(axis=0)
        pts = (pts - centre) * scale + centre
    return pts


def render_two_hand_from_landmarks(two_hand_frame: np.ndarray, canvas: int = CANVAS,
                                   scale: float = 1.0, rotation_deg: float = 0.0,
                                   mirror: bool = False) -> np.ndarray:
    """Convenience: a (42, 3) two-hand-contract frame -- slot0 = frame[:21],
    slot1 = frame[21:], either or both possibly zero-filled for an absent
    hand (see ml/src/landmarks.py) -- rendered as a two-hand skeleton
    image via render_two_hand_skeleton.

    mirror/rotation/scale are applied to BOTH hands' points together, as
    one combined point cloud sharing a single centroid, for the same
    reason centring is combined rather than per-hand: independently
    transforming each hand would scramble their relative arrangement,
    which for a two-handed sign can be part of what distinguishes it.
    Mirroring additionally swaps which hand occupies slot0 vs slot1 --
    mirroring a frame left-right also mirrors which hand is which by the
    x-sorted slot convention (see ml/src/landmarks.py's
    combine_two_hands_by_x) -- so a mirrored two-hand frame with both
    hands present has its slots swapped, matching what `augment.py`
    already does for the landmark-sequence model's own mirror
    augmentation.
    """
    slot0, slot1 = two_hand_frame[:21], two_hand_frame[21:42]
    present0, present1 = np.any(slot0 != 0), np.any(slot1 != 0)

    combined = []
    if present0:
        combined.append(slot0[:, :2])
    if present1:
        combined.append(slot1[:, :2])
    if not combined:
        return np.ones((canvas, canvas, 3), dtype=np.uint8) * 255

    n0 = len(slot0) if present0 else 0
    transformed = _transform_points(np.concatenate(combined, axis=0), mirror, rotation_deg, scale)
    # transformed is still in normalized [0, 1]-ish landmark space --
    # render_two_hand_skeleton expects pixel coordinates, same as
    # render_skeleton does via landmarks_to_pixels in render_from_landmarks.
    # Skipping this step was a real bug caught by inspection: without it,
    # every landmark collapses to pixel (0, 0) after int truncation, which
    # the offset math then centres into a single dot at the canvas middle
    # -- exactly the "tiny red diamond, no bones" image a first render
    # produced.
    padded = np.concatenate([transformed, np.zeros((len(transformed), 1), dtype=np.float32)], axis=1)
    pixels = landmarks_to_pixels(padded)
    t0 = pixels[:n0] if present0 else None
    t1 = pixels[n0:] if present1 else None
    # Mirroring swaps which physical hand sorts to the lower-x slot, so
    # swap which transformed array is treated as "slot0" for drawing --
    # purely cosmetic here (draw order doesn't matter, both are drawn),
    # kept for clarity that the swap is understood, not accidental.
    if mirror:
        t0, t1 = t1, t0
    return render_two_hand_skeleton(t0, t1, canvas)


def hand_pixel_size(landmarks: np.ndarray,
                    frame_w: int = VIRTUAL_FRAME_W,
                    frame_h: int = VIRTUAL_FRAME_H) -> tuple[int, int]:
    """Bounding-box size the hand would occupy, for distribution checks."""
    pts = landmarks_to_pixels(landmarks, frame_w, frame_h)
    return (int(pts[:, 0].max() - pts[:, 0].min()),
            int(pts[:, 1].max() - pts[:, 1].min()))


def measure_drawn_hand_size(image: np.ndarray) -> tuple[int, int]:
    """Bounding box of the drawn skeleton in an existing rendering.

    Used to compare our renderings against theirs on equal terms: it reads
    the green bone pixels back out of a finished image, so it works on
    their JPEGs (where the red joints are largely destroyed) as well as on
    ours.
    """
    b, g, r = image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)
    drawn = (g > 150) & (r < 120) & (b < 120)
    ys, xs = np.where(drawn)
    if not len(xs):
        return (0, 0)
    return (int(xs.max() - xs.min()), int(ys.max() - ys.min()))
