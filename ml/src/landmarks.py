"""Shared landmark geometry: shapes, normalization, resampling.

Two hand "slots" are supported (see TWO_HAND_LANDMARKS below): asl-signs
and BdSLW60 both record a real second hand for two-handed signs, which an
earlier version of this pipeline discarded (kept "whichever hand tracked
better") -- a real, documented accuracy cost for any two-handed sign.

Slot assignment is geometric, not semantic: sorted by wrist x-coordinate
(smaller x -> slot 0), not by true left/right handedness. This is a
deliberate choice, not a shortcut -- the app's hand_landmarker plugin
reports 21-point hands with no handedness field at all, so there is no
left/right signal available at inference time to match against a
semantic label from training data. Geometric sorting is something both
training (from the dataset's raw per-frame coordinates, ignoring its
left_hand/right_hand column labels) and live inference (from the camera
frame's detected hands) can compute identically, which is what actually
matters for a trained model to generalize -- consistency between training
and inference, not anatomical correctness. The known tradeoff: if a
sign's two hands cross on-screen mid-gesture, their slot assignment can
flip; accepted as a minor, honestly-documented limitation rather than
a blocker (see ml/README.md).
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3  # x, y, z
DEFAULT_SEQ_LEN = 40  # ~1.3s of signing at 30fps; isolated signs are short
NUM_HANDS = 2
TWO_HAND_LANDMARKS = NUM_LANDMARKS * NUM_HANDS  # 42

# One extra row appended after the 42 hand landmarks, holding
# (origin_x, origin_y, scale) -- the *raw in-frame* position of the
# reference wrist and the hand's apparent size, i.e. exactly the two
# quantities normalize_two_hand_frame divides out and would otherwise
# discard. See ANCHOR docstring below for why this matters.
ANCHOR_ROW = 1
TWO_HAND_LANDMARKS_WITH_ANCHOR = TWO_HAND_LANDMARKS + ANCHOR_ROW  # 43

# In-frame position for static posed-photo sources, where the hand's
# location in the picture is an artefact of how the photo was cropped and
# carries no information about where the signer held their hand relative
# to their body. A fixed neutral value keeps those rows from injecting
# spurious position signal (see embed_single_hand).
NEUTRAL_ANCHOR_XY = 0.5

WRIST = 0
MIDDLE_MCP = 9


def normalize_hand_frame(frame: np.ndarray) -> np.ndarray:
    """Make one single-hand frame (21, 3) translation- and scale-invariant.

    Translates so the wrist is the origin, then scales by the wrist ->
    middle-finger-MCP distance (a stable proxy for hand size regardless of
    distance from the camera).
    """
    origin = frame[WRIST]
    centered = frame - origin
    scale = np.linalg.norm(centered[MIDDLE_MCP])
    if scale < 1e-6:
        return centered
    return centered / scale


def normalize_hand_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply `normalize_hand_frame` to every frame of a (T, 21, 3) sequence."""
    return np.stack([normalize_hand_frame(f) for f in seq])


def normalize_two_hand_frame(frame: np.ndarray) -> np.ndarray:
    """Make one two-hand frame (42, 3) translation- and scale-invariant.

    Unlike normalizing each hand independently, this uses a *single*
    shared origin and scale (from whichever hand slot is present,
    preferring slot 1) so the relative position between the two hands --
    often the entire signal for a two-handed sign -- survives
    normalization instead of being thrown away.
    """
    left, right = frame[:NUM_LANDMARKS], frame[NUM_LANDMARKS:]
    left_present = np.any(left != 0)
    right_present = np.any(right != 0)
    if not left_present and not right_present:
        return frame
    ref = right if right_present else left
    origin = ref[WRIST]
    scale = np.linalg.norm(ref[MIDDLE_MCP] - origin)
    if scale < 1e-6:
        return frame

    out = frame.copy()
    if left_present:
        out[:NUM_LANDMARKS] = (left - origin) / scale
    if right_present:
        out[NUM_LANDMARKS:] = (right - origin) / scale
    return out


def normalize_two_hand_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply `normalize_two_hand_frame` to every frame of a (T, 42, 3) sequence."""
    return np.stack([normalize_two_hand_frame(f) for f in seq])


def frame_anchor(frame: np.ndarray) -> np.ndarray:
    """The (origin_x, origin_y, scale) that `normalize_two_hand_frame`
    divides out of one (42, 3) frame, returned as a single (3,) row.

    Why this is worth keeping. Normalization makes a frame invariant to
    where the hand is and how big it looks -- which is exactly what you
    want for recognizing a *handshape*, and exactly wrong for recognizing
    a sign whose meaning depends on *where the hand is held*. A large
    number of ASL signs are distinguished only by placement against the
    face or body: identical handshape and motion at the chin versus the
    cheek are different words. After normalization those two become
    near-identical inputs, and the model has no way to tell them apart.

    That cost was measured, not assumed (ml/analyze_face_hypothesis.py):
    47 face-located ASL words average 44.2% per-class recall against
    54.1% for the other 203 -- a 9.9-point gap, Mann-Whitney U one-sided
    p = 0.0034.

    The obvious fix is a face landmark to normalize against, but the
    app's hand_landmarker plugin reports hands only, so training on one
    would reintroduce exactly the train/inference mismatch the geometric
    slot rule was designed to avoid. This is the fix that doesn't: the
    signer faces the camera, so the wrist's raw position *within the
    frame* is already a usable proxy for its position relative to the
    face -- and the plugin hands that number over for free. Nothing new
    is required at inference time; the pipeline simply stops throwing it
    away.
    """
    left, right = frame[:NUM_LANDMARKS], frame[NUM_LANDMARKS:]
    left_present = np.any(left != 0)
    right_present = np.any(right != 0)
    if not left_present and not right_present:
        return np.zeros(FEATURES_PER_LANDMARK, dtype=np.float32)
    ref = right if right_present else left
    origin = ref[WRIST]
    scale = float(np.linalg.norm(ref[MIDDLE_MCP] - origin))
    return np.array([origin[0], origin[1], scale], dtype=np.float32)


def append_anchor(normalized_seq: np.ndarray, raw_seq: np.ndarray) -> np.ndarray:
    """Concatenate the per-frame anchor row onto an already-normalized
    (T, 42, 3) sequence, giving (T, 43, 3).

    Takes both the normalized and the *raw* sequence because the anchor
    has to be computed from the raw coordinates -- by definition it is
    the information normalization removed.
    """
    anchors = np.stack([frame_anchor(f) for f in raw_seq])  # (T, 3)
    return np.concatenate([normalized_seq, anchors[:, np.newaxis, :]], axis=1)


def resample_sequence(
    seq: np.ndarray, target_len: int = DEFAULT_SEQ_LEN, num_landmarks: int = NUM_LANDMARKS
) -> np.ndarray:
    """Resample a (T, num_landmarks, 3) sequence to exactly `target_len` frames.

    Isolated signs vary in duration; the classifier needs a fixed-size
    input. Linear interpolation over the frame axis keeps the motion shape
    instead of naively truncating or zero-padding, which would throw away
    the end (or start) of the gesture.
    """
    t = seq.shape[0]
    if t == target_len:
        return seq
    if t == 0:
        return np.zeros((target_len, num_landmarks, FEATURES_PER_LANDMARK), dtype=np.float32)
    if t == 1:
        return np.repeat(seq, target_len, axis=0)

    src_idx = np.linspace(0, t - 1, num=t)
    dst_idx = np.linspace(0, t - 1, num=target_len)
    flat = seq.reshape(t, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_idx, src_idx, flat[:, col])
    return out.reshape(target_len, num_landmarks, FEATURES_PER_LANDMARK)


def prepare_sequence(seq: np.ndarray, target_len: int = DEFAULT_SEQ_LEN) -> np.ndarray:
    """Full single-hand prep pipeline: resample to a fixed length, then normalize."""
    return normalize_hand_sequence(resample_sequence(seq, target_len))


def prepare_two_hand_sequence(
    seq: np.ndarray,
    target_len: int = DEFAULT_SEQ_LEN,
    with_anchor: bool = True,
    neutral_anchor: bool = False,
) -> np.ndarray:
    """Full two-hand prep pipeline: resample to a fixed length, normalize,
    and (by default) append the in-frame position anchor -- giving
    (target_len, 43, 3). Pass `with_anchor=False` for the older
    (target_len, 42, 3) contract, kept so the pre-anchor models remain
    reproducible for the ablation in ml/README.md.

    `neutral_anchor=True` is for *static posed-photo* sources (the BdSL
    digit/letter CSVs, the ASL alphabet and digit image sets). There the
    hand's position within the picture is an artefact of framing and
    cropping, not a record of where the signer held their hand relative
    to their body, so feeding it in as position signal would be feeding
    in noise. Those rows get a fixed neutral value instead.
    """
    resampled = resample_sequence(seq, target_len, num_landmarks=TWO_HAND_LANDMARKS)
    normalized = normalize_two_hand_sequence(resampled)
    if not with_anchor:
        return normalized
    if neutral_anchor:
        anchors = np.tile(
            np.array([NEUTRAL_ANCHOR_XY, NEUTRAL_ANCHOR_XY, 1.0], dtype=np.float32),
            (len(normalized), 1),
        )
        return np.concatenate([normalized, anchors[:, np.newaxis, :]], axis=1)
    return append_anchor(normalized, resampled)


def embed_single_hand(frame: np.ndarray) -> np.ndarray:
    """Places a single-hand (21, 3) frame into slot 0 of a (42, 3) frame,
    zero-filling slot 1 -- the same place a real live single-handed sign
    lands in (a lone present hand always sorts to slot 0; see
    `combine_two_hands_by_x`). Used to fold single-frame *static* pose
    datasets (BdSL digits/letters, the ASL fingerspelling alphabet) into
    the two-hand contract every sequence model in this project now uses.
    """
    out = np.zeros((TWO_HAND_LANDMARKS, 3), dtype=np.float32)
    out[:NUM_LANDMARKS] = frame
    return out


def sort_hands_by_x(hands: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Assigns up to 2 detected hands (each (21, 3)) to (slot0, slot1) by
    wrist x-coordinate -- see the module docstring for why geometric, not
    semantic (left/right), ordering is used. Returns None for an absent
    slot. Shared by both training-data reconstruction and the app's
    runtime inference so the two sides can't drift apart.
    """
    if not hands:
        return None, None
    if len(hands) == 1:
        return hands[0], None
    a, b = sorted(hands[:2], key=lambda h: h[WRIST][0])
    return a, b


def combine_two_hands_by_x(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Combines two frame-aligned (T, 21, 3) arrays (each zero on frames
    where that semantic hand is absent) into one (T, 42, 3) sequence,
    slotting each frame's present hand(s) by wrist x-coordinate rather
    than the arrays' own left/right identity -- see the module docstring.
    Shared by every two-hand dataset loader (asl-signs, BdSLW60) so the
    combination rule can never drift between them.
    """
    left_present = np.any(left != 0, axis=(1, 2))
    right_present = np.any(right != 0, axis=(1, 2))
    left_x = left[:, WRIST, 0]
    right_x = right[:, WRIST, 0]

    seq = np.zeros((len(left), TWO_HAND_LANDMARKS, 3), dtype=np.float32)
    both = left_present & right_present
    left_only = left_present & ~right_present
    right_only = right_present & ~left_present
    left_first = both & (left_x <= right_x)
    right_first = both & ~left_first

    seq[left_first, :NUM_LANDMARKS] = left[left_first]
    seq[left_first, NUM_LANDMARKS:] = right[left_first]
    seq[right_first, :NUM_LANDMARKS] = right[right_first]
    seq[right_first, NUM_LANDMARKS:] = left[right_first]
    seq[left_only, :NUM_LANDMARKS] = left[left_only]
    seq[right_only, :NUM_LANDMARKS] = right[right_only]
    return seq
