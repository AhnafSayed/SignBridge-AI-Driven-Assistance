"""Measures how often the hands-free control-gesture detector's geometry
misreads a real fingerspelled letter/digit as a control gesture, comparing
the current shipped rule (any non-vertical-thumb fist defaults to
backspace) against a proposed fix (backspace requires a clearly
*horizontal* thumb; a merely tucked/ambiguous thumb is no-gesture instead).

This exists because a real-device report says hands-free control gestures
misfire specifically on fist-shaped letters/digits (ASL S/E/A/M/T/N and
similar BdSL shapes) -- the same collision this project's own
control_gesture.dart docstring already measured for the OLD rule. Before
changing the shipped geometry, the replacement needs the same measurement,
not just reasoning about what ASL handshapes "should" look like.

Usage:
    python eval_control_gesture_collision.py
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
MIDDLE_MCP = 9
# MediaPipe finger-tip / pip landmark pairs used by the shipped
# "any finger up" disqualifier in control_gesture.dart.
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]


def fingers_up(frame: np.ndarray) -> bool:
    return any(frame[tip, 1] < frame[pip, 1] - 0.02 for tip, pip in FINGER_TIP_PIP)


def classify_old(frame: np.ndarray) -> str:
    if fingers_up(frame):
        return "none"
    scale = np.linalg.norm(frame[WRIST] - frame[MIDDLE_MCP])
    if scale < 1e-6:
        return "backspace"
    dx = frame[THUMB_TIP, 0] - frame[THUMB_MCP, 0]
    dy = frame[THUMB_TIP, 1] - frame[THUMB_MCP, 1]
    vertical = abs(dx) < abs(dy) * 0.6 and abs(dy) > scale * 0.75
    if not vertical:
        return "backspace"
    return "clear" if dy > 0 else "speak"


def classify_new(frame: np.ndarray) -> str:
    if fingers_up(frame):
        return "none"
    scale = np.linalg.norm(frame[WRIST] - frame[MIDDLE_MCP])
    if scale < 1e-6:
        return "none"
    dx = frame[THUMB_TIP, 0] - frame[THUMB_MCP, 0]
    dy = frame[THUMB_TIP, 1] - frame[THUMB_MCP, 1]
    vertical = abs(dx) < abs(dy) * 0.6 and abs(dy) > scale * 0.75
    horizontal = abs(dy) < abs(dx) * 0.6 and abs(dx) > scale * 0.75
    if vertical:
        return "clear" if dy > 0 else "speak"
    if horizontal:
        return "backspace"
    return "none"


def report(name: str, X: np.ndarray, y: np.ndarray, label_names: list[str]):
    print(f"\n=== {name} ===")
    old = np.array([classify_old(f) for f in X])
    new = np.array([classify_new(f) for f in X])
    for label in sorted(set(y.tolist())):
        mask = y == label
        n = mask.sum()
        old_bs = float(np.mean(old[mask] == "backspace"))
        new_bs = float(np.mean(new[mask] == "backspace"))
        if old_bs > 0.10 or new_bs > 0.10:
            print(f"  {str(label):12s} n={n:4d}  old backspace={old_bs:.0%}  new backspace={new_bs:.0%}")
    print(f"  overall: old backspace rate={np.mean(old == 'backspace'):.1%}  "
          f"new backspace rate={np.mean(new == 'backspace'):.1%}")


def main():
    letters = np.load(os.path.join(HERE, "data/asl_alphabet_landmarks.npz"), allow_pickle=True)
    report("ASL letters", letters["X"], letters["y"], sorted(set(letters["y"].tolist())))

    digits = np.load(os.path.join(HERE, "data/asl_digit_landmarks.npz"), allow_pickle=True)
    report("ASL digits", digits["X"], digits["y"], sorted(set(digits["y"].tolist())))

    import sys
    sys.path.insert(0, os.path.join(HERE, "src"))
    import dataset_bdsl_static as static_ds

    X_d, y_d, _ = static_ds.load_static_dataset("data/bdsl_digits/CSV Files/Sign Digits - CSV Files")
    report("BdSL digits", X_d, y_d, sorted(set(y_d.tolist())))

    X_l, y_l_idx, _ = static_ds.load_static_dataset("data/bdsl_digits/CSV Files/Sign Letters - CSV Files")
    y_l = np.array([static_ds.LETTER_NAMES[c] for c in y_l_idx])
    report("BdSL letters", X_l, y_l, sorted(set(y_l.tolist())))


if __name__ == "__main__":
    main()
