"""Shared CSV loading helpers for capture.py output (see landmarks.py for schema)."""

import csv
from collections import defaultdict

import numpy as np


def load_full_track(csv_path, hand=None):
    """Returns (t, landmarks, hand_used).

    landmarks has shape (n_frames, 21, 3) for the chosen hand.
    If `hand` is None, uses whichever hand label has the most recorded rows.
    """
    rows_by_hand = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_hand[row["hand"]].append(row)

    if not rows_by_hand:
        raise SystemExit(f"No data rows found in {csv_path}")

    if hand is not None:
        if hand not in rows_by_hand:
            raise SystemExit(
                f"Hand '{hand}' not found in {csv_path}. "
                f"Available: {list(rows_by_hand.keys())}"
            )
        chosen = hand
    else:
        chosen = max(rows_by_hand, key=lambda k: len(rows_by_hand[k]))

    rows = rows_by_hand[chosen]
    t = np.array([float(r["t_sec"]) for r in rows])
    landmarks = np.zeros((len(rows), 21, 3))
    for i, r in enumerate(rows):
        for j in range(21):
            landmarks[i, j, 0] = float(r[f"lm{j}_x"])
            landmarks[i, j, 1] = float(r[f"lm{j}_y"])
            landmarks[i, j, 2] = float(r[f"lm{j}_z"])
    return t, landmarks, chosen


def load_hand_track(csv_path, landmark_idx, hand=None):
    """Returns (t, xy[n,2], hand_used) for a single landmark index."""
    t, landmarks, chosen = load_full_track(csv_path, hand=hand)
    return t, landmarks[:, landmark_idx, :2], chosen


def available_hands(csv_path):
    hands = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hands.add(row["hand"])
    return hands
