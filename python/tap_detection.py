"""Shared thumb-index pinch-tap detection, used by analyze_taps.py and analyze_decrement.py."""

import numpy as np
from scipy import signal

import landmarks as LM


def detect_taps(t, lm, min_tap_interval=0.15, prominence=0.02):
    """Returns dict with uniform time grid, smoothed pinch distance, and detected tap indices."""
    ix, iy = lm[:, LM.INDEX_TIP, 0], lm[:, LM.INDEX_TIP, 1]
    tx, ty = lm[:, LM.THUMB_TIP, 0], lm[:, LM.THUMB_TIP, 1]

    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    t_u = np.arange(t[0], t[-1], dt)
    ix_u, iy_u = np.interp(t_u, t, ix), np.interp(t_u, t, iy)
    tx_u, ty_u = np.interp(t_u, t, tx), np.interp(t_u, t, ty)

    dist = np.sqrt((ix_u - tx_u) ** 2 + (iy_u - ty_u) ** 2)
    win = max(3, int(fs * 0.05) | 1)
    dist_smooth = signal.savgol_filter(dist, win, 2) if win < len(dist) else dist

    min_dist_samples = max(1, int(min_tap_interval * fs))
    taps, _ = signal.find_peaks(-dist_smooth, distance=min_dist_samples, prominence=prominence)

    return {
        "t_u": t_u,
        "fs": fs,
        "dist": dist_smooth,
        "tap_idx": taps,
        "tap_times": t_u[taps],
    }
