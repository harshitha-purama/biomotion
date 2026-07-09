"""
Finger-tap rate & rhythm analysis from a capture.py CSV log.

Digitizes the clinical finger-tapping test (used in Parkinson's / motor
exams): tracks thumb-index pinch distance over time, detects each tap as a
local minimum in that distance, and reports:
  - tap rate (taps/sec)
  - inter-tap interval mean/std
  - rhythm consistency = coefficient of variation (CV) of intervals
    (lower CV = steadier rhythm; clinically, rhythm decrement/irregularity
    is itself a signal, not just raw speed)

See analyze_decrement.py for the amplitude/speed *decrement over the
sequence* metric, which is what MDS-UPDRS finger-tapping scoring actually
weighs most heavily (a clinically normal person keeps amplitude/speed
roughly constant across ~10 taps; bradykinesia shows progressive decay).

Record by tapping your thumb and index fingertip together repeatedly in
front of the webcam (this is literally the clinical test protocol).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_csv import load_full_track
from tap_detection import detect_taps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--hand", choices=["Left", "Right"], default=None)
    parser.add_argument("--min-tap-interval", type=float, default=0.15,
                         help="Minimum seconds between taps (debounce)")
    parser.add_argument("--prominence", type=float, default=0.02,
                         help="Min dip prominence in normalized distance to count as a tap")
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    t, lm, hand_used = load_full_track(args.csv_path, hand=args.hand)
    if len(t) < 20:
        raise SystemExit("Not enough samples in this recording (need >= ~1s of data).")

    d = detect_taps(t, lm, min_tap_interval=args.min_tap_interval, prominence=args.prominence)
    t_u, dist_smooth, taps, tap_times = d["t_u"], d["dist"], d["tap_idx"], d["tap_times"]

    duration = t_u[-1] - t_u[0]
    n_taps = len(tap_times)
    tap_rate = n_taps / duration if duration > 0 else 0.0

    print(f"Hand: {hand_used}")
    print(f"Duration: {duration:.2f} s")
    print(f"Taps detected: {n_taps}")
    print(f"Tap rate: {tap_rate:.2f} taps/sec")

    if n_taps >= 3:
        intervals = np.diff(tap_times)
        mean_iti = intervals.mean()
        std_iti = intervals.std()
        cv = std_iti / mean_iti if mean_iti > 0 else float("nan")
        print(f"Mean inter-tap interval: {mean_iti*1000:.1f} ms")
        print(f"Std inter-tap interval: {std_iti*1000:.1f} ms")
        print(f"Rhythm consistency (CV, lower = steadier): {cv:.3f}")
    else:
        intervals = np.array([])
        print("Not enough taps to compute rhythm consistency (need >= 3).")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(t_u, dist_smooth, label="thumb-index distance")
    axes[0].plot(tap_times, dist_smooth[taps], "rx", label="detected tap")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("normalized distance")
    axes[0].set_title(f"Pinch distance — {hand_used} hand, {n_taps} taps @ {tap_rate:.2f}/s")
    axes[0].legend()

    if len(intervals) > 0:
        axes[1].plot(tap_times[1:], intervals * 1000, "o-")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("inter-tap interval (ms)")
        axes[1].set_title("Rhythm over time (flat line = perfectly steady)")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    out_path = args.save_plot or (
        Path(args.csv_path).with_suffix("").as_posix() + f"_taps_{hand_used}.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
