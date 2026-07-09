"""
Amplitude & speed decrement analysis — the "novel" clinically-grounded metric.

Most hobby finger-tap demos stop at "taps per second." Real MDS-UPDRS
finger-tapping scoring cares less about raw speed and much more about
*decrement*: does opening amplitude and tapping speed progressively shrink
across a ~10-tap sequence (a hallmark of bradykinesia), and are there
hesitations/halts mid-sequence? This script digitizes exactly that.

For each tap cycle:
  - amplitude = max thumb-index distance reached *before* that tap closed
    (i.e. how wide the fingers opened on that cycle)
  - speed = 1 / inter-tap interval

Reports a linear-fit decrement slope for both (normalized to % change over
the sequence, relative to the first few taps) and flags any hesitation
(an inter-tap interval much longer than the running median — a "halt").

Record the same way as analyze_taps.py: tap thumb+index together repeatedly,
ideally >= 10 taps, as fast/wide as you comfortably can, for the whole clip.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_csv import load_full_track
from tap_detection import detect_taps


def cycle_amplitudes(t_u, dist, tap_idx):
    """Max opening distance in the window preceding each tap."""
    amps = []
    prev = 0
    for idx in tap_idx:
        window = dist[prev:idx + 1]
        amps.append(window.max() if len(window) else np.nan)
        prev = idx
    return np.array(amps)


def linear_decrement_pct(values):
    """Fit a line to values-vs-index; report % change from fitted start to end."""
    if len(values) < 3:
        return None, None
    idx = np.arange(len(values))
    slope, intercept = np.polyfit(idx, values, 1)
    start = intercept
    end = slope * (len(values) - 1) + intercept
    pct = ((end - start) / start * 100) if start != 0 else None
    return slope, pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--hand", choices=["Left", "Right"], default=None)
    parser.add_argument("--min-tap-interval", type=float, default=0.15)
    parser.add_argument("--prominence", type=float, default=0.02)
    parser.add_argument("--halt-factor", type=float, default=1.8,
                         help="Flag an interval as a hesitation/halt if it exceeds "
                              "this multiple of the median interval")
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    t, lm, hand_used = load_full_track(args.csv_path, hand=args.hand)
    if len(t) < 20:
        raise SystemExit(f"Only {len(t)} samples for hand '{hand_used}' "
                          f"(need >= ~1s of data). Wrong --hand, or a handedness "
                          f"misdetection blip? Try omitting --hand to auto-pick "
                          f"the hand with the most rows.")
    d = detect_taps(t, lm, min_tap_interval=args.min_tap_interval, prominence=args.prominence)
    t_u, dist, tap_idx, tap_times = d["t_u"], d["dist"], d["tap_idx"], d["tap_times"]

    if len(tap_idx) < 5:
        raise SystemExit(f"Only {len(tap_idx)} taps detected; need >= 5 for a decrement trend.")

    amps = cycle_amplitudes(t_u, dist, tap_idx)
    intervals = np.diff(tap_times)
    speeds = 1.0 / intervals

    amp_slope, amp_pct = linear_decrement_pct(amps)
    speed_slope, speed_pct = linear_decrement_pct(speeds)

    median_iti = np.median(intervals)
    halts = np.where(intervals > args.halt_factor * median_iti)[0]

    print(f"Hand: {hand_used}")
    print(f"Taps detected: {len(tap_idx)}")
    print(f"Amplitude trend: {amp_pct:+.1f}% over the sequence "
          f"(negative = shrinking = decrement)" if amp_pct is not None else "Amplitude trend: n/a")
    print(f"Speed trend: {speed_pct:+.1f}% over the sequence "
          f"(negative = slowing)" if speed_pct is not None else "Speed trend: n/a")
    print(f"Hesitations/halts (interval > {args.halt_factor}x median): {len(halts)}")
    if len(halts):
        halt_times = tap_times[1:][halts]
        print(f"  at t = {', '.join(f'{ht:.2f}s' for ht in halt_times)}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(range(1, len(amps) + 1), amps, "o-", color="tab:orange")
    idx = np.arange(len(amps))
    if amp_slope is not None:
        fit = np.polyval(np.polyfit(idx, amps, 1), idx)
        axes[0].plot(idx + 1, fit, "--", color="gray", alpha=0.7, label="linear trend")
    axes[0].set_xlabel("tap #")
    axes[0].set_ylabel("opening amplitude (normalized)")
    axes[0].set_title(f"Amplitude decrement — {hand_used} hand")
    axes[0].legend()

    axes[1].plot(range(2, len(speeds) + 2), speeds, "o-", color="tab:blue")
    for h in halts:
        axes[1].axvspan(h + 1.5, h + 2.5, color="red", alpha=0.2)
    axes[1].set_xlabel("tap #")
    axes[1].set_ylabel("speed (taps/sec, instantaneous)")
    axes[1].set_title("Speed per cycle (red = flagged hesitation)")

    fig.tight_layout()
    out_path = args.save_plot or (
        Path(args.csv_path).with_suffix("").as_posix() + f"_decrement_{hand_used}.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
