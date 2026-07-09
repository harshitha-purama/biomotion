"""
Tremor-frequency analysis from a capture.py CSV log.

Method:
  1. Resample the (noisy-timed) webcam samples onto a uniform time grid.
  2. Detrend the fingertip x/y trajectory to remove voluntary/slow drift.
  3. Run Welch's method (windowed PSD) on x and y separately, sum the power.
  4. Report the dominant frequency in the physiological tremor band (2-15 Hz)
     and its power, plus a plot of the trajectory + spectrum.

If both hands were recorded, run this once per --hand to compare left vs
right (see analyze_asymmetry.py for an automated bilateral comparison).

Validate this against a known reference before trusting it clinically:
  - Hold your hand still (baseline) -> should show near-zero power in-band.
  - Shake your hand in time with a metronome at a known BPM -> the detected
    peak frequency should match metronome_bpm / 60.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import landmarks as LM
from load_csv import load_hand_track
from tremor_math import dominant_freq, resample_uniform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--hand", choices=["Left", "Right"], default=None,
                         help="Which hand to analyze (default: first one found)")
    parser.add_argument("--band-low", type=float, default=2.0)
    parser.add_argument("--band-high", type=float, default=15.0)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    t, xy, hand_used = load_hand_track(args.csv_path, LM.INDEX_TIP, hand=args.hand)
    x, y = xy[:, 0], xy[:, 1]
    if len(t) < 20:
        raise SystemExit("Not enough samples for this hand (need >= ~1s of data).")

    t_u, x_u, y_u, fs = resample_uniform(t, x, y)
    peak_freq, peak_power, f, power, x_d, y_d = dominant_freq(
        x_u, y_u, fs, band=(args.band_low, args.band_high)
    )

    print(f"Hand: {hand_used}")
    print(f"Effective sample rate: {fs:.1f} Hz")
    print(f"Duration: {t_u[-1] - t_u[0]:.2f} s")
    if peak_freq is None:
        print("No frequency content found in the requested band.")
    else:
        print(f"Dominant frequency in [{args.band_low}, {args.band_high}] Hz band: "
              f"{peak_freq:.2f} Hz")
        print(f"Peak PSD power: {peak_power:.3e}")
        rms = float(np.sqrt(np.mean(x_d**2 + y_d**2)))
        print(f"RMS displacement (normalized coords): {rms:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(t_u, x_d, label="x (detrended)")
    axes[0].plot(t_u, y_d, label="y (detrended)", alpha=0.7)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("normalized displacement")
    axes[0].set_title(f"Fingertip trajectory (detrended) — {hand_used} hand")
    axes[0].legend()

    axes[1].plot(f, power)
    axes[1].axvspan(args.band_low, args.band_high, color="orange", alpha=0.15,
                     label="tremor band")
    if peak_freq is not None:
        axes[1].axvline(peak_freq, color="red", linestyle="--",
                         label=f"peak {peak_freq:.2f} Hz")
    axes[1].set_xlim(0, 20)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power")
    axes[1].set_title("Power spectral density (Welch)")
    axes[1].legend()

    fig.tight_layout()
    out_path = args.save_plot or (
        Path(args.csv_path).with_suffix("").as_posix() + f"_tremor_{hand_used}.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
