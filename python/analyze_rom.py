"""
Per-finger joint-angle / range-of-motion (ROM) analysis.

For each finger, computes the PIP joint angle (angle at the middle joint,
between the MCP->PIP and PIP->DIP vectors) every frame from the 2D landmark
positions, then reports the range of motion (max - min angle) achieved
during the recording. This is a webcam replacement for a physical
goniometer, standard in rehab ROM tracking.

Validate before trusting it: hold a printed protractor in frame, bend your
finger to a few marked angles, and compare against the script's output for
those frames.

Record with capture.py while flexing/extending your fingers through their
full comfortable range.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import landmarks as LM
from load_csv import load_full_track


def joint_angle_deg(a, b, c):
    """Angle at point b formed by rays b->a and b->c, in degrees. a,b,c: (N,2) arrays."""
    v1 = a - b
    v2 = c - b
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot / (n1 * n2 + 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--hand", choices=["Left", "Right"], default=None)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    t, lm, hand_used = load_full_track(args.csv_path, hand=args.hand)
    if len(t) < 5:
        raise SystemExit("Not enough samples in this recording.")

    fig, ax = plt.subplots(figsize=(9, 5))
    print(f"Hand: {hand_used}")
    print(f"{'finger':<8} {'min (deg)':>10} {'max (deg)':>10} {'ROM (deg)':>10}")

    for name, (mcp, pip, dip, tip) in LM.FINGERS.items():
        a = lm[:, mcp, :2]
        b = lm[:, pip, :2]
        c = lm[:, dip, :2]
        angles = joint_angle_deg(a, b, c)
        rom = angles.max() - angles.min()
        print(f"{name:<8} {angles.min():>10.1f} {angles.max():>10.1f} {rom:>10.1f}")
        ax.plot(t, angles, label=f"{name} (ROM {rom:.0f}°)")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("PIP joint angle (degrees)")
    ax.set_title(f"Per-finger joint angle over time — {hand_used} hand")
    ax.legend()

    fig.tight_layout()
    out_path = args.save_plot or (
        Path(args.csv_path).with_suffix("").as_posix() + f"_rom_{hand_used}.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
