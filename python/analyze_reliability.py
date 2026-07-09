"""
Reliability analysis across repeated trials at the same nominal target.

Takes several recordings of the same intended condition (e.g. multiple
separate takes of "shake at 2 Hz") and reports mean absolute error, standard
deviation, coefficient of variation, and detection rate -- the kind of
test-retest reliability numbers a single recording can't show. This script
does not fabricate trials: run capture.py multiple times for the same
target first, then point this at all the resulting CSVs.

Usage:
  python analyze_reliability.py --target-hz 2.0 ../data/2hz_trial*.csv
"""

import argparse
import glob

import numpy as np

from load_csv import load_hand_track
import landmarks as LM
from tremor_math import dominant_freq, resample_uniform


def detected_freq(csv_path, hand=None):
    t, xy, hand_used = load_hand_track(csv_path, LM.INDEX_TIP, hand=hand)
    if len(t) < 20:
        return None, hand_used
    _, x_u, y_u, fs = resample_uniform(t, xy[:, 0], xy[:, 1])
    freq, *_ = dominant_freq(x_u, y_u, fs)
    return freq, hand_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+", help="CSV files, or glob patterns")
    parser.add_argument("--target-hz", type=float, required=True)
    parser.add_argument("--hand", choices=["Left", "Right"], default=None)
    args = parser.parse_args()

    paths = []
    for p in args.csv_paths:
        matches = glob.glob(p)
        paths.extend(matches if matches else [p])

    results = []
    for path in paths:
        freq, hand_used = detected_freq(path, hand=args.hand)
        results.append((path, freq))
        status = f"{freq:.2f} Hz" if freq is not None else "no peak"
        print(f"{path:<40} [{hand_used}]  {status}")

    detected = [f for _, f in results if f is not None]
    n_total = len(results)
    n_detected = len(detected)

    print(f"\nTrials: {n_total}   Detected: {n_detected} "
          f"({100 * n_detected / n_total:.0f}%)" if n_total else "No trials.")

    if n_detected >= 2:
        errors = np.array(detected) - args.target_hz
        mae = np.mean(np.abs(errors))
        std = np.std(detected)
        mean = np.mean(detected)
        cv = std / mean if mean != 0 else float("nan")
        print(f"Mean detected frequency: {mean:.3f} Hz")
        print(f"Mean absolute error vs {args.target_hz} Hz target: {mae:.3f} Hz")
        print(f"Standard deviation: {std:.3f} Hz")
        print(f"Coefficient of variation: {cv:.3f}")
    elif n_detected == 1:
        print("Only one trial detected a peak -- need >= 2 for spread statistics.")
    else:
        print("No trials detected a peak -- nothing to aggregate.")


if __name__ == "__main__":
    main()
