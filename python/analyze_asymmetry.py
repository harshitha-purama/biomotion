"""
Bilateral (left vs. right hand) asymmetry analysis — the other "novel" angle.

Requires a recording where *both* hands were visible (capture.py defaults to
--num-hands 2). Real-world relevance: Parkinson's tremor and bradykinesia
are frequently asymmetric, especially early on — one hand is noticeably
worse than the other. A single-hand demo can't show that at all; comparing
the same metrics computed independently per hand can.

Reports, for tremor and tap rate:
  - each hand's value
  - an asymmetry index = |L - R| / (L + R), 0 = perfectly symmetric,
    1 = fully one-sided (this is the standard normalized-asymmetry
    formula used in gait/motor asymmetry literature)
"""

import argparse

import numpy as np
from scipy import signal

from load_csv import load_full_track, load_hand_track, available_hands
from tap_detection import detect_taps
from tremor_math import dominant_freq, resample_uniform
import landmarks as LM


def asymmetry_index(a, b):
    if a is None or b is None or (a + b) == 0:
        return None
    return abs(a - b) / (a + b)


def tremor_metrics(csv_path, hand):
    t, xy, _ = load_hand_track(csv_path, LM.INDEX_TIP, hand=hand)
    if len(t) < 20:
        return None, None
    t_u, x_u, y_u, fs = resample_uniform(t, xy[:, 0], xy[:, 1])
    freq, power, *_ = dominant_freq(x_u, y_u, fs)
    return freq, power


def amplitude_metric(csv_path, hand):
    """RMS of detrended fingertip displacement -- how much the hand moved overall,
    independent of whether that movement was rhythmic/in-band or not. Useful when
    the movement is too slow for the tremor-band frequency search to catch (see
    README: this is what actually shows asymmetry in a loose "shake one hand
    more" test, vs. the frequency metric which needs a tight, fast oscillation).
    """
    t, xy, _ = load_hand_track(csv_path, LM.INDEX_TIP, hand=hand)
    if len(t) < 20:
        return None
    _, x_u, y_u, _ = resample_uniform(t, xy[:, 0], xy[:, 1])
    x_d, y_d = signal.detrend(x_u), signal.detrend(y_u)
    return float(np.sqrt(np.mean(x_d**2 + y_d**2)))


def tap_rate_metric(csv_path, hand):
    t, lm, _ = load_full_track(csv_path, hand=hand)
    if len(t) < 20:
        return None
    d = detect_taps(t, lm)
    duration = d["t_u"][-1] - d["t_u"][0]
    return len(d["tap_idx"]) / duration if duration > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

    hands = available_hands(args.csv_path)
    if not {"Left", "Right"}.issubset(hands):
        raise SystemExit(
            f"Need both hands recorded for asymmetry analysis; found: {sorted(hands)}. "
            "Re-record with both hands visible to the camera (capture.py --num-hands 2)."
        )

    print(f"{'metric':<22} {'Left':>10} {'Right':>10} {'asymmetry idx':>15}")

    l_freq, l_pow = tremor_metrics(args.csv_path, "Left")
    r_freq, r_pow = tremor_metrics(args.csv_path, "Right")
    freq_asym = asymmetry_index(l_freq, r_freq)
    pow_asym = asymmetry_index(l_pow, r_pow)
    print(f"{'tremor freq (Hz)':<22} "
          f"{l_freq if l_freq is not None else float('nan'):>10.2f} "
          f"{r_freq if r_freq is not None else float('nan'):>10.2f} "
          f"{freq_asym if freq_asym is not None else float('nan'):>15.3f}")
    print(f"{'tremor power':<22} "
          f"{l_pow if l_pow is not None else float('nan'):>10.2e} "
          f"{r_pow if r_pow is not None else float('nan'):>10.2e} "
          f"{pow_asym if pow_asym is not None else float('nan'):>15.3f}")

    l_amp = amplitude_metric(args.csv_path, "Left")
    r_amp = amplitude_metric(args.csv_path, "Right")
    amp_asym = asymmetry_index(l_amp, r_amp)
    print(f"{'movement amplitude':<22} "
          f"{l_amp if l_amp is not None else float('nan'):>10.4f} "
          f"{r_amp if r_amp is not None else float('nan'):>10.4f} "
          f"{amp_asym if amp_asym is not None else float('nan'):>15.3f}")

    l_tap = tap_rate_metric(args.csv_path, "Left")
    r_tap = tap_rate_metric(args.csv_path, "Right")
    tap_asym = asymmetry_index(l_tap, r_tap)
    print(f"{'tap rate (taps/s)':<22} "
          f"{l_tap if l_tap is not None else float('nan'):>10.2f} "
          f"{r_tap if r_tap is not None else float('nan'):>10.2f} "
          f"{tap_asym if tap_asym is not None else float('nan'):>15.3f}")

    print("\nasymmetry index: 0 = symmetric, 1 = fully one-sided.")
    print("Tremor freq/power only means something if the movement was tight, fast "
          "oscillation (try a metronome-paced shake); for slower/looser movement, "
          "'movement amplitude' is the metric that actually reflects asymmetry. "
          "Treat all of this purely as a methods demo, not a diagnostic signal.")


if __name__ == "__main__":
    main()
