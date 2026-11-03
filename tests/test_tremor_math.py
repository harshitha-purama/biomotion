import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tremor_math import dominant_freq  # noqa: E402

FS = 21.0


def _trace(hz, seconds=14, drift_amp=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FS)
    x = 0.01 * np.sin(2 * np.pi * hz * t) if hz else np.zeros_like(t)
    return x + drift_amp * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.001, len(t))


@pytest.mark.parametrize("hz", [2, 3, 5, 8, 9])
def test_recovers_frequency_below_nyquist(hz):
    x = _trace(hz)
    assert dominant_freq(x, 0.5 * x, FS)[0] == pytest.approx(hz, abs=0.1)


def test_arm_motion_below_band_does_not_hide_peak():
    t = np.arange(0, 14, 1 / FS)
    x = 0.05 * np.sin(2 * np.pi * 1.5 * t) + 0.01 * np.sin(2 * np.pi * 5 * t)
    assert dominant_freq(x, x, FS)[0] == pytest.approx(5, abs=0.1)
