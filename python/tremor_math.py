"""Shared tremor-frequency math, used by analyze_tremor.py and analyze_asymmetry.py."""

import numpy as np
from scipy import signal


def resample_uniform(t, x, y):
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    t_uniform = np.arange(t[0], t[-1], dt)
    x_u = np.interp(t_uniform, t, x)
    y_u = np.interp(t_uniform, t, y)
    return t_uniform, x_u, y_u, fs


def dominant_freq(x_u, y_u, fs, band=(2.0, 15.0), min_prominence_ratio=0.25,
                  highpass_hz=1.0, window_sec=6.0):
    """Find a genuine spectral peak inside `band`.

    Two rules keep this honest:
    - A long Welch window (`window_sec`). A 2 s window gives 0.5 Hz resolution,
      too coarse to tell 2.0 Hz from 2.4 Hz.
    - Prominence is relative to the strongest power *inside the band*. Measuring
      it against the whole spectrum lets sub-band arm motion (1-2 Hz) hide a
      real in-band peak. A still hand has no such peak and reports "no peak".

    Peaks are searched on the whole spectrum and then filtered to the band, so
    a peak sitting exactly on the band edge is not lost.
    """
    x_d = signal.detrend(x_u)
    y_d = signal.detrend(y_u)

    if highpass_hz and fs > 2 * highpass_hz:
        sos = signal.butter(4, highpass_hz, btype="highpass", fs=fs, output="sos")
        x_d = signal.sosfiltfilt(sos, x_d)
        y_d = signal.sosfiltfilt(sos, y_d)

    nperseg = min(len(x_d), int(fs * window_sec))
    nfft = 4 * nperseg
    f, pxx = signal.welch(x_d, fs=fs, nperseg=nperseg, nfft=nfft)
    _, pyy = signal.welch(y_d, fs=fs, nperseg=nperseg, nfft=nfft)
    power = pxx + pyy

    upper = min(band[1], 0.9 * fs / 2)  # stay clear of Nyquist (fs / 2)
    in_band = (f >= band[0]) & (f <= upper)
    if power.max() <= 0 or not in_band.any():
        return None, None, f, power, x_d, y_d

    band_max = power[in_band].max()
    peaks, _ = signal.find_peaks(power, prominence=band_max * min_prominence_ratio)
    band_peaks = peaks[(f[peaks] >= band[0]) & (f[peaks] <= upper)]
    if len(band_peaks) == 0:
        return None, None, f, power, x_d, y_d

    best = band_peaks[np.argmax(power[band_peaks])]
    return f[best], power[best], f, power, x_d, y_d
