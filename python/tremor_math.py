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


def dominant_freq(x_u, y_u, fs, band=(2.0, 15.0), min_prominence_ratio=0.15,
                   highpass_hz=1.0):
    """Find a genuine spectral peak inside `band`, not just the highest point in
    that window. A hand that isn't oscillating produces a power spectrum that
    monotonically decays from 0 Hz -- naively taking argmax() over a band window
    would then just report the band's left edge every time, even for a
    perfectly still hand. scipy.signal.find_peaks requires an actual local
    maximum (lower power on both sides) with prominence above a fraction of the
    spectrum's peak power, so flat/monotonic spectra correctly report "no peak."

    Slow arm/wrist drift during a "shake" also tends to dominate the spectrum
    below ~1 Hz and can swamp a real, smaller tremor-band peak -- a Butterworth
    high-pass filter removes that before Welch's method, so the oscillatory
    component (not overall hand travel) is what gets compared against the band.
    """
    x_d = signal.detrend(x_u)
    y_d = signal.detrend(y_u)

    if highpass_hz and fs > 2 * highpass_hz:
        sos = signal.butter(4, highpass_hz, btype="highpass", fs=fs, output="sos")
        x_d = signal.sosfiltfilt(sos, x_d)
        y_d = signal.sosfiltfilt(sos, y_d)

    nperseg = min(len(x_d), int(fs * 2))
    f, pxx = signal.welch(x_d, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(y_d, fs=fs, nperseg=nperseg)
    power = pxx + pyy

    if power.max() <= 0:
        return None, None, f, power, x_d, y_d

    peaks, _ = signal.find_peaks(power, prominence=power.max() * min_prominence_ratio)
    band_peaks = peaks[(f[peaks] >= band[0]) & (f[peaks] <= band[1])]
    if len(band_peaks) == 0:
        return None, None, f, power, x_d, y_d

    best = band_peaks[np.argmax(power[band_peaks])]
    return f[best], power[best], f, power, x_d, y_d
