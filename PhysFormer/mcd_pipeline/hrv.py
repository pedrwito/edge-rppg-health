"""HR and HRV metrics from a 1-D pulsatile signal (PPG or rPPG) or ECG.

All functions return floats; bad/short signals → np.nan.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch


def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float,
              order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq * 0.99
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def hr_welch(sig: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.5) -> float:
    """FFT-domain HR estimate (bpm). Robust dominant-peak."""
    if len(sig) < int(fs * 4):
        return np.nan
    sig = sig - np.mean(sig)
    sig = _bandpass(sig, fs, lo, hi)
    nperseg = min(len(sig), int(fs * 8))
    f, p = welch(sig, fs=fs, nperseg=nperseg)
    band = (f >= lo) & (f <= hi)
    if not band.any() or p[band].max() <= 0:
        return np.nan
    return float(60.0 * f[band][np.argmax(p[band])])


def find_ppg_peaks(sig: np.ndarray, fs: float, lo: float = 0.7,
                   hi: float = 3.5) -> np.ndarray:
    """Detect systolic peaks in a (rPPG/PPG) signal. Returns peak sample indices."""
    if len(sig) < int(fs * 4):
        return np.array([], dtype=int)
    sig = _bandpass(sig - np.mean(sig), fs, lo, hi)
    # min distance between peaks: ~60/220 sec at HR=220 bpm (upper bound)
    min_dist = int(0.27 * fs)
    height = 0.3 * np.std(sig)
    peaks, _ = find_peaks(sig, distance=min_dist, height=height)
    return peaks


def find_ecg_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins-flavored R-peak detection.

    Threshold uses a robust percentile (90th of the moving-window-integrated
    signal) instead of mean+std — std is blown up by occasional clipping
    spikes in MCD ECG, which then masks the real R-peaks.
    """
    if len(ecg) < int(fs * 4):
        return np.array([], dtype=int)
    sig = _bandpass(ecg - np.mean(ecg), fs, 5.0, 15.0)
    sq = np.diff(sig) ** 2
    win = max(int(0.15 * fs), 1)
    integrated = np.convolve(sq, np.ones(win) / win, mode="same")
    # Robust threshold: 25% of the 90th percentile (clipping-safe).
    height = 0.25 * np.percentile(integrated, 90)
    peaks, _ = find_peaks(integrated, distance=int(0.30 * fs), height=height)
    return peaks


def hrv_metrics(peak_idx: np.ndarray, fs: float) -> dict:
    """Time- and frequency-domain HRV from a peak-index array.

    Returns NaN-filled dict if too few beats.
    """
    out = {"hr_peaks": np.nan, "sdnn_ms": np.nan, "rmssd_ms": np.nan,
           "pnn50_pct": np.nan, "lf": np.nan, "hf": np.nan, "lfhf": np.nan,
           "n_beats": int(len(peak_idx))}
    if len(peak_idx) < 4:
        return out
    rr_s = np.diff(peak_idx) / fs
    rr_ms = rr_s * 1000.0
    # Reject obviously bad RRs (outside 0.3-2.0 s, i.e. 30-200 bpm)
    rr_ms = rr_ms[(rr_ms >= 300.0) & (rr_ms <= 2000.0)]
    if len(rr_ms) < 3:
        return out
    out["hr_peaks"] = float(60_000.0 / np.mean(rr_ms))
    out["sdnn_ms"] = float(np.std(rr_ms, ddof=1))
    diffs = np.diff(rr_ms)
    out["rmssd_ms"] = float(np.sqrt(np.mean(diffs ** 2)))
    out["pnn50_pct"] = float(100.0 * np.mean(np.abs(diffs) > 50.0))
    # Frequency domain: interpolate RR series at 4 Hz, Welch PSD
    if len(rr_ms) >= 8:
        t_rr = np.cumsum(rr_s)  # cumulative time at each beat
        t_rr = t_rr - t_rr[0]
        if t_rr[-1] > 0:
            t_uniform = np.arange(0, t_rr[-1], 1.0 / 4.0)
            if len(t_uniform) >= 16:
                rr_uniform = np.interp(t_uniform, t_rr[:len(rr_ms)], rr_ms - np.mean(rr_ms))
                f, p = welch(rr_uniform, fs=4.0, nperseg=min(len(rr_uniform), 256))
                lf = float(np.trapezoid(p[(f >= 0.04) & (f < 0.15)],
                                        f[(f >= 0.04) & (f < 0.15)]))
                hf = float(np.trapezoid(p[(f >= 0.15) & (f < 0.40)],
                                        f[(f >= 0.15) & (f < 0.40)]))
                out["lf"] = lf
                out["hf"] = hf
                out["lfhf"] = lf / hf if hf > 0 else np.nan
    return out


def all_metrics(sig: np.ndarray, fs: float, kind: str = "ppg") -> dict:
    """One-shot: HR (welch) + peak-derived HRV. kind in {'ppg','ecg'}."""
    out = {"hr_welch": hr_welch(sig, fs)}
    peaks = find_ecg_r_peaks(sig, fs) if kind == "ecg" else find_ppg_peaks(sig, fs)
    out.update(hrv_metrics(peaks, fs))
    return out
