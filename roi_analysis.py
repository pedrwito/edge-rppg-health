import os
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from IppgSignalObtainer import IppgSignalObtainer
from Tools.ParametersCalculator import ParametersCalculator
from Tools.signalprocesser import SignalProcessor
from process_ubfc_dataset import load_ubfc_ground_truth


# ------------------------------
# Helpers (ported/adapted from notebook)
# ------------------------------

def _z(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    x = np.asarray(x)
    if not normalize or x.size == 0:
        return x
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / (s + 1e-12)


def _compute_peaks(signal_1d: np.ndarray, fs: float, distance_s: float = 0.3) -> np.ndarray:
    calc = ParametersCalculator()
    return calc.GetPeaks(np.asarray(signal_1d), fs=fs, k_h_max_R=1, distance=distance_s)


def _parabolic_refine(y: np.ndarray, i: int) -> Tuple[float, float]:
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    denom = (y[i - 1] - 2.0 * y[i] + y[i + 1]) + 1e-12
    xv = 0.5 * (y[i - 1] - y[i + 1]) / denom
    pv = y[i] - 0.25 * (y[i - 1] - y[i + 1]) * xv
    return i + xv, pv


def _xcorr_full(x: np.ndarray, y: np.ndarray, fs: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    n = min(len(x), len(y))
    if n == 0:
        return None, None
    x = _z(np.asarray(x[:n]), normalize=True)
    y = _z(np.asarray(y[:n]), normalize=True)
    corr = np.correlate(x, y, mode='full') / n
    lags = np.arange(-n + 1, n) / fs
    return lags, corr


def _xcorr_peak_constrained(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: Optional[float] = None) -> Tuple[float, float]:
    n = min(len(x), len(y))
    if n < 3:
        return float("nan"), float("nan")
    x = (np.asarray(x[:n]) - np.mean(x[:n])) / (np.std(x[:n]) + 1e-12)
    y = (np.asarray(y[:n]) - np.mean(y[:n])) / (np.std(y[:n]) + 1e-12)
    corr = np.correlate(x, y, mode='full') / n
    lags = np.arange(-n + 1, n)
    if max_lag_s is not None:
        k = int(max_lag_s * fs)
        keep = (lags >= -k) & (lags <= k)
        lags = lags[keep]
        corr = corr[keep]
    i = int(np.argmax(corr))
    i_ref, c_ref = _parabolic_refine(corr, i)
    lag_samples = lags[0] + i_ref
    return 1000.0 * lag_samples / fs, float(c_ref)


def _shift_by_ms(x: np.ndarray, fs: float, shift_ms: float) -> np.ndarray:
    t = np.arange(len(x)) / fs
    t_new = t - (shift_ms / 1000.0)
    return np.interp(t, t_new, x, left=x[0], right=x[-1])


def _bandpass_hr(series: np.ndarray, fs: float, hr_bpm: float, half_width_hz: float = 0.5) -> np.ndarray:
    if series is None or len(series) == 0 or hr_bpm is None or not np.isfinite(hr_bpm):
        return np.asarray(series)
    f0 = float(hr_bpm) / 60.0
    low = max(0.1, f0 - half_width_hz)
    high = min(fs / 2 - 0.1, f0 + half_width_hz)
    if low >= high:
        return np.asarray(series)
    return SignalProcessor.bandpass(np.asarray(series), fs, order=3, lowcut=low, highcut=high)

def _global_xcorr_subsample(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: Optional[float] = None) -> Tuple[float, float]:
    n = min(len(x), len(y))
    if n < 3:
        return float("nan"), float("nan")
    x = (np.asarray(x[:n]) - np.mean(x[:n])) / (np.std(x[:n]) + 1e-12)
    y = (np.asarray(y[:n]) - np.mean(y[:n])) / (np.std(y[:n]) + 1e-12)
    corr = np.correlate(x, y, mode='full') / n
    lags = np.arange(-n + 1, n)
    if max_lag_s is not None:
        k = int(max_lag_s * fs)
        keep = (lags >= -k) & (lags <= k)
        corr = corr[keep]
        lags = lags[keep]
    i = int(np.argmax(corr))
    if 0 < i < len(corr) - 1:
        denom = (corr[i-1] - 2*corr[i] + corr[i+1]) + 1e-12
        xv = 0.5 * (corr[i-1] - corr[i+1]) / denom
        peak_val = corr[i] - 0.25 * (corr[i-1] - corr[i+1]) * xv
        i_ref = i + xv
    else:
        i_ref = float(i)
        peak_val = float(corr[i])
    lag_samples = lags[0] + i_ref
    return 1000.0 * lag_samples / fs, float(peak_val)

def _extract_pos_signals(video_path: str, fs: float, window_length: int, start_time: int,
                         forehead: bool = True, cheeks: bool = True, under_nose: bool = False,
                         chin: bool = True) -> Dict[str, np.ndarray]:
    rois_rgb = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
        video_path, fs, window_length=window_length, start_time=start_time,
        forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin, full_face=False, play_video=False
    )
    pos: Dict[str, np.ndarray] = {}
    for roi, ch in rois_rgb.items():
        r, g, b = ch.get('red', []), ch.get('green', []), ch.get('blue', [])
        if len(r) and len(g) and len(b):
            pos_sig = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                r, g, b, fs, normalize=False, derivative=False, bandpass=True, detrend=True, method='pos'
            )
            pos[roi] = np.asarray(pos_sig)
    return pos


# ------------------------------
# 1) Plot ALL ROIs together with peaks and XCorr (single video via precomputed signals)
# ------------------------------

def plot_all_rois_signals_and_xcorr(
    pos_signals: Dict[str, np.ndarray],
    pos_signals_narrow: Dict[str, np.ndarray],
    fs: float,
    show_filtered: bool = True,
    normalize: bool = True,
    plot_individual: bool = False,
) -> None:
    rois = list(pos_signals.keys())

    if plot_individual:
        for roi in rois:
            x = np.asarray(pos_signals.get(roi, []))
            if x.size == 0:
                continue
            t = np.arange(x.size) / fs
            peaks = _compute_peaks(x, fs)
            if peaks is None or len(peaks) == 0:
                continue
            t_peaks = peaks / fs
            plt.figure(figsize=(14, 5))
            plt.scatter(t_peaks, _z(x, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
            plt.plot(t, _z(x, normalize=normalize), label=roi, linewidth=0.9)
            plt.title(f'{roi} - POS (unfiltered)')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
            plt.grid(True)
            plt.legend()
            plt.show()

    # Unfiltered, all ROIs
    plt.figure(figsize=(14, 5))
    for roi in rois:
        x = np.asarray(pos_signals.get(roi, []))
        if x.size == 0:
            continue
        t = np.arange(x.size) / fs
        plt.plot(t, _z(x, normalize=normalize), label=roi, linewidth=0.9)
    for roi in rois:
        x = np.asarray(pos_signals.get(roi, []))
        if x.size == 0:
            continue
        peaks = _compute_peaks(x, fs)
        if peaks is None or len(peaks) == 0:
            continue
        t_peaks = peaks / fs
        plt.scatter(t_peaks, _z(x, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
    plt.title('All ROIs - POS (unfiltered) with peak markers')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
    plt.grid(True)
    plt.legend(ncol=min(len(rois), 4))

    # Filtered, all ROIs
    if show_filtered:
        plt.figure(figsize=(14, 5))
        for roi in rois:
            xf = np.asarray(pos_signals_narrow.get(roi, []))
            if xf.size == 0:
                continue
            t = np.arange(xf.size) / fs
            plt.plot(t, _z(xf, normalize=normalize), label=roi, linewidth=0.9)
        for roi in rois:
            xf = np.asarray(pos_signals_narrow.get(roi, []))
            if xf.size == 0:
                continue
            peaks = _compute_peaks(xf, fs)
            if peaks is None or len(peaks) == 0:
                continue
            t_peaks = peaks / fs
            plt.scatter(t_peaks, _z(xf, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
        plt.title('All ROIs - POS (filtered HR±0.5 Hz) with peak markers')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
        plt.grid(True)
        plt.legend(ncol=min(len(rois), 4))

    # Cross-correlation (unfiltered)
    pairs = list(combinations(rois, 2))
    if len(pairs) > 0:
        ncols = 2 if len(pairs) > 1 else 1
        nrows = int(np.ceil(len(pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
        fig.suptitle('Cross-correlation (unfiltered) - window ±100 ms (sub-sample peak)', y=1.02)
        for idx, (a, b) in enumerate(pairs):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            xa = np.asarray(pos_signals.get(a, []))
            xb = np.asarray(pos_signals.get(b, []))
            lags, corr = _xcorr_full(xa, xb, fs)
            if lags is None:
                ax.set_visible(False)
                continue
            # Restrict to ±100 ms window (plot in milliseconds)
            lags_ms = lags * 1000.0
            window_mask = (lags_ms >= -100.0) & (lags_ms <= 100.0)
            if not np.any(window_mask):
                ax.set_visible(False)
                continue
            lags_w_ms = lags_ms[window_mask]
            corr_w = corr[window_mask]
            i = int(np.argmax(corr_w))
            i_ref, peak_val = _parabolic_refine(corr_w, i)
            lag_step_ms = (lags_w_ms[1] - lags_w_ms[0]) if len(lags_w_ms) > 1 else (1000.0 / fs)
            peak_lag_ms = lags_w_ms[0] + i_ref * lag_step_ms
            # Plot discrete correlation samples as points (no connecting line)
            ax.plot(lags_w_ms, corr_w, marker='o', linestyle='None', color='tab:blue', markersize=3, alpha=0.9)
            # Highlight the three samples used for parabolic interpolation in a different color
            if 0 < i < len(corr_w) - 1:
                ax.plot(lags_w_ms[i-1:i+2], corr_w[i-1:i+2], marker='o', linestyle='None', color='tab:orange', markersize=4, alpha=0.95)
            # Plot the interpolated apex value itself
            ax.plot(peak_lag_ms, peak_val, marker='*', color='tab:purple', markersize=5, alpha=0.95)
            ax.axvline(peak_lag_ms, color='tab:blue', linestyle='--', alpha=0.7)
            ax.set_title(f'{a} vs {b} | peak={peak_lag_ms:.1f} ms (r={peak_val:.3f})')
            ax.set_xlabel('Lag [ms]')
            ax.set_ylabel('Corr')
            ax.grid(True)
        for j in range(len(pairs), nrows * ncols):
            r = j // ncols
            c = j % ncols
            axes[r][c].set_visible(False)
        plt.tight_layout()

    # Cross-correlation (filtered)
    if show_filtered and len(pairs) > 0:
        ncols = 2 if len(pairs) > 1 else 1
        nrows = int(np.ceil(len(pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
        fig.suptitle('Cross-correlation (filtered HR±0.5 Hz) - window ±100 ms (sub-sample peak)', y=1.02)
        for idx, (a, b) in enumerate(pairs):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            xa = np.asarray(pos_signals_narrow.get(a, []))
            xb = np.asarray(pos_signals_narrow.get(b, []))
            lags, corr = _xcorr_full(xa, xb, fs)
            if lags is None:
                ax.set_visible(False)
                continue
            # Restrict to ±100 ms window (plot in milliseconds)
            lags_ms = lags * 1000.0
            window_mask = (lags_ms >= -100.0) & (lags_ms <= 100.0)
            if not np.any(window_mask):
                ax.set_visible(False)
                continue
            lags_w_ms = lags_ms[window_mask]
            corr_w = corr[window_mask]
            i = int(np.argmax(corr_w))
            i_ref, peak_val = _parabolic_refine(corr_w, i)
            lag_step_ms = (lags_w_ms[1] - lags_w_ms[0]) if len(lags_w_ms) > 1 else (1000.0 / fs)
            peak_lag_ms = lags_w_ms[0] + i_ref * lag_step_ms
            # Plot discrete correlation samples as points (no connecting line)
            ax.plot(lags_w_ms, corr_w, marker='o', linestyle='None', color='tab:green', markersize=3, alpha=0.9)
            # Highlight the three samples used for parabolic interpolation in a different color
            if 0 < i < len(corr_w) - 1:
                ax.plot(lags_w_ms[i-1:i+2], corr_w[i-1:i+2], marker='o', linestyle='None', color='tab:orange', markersize=4, alpha=0.95)
            # Plot the interpolated apex value itself
            ax.plot(peak_lag_ms, peak_val, marker='*', color='tab:purple', markersize=5, alpha=0.95)
            ax.axvline(peak_lag_ms, color='tab:green', linestyle='--', alpha=0.7)
            ax.set_title(f'{a} vs {b} | peak={peak_lag_ms:.1f} ms (r={peak_val:.3f})')
            ax.set_xlabel('Lag [ms]')
            ax.set_ylabel('Corr')
            ax.grid(True)
        for j in range(len(pairs), nrows * ncols):
            r = j // ncols
            c = j % ncols
            axes[r][c].set_visible(False)
        plt.tight_layout()

    plt.show()


# ------------------------------
# 2) Sliding xcorr (unfiltered and filtered) for signals
# ------------------------------

def sliding_xcorr_lag(
    pos_signals: Dict[str, np.ndarray],
    pos_signals_narrow: Dict[str, np.ndarray],
    fs: float,
    n_beats: int = 10,
    step_beats: int = 1,
    roi_pairs: Optional[List[Tuple[str, str]]] = None,
    peak_dist_s: float = 0.3,
    max_lag_frac: float = 0.25,
    prealign: bool = True,
    prealign_max_lag_s: float = 0.5,
    plot: bool = False
) -> Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]]:
    calc = ParametersCalculator()
    rois = list(pos_signals.keys())
    if roi_pairs is None:
        roi_pairs = list(combinations(rois, 2))

    def analyze_one(A: np.ndarray, B: np.ndarray):
        if A is None or B is None or len(A) < 3 or len(B) < 3:
            return None
        n = min(len(A), len(B))
        A = np.asarray(A[:n])
        B = np.asarray(B[:n])

        if prealign:
            g_lag_ms, g_r = _xcorr_peak_constrained(A, B, fs, max_lag_s=prealign_max_lag_s)
            B_align = _shift_by_ms(B, fs, -g_lag_ms)
        else:
            g_lag_ms, g_r = 0.0, np.nan
            B_align = B

        pA = calc.GetPeaks(A, fs=fs, k_h_max_R=1, distance=peak_dist_s)
        if len(pA) < n_beats + 1:
            return {'global_lag_ms': g_lag_ms, 'global_r': g_r,
                    'time_centers_s': np.array([]), 'lag_ms': np.array([]), 'r': np.array([])}

        tA = pA / fs
        periods = np.diff(tA)
        medT = np.median(periods) if periods.size else 1.0
        max_lag_s = max_lag_frac * medT

        centers, lags, rs = [], [], []
        i = 0
        while i + n_beats < len(pA):
            s = pA[i]
            e = pA[i + n_beats]
            segA = A[s:e]
            segB = B_align[s:e]
            lag_ms, r = _xcorr_peak_constrained(segA, segB, fs, max_lag_s=max_lag_s)
            centers.append((s + e) / 2.0 / fs)
            lags.append(lag_ms)
            rs.append(r)
            i += step_beats

        return {'global_lag_ms': g_lag_ms, 'global_r': g_r,
                'time_centers_s': np.asarray(centers),
                'lag_ms': np.asarray(lags),
                'r': np.asarray(rs)}

    out = {}
    for a, b in roi_pairs:
        # Unfiltered
        A_unf = pos_signals.get(a)
        B_unf = pos_signals.get(b)
        res_unf = analyze_one(A_unf, B_unf)

        # Filtered
        A_f = pos_signals_narrow.get(a)
        B_f = pos_signals_narrow.get(b)
        res_f = analyze_one(A_f, B_f)

        out[(a, b)] = {'unfiltered': res_unf, 'filtered': res_f}

        if plot and res_unf is not None and res_f is not None:
            # Lag vs time
            plt.figure(figsize=(12, 4))
            if res_unf['time_centers_s'].size:
                plt.plot(res_unf['time_centers_s'], res_unf['lag_ms'],
                         label=f'unfiltered (global={res_unf["global_lag_ms"]:.1f} ms)', color='tab:blue', linewidth=1.0)
            if res_f['time_centers_s'].size:
                plt.plot(res_f['time_centers_s'], res_f['lag_ms'],
                         label=f'filtered (global={res_f["global_lag_ms"]:.1f} ms)', color='tab:green', linewidth=1.0)
            plt.axhline(0, color='k', linestyle=':', alpha=0.6)
            # ±1 frame reference
            res_ms = 1000.0 / fs
            plt.axhline(+res_ms, color='k', linestyle='--', alpha=0.5, linewidth=0.9)
            plt.axhline(-res_ms, color='k', linestyle='--', alpha=0.5, linewidth=0.9)
            plt.title(f'Residual lag (pre-aligned={prealign})  {a} vs {b}   N={n_beats}b, step={step_beats}b')
            plt.xlabel('Time center [s]')
            plt.ylabel('Lag [ms]')
            plt.grid(True)
            plt.legend()

            # r vs time
            plt.figure(figsize=(12, 4))
            if res_unf['time_centers_s'].size:
                plt.plot(res_unf['time_centers_s'], res_unf['r'], label='unfiltered r', color='tab:blue', linewidth=1.0)
            if res_f['time_centers_s'].size:
                plt.plot(res_f['time_centers_s'], res_f['r'], label='filtered r', color='tab:green', linewidth=1.0)
            plt.ylim(-1.05, 1.05)
            plt.title(f'XCorr peak r over time  {a} vs {b}')
            plt.xlabel('Time center [s]')
            plt.ylabel('r')
            plt.grid(True)
            plt.legend()
            plt.show()

    return out


# ------------------------------
# 3) Single-video workflows (including chin ROI)
# ------------------------------

def plot_rois_xcorr_from_video(
    video_path: str,
    fs: float = 30,
    window_length: int = 60,
    start_time: int = 5,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True,
    full_face: bool = False,
    show_filtered: bool = True,
    normalize: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # 1) Extract RGB per ROI
    rois_rgb = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
        video_path, fs,
        window_length=window_length, start_time=start_time,
        forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin, full_face=full_face, play_video=False
    )
    if not isinstance(rois_rgb, dict) or len(rois_rgb) == 0:
        raise ValueError("No ROIs extracted from the video.")

    # 2) Compute POS per ROI
    pos_signals: Dict[str, np.ndarray] = {}
    calc = ParametersCalculator()
    for roi, ch in rois_rgb.items():
        r, g, b = ch.get('red', []), ch.get('green', []), ch.get('blue', [])
        if len(r) and len(g) and len(b):
            pos = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                r, g, b, fs, normalize=False, derivative=False, bandpass=True, detrend=True, method='pos'
            )
            pos_signals[roi] = np.asarray(pos)

    if len(pos_signals) < 1:
        raise ValueError("No valid POS signals by ROI.")

    # 3) HR per ROI and narrow-band filtered versions
    pos_signals_narrow: Dict[str, np.ndarray] = {}
    for roi, sig in pos_signals.items():
        hr_bpm = float(calc.ObtainHeartRate(np.asarray(sig), np.array([]), fs, method='two_peaks_periodogram'))
        pos_signals_narrow[roi] = _bandpass_hr(sig, fs, hr_bpm, half_width_hz=0.5)

    # 4) Plot
    plot_all_rois_signals_and_xcorr(
        pos_signals=pos_signals,
        pos_signals_narrow=pos_signals_narrow,
        fs=fs,
        show_filtered=show_filtered,
        normalize=normalize
    )
    return pos_signals, pos_signals_narrow


def plot_rois_xcorr_from_precomputed_rois(
    rois_rgb: Dict[str, Dict[str, List[float]]],
    fs: float,
    show_filtered: bool = True,
    normalize: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Equivalent to the notebook's plot flow, but starts from an already-extracted
    rois_rgb dict:
      { roi_name: { 'red': [...], 'green': [...], 'blue': [...] } }
    Computes POS per ROI, builds a HR±0.5 Hz filtered version, and calls
    plot_all_rois_signals_and_xcorr. Returns (pos_signals, pos_signals_narrow).
    """
    if not isinstance(rois_rgb, dict) or len(rois_rgb) == 0:
        raise ValueError("rois_rgb is empty or invalid.")

    # Compute POS per ROI
    pos_signals: Dict[str, np.ndarray] = {}
    for roi, ch in rois_rgb.items():
        r, g, b = ch.get('red', []), ch.get('green', []), ch.get('blue', [])
        if len(r) and len(g) and len(b):
            pos = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                r, g, b, fs, normalize=False, derivative=False, bandpass=True, detrend=True, method='pos'
            )
            pos_signals[roi] = np.asarray(pos)

    if len(pos_signals) < 1:
        raise ValueError("No valid POS signals by ROI in rois_rgb.")

    # Narrow-band filtered per ROI using each ROI's HR
    calc = ParametersCalculator()
    pos_signals_narrow: Dict[str, np.ndarray] = {}
    for roi, sig in pos_signals.items():
        hr_bpm = float(calc.ObtainHeartRate(np.asarray(sig), np.array([]), fs, method='two_peaks_periodogram'))
        pos_signals_narrow[roi] = _bandpass_hr(sig, fs, hr_bpm, half_width_hz=0.5)

    # Plot
    plot_all_rois_signals_and_xcorr(
        pos_signals=pos_signals,
        pos_signals_narrow=pos_signals_narrow,
        fs=fs,
        show_filtered=show_filtered,
        normalize=normalize
    )
    return pos_signals, pos_signals_narrow


def analyze_video_pos_green_with_plots(
    video_path: str,
    fs: float = 30,
    window_length: int = 60,
    start_time: int = 5,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True,
    full_face: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    # Extract series by ROI
    rois = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
        video_path, fs,
        window_length=window_length, start_time=start_time,
        forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin, full_face=full_face, play_video=False
    )

    pos_signals: Dict[str, np.ndarray] = {}
    green_signals: Dict[str, np.ndarray] = {}
    calc = ParametersCalculator()

    for roi_name, channels in rois.items():
        red = channels.get('red', [])
        g = channels.get('green', [])
        blue = channels.get('blue', [])
        if len(red) == 0 or len(g) == 0 or len(blue) == 0:
            continue

        pos_sig = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
            red, g, blue, fs,
            normalize=False, derivative=False, bandpass=True, detrend=True, method='pos'
        )
        green_sig = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
            red, g, blue, fs,
            normalize=False, derivative=False, bandpass=True, detrend=True, method='green'
        )

        pos_signals[roi_name] = np.asarray(pos_sig)
        green_signals[roi_name] = np.asarray(green_sig)

    # Compute HR per ROI for POS and GREEN
    hr_pos: Dict[str, float] = {}
    hr_green: Dict[str, float] = {}
    for roi_name in rois.keys():
        ps = pos_signals.get(roi_name)
        gs = green_signals.get(roi_name)
        if ps is not None and len(ps) > 0:
            hr_pos[roi_name] = float(calc.ObtainHeartRate(np.array(ps), np.array([]), fs, method='two_peaks_periodogram'))
        if gs is not None and len(gs) > 0:
            hr_green[roi_name] = float(calc.ObtainHeartRate(np.array(gs), np.array([]), fs, method='two_peaks_periodogram'))

    # Quick plots analogous to the notebook cell (POS and GREEN overlapped)
    def zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.size == 0:
            return x
        x = x - np.mean(x)
        s = np.std(x)
        return x if s == 0 else x / s

    # POS overlay
    plt.figure(figsize=(12, 4))
    for roi_name, sig in pos_signals.items():
        t = np.arange(len(sig)) / fs
        plt.plot(t, zscore(sig), label=roi_name, linewidth=0.8)
    plt.title('POS - iPPG per ROI (overlaid)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (z-score)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # GREEN overlay
    plt.figure(figsize=(12, 4))
    for roi_name, sig in green_signals.items():
        t = np.arange(len(sig)) / fs
        plt.plot(t, zscore(sig), label=roi_name, linewidth=0.8)
    plt.title('GREEN - iPPG per ROI (overlaid)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (z-score)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Per-ROI POS vs GREEN
    for roi_name in rois.keys():
        pos_sig = pos_signals.get(roi_name)
        green_sig = green_signals.get(roi_name)
        if pos_sig is None or green_sig is None or len(pos_sig) == 0 or len(green_sig) == 0:
            continue
        n = min(len(pos_sig), len(green_sig))
        t = np.arange(n) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(t, zscore(pos_sig[:n]), label='POS', color='tab:purple', linewidth=0.8)
        plt.plot(t, zscore(green_sig[:n]), label='GREEN', color='tab:green', alpha=0.85, linewidth=0.8)
        plt.title(f'{roi_name} - POS vs GREEN')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude (z-score)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return pos_signals, green_signals, hr_pos, hr_green


def sliding_xcorr_lag_from_video(
    video_path: str,
    fs: float = 30,
    window_length: int = 60,
    start_time: int = 5,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True,
    full_face: bool = False,
    n_beats: int = 10,
    step_beats: int = 1,
    peak_dist_s: float = 0.3,
    max_lag_frac: float = 0.25,
    prealign: bool = True,
    prealign_max_lag_s: float = 0.5,
    plot: bool = False
) -> Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]]:
    # Extract POS
    rois_rgb = IppgSignalObtainer.extractSeriesRoiRGBFromVideo(
        video_path, fs,
        window_length=window_length, start_time=start_time,
        forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin, full_face=full_face, play_video=False
    )
    calc = ParametersCalculator()
    pos_signals: Dict[str, np.ndarray] = {}
    for roi, ch in rois_rgb.items():
        r, g, b = ch.get('red', []), ch.get('green', []), ch.get('blue', [])
        if len(r) and len(g) and len(b):
            pos = IppgSignalObtainer.GetRppGSeriesfromRGBSeries(
                r, g, b, fs, normalize=False, derivative=False, bandpass=True, detrend=True, method='pos'
            )
            pos_signals[roi] = np.asarray(pos)
    if len(pos_signals) < 2:
        raise ValueError("Need at least two valid ROIs to compute cross-correlations.")
    # Narrow-band per ROI around its HR
    pos_signals_narrow: Dict[str, np.ndarray] = {}
    for roi, sig in pos_signals.items():
        hr_bpm = float(calc.ObtainHeartRate(np.asarray(sig), np.array([]), fs, method='two_peaks_periodogram'))
        pos_signals_narrow[roi] = _bandpass_hr(sig, fs, hr_bpm, half_width_hz=0.5)
    # Sliding xcorr
    return sliding_xcorr_lag(
        pos_signals=pos_signals,
        pos_signals_narrow=pos_signals_narrow,
        fs=fs,
        n_beats=n_beats,
        step_beats=step_beats,
        peak_dist_s=peak_dist_s,
        max_lag_frac=max_lag_frac,
        prealign=prealign,
        prealign_max_lag_s=prealign_max_lag_s,
        plot=plot
    )


# ------------------------------
# 4) UBFC dataset-level analysis (with chin ROI)
# ------------------------------

def analyze_ubfc_lag_metrics_with_hr_filter(
    base_folder: str = 'UBFC_DATASET_MERGED',
    datasets: Tuple[str, ...] = ('DATASET_1', 'DATASET_2'),
    fs: float = 30,
    window_length: int = 60,
    start_time: int = 5,
    n_beats: int = 10,
    step_beats: int = 1,
    hr_err_threshold_bpm: float = 10.0,
    prealign: bool = True,
    save_csv_path: Optional[str] = None,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, list], Dict[str, list]]:
    """
    Ported from Analisis_rois_y_metodos.ipynb with chin support.
    - Extracts POS per ROI and estimates HR per ROI.
    - Keeps only ROI pairs with both HR errors < threshold.
    - Computes global refined xcorr and sliding beat-anchored metrics for unfiltered/filtered signals.
    Returns results, summary, failures, and dicts for included/excluded pairs.
    """
    calc = ParametersCalculator()
    results_rows: List[dict] = []
    failures_rows: List[dict] = []
    pair_err_included: Dict[str, List[dict]] = defaultdict(list)
    pair_err_excluded: Dict[str, List[dict]] = defaultdict(list)

    for dataset_name in datasets:
        dataset_path = os.path.join(base_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"[WARN] Missing {dataset_path}, skipping.")
            continue
        subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

        for subj in subjects:
            vid_folder = os.path.join(dataset_path, subj)
            video_path = os.path.join(vid_folder, 'vid.avi')
            if not os.path.exists(video_path):
                continue

            # Load ground truth
            gt_trace, gt_time, gt_hr, fmt = load_ubfc_ground_truth(vid_folder)
            if gt_hr is None or len(gt_hr) == 0:
                gt_bpm = float(calc.ObtainHeartRate(np.asarray(gt_trace), np.array([]), fs, method='two_peaks_periodogram'))
            else:
                valid = gt_hr[np.isfinite(gt_hr)]
                gt_bpm = float(np.nanmean(valid)) if valid.size else float(np.nan)

            # POS by ROI
            try:
                pos_unf = _extract_pos_signals(
                    video_path, fs, window_length, start_time,
                    forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin
                )
            except Exception as e:
                failures_rows.append({'dataset': dataset_name, 'video_id': subj, 'reason': f'ROI extraction error: {e}'})
                continue
            if len(pos_unf) < 2:
                failures_rows.append({'dataset': dataset_name, 'video_id': subj, 'reason': 'Less than 2 ROIs'})
                continue

            # HR per ROI and errors
            hr_roi: Dict[str, float] = {}
            err_roi: Dict[str, float] = {}
            for roi, sig in pos_unf.items():
                if len(sig) == 0:
                    continue
                hr = float(calc.ObtainHeartRate(np.asarray(sig), np.array([]), fs, method='two_peaks_periodogram'))
                hr_roi[roi] = hr
                err_roi[roi] = abs(hr - gt_bpm)
            if len(err_roi) == 0:
                failures_rows.append({'dataset': dataset_name, 'video_id': subj, 'reason': 'No HR per ROI'})
                continue

            # Track included/excluded by HR threshold
            for a, b in combinations(err_roi.keys(), 2):
                key = f'{a}|{b}'
                entry = {
                    'dataset': dataset_name, 'video_id': subj, 'gt_hr_bpm': gt_bpm,
                    'roi_a': a, 'hr_a': hr_roi.get(a, np.nan), 'err_a': err_roi.get(a, np.nan),
                    'roi_b': b, 'hr_b': hr_roi.get(b, np.nan), 'err_b': err_roi.get(b, np.nan),
                }
                if err_roi.get(a, np.inf) < hr_err_threshold_bpm and err_roi.get(b, np.inf) < hr_err_threshold_bpm:
                    pair_err_included[key].append(entry)
                else:
                    pair_err_excluded[key].append(entry)

            # Valid ROIs
            ok_rois = [roi for roi, e in err_roi.items() if e < hr_err_threshold_bpm]
            if len(ok_rois) < 2:
                row_fail = {'dataset': dataset_name, 'video_id': subj, 'gt_hr_bpm': gt_bpm,
                            'reason': f'Less than 2 ROIs with HR error < {hr_err_threshold_bpm} bpm'}
                for roi in pos_unf.keys():
                    row_fail[f'hr_{roi}'] = hr_roi.get(roi, np.nan)
                    row_fail[f'err_{roi}'] = err_roi.get(roi, np.nan)
                failures_rows.append(row_fail)
                continue

            # Filtered signals around each ROI's HR
            pos_unf_ok = {roi: pos_unf[roi] for roi in ok_rois}
            pos_filt_ok = {roi: _bandpass_hr(pos_unf_ok[roi], fs, hr_roi.get(roi, gt_bpm), half_width_hz=0.5) for roi in ok_rois}

            pairs = list(combinations(ok_rois, 2))

            # Sliding lag on valid ROIs
            lag_res = sliding_xcorr_lag(
                pos_signals=pos_unf_ok,
                pos_signals_narrow=pos_filt_ok,
                fs=fs,
                n_beats=n_beats,
                step_beats=step_beats,
                roi_pairs=pairs,
                peak_dist_s=0.3,
                max_lag_frac=0.25,
                prealign=prealign,
                plot=False
            )

            one_frame_ms = 1000.0 / fs

            # Append rows (per pair and mode)
            for (a, b), bundle in lag_res.items():
                for mode in ['unfiltered', 'filtered']:
                    m = bundle.get(mode)
                    if m is None or m['time_centers_s'].size == 0:
                        continue
                    # Global refined for this mode
                    ga = pos_unf_ok[a] if mode == 'unfiltered' else pos_filt_ok[a]
                    gb = pos_unf_ok[b] if mode == 'unfiltered' else pos_filt_ok[b]
                    g_lag_ms_refined, g_r_refined = _global_xcorr_subsample(ga, gb, fs, max_lag_s=0.5)

                    lags = np.asarray(m['lag_ms'])
                    r = np.asarray(m['r'])
                    lags = lags[np.isfinite(lags)]
                    n_win = int(np.sum(np.isfinite(lags)))
                    if n_win == 0:
                        continue

                    median_signed = float(np.nanmedian(lags))
                    iqr = float(np.nanpercentile(lags, 75) - np.nanpercentile(lags, 25))
                    abs_lags = np.abs(lags)
                    median_abs = float(np.nanmedian(abs_lags))
                    mean_abs = float(np.nanmean(abs_lags))
                    # Histogram-based mode (bin center of the highest-count bin)
                    if abs_lags.size:
                        valid_abs = abs_lags[np.isfinite(abs_lags)]
                        if valid_abs.size:
                            hist, edges = np.histogram(valid_abs, bins='auto')
                            mode_idx = int(np.argmax(hist))
                            mode_abs = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)
                        else:
                            mode_abs = float('nan')
                    else:
                        mode_abs = float('nan')
                    rms = float(np.sqrt(np.nanmean(lags**2)))

                    results_rows.append({
                        'dataset': dataset_name,
                        'video_id': subj,
                        'roi_a': a, 'roi_b': b,
                        'mode': mode,
                        'gt_hr_bpm': gt_bpm,
                        'global_lag_ms_precalc': float(m['global_lag_ms']),
                        'global_r_precalc': float(m['global_r']),
                        'global_lag_ms_refined': float(g_lag_ms_refined),
                        'global_r_refined': float(g_r_refined),
                        'n_windows': n_win,
                        'median_residual_lag_ms': median_signed,
                        'iqr_residual_lag_ms': iqr,
                        'median_abs_residual_lag_ms': median_abs,
                        'mean_abs_residual_lag_ms': mean_abs,
                        'mode_abs_residual_lag_ms': mode_abs,
                        'rms_residual_lag_ms': rms,
                        'median_r': float(np.nanmedian(r)),
                        'pct_windows_|lag|<=1frame': float(np.mean(np.abs(lags) <= one_frame_ms) * 100.0),
                        'pct_windows_r>=0.8': float(np.mean(r >= 0.8) * 100.0),
                    })

    df_results = pd.DataFrame(results_rows).sort_values(['dataset', 'video_id', 'roi_a', 'roi_b', 'mode'])
    df_failures = pd.DataFrame(failures_rows).sort_values(['dataset', 'video_id']) if failures_rows else pd.DataFrame()

    # Aggregated summary per dataset, pair, mode
    if not df_results.empty:
        df_summary = (
            df_results.groupby(['dataset', 'roi_a', 'roi_b', 'mode'])
            .agg(
                videos=('video_id', 'nunique'),
                n_windows=('n_windows', 'sum'),
                median_global_lag_ms=('global_lag_ms_refined', 'median'),
                median_global_r=('global_r_refined', 'median'),
                median_residual_lag_ms=('median_residual_lag_ms', 'median'),
                median_iqr_lag_ms=('iqr_residual_lag_ms', 'median'),
                median_abs_residual_lag_ms=('median_abs_residual_lag_ms', 'median'),
                mean_abs_residual_lag_ms=('mean_abs_residual_lag_ms', 'mean'),
                mode_abs_residual_lag_ms=('mode_abs_residual_lag_ms', 'median'),
                median_rms_residual_lag_ms=('rms_residual_lag_ms', 'median'),
                mean_pct_within_1frame=('pct_windows_|lag|<=1frame', 'mean'),
                mean_pct_high_r=('pct_windows_r>=0.8', 'mean'),
            )
            .reset_index()
        )
    else:
        df_summary = pd.DataFrame()

    if save_csv_path:
        base, ext = os.path.splitext(save_csv_path)
        ext = ext if ext else '.csv'
        df_results.to_csv(f"{base}{ext}", index=False)
        df_summary.to_csv(f"{base}_summary{ext}", index=False)
        if pair_err_included:
            inc_rows = [dict(pair=pair, **row) for pair, lst in pair_err_included.items() for row in lst]
            pd.DataFrame(inc_rows).to_csv(f"{base}_pair_err_included{ext}", index=False)
        if pair_err_excluded:
            exc_rows = [dict(pair=pair, **row) for pair, lst in pair_err_excluded.items() for row in lst]
            pd.DataFrame(exc_rows).to_csv(f"{base}_pair_err_excluded{ext}", index=False)
        if not df_failures.empty:
            df_failures.to_csv(f"{base}_failures{ext}", index=False)

    return df_results, df_summary, df_failures, pair_err_included, pair_err_excluded


