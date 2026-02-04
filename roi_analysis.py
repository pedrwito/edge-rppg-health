import os
import re
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.lines import Line2D
import builtins

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


def _shift_by_samples(x: np.ndarray, shift_samples: int) -> np.ndarray:
    """
    Shift signal by integer number of samples (no interpolation).
    Positive shift_samples means shift right (delay), negative means shift left (advance).
    """
    if shift_samples == 0:
        return np.asarray(x)
    x = np.asarray(x)
    if abs(shift_samples) >= len(x):
        # Shift is larger than signal length
        return np.zeros_like(x)
    if shift_samples > 0:
        # Shift right (delay): pad with zeros at the beginning
        return np.concatenate([np.zeros(shift_samples), x[:-shift_samples]])
    else:
        # Shift left (advance): pad with zeros at the end
        return np.concatenate([x[-shift_samples:], np.zeros(-shift_samples)])


def _round_to_discrete_lag(lag_ms: float, fs: float) -> float:
    """
    Round a lag value in ms to the nearest discrete lag value (integer multiple of sample period).
    For 60 fps: sample period = 16.666... ms, so lags are 0, ±16.67, ±33.33, etc.
    """
    if not np.isfinite(lag_ms):
        return lag_ms
    sample_period_ms = 1000.0 / fs
    # Convert to samples, round to nearest integer, convert back to ms
    lag_samples = round(lag_ms / sample_period_ms)
    return lag_samples * sample_period_ms


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


def _xcorr_peak_constrained_discrete(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: Optional[float] = None) -> Tuple[float, float]:
    """
    Discrete version: no interpolation, lag is integer multiple of sample period.
    Returns lag in ms and correlation value at that discrete sample.
    """
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
    lag_samples = int(lags[i])  # Integer sample lag
    peak_val = float(corr[i])  # Actual correlation at that sample
    return 1000.0 * lag_samples / fs, peak_val


def _global_xcorr_discrete(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: Optional[float] = None) -> Tuple[float, float]:
    """
    Discrete version: no interpolation, lag is integer multiple of sample period.
    Returns lag in ms and correlation value at that discrete sample.
    """
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
    lag_samples = int(lags[i])  # Integer sample lag
    peak_val = float(corr[i])  # Actual correlation at that sample
    return 1000.0 * lag_samples / fs, peak_val

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
    subsample_peak: bool = True,
    range: Optional[Tuple[float, float]] = None,
    save_for_paper: bool = False,
    save_prefix: Optional[str] = None,
    paper_style: bool = False,
) -> None:
    rois = list(pos_signals.keys())

    def _pretty_roi(name: str) -> str:
        # left_cheek -> Left Cheek
        return str(name).replace("_", " ").strip().title()

    # Optional time-windowing (for plotting only)
    # NOTE: `range` is a (t_start_s, t_end_s) tuple, in seconds, relative to the extracted signal start.
    pos_plot = pos_signals
    pos_narrow_plot = pos_signals_narrow
    t0_s: float = 0.0
    t1_s: Optional[float] = None

    if range is not None:
        if (not isinstance(range, (list, tuple))) or len(range) != 2:
            raise ValueError("range must be None or a (t_start_s, t_end_s) tuple")
        t0_s = float(range[0])
        t1_s = float(range[1])
        if not np.isfinite(t0_s) or not np.isfinite(t1_s) or t1_s <= t0_s:
            raise ValueError("range must be finite and satisfy t_end_s > t_start_s")
        if fs <= 0:
            raise ValueError("fs must be positive")

        # Find a stable max length across available ROIs so all plots align.
        lengths = [int(np.asarray(pos_signals.get(roi, [])).size) for roi in rois]
        max_len = min([n for n in lengths if n > 0], default=0)
        if max_len <= 0:
            raise ValueError("No non-empty ROI signals available for plotting.")

        i0 = max(0, int(np.floor(t0_s * fs)))
        i1 = min(max_len, int(np.ceil(t1_s * fs)))
        if i1 - i0 < 3:
            raise ValueError(f"range is too small for plotting after clipping: samples={i1 - i0}")

        def _slice_dict(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            for k, v in d.items():
                x = np.asarray(v)
                if x.size == 0:
                    out[k] = x
                    continue
                out[k] = x[i0:i1]
            return out

        pos_plot = _slice_dict(pos_signals)
        pos_narrow_plot = _slice_dict(pos_signals_narrow)
        # Keep exact clipped window for filenames / axes
        t0_s = i0 / fs
        t1_s = i1 / fs

    # Optional SVG saving
    save_base = save_prefix or "plot"
    time_suffix = ""
    if t1_s is not None and t0_s is not None and (range is not None):
        # Make a filesystem-safe suffix
        time_suffix = f"_t{t0_s:.2f}-{t1_s:.2f}s".replace(".", "p")

    def _maybe_save(fig: "plt.Figure", tag: str) -> None:
        if not save_for_paper:
            return
        out_path = f"{save_base}{time_suffix}_{tag}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")

    # Paper-friendly matplotlib style (local to this function)
    rc = {}
    if paper_style or save_for_paper:
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }

    with plt.rc_context(rc):

        if plot_individual:
            for roi in rois:
                x = np.asarray(pos_plot.get(roi, []))
                if x.size == 0:
                    continue
                t = (np.arange(x.size) / fs) + t0_s
                peaks = _compute_peaks(x, fs)
                if peaks is None or len(peaks) == 0:
                    continue
                t_peaks = (peaks / fs) + t0_s
                fig = plt.figure(figsize=(14, 5))
                plt.scatter(t_peaks, _z(x, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
                plt.plot(t, _z(x, normalize=normalize), label=_pretty_roi(roi), linewidth=0.9)
                if not (paper_style or save_for_paper):
                    plt.title(f'{roi} - POS (unfiltered)')
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
                plt.grid(True)
                plt.legend()
                _maybe_save(fig, f"signal_{roi}_unfiltered")
                plt.show()

        # Unfiltered, all ROIs
        fig = plt.figure(figsize=(14, 5))
        for roi in rois:
            x = np.asarray(pos_plot.get(roi, []))
            if x.size == 0:
                continue
            t = (np.arange(x.size) / fs) + t0_s
            plt.plot(t, _z(x, normalize=normalize), label=_pretty_roi(roi), linewidth=0.9)
        for roi in rois:
            x = np.asarray(pos_plot.get(roi, []))
            if x.size == 0:
                continue
            peaks = _compute_peaks(x, fs)
            if peaks is None or len(peaks) == 0:
                continue
            t_peaks = (peaks / fs) + t0_s
            plt.scatter(t_peaks, _z(x, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
        if not (paper_style or save_for_paper):
            plt.title('All ROIs - POS (unfiltered) with peak markers')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
        plt.grid(True)
        plt.legend(ncol=min(len(rois), 4))
        _maybe_save(fig, "signals_all_unfiltered")

        # Filtered, all ROIs
        if show_filtered:
            fig = plt.figure(figsize=(14, 5))
            for roi in rois:
                xf = np.asarray(pos_narrow_plot.get(roi, []))
                if xf.size == 0:
                    continue
                t = (np.arange(xf.size) / fs) + t0_s
                plt.plot(t, _z(xf, normalize=normalize), label=_pretty_roi(roi), linewidth=0.9)
            for roi in rois:
                xf = np.asarray(pos_narrow_plot.get(roi, []))
                if xf.size == 0:
                    continue
                peaks = _compute_peaks(xf, fs)
                if peaks is None or len(peaks) == 0:
                    continue
                t_peaks = (peaks / fs) + t0_s
                plt.scatter(t_peaks, _z(xf, normalize=normalize)[peaks], s=12, marker='o', alpha=0.8)
            if not (paper_style or save_for_paper):
                plt.title('All ROIs - POS (filtered HR±0.5 Hz) with peak markers')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude' + (' (z-score)' if normalize else ''))
            plt.grid(True)
            plt.legend(ncol=min(len(rois), 4))
            _maybe_save(fig, "signals_all_filtered")

        # Cross-correlation (unfiltered)
        pairs = list(combinations(rois, 2))
        if len(pairs) > 0:
            ncols = 2 if len(pairs) > 1 else 1
            nrows = int(np.ceil(len(pairs) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
            if not (paper_style or save_for_paper):
                fig.suptitle(
                    'Cross-correlation (unfiltered) - window ±100 ms (sub-sample peak)'
                    if subsample_peak else
                    'Cross-correlation (unfiltered) - window ±100 ms (discrete peak, no interpolation)',
                    y=1.02
                )
            for idx, (a, b) in enumerate(pairs):
                rr = idx // ncols
                cc = idx % ncols
                ax = axes[rr][cc]
                xa = np.asarray(pos_plot.get(a, []))
                xb = np.asarray(pos_plot.get(b, []))
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
                if subsample_peak:
                    i_ref, peak_val = _parabolic_refine(corr_w, i)
                    lag_step_ms = (lags_w_ms[1] - lags_w_ms[0]) if len(lags_w_ms) > 1 else (1000.0 / fs)
                    peak_lag_ms = lags_w_ms[0] + i_ref * lag_step_ms
                else:
                    peak_val = float(corr_w[i])
                    peak_lag_ms = float(lags_w_ms[i])

                # Plot discrete correlation samples as points (no connecting line)
                ax.plot(lags_w_ms, corr_w, marker='o', linestyle='None', color='tab:blue', markersize=3, alpha=0.9)

                if subsample_peak and 0 < i < len(corr_w) - 1:
                    # Highlight the three samples used for parabolic interpolation in a different color
                    x3 = lags_w_ms[i - 1 : i + 2]
                    y3 = corr_w[i - 1 : i + 2]
                    ax.plot(x3, y3, marker='o', linestyle='None', color='tab:orange', markersize=4, alpha=0.95)

                    # Overlay the parabola fit to visually justify the sub-sample peak
                    try:
                        coeff = np.polyfit(x3, y3, deg=2)
                        x_fit = np.linspace(float(x3[0]), float(x3[-1]), 80)
                        y_fit = np.polyval(coeff, x_fit)
                        ax.plot(x_fit, y_fit, color='tab:orange', linewidth=1.2, alpha=0.9)
                    except Exception:
                        pass

                    # Plot the interpolated apex value itself
                    ax.plot(peak_lag_ms, peak_val, marker='*', color='tab:purple', markersize=6, alpha=0.95)

                ax.axvline(peak_lag_ms, color='tab:blue', linestyle='--', alpha=0.7)

                # Cleaner, paper-style titles
                a_name, b_name = _pretty_roi(a), _pretty_roi(b)
                if paper_style or save_for_paper:
                    ax.set_title(rf'{a_name} – {b_name} ($\tau_{{peak}}={peak_lag_ms:.1f}$ ms, $r={peak_val:.2f}$)')
                else:
                    ax.set_title(f'{a} vs {b} | peak={peak_lag_ms:.1f} ms (r={peak_val:.3f})')

                # De-clutter: axis labels only on left column / bottom row (for publication)
                if paper_style or save_for_paper:
                    if cc != 0:
                        ax.set_ylabel("")
                        ax.tick_params(labelleft=False)
                    else:
                        ax.set_ylabel("Corr")
                    if rr != (nrows - 1):
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xlabel("Lag [ms]")
                else:
                    ax.set_xlabel('Lag [ms]')
                    ax.set_ylabel('Corr')

                ax.grid(True)

            for j in builtins.range(len(pairs), nrows * ncols):
                rr = j // ncols
                cc = j % ncols
                axes[rr][cc].set_visible(False)

            if paper_style or save_for_paper:
                # Single legend for the whole grid (explains the blue/orange/purple markers)
                handles = [
                    Line2D([0], [0], marker='o', color='tab:blue', linestyle='None', markersize=4, label='XCorr samples'),
                    Line2D([0], [0], marker='o', color='tab:orange', linestyle='None', markersize=5, label='Fit samples'),
                    Line2D([0], [0], color='tab:orange', linewidth=1.2, label='Parabolic fit'),
                    Line2D([0], [0], marker='*', color='tab:purple', linestyle='None', markersize=7, label=r'Sub-sample peak'),
                ]
                fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout()
            _maybe_save(fig, "xcorr_unfiltered")

        # Cross-correlation (filtered)
        if show_filtered and len(pairs) > 0:
            ncols = 2 if len(pairs) > 1 else 1
            nrows = int(np.ceil(len(pairs) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
            if not (paper_style or save_for_paper):
                fig.suptitle(
                    'Cross-correlation (filtered HR±0.5 Hz) - window ±100 ms (sub-sample peak)'
                    if subsample_peak else
                    'Cross-correlation (filtered HR±0.5 Hz) - window ±100 ms (discrete peak, no interpolation)',
                    y=1.02
                )
            for idx, (a, b) in enumerate(pairs):
                rr = idx // ncols
                cc = idx % ncols
                ax = axes[rr][cc]
                xa = np.asarray(pos_narrow_plot.get(a, []))
                xb = np.asarray(pos_narrow_plot.get(b, []))
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
                if subsample_peak:
                    i_ref, peak_val = _parabolic_refine(corr_w, i)
                    lag_step_ms = (lags_w_ms[1] - lags_w_ms[0]) if len(lags_w_ms) > 1 else (1000.0 / fs)
                    peak_lag_ms = lags_w_ms[0] + i_ref * lag_step_ms
                else:
                    peak_val = float(corr_w[i])
                    peak_lag_ms = float(lags_w_ms[i])

                # Plot discrete correlation samples as points (no connecting line)
                ax.plot(lags_w_ms, corr_w, marker='o', linestyle='None', color='tab:green', markersize=3, alpha=0.9)

                if subsample_peak and 0 < i < len(corr_w) - 1:
                    x3 = lags_w_ms[i - 1 : i + 2]
                    y3 = corr_w[i - 1 : i + 2]
                    ax.plot(x3, y3, marker='o', linestyle='None', color='tab:orange', markersize=4, alpha=0.95)
                    try:
                        coeff = np.polyfit(x3, y3, deg=2)
                        x_fit = np.linspace(float(x3[0]), float(x3[-1]), 80)
                        y_fit = np.polyval(coeff, x_fit)
                        ax.plot(x_fit, y_fit, color='tab:orange', linewidth=1.2, alpha=0.9)
                    except Exception:
                        pass
                    ax.plot(peak_lag_ms, peak_val, marker='*', color='tab:purple', markersize=6, alpha=0.95)

                ax.axvline(peak_lag_ms, color='tab:green', linestyle='--', alpha=0.7)

                a_name, b_name = _pretty_roi(a), _pretty_roi(b)
                if paper_style or save_for_paper:
                    ax.set_title(rf'{a_name} – {b_name} ($\tau_{{peak}}={peak_lag_ms:.1f}$ ms, $r={peak_val:.2f}$)')
                else:
                    ax.set_title(f'{a} vs {b} | peak={peak_lag_ms:.1f} ms (r={peak_val:.3f})')

                if paper_style or save_for_paper:
                    if cc != 0:
                        ax.set_ylabel("")
                        ax.tick_params(labelleft=False)
                    else:
                        ax.set_ylabel("Corr")
                    if rr != (nrows - 1):
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xlabel("Lag [ms]")
                else:
                    ax.set_xlabel('Lag [ms]')
                    ax.set_ylabel('Corr')

                ax.grid(True)

            for j in builtins.range(len(pairs), nrows * ncols):
                rr = j // ncols
                cc = j % ncols
                axes[rr][cc].set_visible(False)

            if paper_style or save_for_paper:
                handles = [
                    Line2D([0], [0], marker='o', color='tab:green', linestyle='None', markersize=4, label='XCorr samples'),
                    Line2D([0], [0], marker='o', color='tab:orange', linestyle='None', markersize=5, label='Fit samples'),
                    Line2D([0], [0], color='tab:orange', linewidth=1.2, label='Parabolic fit'),
                    Line2D([0], [0], marker='*', color='tab:purple', linestyle='None', markersize=7, label=r'Sub-sample peak'),
                ]
                fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout()
            _maybe_save(fig, "xcorr_filtered")

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


def sliding_xcorr_lag_discrete(
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
    """
    Discrete version: no interpolation, lags are integer multiples of sample period.
    """
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
            g_lag_ms, g_r = _xcorr_peak_constrained_discrete(A, B, fs, max_lag_s=prealign_max_lag_s)
            # Convert lag_ms to integer samples for discrete shifting
            shift_samples = int(round(g_lag_ms * fs / 1000.0))
            B_align = _shift_by_samples(B, -shift_samples)
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
            lag_ms, r = _xcorr_peak_constrained_discrete(segA, segB, fs, max_lag_s=max_lag_s)
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
            plt.title(f'Residual lag (pre-aligned={prealign}, DISCRETE)  {a} vs {b}   N={n_beats}b, step={step_beats}b')
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
            plt.title(f'XCorr peak r over time (DISCRETE)  {a} vs {b}')
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
    normalize: bool = True,
    subsample_peak: bool = True,
    range: Optional[Tuple[float, float]] = None,
    save_for_paper: bool = False,
    paper_style: bool = False,
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
        normalize=normalize,
        subsample_peak=subsample_peak,
        range=range,
        save_for_paper=save_for_paper,
        save_prefix=os.path.splitext(os.path.basename(video_path))[0],
        paper_style=(paper_style or save_for_paper),
    )
    return pos_signals, pos_signals_narrow


def plot_rois_xcorr_from_precomputed_rois(
    rois_rgb: Dict[str, Dict[str, List[float]]],
    fs: float,
    show_filtered: bool = True,
    normalize: bool = True,
    subsample_peak: bool = True,
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
        normalize=normalize,
        subsample_peak=subsample_peak,
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
    chin: bool = True,
    min_global_r: Optional[float] = None,
    use_interpolation: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, list], Dict[str, list]]:
    """
    Ported from Analisis_rois_y_metodos.ipynb with chin support.
    - Extracts POS per ROI and estimates HR per ROI.
    - Keeps only ROI pairs with both HR errors < threshold.
    - Optionally filters out pair/mode results with global refined xcorr peak r < min_global_r.
    - Optionally uses discrete lags (no interpolation) via use_interpolation=False.
    - Computes global refined xcorr and sliding beat-anchored metrics for unfiltered/filtered signals.
    Returns results, summary, failures, and dicts for included/excluded pairs.
    """
    calc = ParametersCalculator()
    results_rows: List[dict] = []
    failures_rows: List[dict] = []
    pair_err_included: Dict[str, List[dict]] = defaultdict(list)
    pair_err_excluded: Dict[str, List[dict]] = defaultdict(list)

    # Choose lag/correlation implementations
    lag_fn = sliding_xcorr_lag if use_interpolation else sliding_xcorr_lag_discrete
    global_xcorr_fn = _global_xcorr_subsample if use_interpolation else _global_xcorr_discrete

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
            lag_res = lag_fn(
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
                    g_lag_ms_refined, g_r_refined = global_xcorr_fn(ga, gb, fs, max_lag_s=0.5)

                    if min_global_r is not None and np.isfinite(min_global_r) and np.isfinite(g_r_refined):
                        if g_r_refined < float(min_global_r):
                            continue

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
                        'pct_windows_|lag|>1frame': float(np.mean(np.abs(lags) > one_frame_ms) * 100.0),
                        'pct_windows_|lag|>2frames': float(np.mean(np.abs(lags) > 2.0 * one_frame_ms) * 100.0),
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
                mean_pct_over_1frame=('pct_windows_|lag|>1frame', 'mean'),
                mean_pct_over_2frames=('pct_windows_|lag|>2frames', 'mean'),
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


# ------------------------------
# 5) Local lossless videos analysis (60 fps, HR from filename, correlation filter)
# ------------------------------

def _parse_hr_from_filename(filename: str) -> Optional[float]:
    """
    Parse HR from filename patterns like:
    - video_lossless_*_87bpm_*.mkv -> 87
    - video_lossless_*_60-72bpm_*.mkv -> 66 (average)
    - video_lossless_*_66bpm_*.mkv -> 66
    Returns None if no HR pattern found.
    """
    # Pattern 1: Range like "60-72bpm" -> take average (check first to avoid matching "60" in range)
    match_range = re.search(r'(\d+)-(\d+)bpm', filename)
    if match_range:
        low = float(match_range.group(1))
        high = float(match_range.group(2))
        return (low + high) / 2.0
    
    # Pattern 2: Single HR like "87bpm" or "66bpm"
    match_single = re.search(r'(\d+)bpm', filename)
    if match_single:
        return float(match_single.group(1))
    
    return None


def analyze_lossless_videos_lag_metrics(
    video_folder: str = '.',
    video_pattern: str = 'video_lossless_*.mkv',
    fs: float = 60,
    window_length: int = 60,
    start_time: int = 5,
    n_beats: int = 10,
    step_beats: int = 1,
    min_correlation: float = 0.5,
    prealign: bool = True,
    save_csv_path: Optional[str] = None,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze local lossless videos (60 fps) with HR extracted from filenames.
    - Extracts POS per ROI and estimates HR per ROI.
    - Keeps only ROI pairs with correlation >= min_correlation.
    - Computes global refined xcorr and sliding beat-anchored metrics for unfiltered/filtered signals.
    Returns results, summary, and failures dataframes.
    """
    import glob
    
    calc = ParametersCalculator()
    results_rows: List[dict] = []
    failures_rows: List[dict] = []
    
    # Find all matching video files
    video_files = glob.glob(os.path.join(video_folder, video_pattern))
    video_files = sorted([f for f in video_files if os.path.isfile(f)])
    
    if len(video_files) == 0:
        print(f"[WARN] No videos found matching pattern: {video_pattern}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(video_files)} videos to analyze")
    
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        
        # Parse HR from filename
        gt_bpm = _parse_hr_from_filename(video_filename)
        if gt_bpm is None:
            failures_rows.append({
                'video_id': video_id,
                'reason': f'Could not parse HR from filename: {video_filename}'
            })
            continue
        
        print(f"Processing {video_filename} (GT HR: {gt_bpm:.1f} bpm)")
        
        # POS by ROI
        try:
            pos_unf = _extract_pos_signals(
                video_path, fs, window_length, start_time,
                forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin
            )
        except Exception as e:
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': f'ROI extraction error: {e}'
            })
            continue
        
        if len(pos_unf) < 2:
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': 'Less than 2 ROIs'
            })
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
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': 'No HR per ROI'
            })
            continue
        
        # Filtered signals around each ROI's HR
        pos_filt = {roi: _bandpass_hr(pos_unf[roi], fs, hr_roi.get(roi, gt_bpm), half_width_hz=0.5) for roi in pos_unf.keys()}
        
        pairs = list(combinations(pos_unf.keys(), 2))
        
        # Sliding lag on all ROIs
        lag_res = sliding_xcorr_lag(
            pos_signals=pos_unf,
            pos_signals_narrow=pos_filt,
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
        
        # Append rows (per pair and mode) - filter by correlation
        for (a, b), bundle in lag_res.items():
            for mode in ['unfiltered', 'filtered']:
                m = bundle.get(mode)
                if m is None or m['time_centers_s'].size == 0:
                    continue
                
                # Global refined for this mode
                ga = pos_unf[a] if mode == 'unfiltered' else pos_filt[a]
                gb = pos_unf[b] if mode == 'unfiltered' else pos_filt[b]
                g_lag_ms_refined, g_r_refined = _global_xcorr_subsample(ga, gb, fs, max_lag_s=0.5)
                
                # Filter by correlation threshold
                if g_r_refined < min_correlation:
                    continue
                
                lags = np.asarray(m['lag_ms'])
                r = np.asarray(m['r'])
                valid_mask = np.isfinite(lags)
                lags = lags[valid_mask]
                r = r[valid_mask]  # Filter r to match valid lags
                n_win = int(len(lags))
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
                    'video_id': video_id,
                    'roi_a': a, 'roi_b': b,
                    'mode': mode,
                    'gt_hr_bpm': gt_bpm,
                    'hr_a': hr_roi.get(a, np.nan),
                    'hr_b': hr_roi.get(b, np.nan),
                    'err_a': err_roi.get(a, np.nan),
                    'err_b': err_roi.get(b, np.nan),
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
    
    df_results = pd.DataFrame(results_rows).sort_values(['video_id', 'roi_a', 'roi_b', 'mode'])
    df_failures = pd.DataFrame(failures_rows).sort_values(['video_id']) if failures_rows else pd.DataFrame()
    
    # Aggregated summary per pair, mode
    if not df_results.empty:
        df_summary = (
            df_results.groupby(['roi_a', 'roi_b', 'mode'])
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
        if not df_failures.empty:
            df_failures.to_csv(f"{base}_failures{ext}", index=False)
    
    return df_results, df_summary, df_failures


# ------------------------------
# 5b) Local lossless videos analysis (HR from filename, HR-error filter + corr threshold, optional discrete)
# ------------------------------

def analyze_lossless_videos_lag_metrics_with_hr_filter(
    video_folder: str = '.',
    video_pattern: str = 'video_lossless_*.mkv',
    fs: float = 60,
    window_length: int = 60,
    start_time: int = 5,
    n_beats: int = 10,
    step_beats: int = 1,
    hr_err_threshold_bpm: float = 10.0,
    min_correlation: Optional[float] = 0.5,
    use_interpolation: bool = True,
    prealign: bool = True,
    save_csv_path: Optional[str] = None,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze local lossless videos with HR extracted from filenames (e.g. "*_54bpm_*.mkv")
    using an HR-error filter (like UBFC) and an optional global correlation threshold.

    - Extracts POS per ROI and estimates HR per ROI.
    - Keeps only ROIs with |HR_roi - HR_gt| < hr_err_threshold_bpm.
    - Computes global xcorr and sliding beat-anchored metrics for unfiltered/filtered signals.
    - Keeps only ROI pairs with global refined xcorr peak r >= min_correlation (if not None).
    - use_interpolation=False uses discrete lags (integer multiples of sample period).

    Example (your folder):
        analyze_lossless_videos_lag_metrics_with_hr_filter(
            video_folder=r"C:\\Users\\pedro\\edge-rppg-health\\VIDEOS 60 FPS",
            video_pattern="video_lossless_*.mkv",
            fs=60,
            hr_err_threshold_bpm=10,
            min_correlation=0.5,
            use_interpolation=False,
            save_csv_path="videos60_lag_metrics_hrfilter_discrete.csv",
        )
    """
    import glob

    calc = ParametersCalculator()
    results_rows: List[dict] = []
    failures_rows: List[dict] = []

    lag_fn = sliding_xcorr_lag if use_interpolation else sliding_xcorr_lag_discrete
    global_xcorr_fn = _global_xcorr_subsample if use_interpolation else _global_xcorr_discrete

    video_files = glob.glob(os.path.join(video_folder, video_pattern))
    video_files = sorted([f for f in video_files if os.path.isfile(f)])
    if len(video_files) == 0:
        print(f"[WARN] No videos found matching pattern: {video_pattern} in {video_folder}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(video_files)} videos to analyze ({'INTERP' if use_interpolation else 'DISCRETE'} mode)")

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]

        gt_bpm = _parse_hr_from_filename(video_filename)
        if gt_bpm is None or not np.isfinite(gt_bpm):
            failures_rows.append({'video_id': video_id, 'reason': f'Could not parse HR from filename: {video_filename}'})
            continue

        try:
            pos_unf = _extract_pos_signals(
                video_path, fs, window_length, start_time,
                forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin
            )
        except Exception as e:
            failures_rows.append({'video_id': video_id, 'gt_hr_bpm': gt_bpm, 'reason': f'ROI extraction error: {e}'})
            continue

        if len(pos_unf) < 2:
            failures_rows.append({'video_id': video_id, 'gt_hr_bpm': gt_bpm, 'reason': 'Less than 2 ROIs'})
            continue

        # HR per ROI and errors
        hr_roi: Dict[str, float] = {}
        err_roi: Dict[str, float] = {}
        for roi, sig in pos_unf.items():
            if len(sig) == 0:
                continue
            hr = float(calc.ObtainHeartRate(np.asarray(sig), np.array([]), fs, method='two_peaks_periodogram'))
            hr_roi[roi] = hr
            err_roi[roi] = abs(hr - float(gt_bpm))

        if len(err_roi) == 0:
            failures_rows.append({'video_id': video_id, 'gt_hr_bpm': gt_bpm, 'reason': 'No HR per ROI'})
            continue

        ok_rois = [roi for roi, e in err_roi.items() if np.isfinite(e) and e < hr_err_threshold_bpm]
        if len(ok_rois) < 2:
            row_fail = {
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': f'Less than 2 ROIs with HR error < {hr_err_threshold_bpm} bpm'
            }
            for roi in sorted(pos_unf.keys()):
                row_fail[f'hr_{roi}'] = hr_roi.get(roi, np.nan)
                row_fail[f'err_{roi}'] = err_roi.get(roi, np.nan)
            failures_rows.append(row_fail)
            continue

        pos_unf_ok = {roi: pos_unf[roi] for roi in ok_rois}
        pos_filt_ok = {roi: _bandpass_hr(pos_unf_ok[roi], fs, hr_roi.get(roi, gt_bpm), half_width_hz=0.5) for roi in ok_rois}

        pairs = list(combinations(ok_rois, 2))
        lag_res = lag_fn(
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

        for (a, b), bundle in lag_res.items():
            for mode in ['unfiltered', 'filtered']:
                m = bundle.get(mode)
                if m is None or m['time_centers_s'].size == 0:
                    continue

                ga = pos_unf_ok[a] if mode == 'unfiltered' else pos_filt_ok[a]
                gb = pos_unf_ok[b] if mode == 'unfiltered' else pos_filt_ok[b]
                g_lag_ms_refined, g_r_refined = global_xcorr_fn(ga, gb, fs, max_lag_s=0.5)
                if not use_interpolation:
                    g_lag_ms_refined = _round_to_discrete_lag(g_lag_ms_refined, fs)

                if min_correlation is not None and np.isfinite(min_correlation) and np.isfinite(g_r_refined):
                    if float(g_r_refined) < float(min_correlation):
                        continue

                lags = np.asarray(m['lag_ms'])
                r = np.asarray(m['r'])
                valid_mask = np.isfinite(lags)
                lags = lags[valid_mask]
                r = r[valid_mask]
                n_win = int(len(lags))
                if n_win == 0:
                    continue

                if use_interpolation:
                    median_signed = float(np.nanmedian(lags))
                    iqr = float(np.nanpercentile(lags, 75) - np.nanpercentile(lags, 25))
                    abs_lags = np.abs(lags)
                    median_abs = float(np.nanmedian(abs_lags))
                    mean_abs = float(np.nanmean(abs_lags))
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
                else:
                    sample_period_ms = 1000.0 / fs
                    lags_samples = np.round(lags / sample_period_ms).astype(int)
                    median_samples = int(np.round(np.median(lags_samples)))
                    median_signed = float(median_samples * sample_period_ms)
                    q75_samples = int(np.round(np.percentile(lags_samples, 75)))
                    q25_samples = int(np.round(np.percentile(lags_samples, 25)))
                    iqr = float((q75_samples - q25_samples) * sample_period_ms)
                    abs_lags_samples = np.abs(lags_samples)
                    median_abs_samples = int(np.round(np.median(abs_lags_samples)))
                    median_abs = float(median_abs_samples * sample_period_ms)
                    mean_abs_samples = np.mean(abs_lags_samples)
                    mean_abs = float(round(mean_abs_samples) * sample_period_ms)
                    if abs_lags_samples.size:
                        unique, counts = np.unique(abs_lags_samples, return_counts=True)
                        mode_samples = int(unique[np.argmax(counts)])
                        mode_abs = float(mode_samples * sample_period_ms)
                    else:
                        mode_abs = float('nan')
                    rms_samples = np.sqrt(np.mean(lags_samples**2))
                    rms = float(round(rms_samples) * sample_period_ms)

                abs_lags = np.abs(lags)
                results_rows.append({
                    'video_id': video_id,
                    'roi_a': a, 'roi_b': b,
                    'mode': mode,
                    'gt_hr_bpm': float(gt_bpm),
                    'hr_a': hr_roi.get(a, np.nan),
                    'hr_b': hr_roi.get(b, np.nan),
                    'err_a': err_roi.get(a, np.nan),
                    'err_b': err_roi.get(b, np.nan),
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
                    'pct_windows_|lag|<=1frame': float(np.mean(abs_lags <= one_frame_ms) * 100.0),
                    'pct_windows_|lag|>1frame': float(np.mean(abs_lags > one_frame_ms) * 100.0),
                    'pct_windows_|lag|>2frames': float(np.mean(abs_lags > 2.0 * one_frame_ms) * 100.0),
                    'pct_windows_r>=0.8': float(np.mean(r >= 0.8) * 100.0),
                })

    df_results = (
        pd.DataFrame(results_rows).sort_values(['video_id', 'roi_a', 'roi_b', 'mode'])
        if results_rows else pd.DataFrame()
    )
    df_failures = pd.DataFrame(failures_rows).sort_values(['video_id']) if failures_rows else pd.DataFrame()

    if not df_results.empty:
        df_summary = (
            df_results.groupby(['roi_a', 'roi_b', 'mode'])
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
                mean_pct_over_1frame=('pct_windows_|lag|>1frame', 'mean'),
                mean_pct_over_2frames=('pct_windows_|lag|>2frames', 'mean'),
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
        if not df_failures.empty:
            df_failures.to_csv(f"{base}_failures{ext}", index=False)

    return df_results, df_summary, df_failures


# ------------------------------
# 4b) UBFC CSV post-analysis helpers
# ------------------------------

def summarize_ubfc_global_lag_by_frames(
    csv_path: str = "ubfc_lag_metrics.csv",
    fs: float = 30.0,
    lag_col: str = "global_lag_ms_refined",
    by: Optional[List[str]] = None,
    use_abs: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Load `ubfc_lag_metrics.csv` (results-level rows) and compute the percentage of
    ROI-pairs whose global lag is:
    - 0 steps (rounded-to-nearest integer number of frames equals 0)
    - 1 or more steps (>= 1 frame step)
    - 2 or more steps (>= 2 frame steps)

    Notes:
    - Percentages are computed over rows (dataset, video_id, roi_a, roi_b, mode) present in the CSV.
    - Uses |lag| by default (use_abs=True).
    - Set `by=['dataset','mode']` (or similar) to get a breakdown; default is overall.
    """
    df = pd.read_csv(csv_path)
    if lag_col not in df.columns:
        raise ValueError(f"Column '{lag_col}' not found in {csv_path}. Available: {list(df.columns)}")
    if fs <= 0:
        raise ValueError("fs must be positive")

    lag = df[lag_col].astype(float)
    if dropna:
        mask = np.isfinite(lag.to_numpy())
        df = df.loc[mask].copy()
        lag = df[lag_col].astype(float)

    lag_ms = lag.abs() if use_abs else lag
    one_frame_ms = 1000.0 / float(fs)

    # Convert lag (ms) -> integer lag steps (frames) by rounding to nearest step.
    # Use floor(x + 0.5) instead of np.round to avoid banker's rounding at x.5.
    lag_steps = np.floor((lag_ms / one_frame_ms) + 0.5).astype(int)
    lag_steps = np.abs(lag_steps.to_numpy()) if hasattr(lag_steps, "to_numpy") else np.abs(lag_steps)
    df["_lag_steps"] = lag_steps

    df["_is_0step"] = (df["_lag_steps"] == 0)
    df["_ge_1step"] = (df["_lag_steps"] >= 1)
    df["_ge_2step"] = (df["_lag_steps"] >= 2)

    group_cols = by or []

    def _summ(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))
        if n == 0:
            return pd.Series(
                {
                    "n_rows": 0,
                    "pct_0step": float("nan"),
                    "pct_ge_1step": float("nan"),
                    "pct_ge_2step": float("nan"),
                }
            )
        return pd.Series(
            {
                "n_rows": n,
                "pct_0step": float(g["_is_0step"].mean() * 100.0),
                "pct_ge_1step": float(g["_ge_1step"].mean() * 100.0),
                "pct_ge_2step": float(g["_ge_2step"].mean() * 100.0),
            }
        )

    if group_cols:
        out = df.groupby(group_cols, dropna=False).apply(_summ).reset_index()
    else:
        out = _summ(df).to_frame().T

    return out


def summarize_ubfc_global_lag_by_frames_per_roi(
    csv_path: str = "ubfc_lag_metrics.csv",
    fs: float = 30.0,
    lag_col: str = "global_lag_ms_refined",
    use_abs: bool = True,
    dropna: bool = True,
    include_total_mode: bool = True,
    extra_group_cols: Optional[List[str]] = None,
    include_abs_median_lag: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Like `summarize_ubfc_global_lag_by_frames`, but returns ROI-centric summaries.

    Computes, from `ubfc_lag_metrics.csv`, the percentages of (ROI pairs) whose global lag is:
    - 0 steps (rounded-to-nearest integer number of frames equals 0)
    - 1 or more steps (>= 1 frame step)
    - 2 or more steps (>= 2 frame steps)

    Outputs (3 dataframes):
    1) overall_by_mode: one row per mode (filtered/unfiltered) [+ optional total]
    2) per_roi_total: one row per (mode, roi) summarizing all pairs that include that roi
    3) per_pair: one row per (mode, roi_a, roi_b) summarizing each unordered pair ONCE (no A/B duplicates)

    Notes:
    - The CSV stores unordered pairs in (roi_a, roi_b). This helper expands each row into two directed
      rows: (roi=roi_a, other_roi=roi_b) and (roi=roi_b, other_roi=roi_a), so per-roi stats are symmetric.
    - Percentages are over rows in the CSV (dataset, video_id, roi_a, roi_b, mode).
    - Set `extra_group_cols` (e.g. ['dataset']) to compute the same summaries per dataset as well.
    """
    df = pd.read_csv(csv_path)
    required = {"roi_a", "roi_b", "mode", lag_col}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}. Available: {list(df.columns)}")
    if fs <= 0:
        raise ValueError("fs must be positive")

    lag = df[lag_col].astype(float)
    if dropna:
        mask = np.isfinite(lag.to_numpy())
        df = df.loc[mask].copy()
        lag = df[lag_col].astype(float)

    # Keep both signed + absolute lag (ms) available for optional summaries.
    df["_signed_lag_ms"] = lag.astype(float)
    df["_abs_lag_ms"] = lag.astype(float).abs()

    lag_ms = lag.abs() if use_abs else lag
    one_frame_ms = 1000.0 / float(fs)

    # Convert lag (ms) -> integer lag steps (frames) by rounding to nearest step.
    lag_steps = np.floor((lag_ms / one_frame_ms) + 0.5).astype(int)
    lag_steps = np.abs(lag_steps.to_numpy()) if hasattr(lag_steps, "to_numpy") else np.abs(lag_steps)
    df["_lag_steps"] = lag_steps

    df["_is_0step"] = (df["_lag_steps"] == 0)
    df["_ge_1step"] = (df["_lag_steps"] >= 1)
    df["_ge_2step"] = (df["_lag_steps"] >= 2)

    extra = extra_group_cols or []

    def _summ(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))
        if n == 0:
            return pd.Series(
                {
                    "n_rows": 0,
                    "pct_0step": float("nan"),
                    "pct_ge_1step": float("nan"),
                    "pct_ge_2step": float("nan"),
                    **(
                        {
                            "median_abs_global_lag_ms": float("nan"),
                            "median_abs_global_lag_steps": float("nan"),
                        }
                        if include_abs_median_lag
                        else {}
                    ),
                }
            )
        out = {
            "n_rows": n,
            "pct_0step": float(g["_is_0step"].mean() * 100.0),
            "pct_ge_1step": float(g["_ge_1step"].mean() * 100.0),
            "pct_ge_2step": float(g["_ge_2step"].mean() * 100.0),
        }

        if include_abs_median_lag:
            # Median of absolute global lag (ms), and its discretized (rounded) frame-step version.
            out["median_abs_global_lag_ms"] = float(np.nanmedian(g["_abs_lag_ms"].to_numpy()))
            out["median_abs_global_lag_steps"] = float(np.nanmedian(g["_lag_steps"].to_numpy()))

        return pd.Series(out)

    # Overall by mode
    overall_by_mode = df.groupby(extra + ["mode"], dropna=False).apply(_summ).reset_index()

    # Expand to directed rows for ROI-centric summaries
    a = df.copy()
    a["roi"] = a["roi_a"]
    a["other_roi"] = a["roi_b"]
    b = df.copy()
    b["roi"] = b["roi_b"]
    b["other_roi"] = b["roi_a"]
    long_df = pd.concat([a, b], ignore_index=True)

    per_roi_total = long_df.groupby(extra + ["mode", "roi"], dropna=False).apply(_summ).reset_index()

    # Unordered (unique) pair table: enforce canonical ordering so (A,B) and (B,A) are the same
    pair_df = df.copy()
    pair_a = np.minimum(pair_df["roi_a"].astype(str), pair_df["roi_b"].astype(str))
    pair_b = np.maximum(pair_df["roi_a"].astype(str), pair_df["roi_b"].astype(str))
    pair_df["roi_a"] = pair_a
    pair_df["roi_b"] = pair_b
    per_pair = pair_df.groupby(extra + ["mode", "roi_a", "roi_b"], dropna=False).apply(_summ).reset_index()

    if include_total_mode:
        df_total = df.copy()
        df_total["mode"] = "total"
        overall_total = df_total.groupby(extra + ["mode"], dropna=False).apply(_summ).reset_index()
        overall_by_mode = pd.concat([overall_by_mode, overall_total], ignore_index=True)

        long_total = long_df.copy()
        long_total["mode"] = "total"
        per_roi_total = pd.concat(
            [per_roi_total, long_total.groupby(extra + ["mode", "roi"], dropna=False).apply(_summ).reset_index()],
            ignore_index=True,
        )

        pair_total = pair_df.copy()
        pair_total["mode"] = "total"
        per_pair = pd.concat(
            [per_pair, pair_total.groupby(extra + ["mode", "roi_a", "roi_b"], dropna=False).apply(_summ).reset_index()],
            ignore_index=True,
        )

    # Optional stable ordering
    sort_cols_overall = extra + ["mode"]
    sort_cols_roi_total = extra + ["mode", "roi"]
    sort_cols_pair = extra + ["mode", "roi_a", "roi_b"]
    overall_by_mode = overall_by_mode.sort_values(sort_cols_overall).reset_index(drop=True)
    per_roi_total = per_roi_total.sort_values(sort_cols_roi_total).reset_index(drop=True)
    per_pair = per_pair.sort_values(sort_cols_pair).reset_index(drop=True)

    return overall_by_mode, per_roi_total, per_pair


def recreate_lag_metrics_summary_from_results_csv(
    csv_path: str,
    group_cols: List[str],
    modes: Optional[List[str]] = None,
    global_lag_col: str = "global_lag_ms_refined",
    n_windows_col: str = "n_windows",
    local_mean_abs_col: str = "mean_abs_residual_lag_ms",
    local_rms_col: str = "rms_residual_lag_ms",
    local_signed_col: str = "median_residual_lag_ms",
) -> pd.DataFrame:
    """
    Recreate the `summary_df` (pair/mode aggregate) from a results-level CSV produced by:
    - `analyze_lossless_videos_lag_metrics_with_hr_filter(..., save_csv_path=...)`
    - `analyze_ubfc_lag_metrics_with_hr_filter(..., save_csv_path=...)`

    The recreated summary focuses on MEANS of ABSOLUTE lag metrics:
    - mean_abs_global_lag_ms: mean(|global_lag_col|)
    - mean_abs_residual_lag_ms: window-weighted mean of per-row mean(|local lag|) if available

    If `local_mean_abs_col` is missing but `local_rms_col` exists, it will estimate:
        mean(|X|) ≈ sqrt(2/pi) * rms(X)
    (reasonable if residuals are roughly zero-mean / symmetric).
    """
    df = pd.read_csv(csv_path)
    if modes is not None:
        if not isinstance(modes, (list, tuple)) or not len(modes):
            raise ValueError("modes must be None or a non-empty list like ['filtered']")
        if "mode" not in df.columns:
            raise ValueError(f"CSV {csv_path} has no 'mode' column, cannot filter modes.")
        df = df[df["mode"].isin(list(modes))].copy()

    required = set(group_cols) | {"video_id", global_lag_col}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}. Available: {list(df.columns)}")

    # Absolute global lag per row
    df["_abs_global_lag_ms"] = df[global_lag_col].astype(float).abs()
    df["_signed_global_lag_ms"] = df[global_lag_col].astype(float)

    has_windows = n_windows_col in df.columns
    if has_windows:
        df[n_windows_col] = df[n_windows_col].astype(float)
        df["_w"] = df[n_windows_col].where(np.isfinite(df[n_windows_col].to_numpy()), np.nan)
    else:
        df["_w"] = np.nan

    # Per-row mean(|local lag|) if available, otherwise estimate from RMS if possible
    if local_mean_abs_col in df.columns:
        df["_mean_abs_local_ms"] = df[local_mean_abs_col].astype(float)
        df["_mean_abs_local_ms_is_est"] = False
    elif local_rms_col in df.columns:
        # E[|X|] ≈ sqrt(2/pi) * RMS for symmetric zero-mean distributions
        df["_mean_abs_local_ms"] = df[local_rms_col].astype(float) * float(np.sqrt(2.0 / np.pi))
        df["_mean_abs_local_ms_is_est"] = True
    else:
        df["_mean_abs_local_ms"] = np.nan
        df["_mean_abs_local_ms_is_est"] = True

    # Per-row signed local lag (best-effort)
    # Prefer an explicit mean column if present, otherwise fall back to median residual lag.
    if "mean_residual_lag_ms" in df.columns:
        df["_mean_signed_local_ms"] = df["mean_residual_lag_ms"].astype(float)
        df["_mean_signed_local_ms_source"] = "mean_residual_lag_ms"
    elif local_signed_col in df.columns:
        df["_mean_signed_local_ms"] = df[local_signed_col].astype(float)
        df["_mean_signed_local_ms_source"] = local_signed_col
    else:
        df["_mean_signed_local_ms"] = np.nan
        df["_mean_signed_local_ms_source"] = "missing"

    # Per-row MEDIAN local lag (signed and abs) if available.
    # Many of our results CSVs already store:
    # - median_residual_lag_ms (signed)
    # - median_abs_residual_lag_ms (abs)
    if local_signed_col in df.columns:
        df["_median_signed_local_ms"] = df[local_signed_col].astype(float)
        df["_median_signed_local_ms_source"] = local_signed_col
    else:
        df["_median_signed_local_ms"] = np.nan
        df["_median_signed_local_ms_source"] = "missing"

    if "median_abs_residual_lag_ms" in df.columns:
        df["_median_abs_local_ms"] = df["median_abs_residual_lag_ms"].astype(float)
        df["_median_abs_local_ms_source"] = "median_abs_residual_lag_ms"
    elif local_signed_col in df.columns:
        # Fallback: abs(median signed) is not identical to median(abs(.)),
        # but it is a reasonable best-effort if the abs-median column is missing.
        df["_median_abs_local_ms"] = df[local_signed_col].astype(float).abs()
        df["_median_abs_local_ms_source"] = f"abs({local_signed_col})"
    else:
        df["_median_abs_local_ms"] = np.nan
        df["_median_abs_local_ms_source"] = "missing"

    # Weighted sum for local mean abs across windows (best-effort)
    if has_windows:
        df["_w_abs_local_sum"] = df["_mean_abs_local_ms"] * df["_w"]
        df["_w_signed_local_sum"] = df["_mean_signed_local_ms"] * df["_w"]
    else:
        df["_w_abs_local_sum"] = np.nan
        df["_w_signed_local_sum"] = np.nan

    # Aggregate
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            videos=("video_id", "nunique"),
            n_rows=("video_id", "size"),
            n_windows=(n_windows_col, "sum") if has_windows else ("video_id", "size"),
            mean_global_lag_ms=("_signed_global_lag_ms", "mean"),
            median_global_lag_ms=("_signed_global_lag_ms", "median"),
            mean_abs_global_lag_ms=("_abs_global_lag_ms", "mean"),
            median_abs_global_lag_ms=("_abs_global_lag_ms", "median"),
            # Local medians (typically already computed per-row by the analysis pipeline)
            median_residual_lag_ms=("_median_signed_local_ms", "median"),
            median_abs_residual_lag_ms=("_median_abs_local_ms", "median"),
            # Local: window-weighted mean if we have windows, else plain mean over rows
            w_abs_local_sum=("_w_abs_local_sum", "sum") if has_windows else ("video_id", "size"),
            w_signed_local_sum=("_w_signed_local_sum", "sum") if has_windows else ("video_id", "size"),
            w_sum=("_w", "sum") if has_windows else ("video_id", "size"),
            mean_abs_residual_lag_ms_rowmean=("_mean_abs_local_ms", "mean"),
            mean_signed_residual_lag_ms_rowmean=("_mean_signed_local_ms", "mean"),
            any_local_mean_is_est=("_mean_abs_local_ms_is_est", "max"),
            local_signed_source=("_mean_signed_local_ms_source", "first"),
            local_median_signed_source=("_median_signed_local_ms_source", "first"),
            local_median_abs_source=("_median_abs_local_ms_source", "first"),
        )
        .reset_index()
    )

    if has_windows:
        # weighted mean across windows
        agg["mean_abs_residual_lag_ms"] = agg["w_abs_local_sum"] / agg["w_sum"]
        agg["mean_signed_residual_lag_ms"] = agg["w_signed_local_sum"] / agg["w_sum"]
    else:
        agg["mean_abs_residual_lag_ms"] = agg["mean_abs_residual_lag_ms_rowmean"]
        agg["mean_signed_residual_lag_ms"] = agg["mean_signed_residual_lag_ms_rowmean"]

    # Alias to match the naming some notebooks/scripts expect.
    agg["median_signed_residual_lag_ms"] = agg["median_residual_lag_ms"]

    # Clean up helper cols
    agg = agg.drop(
        columns=[
            "w_abs_local_sum",
            "w_signed_local_sum",
            "w_sum",
            "mean_abs_residual_lag_ms_rowmean",
            "mean_signed_residual_lag_ms_rowmean",
        ]
    )

    # Stable ordering
    agg = agg.sort_values(group_cols).reset_index(drop=True)
    return agg


def recreate_lossless_summary_df_from_results_csv(
    csv_path: str,
    modes: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper for lossless videos results CSV.
    Expected grouping: (roi_a, roi_b, mode)
    """
    return recreate_lag_metrics_summary_from_results_csv(
        csv_path=csv_path,
        group_cols=["roi_a", "roi_b", "mode"],
        modes=modes,
        **kwargs,
    )


def recreate_ubfc_summary_df_from_results_csv(
    csv_path: str,
    modes: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper for UBFC results CSV.
    Expected grouping: (dataset, roi_a, roi_b, mode)
    """
    return recreate_lag_metrics_summary_from_results_csv(
        csv_path=csv_path,
        group_cols=["dataset", "roi_a", "roi_b", "mode"],
        modes=modes,
        **kwargs,
    )


def summarize_global_lag_abs_thresholds_per_roi(
    csv_path: str,
    lag_col: str = "global_lag_ms_refined",
    use_abs: bool = True,
    dropna: bool = True,
    thresholds_ms: Tuple[float, float, float] = (10.0, 16.66, 20.0),
    modes: Optional[List[str]] = None,
    include_total_mode: bool = True,
    extra_group_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ROI-centric summary that ignores "frame steps" and instead reports absolute-lag thresholds in ms.

    Given a results-level CSV (e.g. `ubfc_lag_metrics_interpolated.csv`,
    `videos60_lag_metrics_hrfilter_interpolated.csv`) with columns:
    - roi_a, roi_b: ROI pair identifiers
    - mode: 'filtered'/'unfiltered' (or similar)
    - lag_col: lag value in milliseconds
    this function computes the percentage of rows whose |lag| satisfies:
    - |lag| <= thresholds_ms[0]
    - |lag| >= thresholds_ms[1]
    - |lag| >= thresholds_ms[2]

    Outputs (3 dataframes):
    1) overall_by_mode: one row per mode (and optional total)
    2) per_roi_total: one row per (mode, roi) summarizing all pairs that include that roi
    3) per_pair: one row per (mode, roi_a, roi_b) summarizing each unordered pair ONCE (no A/B duplicates)

    Notes:
    - Uses absolute value by default (use_abs=True). If use_abs=False, thresholds apply to signed lags.
    - Set `extra_group_cols` (e.g. ['dataset']) to compute the same summaries per dataset as well.
    """
    df = pd.read_csv(csv_path)
    required = {"roi_a", "roi_b", "mode", lag_col}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}. Available: {list(df.columns)}")

    if modes is not None:
        if not isinstance(modes, (list, tuple)) or not len(modes):
            raise ValueError("modes must be None or a non-empty list like ['filtered']")
        df = df[df["mode"].isin(list(modes))].copy()

    if thresholds_ms is None or len(thresholds_ms) != 3:
        raise ValueError("thresholds_ms must be a 3-tuple like (10.0, 16.66, 20.0)")

    t_le, t_ge1, t_ge2 = (float(thresholds_ms[0]), float(thresholds_ms[1]), float(thresholds_ms[2]))

    lag = df[lag_col].astype(float)
    if dropna:
        mask = np.isfinite(lag.to_numpy())
        df = df.loc[mask].copy()
        lag = df[lag_col].astype(float)

    lag_ms = lag.abs() if use_abs else lag
    df["_abs_lag_ms"] = lag_ms

    df["_abs_le_t0"] = (df["_abs_lag_ms"] <= t_le)
    df["_abs_ge_t1"] = (df["_abs_lag_ms"] >= t_ge1)
    df["_abs_ge_t2"] = (df["_abs_lag_ms"] >= t_ge2)

    extra = extra_group_cols or []

    pct_le_col = f"pct_|lag|<={t_le}ms"
    pct_ge1_col = f"pct_|lag|>={t_ge1}ms"
    pct_ge2_col = f"pct_|lag|>={t_ge2}ms"

    def _summarize(group_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        out = (
            group_df.groupby(group_cols, dropna=False)
            .agg(
                n_rows=("_abs_lag_ms", "size"),
                **{
                    pct_le_col: ("_abs_le_t0", "mean"),
                    pct_ge1_col: ("_abs_ge_t1", "mean"),
                    pct_ge2_col: ("_abs_ge_t2", "mean"),
                },
            )
            .reset_index()
        )
        out[pct_le_col] = out[pct_le_col].astype(float) * 100.0
        out[pct_ge1_col] = out[pct_ge1_col].astype(float) * 100.0
        out[pct_ge2_col] = out[pct_ge2_col].astype(float) * 100.0
        out["n_rows"] = out["n_rows"].astype(int)
        return out

    # Overall by mode
    overall_by_mode = _summarize(df, extra + ["mode"])

    # Expand to directed rows for ROI-centric summaries
    a = df.copy()
    a["roi"] = a["roi_a"]
    a["other_roi"] = a["roi_b"]
    b = df.copy()
    b["roi"] = b["roi_b"]
    b["other_roi"] = b["roi_a"]
    long_df = pd.concat([a, b], ignore_index=True)
    per_roi_total = _summarize(long_df, extra + ["mode", "roi"])

    # Unordered (unique) pair table: enforce canonical ordering so (A,B) and (B,A) are the same
    pair_df = df.copy()
    pair_a = np.minimum(pair_df["roi_a"].astype(str), pair_df["roi_b"].astype(str))
    pair_b = np.maximum(pair_df["roi_a"].astype(str), pair_df["roi_b"].astype(str))
    pair_df["roi_a"] = pair_a
    pair_df["roi_b"] = pair_b
    per_pair = _summarize(pair_df, extra + ["mode", "roi_a", "roi_b"])

    if include_total_mode:
        df_total = df.copy()
        df_total["mode"] = "total"
        overall_total = _summarize(df_total, extra + ["mode"])
        overall_by_mode = pd.concat([overall_by_mode, overall_total], ignore_index=True)

        long_total = long_df.copy()
        long_total["mode"] = "total"
        per_roi_total = pd.concat(
            [per_roi_total, _summarize(long_total, extra + ["mode", "roi"])],
            ignore_index=True,
        )

        pair_total = pair_df.copy()
        pair_total["mode"] = "total"
        per_pair = pd.concat(
            [per_pair, _summarize(pair_total, extra + ["mode", "roi_a", "roi_b"])],
            ignore_index=True,
        )

    # Optional stable ordering
    overall_by_mode = overall_by_mode.sort_values(extra + ["mode"]).reset_index(drop=True)
    per_roi_total = per_roi_total.sort_values(extra + ["mode", "roi"]).reset_index(drop=True)
    per_pair = per_pair.sort_values(extra + ["mode", "roi_a", "roi_b"]).reset_index(drop=True)

    return overall_by_mode, per_roi_total, per_pair


def analyze_lossless_videos_lag_metrics_discrete(
    video_folder: str = '.',
    video_pattern: str = 'video_lossless_*.mkv',
    fs: float = 60,
    window_length: int = 60,
    start_time: int = 5,
    n_beats: int = 10,
    step_beats: int = 1,
    min_correlation: float = 0.5,
    prealign: bool = True,
    save_csv_path: Optional[str] = None,
    forehead: bool = True,
    cheeks: bool = True,
    under_nose: bool = False,
    chin: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discrete version: no interpolation, lags are integer multiples of sample period.
    Analyze local lossless videos (60 fps) with HR extracted from filenames.
    - Extracts POS per ROI and estimates HR per ROI.
    - Keeps only ROI pairs with correlation >= min_correlation.
    - Computes global discrete xcorr and sliding beat-anchored metrics for unfiltered/filtered signals.
    Returns results, summary, and failures dataframes.
    """
    import glob
    
    calc = ParametersCalculator()
    results_rows: List[dict] = []
    failures_rows: List[dict] = []
    
    # Find all matching video files
    video_files = glob.glob(os.path.join(video_folder, video_pattern))
    video_files = sorted([f for f in video_files if os.path.isfile(f)])
    
    if len(video_files) == 0:
        print(f"[WARN] No videos found matching pattern: {video_pattern}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(video_files)} videos to analyze (DISCRETE mode)")
    
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        
        # Parse HR from filename
        gt_bpm = _parse_hr_from_filename(video_filename)
        if gt_bpm is None:
            failures_rows.append({
                'video_id': video_id,
                'reason': f'Could not parse HR from filename: {video_filename}'
            })
            continue
        
        print(f"Processing {video_filename} (GT HR: {gt_bpm:.1f} bpm)")
        
        # POS by ROI
        try:
            pos_unf = _extract_pos_signals(
                video_path, fs, window_length, start_time,
                forehead=forehead, cheeks=cheeks, under_nose=under_nose, chin=chin
            )
        except Exception as e:
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': f'ROI extraction error: {e}'
            })
            continue
        
        if len(pos_unf) < 2:
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': 'Less than 2 ROIs'
            })
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
            failures_rows.append({
                'video_id': video_id,
                'gt_hr_bpm': gt_bpm,
                'reason': 'No HR per ROI'
            })
            continue
        
        # Filtered signals around each ROI's HR
        pos_filt = {roi: _bandpass_hr(pos_unf[roi], fs, hr_roi.get(roi, gt_bpm), half_width_hz=0.5) for roi in pos_unf.keys()}
        
        pairs = list(combinations(pos_unf.keys(), 2))
        
        # Sliding lag on all ROIs (DISCRETE version)
        lag_res = sliding_xcorr_lag_discrete(
            pos_signals=pos_unf,
            pos_signals_narrow=pos_filt,
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
        
        # Append rows (per pair and mode) - filter by correlation
        for (a, b), bundle in lag_res.items():
            for mode in ['unfiltered', 'filtered']:
                m = bundle.get(mode)
                if m is None or m['time_centers_s'].size == 0:
                    continue
                
                # Global discrete for this mode
                ga = pos_unf[a] if mode == 'unfiltered' else pos_filt[a]
                gb = pos_unf[b] if mode == 'unfiltered' else pos_filt[b]
                g_lag_ms_refined, g_r_refined = _global_xcorr_discrete(ga, gb, fs, max_lag_s=0.5)
                # Ensure global lag is discrete (should already be, but round to be safe)
                g_lag_ms_refined = _round_to_discrete_lag(g_lag_ms_refined, fs)
                
                # Filter by correlation threshold
                if g_r_refined < min_correlation:
                    continue
                
                lags = np.asarray(m['lag_ms'])
                r = np.asarray(m['r'])
                valid_mask = np.isfinite(lags)
                lags = lags[valid_mask]
                r = r[valid_mask]  # Filter r to match valid lags
                n_win = int(len(lags))
                if n_win == 0:
                    continue
                
                # For discrete mode: compute statistics on integer sample lags, then convert back to ms
                # This ensures all statistics are discrete values (multiples of sample period)
                sample_period_ms = 1000.0 / fs
                lags_samples = np.round(lags / sample_period_ms).astype(int)
                
                # Median: use the actual median sample lag, convert to ms
                median_samples = int(np.round(np.median(lags_samples)))
                median_signed = float(median_samples * sample_period_ms)
                
                # IQR: compute on sample lags, convert to ms
                q75_samples = int(np.round(np.percentile(lags_samples, 75)))
                q25_samples = int(np.round(np.percentile(lags_samples, 25)))
                iqr = float((q75_samples - q25_samples) * sample_period_ms)
                
                # Absolute lags
                abs_lags_samples = np.abs(lags_samples)
                median_abs_samples = int(np.round(np.median(abs_lags_samples)))
                median_abs = float(median_abs_samples * sample_period_ms)
                
                # Mean absolute: compute on samples, convert to ms
                mean_abs_samples = np.mean(abs_lags_samples)
                mean_abs = float(round(mean_abs_samples) * sample_period_ms)
                
                # Mode: find most common sample lag
                if abs_lags_samples.size:
                    unique, counts = np.unique(abs_lags_samples, return_counts=True)
                    mode_samples = int(unique[np.argmax(counts)])
                    mode_abs = float(mode_samples * sample_period_ms)
                else:
                    mode_abs = float('nan')
                
                # RMS: compute on sample lags, convert to ms
                rms_samples = np.sqrt(np.mean(lags_samples**2))
                rms = float(round(rms_samples) * sample_period_ms)
                
                results_rows.append({
                    'video_id': video_id,
                    'roi_a': a, 'roi_b': b,
                    'mode': mode,
                    'gt_hr_bpm': gt_bpm,
                    'hr_a': hr_roi.get(a, np.nan),
                    'hr_b': hr_roi.get(b, np.nan),
                    'err_a': err_roi.get(a, np.nan),
                    'err_b': err_roi.get(b, np.nan),
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
    
    df_results = pd.DataFrame(results_rows).sort_values(['video_id', 'roi_a', 'roi_b', 'mode'])
    df_failures = pd.DataFrame(failures_rows).sort_values(['video_id']) if failures_rows else pd.DataFrame()
    
    # Aggregated summary per pair, mode
    if not df_results.empty:
        df_summary = (
            df_results.groupby(['roi_a', 'roi_b', 'mode'])
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
        if not df_failures.empty:
            df_failures.to_csv(f"{base}_failures{ext}", index=False)
    
    return df_results, df_summary, df_failures


