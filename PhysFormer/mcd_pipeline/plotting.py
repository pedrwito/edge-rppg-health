"""Helpers for loading/plotting MCD rPPG + ECG signals. Used by the notebook."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

from mcd_pipeline.utils import R_peaks

DEFAULT_INDEX = "data/MCD_test.txt"
DEFAULT_ECG_DIR = "data/mcd_raw/ecg"
DEFAULT_PPG_DIR = "data/mcd_raw/ppg_sync"
DEFAULT_RPPG_DIR = "Inference_MCD_PhysFormer"


def load_index(path: str = DEFAULT_INDEX) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=" ", header=None,
                       names=["video_id", "n_clips", "fps", "hr_gt"])


def load_ecg(video_id: str, ecg_dir: str = DEFAULT_ECG_DIR):
    parts = video_id.split("_")
    subject, state = parts[0], parts[-1]
    with open(Path(ecg_dir) / f"{subject}_{state}.json") as f:
        d = json.load(f)
    fs = float(d["frequency"])
    lead = next((c for c in d["data"] if c.get("title") == "I"), d["data"][0])
    return np.asarray(lead["values"], dtype=float), fs


def load_rppg(video_id: str, rppg_dir: str = DEFAULT_RPPG_DIR) -> np.ndarray:
    return np.load(Path(rppg_dir) / f"{video_id}.npy")


def load_real_ppg(video_id: str, ppg_dir: str = DEFAULT_PPG_DIR,
                  fps: float | None = None,
                  index_path: str = DEFAULT_INDEX) -> tuple[np.ndarray, float]:
    """Contact-sensor PPG (ppg_sync/<video>.txt) — synchronised to the video framerate.

    One sample per video frame (col 0 = amplitude, col 1 = per-sample timing).
    If fps is omitted, read it from the MCD index.
    """
    path = Path(ppg_dir) / f"{video_id}.txt"
    arr = np.loadtxt(path)
    if arr.ndim == 2:
        arr = arr[:, 0]
    if fps is None:
        idx = load_index(index_path)
        fps = float(idx[idx.video_id == video_id].iloc[0]["fps"])
    return arr.astype(float), fps


def bandpass(sig: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.5) -> np.ndarray:
    b, a = butter(3, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, sig - np.mean(sig))


def hr_fft(sig: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.5) -> float:
    """Whole-window HR: Welch PSD peak in [lo, hi] Hz."""
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    m = (f >= lo) & (f <= hi)
    return 60.0 * f[m][np.argmax(p[m])] if m.any() else 0.0


def hr_per_clip_median(rppg: np.ndarray, fs: float, clip_len: int = 160,
                      lo: float = 0.7, hi: float = 3.5) -> tuple[float, np.ndarray]:
    """Per-clip FFT HR, then median. More robust to waveform artifacts than whole-FFT."""
    n_clips = len(rppg) // clip_len
    hrs = []
    for c in range(n_clips):
        seg = rppg[c * clip_len:(c + 1) * clip_len]
        seg = bandpass(seg, fs, lo, hi)
        hrs.append(hr_fft(seg, fs, lo, hi))
    arr = np.array([h for h in hrs if h > 0])
    return (float(np.median(arr)) if len(arr) else 0.0), arr


def hr_from_ecg(ecg: np.ndarray, fs: float) -> tuple[float, np.ndarray]:
    """Pan-Tompkins HR (bpm) + R-peak indices."""
    peaks = R_peaks(ecg, fs)
    if len(peaks) < 2:
        return 0.0, peaks
    return float(60.0 / np.mean(np.diff(peaks) / fs)), peaks


def compute_hrv_table(*, window_s: float | None = 30.0,
                      index_path: str = DEFAULT_INDEX,
                      rppg_dir: str = DEFAULT_RPPG_DIR,
                      ppg_dir: str = DEFAULT_PPG_DIR,
                      ecg_dir: str = DEFAULT_ECG_DIR) -> pd.DataFrame:
    """Compute HRV metrics for every video in the index across 4 sources:
    contact PPG (truth), POS (my method), PhysFormer, ECG (per subject+state).

    Returns a long-format DataFrame with one row per (video_id, method).
    """
    from mcd_pipeline.my_method import (
        pos_rppg, hrv_from_rppg, hrv_from_ecg, HRV_KEYS,
    )

    idx = load_index(index_path)
    rows = []
    ecg_cache: dict[str, tuple[np.ndarray, float]] = {}

    for _, r in idx.iterrows():
        video_id = r["video_id"]
        fps = float(r["fps"])
        parts = video_id.split("_")
        subject, camera, state = parts[0], parts[1], parts[-1]

        # Crop window relative to each source's native fs
        def _crop(sig: np.ndarray, fs: float) -> np.ndarray:
            if window_s is None:
                return sig
            return sig[: int(window_s * fs)]

        try:
            ppg_real, ppg_fs = load_real_ppg(video_id, ppg_dir, fps=fps)
            hrv_ppg = hrv_from_rppg(_crop(ppg_real, ppg_fs), ppg_fs)
        except Exception:
            hrv_ppg = {k: float("nan") for k in HRV_KEYS}

        try:
            pos = pos_rppg(video_id, fps)
            hrv_pos = hrv_from_rppg(_crop(pos, fps), fps)
        except Exception:
            hrv_pos = {k: float("nan") for k in HRV_KEYS}

        try:
            phys = load_rppg(video_id, rppg_dir)
            hrv_phys = hrv_from_rppg(_crop(phys, fps), fps)
        except Exception:
            hrv_phys = {k: float("nan") for k in HRV_KEYS}

        ecg_key = f"{subject}_{state}"
        if ecg_key not in ecg_cache:
            ecg_cache[ecg_key] = load_ecg(video_id, ecg_dir)
        ecg, ecg_fs = ecg_cache[ecg_key]
        try:
            hrv_ecg = hrv_from_ecg(_crop(ecg, ecg_fs), ecg_fs)
        except Exception:
            hrv_ecg = {k: float("nan") for k in HRV_KEYS}

        for method, hrv in (("ppg_truth", hrv_ppg), ("pos", hrv_pos),
                            ("physformer", hrv_phys), ("ecg", hrv_ecg)):
            rows.append({
                "video_id": video_id, "subject": subject, "camera": camera,
                "state": state, "method": method, **hrv,
            })
    return pd.DataFrame(rows)


def compute_hrv_mae(df: pd.DataFrame,
                    ref_method: str = "ecg",
                    methods: tuple[str, ...] = ("ppg_truth", "pos", "physformer")
                    ) -> pd.DataFrame:
    """For each method, mean absolute error of every HRV metric vs `ref_method`
    (default: ECG) across all videos. Returns a method × metric DataFrame.
    """
    non_metric = {"video_id", "subject", "camera", "state", "method"}
    metrics = [c for c in df.columns if c not in non_metric]
    ref = df[df.method == ref_method].set_index("video_id")[metrics]
    rows = []
    for m in methods:
        sub = df[df.method == m].set_index("video_id")[metrics].reindex(ref.index)
        mae = (sub - ref).abs().mean(numeric_only=True)
        rows.append({"method": m, **mae.to_dict()})
    return pd.DataFrame(rows).set_index("method")


def plot_hrv_comparison(df: pd.DataFrame,
                        metrics: list[str] = ("HR", "SDNN_ms", "RMSSD_ms", "dif50", "LF/HF", "stress"),
                        savepath: str | None = None):
    """Grouped bar chart: one subplot per metric, videos on x-axis, colour = method."""
    method_colors = {"ecg": "black", "ppg_truth": "gray",
                     "pos": "darkorange", "physformer": "steelblue"}
    methods = ["ecg", "ppg_truth", "pos", "physformer"]
    videos = df["video_id"].drop_duplicates().tolist()
    x = np.arange(len(videos))
    width = 0.2

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 1.5 * len(videos)), 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for i, m in enumerate(methods):
            sub = df[df.method == m].set_index("video_id").reindex(videos)
            ax.bar(x + (i - 1.5) * width, sub[metric].values, width=width,
                   label=m, color=method_colors[m], edgecolor="k", linewidth=0.3)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="upper right", ncol=4)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([v.replace("1020_", "") for v in videos], rotation=25, ha="right")
    fig.suptitle("HRV comparison across methods & cameras", fontsize=12)
    plt.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=120)
    return fig


def _znorm(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    return x / (np.std(x) + 1e-8)


def plot_three_rppg_signals(video_id: str, *, window_s: float = 30.0,
                            bandpass_lo: float = 0.7, bandpass_hi: float = 3.5,
                            index_path: str = DEFAULT_INDEX,
                            rppg_dir: str = DEFAULT_RPPG_DIR,
                            ppg_dir: str = DEFAULT_PPG_DIR,
                            ecg_dir: str = DEFAULT_ECG_DIR,
                            savepath: str | None = None):
    """3-panel comparison: contact PPG (truth), POS rPPG (edge-rppg), PhysFormer rPPG.

    Each panel shows a bandpassed + z-normalised signal over the overlap window.
    Titles include per-method HR; POS HR uses ParametersCalculator, PhysFormer HR
    uses per-clip-median, and truth HR uses ECG Pan-Tompkins.
    """
    from mcd_pipeline.my_method import pos_rppg, hr_from_pos

    idx = load_index(index_path)
    row = idx[idx.video_id == video_id].iloc[0]
    fps = float(row["fps"])

    # --- Load all three sources (ppg_sync is at video fps, one sample per frame) ---
    rppg_phys = load_rppg(video_id, rppg_dir)
    ppg_real, ppg_fs = load_real_ppg(video_id, ppg_dir, fps=fps)
    rppg_pos = pos_rppg(video_id, fps)

    # --- HR ground truth from ECG (comparable reference) ---
    ecg, ecg_fs = load_ecg(video_id, ecg_dir)

    # --- Crop everything to the overlap window ---
    max_win = min(len(ppg_real) / ppg_fs, len(rppg_phys) / fps,
                  len(rppg_pos) / fps, len(ecg) / ecg_fs)
    win = min(window_s, max_win)
    ppg_real = ppg_real[: int(win * ppg_fs)]
    rppg_pos = rppg_pos[: int(win * fps)]
    rppg_phys = rppg_phys[: int(win * fps)]
    ecg_w = ecg[: int(win * ecg_fs)]

    # --- Bandpass + normalise for display ---
    ppg_real_f = _znorm(bandpass(ppg_real, ppg_fs, bandpass_lo, bandpass_hi))
    rppg_pos_f = _znorm(bandpass(rppg_pos, fps, bandpass_lo, bandpass_hi))
    rppg_phys_f = _znorm(bandpass(rppg_phys, fps, bandpass_lo, bandpass_hi))

    # --- HRs ---
    hr_ecg, _ = hr_from_ecg(ecg_w, ecg_fs)
    hr_ppg = hr_fft(ppg_real_f, ppg_fs, bandpass_lo, bandpass_hi)
    hr_pos = hr_from_pos(rppg_pos, fps)
    hr_phys, _ = hr_per_clip_median(rppg_phys, fps, lo=bandpass_lo, hi=bandpass_hi)

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t_ppg = np.arange(len(ppg_real_f)) / ppg_fs
    t_pos = np.arange(len(rppg_pos_f)) / fps
    t_phys = np.arange(len(rppg_phys_f)) / fps

    axes[0].plot(t_ppg, ppg_real_f, color="black", lw=0.8)
    axes[0].set_ylabel("Contact PPG\n(z-norm)")
    axes[0].set_title(f"Ground truth — contact PPG   HR(FFT)={hr_ppg:.1f}   ECG HR={hr_ecg:.1f} bpm")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_pos, rppg_pos_f, color="darkorange", lw=1.0)
    axes[1].set_ylabel("POS rPPG\n(z-norm)")
    axes[1].set_title(f"My method (POS + ParametersCalculator)   HR={hr_pos:.1f} bpm   "
                      f"|MAE vs ECG={abs(hr_pos - hr_ecg):.1f}|")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_phys, rppg_phys_f, color="steelblue", lw=1.0)
    axes[2].set_ylabel("PhysFormer rPPG\n(z-norm)")
    axes[2].set_title(f"PhysFormer (VIPL-HR fold 1)   HR(per-clip median)={hr_phys:.1f} bpm   "
                      f"|MAE vs ECG={abs(hr_phys - hr_ecg):.1f}|")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t_ppg, ppg_real_f, color="black", lw=1.0, alpha=0.9, label="Contact PPG (truth)")
    axes[3].plot(t_pos, rppg_pos_f, color="darkorange", lw=1.0, alpha=0.8, label="POS (my method)")
    axes[3].plot(t_phys, rppg_phys_f, color="steelblue", lw=1.0, alpha=0.8, label="PhysFormer")
    axes[3].set_ylabel("All three\n(z-norm)")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title("Superimposed")
    axes[3].legend(loc="upper right", ncol=3)
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"{video_id}   fps={fps:.1f}   window={win:.1f}s", fontsize=12)
    plt.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=120)
    return fig, {"hr_ppg_fft": hr_ppg, "hr_pos": hr_pos,
                 "hr_physformer": hr_phys, "hr_ecg": hr_ecg, "window_s": win}


def plot_rppg_vs_ecg(video_id: str, *, window_s: float | None = 30.0,
                     bandpass_lo: float = 0.7, bandpass_hi: float = 3.5,
                     index_path: str = DEFAULT_INDEX,
                     rppg_dir: str = DEFAULT_RPPG_DIR,
                     ecg_dir: str = DEFAULT_ECG_DIR,
                     savepath: str | None = None):
    """2-panel plot: predicted rPPG (top) + ECG with R-peaks (bottom)."""
    idx = load_index(index_path)
    row = idx[idx.video_id == video_id].iloc[0]
    fps = float(row["fps"])

    rppg = load_rppg(video_id, rppg_dir)
    ecg, ecg_fs = load_ecg(video_id, ecg_dir)

    max_win = min(len(ecg) / ecg_fs, len(rppg) / fps)
    win = max_win if window_s is None else min(window_s, max_win)
    rppg = rppg[: int(win * fps)]
    ecg = ecg[: int(win * ecg_fs)]

    rppg_f = bandpass(rppg, fps, bandpass_lo, bandpass_hi)
    hr_whole = hr_fft(rppg_f, fps, bandpass_lo, bandpass_hi)
    hr_med, _ = hr_per_clip_median(rppg, fps, lo=bandpass_lo, hi=bandpass_hi)
    hr_ecg, peaks = hr_from_ecg(ecg, ecg_fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t_r = np.arange(len(rppg_f)) / fps
    axes[0].plot(t_r, rppg_f, color="steelblue", lw=1.0)
    axes[0].set_ylabel("rPPG (bandpassed)")
    axes[0].set_title(
        f"{video_id}   fps={fps:.1f}   window={win:.1f}s\n"
        f"PhysFormer HR (whole-FFT)={hr_whole:.1f}   "
        f"PhysFormer HR (per-clip median)={hr_med:.1f}   "
        f"ECG HR={hr_ecg:.1f} bpm")
    axes[0].grid(alpha=0.3)

    t_e = np.arange(len(ecg)) / ecg_fs
    axes[1].plot(t_e, ecg, color="black", lw=0.6)
    axes[1].plot(t_e[peaks], ecg[peaks], "ro", ms=3, label=f"R-peaks (n={len(peaks)})")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("ECG lead I")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=120)
    return fig, {"hr_whole": hr_whole, "hr_per_clip_median": hr_med, "hr_ecg": hr_ecg,
                 "window_s": win}
