"""Bridge to edge-rppg-health: extract POS rPPG + HR from the preprocessed MCD frames.

We skip edge-rppg's MediaPipe-based face ROI extraction (mp.solutions.face_mesh is
broken for our Apple Silicon / numpy 2.x / Python 3.12 stack) and reuse the
already-face-cropped 128x128 PNGs written by mcd_pipeline.preprocess. Forehead mean
RGB per frame feeds POS_WANG, then ParametersCalculator gives HR.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path("/Users/pedrolucasbarrera/edge-rppg-health")
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

# numpy 2.x removed np.mat; edge-rppg still uses it
if not hasattr(np, "mat"):
    np.mat = np.asmatrix  # type: ignore[attr-defined]

from Methods.pos import POS_WANG  # noqa: E402
from Tools.ParametersCalculator import ParametersCalculator  # noqa: E402


def rgb_series_from_frames(video_id: str, frames_root: str = "data/VIPL_frames",
                           forehead_only: bool = True) -> np.ndarray:
    """Mean RGB per frame from the 128x128 preprocessed face crops.

    With forehead_only=True we take the top strip (~forehead region) of each crop;
    that region tends to have less motion + occlusion than cheeks for rPPG.

    Returns: (3, N) array in R/G/B order — shape POS_WANG expects.
    """
    frames = sorted(Path(frames_root, video_id).glob("image_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {frames_root}/{video_id}")

    r, g, b = [], [], []
    for f in frames:
        img = cv2.imread(str(f))  # BGR uint8, 128x128
        if forehead_only:
            img = img[:42, 16:112]  # top 1/3, ignore lateral edges (hair/ears)
        b_mean, g_mean, r_mean = cv2.mean(img)[:3]
        r.append(r_mean); g.append(g_mean); b.append(b_mean)
    return np.asarray([r, g, b], dtype=float)


def pos_rppg(video_id: str, fps: float, frames_root: str = "data/VIPL_frames",
             forehead_only: bool = True) -> np.ndarray:
    """POS rPPG signal (same framerate as the video)."""
    rgb = rgb_series_from_frames(video_id, frames_root, forehead_only)
    return POS_WANG(rgb, fps)


def hr_from_pos(signal: np.ndarray, fs: float) -> float:
    """HR (bpm) via ParametersCalculator. Falls back to Welch-FFT on failure."""
    calc = ParametersCalculator()
    try:
        return float(calc.GetParameters(signal, int(round(fs)))["HR"])
    except Exception:
        from scipy.signal import welch
        f, p = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 8)))
        m = (f >= 0.7) & (f <= 3.5)
        return 60.0 * f[m][np.argmax(p[m])] if m.any() else 0.0


# --- HRV -----------------------------------------------------------------

HRV_KEYS = [
    "HR", "SDNN_ms", "RMSSD_ms", "dif50", "Pcv", "P_NMASD",
    "LF/HF", "Power LF (ms2)", "Power HF (ms2)", "stress",
]


def _hrv_from_ibi(ibi: np.ndarray, fs_interpolation: int = 4) -> dict:
    """Compute HRV metrics from an inter-beat-interval (seconds) series.

    Mirrors what ParametersCalculator.GetParameters does after peak detection,
    which lets us feed ECG-derived IBIs (from Pan-Tompkins) into the same code.
    """
    from Tools.signalprocesser import SignalProcessor
    calc = ParametersCalculator()

    ibi_clean, peaks_aux, _ = SignalProcessor.process_serie_IBI_absolute(
        list(ibi), peaks=[], int_min=0.3, int_max=2)
    ibi_clean, peaks_aux, _ = SignalProcessor.process_serie_IBI_relative(
        ibi_clean, peaks_aux, i_del=[], k_min=1/2.5, k_max=2.5)
    ibi_clean = np.asarray(ibi_clean, dtype=float)
    if len(ibi_clean) < 4:
        return {k: float("nan") for k in HRV_KEYS}

    t_ibi = np.cumsum(ibi_clean)
    try:
        interp_t, interp_s = SignalProcessor.cubic_interpolation(t_ibi, ibi_clean, fs_interpolation)
    except Exception:
        interp_t, interp_s = SignalProcessor.linear_interpolation(t_ibi, ibi_clean, fs_interpolation)

    results, *_ = calc.HeartRateVariability(ibi_clean, interp_s, fs=fs_interpolation)
    # HeartRateVariability returns SDNN/RMSSD in seconds (IBI units); convert to ms
    results["SDNN_ms"] = results.get("SDNN", float("nan")) * 1000.0
    results["RMSSD_ms"] = results.get("RMSSD", float("nan")) * 1000.0
    results["HR"] = 60.0 / float(np.mean(ibi_clean))
    try:
        results["stress"] = float(calc.GetStress(ibi_clean))
    except Exception:
        results["stress"] = float("nan")
    return {k: float(results.get(k, float("nan"))) for k in HRV_KEYS}


def hrv_from_rppg(signal: np.ndarray, fs: float) -> dict:
    """HRV from an rPPG-style signal (POS, PhysFormer, contact PPG).

    Replicates ParametersCalculator.GetParameters' two-pass peak pipeline
    (raw → rough HR → bandpass around HR → re-detect peaks → refine in raw)
    but feeds the correct interpolated IBI *series* to HeartRateVariability
    (GetParameters passes the time array instead, so LF/HF comes out bogus).
    """
    from Tools.signalprocesser import SignalProcessor
    calc = ParametersCalculator()
    fs_int = int(round(fs))

    try:
        ibi_rough, _, peaks_rough = calc.GetIBISeries(signal, fs_int)
        if len(peaks_rough) < 3:
            return {k: float("nan") for k in HRV_KEYS}
        hr_est = calc.ObtainHeartRate(signal, ibi_rough, fs_int,
                                      method="two_peaks_periodogram")
        hr_hz = hr_est / 60.0
        signal_bp = SignalProcessor.bandpass(signal, fs_int,
                                             lowcut=max(0.5, hr_hz - 0.5),
                                             highcut=hr_hz + 0.5)
        _, _, peaks_bp = calc.GetIBISeries(signal_bp, fs_int)
        win = int(0.2 * fs_int)
        peaks_refined = [
            max(0, p - win) + int(np.argmax(signal[max(0, p - win):min(len(signal), p + win + 1)]))
            for p in peaks_bp
        ]
        peaks_refined = np.asarray(sorted(set(peaks_refined)))
        if len(peaks_refined) < 3:
            return {k: float("nan") for k in HRV_KEYS}
        ibi = np.diff(peaks_refined) / fs
        return _hrv_from_ibi(ibi)
    except Exception:
        return {k: float("nan") for k in HRV_KEYS}


def hrv_from_ecg(ecg: np.ndarray, fs: float) -> dict:
    """HRV from an ECG signal using Pan-Tompkins R-peak detection."""
    from mcd_pipeline.utils import R_peaks
    peaks = R_peaks(ecg, fs)
    if len(peaks) < 3:
        return {k: float("nan") for k in HRV_KEYS}
    ibi = np.diff(peaks) / fs
    return _hrv_from_ibi(ibi)
