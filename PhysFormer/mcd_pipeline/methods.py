"""Classical rPPG methods that turn an (T, 3) RGB trace into a 1-D pulsatile signal.

All take rgb of shape (T, 3) channel order R, G, B, fps. Return signal of shape (T,).
"""
import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(sig: np.ndarray, fps: float, lo: float = 0.7, hi: float = 3.5,
              order: int = 3) -> np.ndarray:
    nyq = fps / 2.0
    if hi >= nyq:
        hi = nyq * 0.99
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def green(rgb: np.ndarray, fps: float) -> np.ndarray:
    """G channel detrended + bandpassed. Crude but a useful floor reference."""
    g = rgb[:, 1].astype(np.float64)
    g = g - np.mean(g)
    return _bandpass(g, fps)


def chrom(rgb: np.ndarray, fps: float) -> np.ndarray:
    """De Haan & Jeanne 2013 — chrominance projection on normalized RGB."""
    rgb = rgb.astype(np.float64)
    mean = np.mean(rgb, axis=0, keepdims=True)
    norm = rgb / (mean + 1e-9)
    Xs = 3 * norm[:, 0] - 2 * norm[:, 1]
    Ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
    Xf = _bandpass(Xs, fps)
    Yf = _bandpass(Ys, fps)
    alpha = np.std(Xf) / (np.std(Yf) + 1e-9)
    return Xf - alpha * Yf


def pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Wang et al. 2017 — Plane-Orthogonal-to-Skin (POS), sliding window."""
    rgb = rgb.astype(np.float64)
    T = len(rgb)
    win = int(1.6 * fps)
    if T < win + 1:
        return np.zeros(T)
    H = np.zeros(T)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(win, T):
        Cn = rgb[n - win:n]
        mu = np.mean(Cn, axis=0, keepdims=True)
        Cn_norm = Cn / (mu + 1e-9)
        S = (P @ Cn_norm.T)
        h = S[0] + (np.std(S[0]) / (np.std(S[1]) + 1e-9)) * S[1]
        h = h - np.mean(h)
        H[n - win:n] += h
    return _bandpass(H, fps)


def pbv(rgb: np.ndarray, fps: float) -> np.ndarray:
    """De Haan & van Leest 2014 — blood volume pulse signature projection."""
    rgb = rgb.astype(np.float64)
    mean = np.mean(rgb, axis=0, keepdims=True)
    norm = rgb / (mean + 1e-9) - 1.0  # AC component
    # Empirical PBV signature for skin (mean across many subjects)
    pbv_sig = np.array([0.13, 0.79, 0.59])
    pbv_sig = pbv_sig / np.linalg.norm(pbv_sig)
    # Project AC onto PBV direction
    cov = norm.T @ norm
    w = np.linalg.solve(cov + 1e-6 * np.eye(3), pbv_sig)
    sig = norm @ w
    return _bandpass(sig, fps)


METHODS = {"green": green, "chrom": chrom, "pos": pos, "pbv": pbv}
