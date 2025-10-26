def POS_WANG(RGB, fs, normalize=False, detrend=True, bandpass=True, derivative=False, plot_steps=False):
    import numpy as np
    from scipy import signal
    import math
    from Tools.signalprocesser import SignalProcessor

    color = 'purple'
    WinSec = 1.6
    N = RGB.shape[1]  # RGB expected shape: (3, N)
    H = np.zeros(N, dtype=float)
    window_length = int(math.ceil(WinSec * fs))

    # Align with MATLAB windowing: first valid n = window_length, include endpoint via slice m:n
    for n in range(window_length, N + 1):
        m = n - window_length
        # Temporal normalization: per-channel mean over the window (mean across time)
        denom = np.mean(RGB[:, m:n], axis=1, keepdims=True)  # shape (3, 1)
        Cn = RGB[:, m:n] / (denom + 1e-12)  # shape (3, window_length)

        # Projection and tuning
        S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float) @ Cn  # shape (2, window_length)
        s0, s1 = S[0, :], S[1, :]
        ratio = np.std(s0) / (np.std(s1) + 1e-12)
        h = s0 + ratio * s1
        h = h - np.mean(h)

        # Overlap-add
        H[m:n] += h

    BVP = H

    if normalize:
        BVP = SignalProcessor.normalize(BVP, fs=fs, plot=plot_steps, color=color)

    if detrend:
        # keep interface with your detrend (expects column vector)
        BVP = SignalProcessor.detrend(np.asarray(BVP)[None, :].T, 100, fs=fs, plot=plot_steps, color=color)
        BVP = np.asarray(BVP.T)[0]

    if bandpass:
        # Match reference cutoffs 0.7–2.5 Hz
        b, a = signal.butter(1, [0.5 / (fs / 2.0), 5 / (fs / 2.0)], btype='bandpass')
        BVP = signal.filtfilt(b, a, BVP.astype(np.double))

    if derivative:
        BVP = SignalProcessor.derivativeFilter(BVP, fs=fs, plot=plot_steps, color=color)

    return BVP