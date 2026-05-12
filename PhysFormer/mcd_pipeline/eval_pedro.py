"""Compute HR + HRV parameters for every (video, method) using Pedro's
codigo pedro rppg classes:
  - Methods.pos.POS_WANG, Methods.chrom.CHROM  (rPPG signal extraction)
  - Tools.ParametersCalculator.ParametersCalculator.GetParameters
    (HR, HRV time/freq metrics, stress)

Methods evaluated per video:
  - GT ECG (lead I, 500 Hz)
  - GT synced PPG (~30 Hz, per camera)
  - PhysFormer reconstructed PPG (Inference_MCD_PhysFormer_stream/<id>.npy)
  - POS_WANG on rgb_traces_mp/<id>.npz   (Pedro's POS)
  - CHROM on rgb_traces_mp/<id>.npz       (Pedro's CHROM)

Output: data/eval_pedro.csv (one row per video; wide format).
        data/eval_pedro_summary.csv (per camera × method × param: MAE + GT typical value).
"""
import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Pedro's repo lives in this subdir; add it to sys.path so the relative imports
# inside Methods/ and Tools/ resolve.
PEDRO_DIR = Path(__file__).resolve().parent.parent / "codigo pedro rppg"


def _ensure_pedro_on_path():
    s = str(PEDRO_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
    # Pedro's code uses np.mat (removed in NumPy 2.0). Restore the alias.
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix


# Constant list of HRV/HR keys we care about (for orderly columns + summaries).
PARAM_KEYS = [
    "HR", "RR",
    "Pcv", "P_NMASD", "SDNN", "RMSSD", "dif50",
    "I_shan", "I_CSamEn", "p2", "p3",
    "std_0_nom", "std_ed_norm",
    "Power VLF (ms2)", "Power LF (ms2)", "Power HF (ms2)", "Power Total (ms2)",
    "LF/HF", "Fraction LF (nu)", "Fraction HF (nu)",
    "Peak VLF (Hz)", "Peak LF (Hz)", "Peak HF (Hz)",
    "stress",
]

METHODS = ["physformer", "pos", "chrom", "ppg_sync", "ecg"]


def safe_get_params(sig: np.ndarray, fs: float, calculator) -> dict:
    """rPPG path: run Pedro's GetParameters (linear interp avoids cubic-dup bug).

    Pedro's stack is built for rPPG-shaped signals — a smooth pulsatile wave
    with one peak per beat at amplitude ~ mean. PhysFormer / POS / CHROM /
    PPG-sync all fit. Use ecg_get_params() for raw ECG instead.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = calculator.GetParameters(sig.astype(np.float64), fs=fs,
                                              interpolation="linear")
        return {k: float(params[k]) if k in params and np.isfinite(params[k])
                else np.nan for k in PARAM_KEYS}
    except Exception:
        return {k: np.nan for k in PARAM_KEYS}


def ecg_get_params(ecg: np.ndarray, fs: float, calculator) -> dict:
    """Reuse Pedro's HRV machinery on R-peaks detected by Pan-Tompkins.

    GetParameters peak-detector assumes PPG-shape; ECG QRS spikes break it.
    Workflow: Pan-Tompkins-flavored R-peaks → IBI series → Pedro's
    HeartRateVariability + GetStress.
    """
    from scipy.signal import butter, filtfilt, find_peaks
    from mcd_pipeline.hrv import find_ecg_r_peaks
    out = {k: np.nan for k in PARAM_KEYS}
    try:
        peaks = find_ecg_r_peaks(ecg, fs)
        if len(peaks) < 5:
            return out
        rr_s = np.diff(peaks) / fs
        rr_s = rr_s[(rr_s >= 0.30) & (rr_s <= 2.0)]
        if len(rr_s) < 5:
            return out
        out["HR"] = float(60.0 / np.mean(rr_s))
        # Build interpolated series at 4 Hz for frequency-domain HRV
        from Tools.signalprocesser import SignalProcessor
        t_rr = np.cumsum(rr_s)
        t_uniform, rr_uniform = SignalProcessor.linear_interpolation(t_rr, rr_s, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hrv, _, _ = calculator.HeartRateVariability(rr_s, rr_uniform, fs=4)
        for k, v in hrv.items():
            if k in out and v is not None and np.isfinite(v):
                out[k] = float(v)
        try:
            out["stress"] = float(calculator.GetStress(rr_s))
        except Exception:
            pass
    except Exception:
        pass
    return out


def load_ppg_sync(path: Path, video_fps: float) -> np.ndarray:
    """ppg_sync is sampled once per video frame (1 sample : 1 frame), so fs == video fps."""
    return np.loadtxt(path)[:, 0].astype(np.float64)


def load_ecg_lead(path: Path) -> tuple[np.ndarray, float]:
    with open(path) as f:
        d = json.load(f)
    fs = float(d["frequency"])
    lead = next((c for c in d["data"] if c.get("title") == "I"), d["data"][0])
    return np.asarray(lead["values"], dtype=np.float64), fs


def process_one(args: tuple) -> dict:
    (video_id, fps, paths) = args
    _ensure_pedro_on_path()
    from Methods.pos import POS_WANG
    from Methods.chrom import CHROM
    from Tools.ParametersCalculator import ParametersCalculator
    calc = ParametersCalculator()

    parts = video_id.split("_")
    subject, camera, state = parts[0], parts[1], parts[-1]
    out: dict = {"video_id": video_id, "subject": subject, "camera": camera,
                 "state": state, "fps": fps}

    # GT — ECG (lead I, 500 Hz). Use Pan-Tompkins R-peaks → Pedro's HRV.
    ecg_path = paths["ecg_root"] / f"{subject}_{state}.json"
    if ecg_path.exists():
        try:
            sig, fs = load_ecg_lead(ecg_path)
            for k, v in ecg_get_params(sig, fs, calc).items():
                out[f"ecg_{k}"] = v
        except Exception as e:
            out["ecg_err"] = str(e)[:80]

    # GT — synced PPG (1 sample per video frame → fs == video fps)
    ppg_path = paths["ppg_sync_root"] / f"{video_id}.txt"
    if ppg_path.exists():
        try:
            sig = load_ppg_sync(ppg_path, fps)
            for k, v in safe_get_params(sig, fps, calc).items():
                out[f"ppg_sync_{k}"] = v
        except Exception as e:
            out["ppg_sync_err"] = str(e)[:80]

    # PhysFormer reconstructed PPG (use video fps)
    pf_path = paths["physformer_root"] / f"{video_id}.npy"
    if pf_path.exists():
        try:
            sig = np.load(pf_path).astype(np.float64)
            for k, v in safe_get_params(sig, fps, calc).items():
                out[f"physformer_{k}"] = v
        except Exception as e:
            out["physformer_err"] = str(e)[:80]

    # POS + CHROM on RGB trace (Pedro's implementations expect channels-first)
    rgb_path = paths["rgb_root"] / f"{video_id}.npz"
    if rgb_path.exists():
        try:
            d = np.load(rgb_path)
            rgb = d["rgb"].astype(np.float64)        # shape (T, 3) R,G,B
            rgb_fs = float(d["fps"]) if "fps" in d.files else fps
            rgb_T = rgb.T                            # -> (3, T)

            try:
                bvp_pos = POS_WANG(rgb_T, rgb_fs)
                for k, v in safe_get_params(np.asarray(bvp_pos), rgb_fs, calc).items():
                    out[f"pos_{k}"] = v
            except Exception as e:
                out["pos_err"] = str(e)[:80]
            try:
                bvp_chrom = CHROM(rgb_T, rgb_fs)
                for k, v in safe_get_params(np.asarray(bvp_chrom), rgb_fs, calc).items():
                    out[f"chrom_{k}"] = v
            except Exception as e:
                out["chrom_err"] = str(e)[:80]
        except Exception as e:
            out["rgb_err"] = str(e)[:80]
    return out


def per_video_csv(args):
    info = pd.read_csv(args.index, delimiter=" ", header=None,
                       names=["video_id", "n_clips", "fps", "hr_gt"])
    paths = {"ecg_root": args.ecg_root, "ppg_sync_root": args.ppg_sync_root,
             "rgb_root": args.rgb_root, "physformer_root": args.physformer_root}
    work = [(str(r["video_id"]), float(r["fps"]), paths)
            for _, r in info.iterrows()]
    if args.limit:
        work = work[:args.limit]

    print(f"[scan]   {len(work)} videos, workers={args.workers}")
    rows = []
    if args.workers <= 1:
        for i, w in enumerate(work):
            rows.append(process_one(w))
            if (i + 1) % 25 == 0:
                print(f"[ok]     {i+1}/{len(work)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, w) for w in work]
            for i, fut in enumerate(as_completed(futs)):
                rows.append(fut.result())
                if (i + 1) % 100 == 0:
                    print(f"[ok]     {i+1}/{len(work)}")
    df = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)

    # Per-method × per-param error columns (vs ECG gold standard).
    for method in ["physformer", "pos", "chrom", "ppg_sync"]:
        for k in PARAM_KEYS:
            gt = df.get(f"ecg_{k}")
            pr = df.get(f"{method}_{k}")
            if gt is None or pr is None:
                continue
            df[f"err_{method}_{k}"] = pr - gt
            df[f"abserr_{method}_{k}"] = (pr - gt).abs()
    df.to_csv(args.out, index=False)
    print(f"[done]   wrote {len(df)} rows to {args.out} ({len(df.columns)} cols)")
    return df


def summary_csv(df: pd.DataFrame, out_path: Path):
    """Per camera × method × param: MAE, bias, GT mean/median, n."""
    rows = []
    for camera in sorted(df["camera"].dropna().unique()):
        sub = df[df["camera"] == camera]
        for k in PARAM_KEYS:
            gt = sub.get(f"ecg_{k}")
            if gt is None:
                continue
            gt_valid = gt.dropna()
            if len(gt_valid) == 0:
                continue
            for method in ["physformer", "pos", "chrom", "ppg_sync"]:
                pr = sub.get(f"{method}_{k}")
                if pr is None:
                    continue
                m = gt.notna() & pr.notna()
                if m.sum() < 5:
                    continue
                err = pr[m] - gt[m]
                rows.append({
                    "camera": camera, "method": method, "param": k,
                    "n": int(m.sum()),
                    "mae": float(err.abs().mean()),
                    "bias": float(err.mean()),
                    "gt_mean": float(gt[m].mean()),
                    "gt_median": float(gt[m].median()),
                    "gt_std": float(gt[m].std()),
                    "pred_mean": float(pr[m].mean()),
                    "corr": (float(gt[m].corr(pr[m]))
                             if pr[m].nunique() > 1 else np.nan),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"[done]   wrote summary {len(summary)} rows to {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/MCD_test_stream.txt"))
    ap.add_argument("--ecg_root", type=Path, default=Path("data/mcd_raw/ecg"))
    ap.add_argument("--ppg_sync_root", type=Path, default=Path("data/mcd_raw/ppg_sync"))
    ap.add_argument("--rgb_root", type=Path, default=Path("data/rgb_traces_mp"))
    ap.add_argument("--physformer_root", type=Path,
                    default=Path("Inference_MCD_PhysFormer_stream"))
    ap.add_argument("--out", type=Path, default=Path("data/eval_pedro.csv"))
    ap.add_argument("--summary_out", type=Path,
                    default=Path("data/eval_pedro_summary.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = per_video_csv(a)
    summary_csv(df, a.summary_out)


if __name__ == "__main__":
    main()
