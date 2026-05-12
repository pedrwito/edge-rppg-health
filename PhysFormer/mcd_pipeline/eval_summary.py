"""Summarize data/eval_methods.csv:
  1. GT distributions (HR + HRV from ECG, range, IQR, mean+/-std)
  2. Per-method × per-camera × per-state MAE for HR + HRV metrics
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["physformer", "pos", "chrom", "pbv", "green"]
HR_KEYS = ["hr_welch", "hr_peaks"]
HRV_KEYS = ["sdnn_ms", "rmssd_ms", "pnn50_pct", "lfhf"]


def fmt(x, prec=2):
    if pd.isna(x):
        return "  -"
    return f"{x:.{prec}f}"


def gt_distribution(df):
    print("\n" + "=" * 78)
    print("GROUND-TRUTH DISTRIBUTIONS (from ECG, all 500 subjects × 2 states)")
    print("=" * 78)
    # ECG is per subject_state, so dedupe
    ecg = df.drop_duplicates(subset=["subject", "state"]).copy()
    rows = []
    for col, label, unit in [
        ("gt_ecg_hr_peaks", "HR (R-peaks)",      "bpm"),
        ("gt_ecg_hr_welch", "HR (Welch FFT)",    "bpm"),
        ("gt_ecg_sdnn_ms",  "SDNN",              "ms"),
        ("gt_ecg_rmssd_ms", "RMSSD",             "ms"),
        ("gt_ecg_pnn50_pct","pNN50",             "%"),
        ("gt_ecg_lfhf",     "LF/HF ratio",       "-"),
    ]:
        s = ecg[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "metric": label, "unit": unit, "n": len(s),
            "mean": s.mean(), "std": s.std(),
            "min": s.min(), "p25": s.quantile(0.25),
            "median": s.median(), "p75": s.quantile(0.75),
            "max": s.max(),
        })
    g = pd.DataFrame(rows)
    print(g.to_string(index=False, formatters={
        "mean": lambda x: f"{x:7.2f}", "std": lambda x: f"{x:6.2f}",
        "min": lambda x: f"{x:6.2f}", "p25": lambda x: f"{x:6.2f}",
        "median": lambda x: f"{x:6.2f}", "p75": lambda x: f"{x:6.2f}",
        "max": lambda x: f"{x:6.2f}",
    }))
    # Compare states (before vs after)
    print("\nBefore vs After (resting vs post-exercise) HR:")
    for st in ["before", "after"]:
        s = ecg[ecg["state"] == st]["gt_ecg_hr_peaks"].dropna()
        print(f"  {st:6s}  n={len(s):3d}  mean={s.mean():.1f} ± {s.std():.1f} bpm  "
              f"range [{s.min():.0f}, {s.max():.0f}]")


def per_camera_state_mae(df, gt_col, pred_template, label, unit):
    print(f"\n{label} ({unit}) — MAE vs {gt_col}:")
    print(f"  {'method':12s}  {'camera':14s}  {'state':7s}  {'n':>4s}  "
          f"{'MAE':>7s}  {'bias':>7s}  {'GT mean':>8s}  {'pred mean':>9s}")
    for method in METHODS:
        col = pred_template.format(method=method)
        if col not in df.columns:
            continue
        for camera in ["FullHDwebcam", "IriunWebcam", "USBVideo"]:
            for state in ["before", "after"]:
                sub = df[(df["camera"] == camera) & (df["state"] == state)]
                gt = sub[gt_col].astype(float)
                pr = sub[col].astype(float)
                m = gt.notna() & pr.notna()
                if m.sum() < 5:
                    continue
                err = pr[m] - gt[m]
                print(f"  {method:12s}  {camera:14s}  {state:7s}  {m.sum():4d}  "
                      f"{err.abs().mean():7.2f}  {err.mean():7.2f}  "
                      f"{gt[m].mean():8.2f}  {pr[m].mean():9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/eval_methods.csv"))
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print(f"loaded {len(df)} rows from {a.csv}")

    gt_distribution(df)

    print("\n" + "=" * 78)
    print("PER-METHOD HR ERROR (gt = ECG R-peaks, gold standard)")
    print("=" * 78)
    per_camera_state_mae(df, "gt_ecg_hr_peaks", "{method}_hr_welch",
                         "HR (Welch on rPPG vs ECG R-peak HR)", "bpm")

    print("\n" + "=" * 78)
    print("PER-METHOD HRV ERROR (gt = ECG R-peaks)")
    print("=" * 78)
    for key, label, unit in [
        ("sdnn_ms",  "SDNN",  "ms"),
        ("rmssd_ms", "RMSSD", "ms"),
        ("pnn50_pct","pNN50", "%"),
        ("lfhf",     "LF/HF", "-"),
    ]:
        per_camera_state_mae(df, f"gt_ecg_{key}", f"{{method}}_{key}", label, unit)

    print("\n" + "=" * 78)
    print("AGGREGATE (all cameras, all states pooled)")
    print("=" * 78)
    print(f"  {'method':12s}  {'metric':10s}  {'n':>4s}  {'MAE':>7s}  "
          f"{'bias':>7s}  {'corr':>6s}")
    for method in METHODS:
        for key, label in [("hr_welch", "HR_welch"), ("sdnn_ms", "SDNN"),
                            ("rmssd_ms", "RMSSD"), ("pnn50_pct", "pNN50"),
                            ("lfhf", "LFHF")]:
            gt_col = f"gt_ecg_{key}" if key != "hr_welch" else "gt_ecg_hr_peaks"
            col = f"{method}_{key}"
            if col not in df.columns or gt_col not in df.columns:
                continue
            gt = df[gt_col].astype(float)
            pr = df[col].astype(float)
            m = gt.notna() & pr.notna()
            if m.sum() < 5:
                continue
            err = pr[m] - gt[m]
            corr = gt[m].corr(pr[m]) if pr[m].nunique() > 1 else np.nan
            print(f"  {method:12s}  {label:10s}  {m.sum():4d}  "
                  f"{err.abs().mean():7.2f}  {err.mean():7.2f}  {fmt(corr):>6s}")


if __name__ == "__main__":
    main()
