"""Rebuild data/MCD_test.txt from existing data/VIPL_frames/ folders.

For each preprocessed folder, computes:
  - n_clips from frame count (PhysFormer reads 220-frame clips with stride 160)
  - fps from _fps.txt
  - hr_gt from data/mcd_raw/ecg/<subject>_<state>.json (lead I, peak detection)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from mcd_pipeline.utils import R_peaks

CLIP_STRIDE = 160


def hr_from_ecg(ecg: np.ndarray, fs: float) -> float:
    peaks = R_peaks(ecg, fs)
    if len(peaks) < 2:
        return 0.0
    rr_intervals_s = np.diff(peaks) / fs
    return float(60.0 / np.mean(rr_intervals_s))


def load_ecg_hr(subject: str, state: str, ecg_root: Path) -> float:
    path = ecg_root / f"{subject}_{state}.json"
    if not path.exists():
        return 0.0
    with open(path) as f:
        d = json.load(f)
    fs = float(d["frequency"])
    lead_i = next((c for c in d["data"] if c.get("title") == "I"), d["data"][0])
    sig = np.asarray(lead_i["values"], dtype=float)
    return hr_from_ecg(sig, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", type=Path, default=Path("data/VIPL_frames"))
    ap.add_argument("--ecg_root", type=Path, default=Path("data/mcd_raw/ecg"))
    ap.add_argument("--out", type=Path, default=Path("data/MCD_test.txt"))
    args = ap.parse_args()

    folders = sorted(p for p in args.frames_root.iterdir() if p.is_dir())
    print(f"[scan]   {len(folders)} folders in {args.frames_root}")

    hr_cache: dict[str, float] = {}
    rows = []
    skipped = 0
    for folder in folders:
        video_id = folder.name
        fps_file = folder / "_fps.txt"
        if not fps_file.exists():
            print(f"[skip]   {video_id}: missing _fps.txt")
            skipped += 1
            continue
        fps = float(fps_file.read_text())
        n_frames = len(list(folder.glob("image_*.png")))
        if n_frames < 280:
            print(f"[skip]   {video_id}: {n_frames} frames (<280)")
            skipped += 1
            continue
        n_clips = max((n_frames - 120) // CLIP_STRIDE, 0)

        parts = video_id.split("_")
        subject, state = parts[0], parts[-1]
        key = f"{subject}_{state}"
        if key not in hr_cache:
            hr_cache[key] = load_ecg_hr(subject, state, args.ecg_root)
        hr_gt = hr_cache[key]
        if hr_gt == 0.0:
            print(f"[warn]   {video_id}: no/invalid ECG, hr_gt=0")

        rows.append(f"{video_id} {n_clips} {fps:.4f} {hr_gt:.4f}")

    args.out.write_text("\n".join(rows) + "\n")
    print(f"[done]   wrote {len(rows)} entries to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
