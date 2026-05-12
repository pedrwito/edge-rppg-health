"""Build MCD_test_stream.txt index from raw .avi + Stage-0 landmarks.

For each video that has both a .avi and a face_landmarks/<id>.npz, computes:
  n_clips = max((n_valid_frames - 120) // 160, 0)   (PhysFormer 220-frame clip with stride 160)
  fps     from landmarks .npz
  hr_gt   from data/mcd_raw/ecg/<subject>_<state>.json (lead I, peak detection)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mcd_pipeline.utils import R_peaks

CLIP_STRIDE = 160
CLIP_FRAMES = 220


def hr_from_ecg(ecg: np.ndarray, fs: float) -> float:
    peaks = R_peaks(ecg, fs)
    if len(peaks) < 2:
        return 0.0
    rr_s = np.diff(peaks) / fs
    return float(60.0 / np.mean(rr_s))


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
    ap.add_argument("--videos_root", type=Path, default=Path("data/mcd_videos"))
    ap.add_argument("--landmarks_root", type=Path, default=Path("data/face_landmarks"))
    ap.add_argument("--ecg_root", type=Path, default=Path("data/mcd_raw/ecg"))
    ap.add_argument("--out", type=Path, default=Path("data/MCD_test_stream.txt"))
    args = ap.parse_args()

    avis = sorted(args.videos_root.glob("*.avi"))
    print(f"[scan]   {len(avis)} .avi in {args.videos_root}")

    hr_cache: dict[str, float] = {}
    rows = []
    skipped = 0
    for avi in avis:
        video_id = avi.stem
        lm_path = args.landmarks_root / f"{video_id}.npz"
        if not lm_path.exists():
            skipped += 1
            continue
        lm = np.load(lm_path)
        valid = lm["valid"]
        n_valid = int(valid.sum())
        if n_valid < CLIP_FRAMES + CLIP_STRIDE:
            skipped += 1
            continue
        n_clips = max((n_valid - (CLIP_FRAMES - CLIP_STRIDE)) // CLIP_STRIDE, 0)
        fps = float(lm["fps"])

        parts = video_id.split("_")
        subject, state = parts[0], parts[-1]
        key = f"{subject}_{state}"
        if key not in hr_cache:
            hr_cache[key] = load_ecg_hr(subject, state, args.ecg_root)
        hr_gt = hr_cache[key]

        rows.append(f"{video_id} {n_clips} {fps:.4f} {hr_gt:.4f}")

    args.out.write_text("\n".join(rows) + "\n")
    print(f"[done]   wrote {len(rows)} entries to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
