"""
Download a subset of the MCD-rPPG dataset and preprocess it into the VIPL frame
layout expected by inference_OneSample_VIPL_PhysFormer.py.

Deps: huggingface_hub, opencv-python, scipy, numpy, pandas

Outputs:
  data/VIPL_frames/<video_id>/image_00001.png ...   (128x128 face crops)
  data/MCD_test.txt                                 (space-delimited index)

Index columns (matches Loadtemporal_data_test.VIPL):
  <video_id> <total_clips> <framerate> <clip_average_HR>
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from mcd_pipeline.utils import R_peaks


REPO_ID = "kyegorov/mcd_rppg"
CLIP_LEN = 160 + 60  # loader reads 220 frames per clip
CLIP_STRIDE = 160


CAMERAS = ("FullHDwebcam", "IriunWebcam", "USBVideo")


def pick_video_files(state_filter: str, n_subjects: int, all_files: list[str]):
    """Return list of video repo paths across ALL 3 cameras for the first N subjects."""
    vids = [f for f in all_files if f.startswith("video/") and f.endswith(".avi")]
    if state_filter != "all":
        vids = [f for f in vids if f.endswith(f"_{state_filter}.avi")]
    subjects = sorted({os.path.basename(f).split("_")[0] for f in vids})[:n_subjects]
    return [f for f in vids if os.path.basename(f).split("_")[0] in subjects]


def parse_video_id(video_basename: str) -> tuple[str, str, str]:
    """'1020_FullHDwebcam_after.avi' -> ('1020', 'FullHDwebcam', 'after')"""
    stem = video_basename.replace(".avi", "")
    parts = stem.split("_")
    return parts[0], parts[1], parts[-1]


def download_signals_for_video(video_basename: str, all_files: list[str],
                               out_root: Path, ecg_cache: dict) -> tuple[str | None, str | None]:
    """Download ppg_sync (per-video), ppg raw (per subject+state), ecg (per subject+state).
    Returns (ppg_sync_local_path, ecg_local_path)."""
    subject, _, state = parse_video_id(video_basename)
    stem = video_basename.replace(".avi", "")

    # Per-video: ppg_sync + meta
    ppg_sync_local = None
    for f in all_files:
        if (f.startswith("ppg_sync/") or f.startswith("meta/")) and stem in f:
            local = hf_hub_download(REPO_ID, f, repo_type="dataset",
                                    local_dir=str(out_root))
            if f.startswith("ppg_sync/"):
                ppg_sync_local = local

    # Per-subject+state: ppg raw + ecg (cached across cameras)
    key = f"{subject}_{state}"
    if key not in ecg_cache:
        ecg_local = None
        for f in all_files:
            if f.startswith(("ppg/", "ecg/")) and os.path.basename(f).startswith(f"{subject}_{state}."):
                local = hf_hub_download(REPO_ID, f, repo_type="dataset",
                                        local_dir=str(out_root))
                if f.startswith("ecg/"):
                    ecg_local = local
        ecg_cache[key] = ecg_local
    return ppg_sync_local, ecg_cache[key]


def load_ecg(path: str) -> tuple[np.ndarray, float]:
    """Load ECG JSON (lead I) -> (signal, fs)."""
    with open(path) as f:
        d = json.load(f)
    fs = float(d["frequency"])
    lead_i = next((c for c in d["data"] if c.get("title") == "I"), d["data"][0])
    return np.asarray(lead_i["values"], dtype=float), fs


def hr_from_ecg(ecg: np.ndarray, fs: float) -> float:
    """Compute average HR (bpm) from an ECG signal using Pan-Tompkins R-peaks."""
    peaks = R_peaks(ecg, fs)
    if len(peaks) < 2:
        return 0.0
    rr_intervals_s = np.diff(peaks) / fs
    return float(60.0 / np.mean(rr_intervals_s))


def build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def crop_face_128(frame: np.ndarray, detector) -> np.ndarray | None:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                      minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick largest face
    x, y, bw, bh = max(faces, key=lambda b: b[2] * b[3])
    side = max(bw, bh)
    cx, cy = x + bw // 2, y + bh // 2
    x0 = max(cx - side // 2, 0)
    y0 = max(cy - side // 2, 0)
    x1 = min(x0 + side, w)
    y1 = min(y0 + side, h)
    face = frame[y0:y1, x0:x1]
    return cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)


def extract_frames(video_path: str, out_dir: Path, detector) -> tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    last_face = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        face = crop_face_128(frame, detector)
        if face is None:
            if last_face is None:
                continue  # skip until first detection
            face = last_face
        else:
            last_face = face
        cv2.imwrite(str(out_dir / f"image_{idx:05d}.png"), face)
        idx += 1
    cap.release()
    return idx - 1, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_subjects", type=int, default=10)
    ap.add_argument("--state", default="all", choices=["before", "after", "all"])
    ap.add_argument("--out_root", type=Path, default=Path("data"))
    ap.add_argument("--keep_raw", action="store_true",
                    help="Keep downloaded .avi files (default deletes after preprocessing)")
    args = ap.parse_args()

    frames_root = args.out_root / "VIPL_frames"
    index_path = args.out_root / "MCD_test.txt"
    signals_root = args.out_root / "mcd_raw"  # ppg/ppg_sync/ecg/meta mirror HF layout
    frames_root.mkdir(parents=True, exist_ok=True)
    signals_root.mkdir(parents=True, exist_ok=True)

    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    videos = pick_video_files(args.state, args.n_subjects, all_files)
    subject_ids = sorted({os.path.basename(f).split("_")[0] for f in videos})
    ecg_cache: dict[str, str | None] = {}
    hr_cache: dict[str, float] = {}

    # Top-level metadata (db.csv + readmes) — downloaded once via single snapshot call.
    # meta/ folder restricted to selected subjects (one pattern per subject).
    meta_patterns = ["db.csv", "README.md", "readme.txt"]
    meta_patterns += [f"meta/{sid}_*" for sid in subject_ids]
    print(f"[meta]     snapshot_download for {len(subject_ids)} subjects' meta + db.csv")
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=str(signals_root),
                      allow_patterns=meta_patterns)
    print(f"Selected {len(videos)} videos across {len(CAMERAS)} cameras")

    mp_face = build_face_detector()

    rows = []
    for vrepo in videos:
        video_id = os.path.basename(vrepo).replace(".avi", "")
        out_dir = frames_root / video_id
        fps_file = out_dir / "_fps.txt"
        if out_dir.exists() and any(out_dir.glob("image_*.png")) and fps_file.exists():
            print(f"[skip]     {video_id} already preprocessed")
        else:
            print(f"[download] {vrepo}")
            vpath = hf_hub_download(REPO_ID, vrepo, repo_type="dataset")
            print(f"[frames]   extracting -> {out_dir}")
            n_frames, fps = extract_frames(vpath, out_dir, mp_face)
            fps_file.write_text(str(fps))
            if not args.keep_raw:
                try: os.remove(vpath)
                except OSError: pass
        n_frames = len(list(out_dir.glob("image_*.png")))
        fps = float(fps_file.read_text())

        # Download ground-truth signals (ppg_sync + ppg + ecg + meta); ecg is cached per subject+state
        _ppg_sync_path, ecg_path = download_signals_for_video(
            os.path.basename(vrepo), all_files, signals_root, ecg_cache)
        # Each clip reads 220 frames starting at tt*160 + 61; last frame = tt*160 + 280.
        # So total_clips = (n_frames - 280) // 160 + 1, valid only when n_frames >= 280.
        total_clips = max((n_frames - 120) // CLIP_STRIDE, 0) if n_frames >= 280 else 0

        subject, _, state = parse_video_id(os.path.basename(vrepo))
        ecg_key = f"{subject}_{state}"
        if ecg_path is None:
            print(f"[warn]     no ECG for {video_id}, writing 0 HR")
            avg_hr = 0.0
        else:
            if ecg_key not in hr_cache:
                ecg, ecg_fs = load_ecg(ecg_path)
                hr_cache[ecg_key] = hr_from_ecg(ecg, ecg_fs)
                print(f"[ecg-hr]   {ecg_key}: {hr_cache[ecg_key]:.1f} bpm "
                      f"(fs={ecg_fs:.0f}, dur={len(ecg)/ecg_fs:.1f}s)")
            avg_hr = hr_cache[ecg_key]

        if total_clips <= 0:
            print(f"[warn]  {video_id} too short ({n_frames} frames), skipping")
            shutil.rmtree(out_dir, ignore_errors=True)
            continue

        rows.append(f"{video_id} {total_clips} {fps:.4f} {avg_hr:.4f}")
        print(f"[ok]    {video_id}: {n_frames} frames, {total_clips} clips, HR={avg_hr:.1f}")

    index_path.write_text("\n".join(rows) + "\n")
    print(f"\nWrote index: {index_path}")
    print(f"Frames root: {frames_root}")


if __name__ == "__main__":
    main()
