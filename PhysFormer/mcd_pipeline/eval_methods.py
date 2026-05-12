"""Compute HR + HRV metrics for every (video, method) and ground truth.

For each video in MCD_test_stream.txt, produces one row with:
  - GT from ECG (gold standard, 500 Hz)
  - GT from synced PPG (~30 Hz, per camera)
  - GT pulse from db.csv (single medical-grade HR value)
  - rPPG estimates from PhysFormer + classical methods on rgb_traces_mp/

Output: data/eval_methods.csv (one row per video × method)
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from mcd_pipeline.methods import METHODS
from mcd_pipeline.hrv import all_metrics


def load_ppg_sync(path: Path) -> tuple[np.ndarray, float]:
    """ppg_sync/<video>.txt: each line `value timestamp_or_dt`. Returns sig, fs."""
    arr = np.loadtxt(path)
    sig = arr[:, 0].astype(np.float64)
    # Second column is delta-t (seconds, ~0.033 for 30 Hz).
    dt = arr[1:, 1]
    dt = dt[(dt > 0.001) & (dt < 0.5)]
    fs = float(1.0 / np.median(dt)) if len(dt) else 30.0
    return sig, fs


def load_ecg_lead(path: Path) -> tuple[np.ndarray, float]:
    """ecg/<subject_state>.json: lead-I values + frequency. Returns sig, fs."""
    with open(path) as f:
        d = json.load(f)
    fs = float(d["frequency"])
    lead = next((c for c in d["data"] if c.get("title") == "I"), d["data"][0])
    sig = np.asarray(lead["values"], dtype=np.float64)
    return sig, fs


def process_one(args: tuple) -> dict:
    (video_id, fps, hr_pulse, paths) = args
    out = {"video_id": video_id, "n_clips": None, "fps": fps,
           "hr_pulse_db": hr_pulse}
    parts = video_id.split("_")
    subject, camera, state = parts[0], parts[1], parts[-1]
    out.update({"subject": subject, "camera": camera, "state": state})

    # GT — ECG
    ecg_path = paths["ecg_root"] / f"{subject}_{state}.json"
    if ecg_path.exists():
        try:
            sig, fs = load_ecg_lead(ecg_path)
            for k, v in all_metrics(sig, fs, kind="ecg").items():
                out[f"gt_ecg_{k}"] = v
        except Exception as e:
            out["ecg_err"] = str(e)[:80]

    # GT — synced PPG
    pps_path = paths["ppg_sync_root"] / f"{video_id}.txt"
    if pps_path.exists():
        try:
            sig, fs = load_ppg_sync(pps_path)
            for k, v in all_metrics(sig, fs, kind="ppg").items():
                out[f"gt_ppg_{k}"] = v
        except Exception as e:
            out["ppg_err"] = str(e)[:80]

    # PhysFormer rPPG
    pf_path = paths["physformer_root"] / f"{video_id}.npy"
    if pf_path.exists():
        try:
            sig = np.load(pf_path).astype(np.float64)
            for k, v in all_metrics(sig, fps, kind="ppg").items():
                out[f"physformer_{k}"] = v
        except Exception as e:
            out["physformer_err"] = str(e)[:80]

    # Classical methods on RGB trace
    rgb_path = paths["rgb_root"] / f"{video_id}.npz"
    if rgb_path.exists():
        try:
            d = np.load(rgb_path)
            rgb = d["rgb"]
            rgb_fps = float(d["fps"]) if "fps" in d.files else fps
            for name, fn in METHODS.items():
                try:
                    sig = fn(rgb, rgb_fps)
                    for k, v in all_metrics(sig, rgb_fps, kind="ppg").items():
                        out[f"{name}_{k}"] = v
                except Exception as e:
                    out[f"{name}_err"] = str(e)[:80]
        except Exception as e:
            out["rgb_err"] = str(e)[:80]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/MCD_test_stream.txt"))
    ap.add_argument("--db", type=Path, default=Path("data/mcd_raw/db.csv"))
    ap.add_argument("--ecg_root", type=Path, default=Path("data/mcd_raw/ecg"))
    ap.add_argument("--ppg_sync_root", type=Path, default=Path("data/mcd_raw/ppg_sync"))
    ap.add_argument("--rgb_root", type=Path, default=Path("data/rgb_traces_mp"))
    ap.add_argument("--physformer_root", type=Path,
                    default=Path("Inference_MCD_PhysFormer_stream"))
    ap.add_argument("--out", type=Path, default=Path("data/eval_methods.csv"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    info = pd.read_csv(a.index, delimiter=" ", header=None,
                       names=["video_id", "n_clips", "fps", "hr_gt"])
    db = pd.read_csv(a.db)
    db["state"] = db["step"].astype(str)
    db["camera"] = db["camera"].astype(str)
    db["subject"] = db["patient_id"].astype(str)
    db["video_id"] = db["subject"] + "_" + db["camera"] + "_" + db["state"]
    pulse_map = dict(zip(db["video_id"], db["pulse"]))

    paths = {"ecg_root": a.ecg_root, "ppg_sync_root": a.ppg_sync_root,
             "rgb_root": a.rgb_root, "physformer_root": a.physformer_root}

    work = []
    for _, r in info.iterrows():
        vid = str(r["video_id"])
        work.append((vid, float(r["fps"]), pulse_map.get(vid, np.nan), paths))

    print(f"[scan]   {len(work)} videos, workers={a.workers}")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_one, w) for w in work]
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 100 == 0:
                print(f"[ok]     {i+1}/{len(work)}")

    df = pd.DataFrame(rows).set_index("video_id").sort_index()
    df.to_csv(a.out)
    print(f"[done]   wrote {len(df)} rows -> {a.out}  ({len(df.columns)} columns)")


if __name__ == "__main__":
    main()
