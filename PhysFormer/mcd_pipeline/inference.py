"""Run PhysFormer inference on preprocessed MCD-rPPG clips.

Uses data/MCD_test.txt + data/VIPL_frames/ produced by preprocess_mcd_rppg.py
and Physformer_VIPL_fold1.pkl (author-released VIPL-HR fold-1 checkpoint).

Outputs:
  Inference_MCD_PhysFormer/<video_id>.npy  -- predicted rPPG signal per video
  Inference_MCD_PhysFormer/results.csv     -- per-video HR_pred vs HR_gt (ECG)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, welch
from torch.utils.data import DataLoader
from torchvision import transforms

from Loadtemporal_data_test import VIPL, Normaliztion, ToTensor
from model import ViT_ST_ST_Compact3_TDC_gra_sharp


def hr_from_rppg(rppg: np.ndarray, fps: float, f_low: float = 0.7,
                 f_high: float = 3.5) -> float:
    """FFT-based HR from an rPPG waveform. Dominant peak in 42-210 bpm band."""
    sig = rppg - np.mean(rppg)
    if len(sig) < int(fps * 4):
        return 0.0
    b, a = butter(3, [f_low, f_high], btype="band", fs=fps)
    sig = filtfilt(b, a, sig)
    f, p = welch(sig, fs=fps, nperseg=min(len(sig), int(fps * 8)))
    band = (f >= f_low) & (f <= f_high)
    if not band.any():
        return 0.0
    return 60.0 * f[band][np.argmax(p[band])]


def pick_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/MCD_test.txt")
    ap.add_argument("--frames_root", default="data/VIPL_frames/")
    ap.add_argument("--checkpoint", default="Physformer_VIPL_fold1.pkl")
    ap.add_argument("--out", default="Inference_MCD_PhysFormer")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"[device]   {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
        num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    gra_sharp = 2.0
    print(f"[model]    loaded {args.checkpoint}")

    ds = VIPL(args.index, args.frames_root,
              transform=transforms.Compose([Normaliztion(), ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    index_df = pd.read_csv(args.index, delimiter=" ", header=None,
                           names=["video_id", "n_clips", "fps", "hr_gt"])

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            video_id = index_df.iloc[i]["video_id"]
            fps = float(index_df.iloc[i]["fps"])
            hr_gt = float(index_df.iloc[i]["hr_gt"])
            inputs = batch["video_x"].to(device)  # [1, n_clips, C, T, H, W]

            rppg_segments = []
            hr_per_clip = []
            for c in range(inputs.shape[1]):
                rppg, *_ = model(inputs[:, c], gra_sharp)
                seg = rppg[0, 30:30 + 160].cpu().numpy()
                rppg_segments.append(seg)
                hr_per_clip.append(hr_from_rppg(seg, fps))
            rppg_long = np.concatenate(rppg_segments)
            hr_pred = float(np.median([h for h in hr_per_clip if h > 0])) \
                if any(h > 0 for h in hr_per_clip) else 0.0
            hr_mean_clip = float(np.mean([h for h in hr_per_clip if h > 0])) \
                if any(h > 0 for h in hr_per_clip) else 0.0

            np.save(out_dir / f"{video_id}.npy", rppg_long)
            rows.append({"video_id": video_id, "fps": fps, "hr_gt": hr_gt,
                         "hr_pred_median": hr_pred, "hr_pred_mean": hr_mean_clip,
                         "n_clips": len(hr_per_clip)})
            print(f"[ok]       {video_id}: HR_pred={hr_pred:.1f} (mean={hr_mean_clip:.1f}) "
                  f"vs HR_gt={hr_gt:.1f}  |  MAE={abs(hr_pred - hr_gt):.1f}")

    df = pd.DataFrame(rows)
    csv_out = out_dir / "results.csv"
    df.to_csv(csv_out, index=False)
    mae_median = (df["hr_pred_median"] - df["hr_gt"]).abs().mean()
    print(f"\n[summary]  MAE(median) = {mae_median:.2f} bpm over {len(df)} videos")
    print(f"[summary]  wrote {csv_out}")


if __name__ == "__main__":
    main()
