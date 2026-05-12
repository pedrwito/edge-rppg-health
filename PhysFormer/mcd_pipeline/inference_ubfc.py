"""Streaming PhysFormer inference on UBFC.

Mirrors mcd_pipeline/inference_stream.py but resolves UBFC's subfolder layout
(<videos_root>/<subject>/vid.avi) and writes Inference_UBFC_PhysFormer/<subject>.npy.

Per-subject row in results_ubfc.csv:
    subject, fps, hr_gt (from ground_truth.txt line 2), hr_pred_median, hr_pred_mean, n_clips
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Loadtemporal_data_test import Normaliztion, ToTensor
from model import ViT_ST_ST_Compact3_TDC_gra_sharp
from mcd_pipeline.inference_stream import (
    CLIP_FRAMES, CLIP_STRIDE, load_valid_crops, hr_from_rppg, pick_device,
)


class UBFCStream(Dataset):
    """Like VIPLStream but resolves <videos_root>/<subject>/vid.avi paths."""
    def __init__(self, info_list, videos_root, landmarks_root, transform=None):
        self.info = pd.read_csv(info_list, delimiter=" ", header=None,
                                names=["subject", "n_clips", "fps", "hr_gt"])
        self.videos_root = Path(videos_root)
        self.landmarks_root = Path(landmarks_root)
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        subject = str(self.info.iloc[idx]["subject"])
        n_clips = int(self.info.iloc[idx]["n_clips"])
        fps = float(self.info.iloc[idx]["fps"])
        hr_gt = float(self.info.iloc[idx]["hr_gt"])

        avi = self.videos_root / subject / "vid.avi"
        lm = self.landmarks_root / f"{subject}.npz"
        crops = load_valid_crops(avi, lm)

        import cv2
        video_x = np.zeros((n_clips, CLIP_FRAMES, 128, 128, 3), dtype=np.float32)
        for tt in range(n_clips):
            for i in range(CLIP_FRAMES):
                f_idx = tt * CLIP_STRIDE + 60 + i
                if f_idx >= len(crops):
                    f_idx = len(crops) - 1
                img = crops[f_idx]
                img = cv2.resize(img, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                video_x[tt, i] = img

        sample = {"video_x": video_x, "framerate": fps, "clip_average_HR_peaks": hr_gt}
        if self.transform:
            sample = self.transform(sample)
        return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/UBFC_test_stream.txt")
    ap.add_argument("--videos_root", required=True,
                    help="UBFC root with subjectN/vid.avi")
    ap.add_argument("--landmarks_root", default="data/face_landmarks_ubfc")
    ap.add_argument("--checkpoint", default="Physformer_VIPL_fold1.pkl")
    ap.add_argument("--out", default="Inference_UBFC_PhysFormer")
    ap.add_argument("--cpu", action="store_true")
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

    full_df = pd.read_csv(args.index, delimiter=" ", header=None,
                          names=["subject", "n_clips", "fps", "hr_gt"])
    todo_mask = ~full_df["subject"].apply(lambda s: (out_dir / f"{s}.npy").exists())
    todo_df = full_df[todo_mask].reset_index(drop=True)
    print(f"[index]    {len(full_df)} total, {len(todo_df)} todo, {len(full_df) - len(todo_df)} done")
    if len(todo_df) == 0:
        print("[done]     nothing to do"); return

    todo_index = out_dir / "_todo_index.txt"
    todo_df.to_csv(todo_index, sep=" ", header=False, index=False)
    ds = UBFCStream(todo_index, args.videos_root, args.landmarks_root,
                    transform=transforms.Compose([Normaliztion(), ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    mps_clear = (device.type == "mps") and hasattr(torch.mps, "empty_cache")

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            subject = todo_df.iloc[i]["subject"]
            fps = float(todo_df.iloc[i]["fps"])
            hr_gt = float(todo_df.iloc[i]["hr_gt"])
            out_npy = out_dir / f"{subject}.npy"

            inputs = batch["video_x"].to(device)
            rppg_segments = []
            hr_per_clip = []
            for c in range(inputs.shape[1]):
                rppg, *_ = model(inputs[:, c], gra_sharp)
                seg = rppg[0, 30:30 + 160].cpu().numpy()
                rppg_segments.append(seg)
                hr_per_clip.append(hr_from_rppg(seg, fps))
            rppg_long = np.concatenate(rppg_segments)
            valid_hr = [h for h in hr_per_clip if h > 0]
            hr_pred = float(np.median(valid_hr)) if valid_hr else 0.0
            hr_mean = float(np.mean(valid_hr)) if valid_hr else 0.0

            np.save(out_npy, rppg_long)
            rows.append({"subject": subject, "fps": fps, "hr_gt": hr_gt,
                         "hr_pred_median": hr_pred, "hr_pred_mean": hr_mean,
                         "n_clips": len(hr_per_clip)})
            print(f"[ok]       {subject}: HR_pred={hr_pred:.1f} (mean={hr_mean:.1f}) "
                  f"vs HR_gt={hr_gt:.1f}  |  MAE={abs(hr_pred - hr_gt):.1f}", flush=True)
            del inputs, rppg_segments
            if mps_clear:
                torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    csv_out = out_dir / "results_ubfc.csv"
    df.to_csv(csv_out, index=False)
    valid = df[df["hr_gt"] > 0]
    if len(valid):
        mae = (valid["hr_pred_median"] - valid["hr_gt"]).abs().mean()
        print(f"\n[summary]  MAE(median) = {mae:.2f} bpm over {len(valid)} subjects")
    print(f"[summary]  wrote {csv_out}")


if __name__ == "__main__":
    main()
