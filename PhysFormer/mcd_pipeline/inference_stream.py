"""Streaming PhysFormer inference: read .avi + Stage-0 landmarks in memory.

No PNG intermediate on disk. For each entry in MCD_test_stream.txt:
  - opens raw .avi
  - reuses MediaPipe smoothed bboxes from data/face_landmarks/<id>.npz
  - crops + resizes to 128x128 in memory
  - feeds 220-frame clips to PhysFormer
  - saves reconstructed PPG signal as Inference_MCD_PhysFormer/<id>.npy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, welch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Loadtemporal_data_test import Normaliztion, ToTensor
from model import ViT_ST_ST_Compact3_TDC_gra_sharp


CLIP_FRAMES = 220
CLIP_STRIDE = 160


def load_valid_crops(avi_path: Path, lm_path: Path) -> np.ndarray:
    """Decode .avi once, return uint8 array of shape (N_valid, 128, 128, 3)."""
    lm = np.load(lm_path)
    bboxes = lm["bboxes"].astype(np.int32)
    valid = lm["valid"]
    width = int(lm["width"]); height = int(lm["height"])

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {avi_path}")

    crops: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(valid):
            break
        if valid[frame_idx]:
            x1, y1, x2, y2 = bboxes[frame_idx]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(width, x2); y2 = min(height, y2)
            if x2 - x1 >= 10 and y2 - y1 >= 10:
                crop = frame[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                if h != w:
                    side = max(h, w)
                    top = (side - h) // 2; bottom = side - h - top
                    left = (side - w) // 2; right = side - w - left
                    crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                              cv2.BORDER_REPLICATE)
                face_128 = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
                crops.append(face_128)
        frame_idx += 1
    cap.release()

    if not crops:
        return np.zeros((0, 128, 128, 3), dtype=np.uint8)
    return np.stack(crops, axis=0)


class VIPLStream(Dataset):
    def __init__(self, info_list, videos_root, landmarks_root, transform=None):
        self.info = pd.read_csv(info_list, delimiter=" ", header=None,
                                names=["video_id", "n_clips", "fps", "hr_gt"])
        self.videos_root = Path(videos_root)
        self.landmarks_root = Path(landmarks_root)
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        video_id = str(self.info.iloc[idx]["video_id"])
        n_clips = int(self.info.iloc[idx]["n_clips"])
        fps = float(self.info.iloc[idx]["fps"])
        hr_gt = float(self.info.iloc[idx]["hr_gt"])

        avi = self.videos_root / f"{video_id}.avi"
        lm = self.landmarks_root / f"{video_id}.npz"
        crops = load_valid_crops(avi, lm)  # (N, 128, 128, 3) uint8

        # PhysFormer reference: each frame is resized 132x132 then center-cropped to 128
        # (a small blur step inherited from VIPL training pipeline; keep for parity).
        video_x = np.zeros((n_clips, CLIP_FRAMES, 128, 128, 3), dtype=np.float32)
        for tt in range(n_clips):
            for i in range(CLIP_FRAMES):
                f_idx = tt * CLIP_STRIDE + 60 + i  # matches VIPL's image_id = tt*160+61+i (1-indexed)
                if f_idx >= len(crops):
                    f_idx = len(crops) - 1  # last-frame replication
                img = crops[f_idx]
                img = cv2.resize(img, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                video_x[tt, i] = img

        sample = {"video_x": video_x, "framerate": fps, "clip_average_HR_peaks": hr_gt}
        if self.transform:
            sample = self.transform(sample)
        return sample


def hr_from_rppg(rppg: np.ndarray, fps: float, f_low: float = 0.7,
                 f_high: float = 3.5) -> float:
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
    ap.add_argument("--index", default="data/MCD_test_stream.txt")
    ap.add_argument("--videos_root", default="data/mcd_videos")
    ap.add_argument("--landmarks_root", default="data/face_landmarks")
    ap.add_argument("--checkpoint", default="Physformer_VIPL_fold1.pkl")
    ap.add_argument("--out", default="Inference_MCD_PhysFormer")
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

    # Filter index to entries that don't already have an output, so the DataLoader
    # doesn't waste time decoding videos we'll just skip.
    full_df = pd.read_csv(args.index, delimiter=" ", header=None,
                          names=["video_id", "n_clips", "fps", "hr_gt"])
    todo_mask = ~full_df["video_id"].apply(lambda v: (out_dir / f"{v}.npy").exists())
    todo_df = full_df[todo_mask].reset_index(drop=True)
    print(f"[index]    {len(full_df)} total, {len(todo_df)} todo, {len(full_df) - len(todo_df)} done")
    if len(todo_df) == 0:
        print("[done]     nothing to do")
        return
    todo_index = out_dir / "_todo_index.txt"
    todo_df.to_csv(todo_index, sep=" ", header=False, index=False)

    ds = VIPLStream(todo_index, args.videos_root, args.landmarks_root,
                    transform=transforms.Compose([Normaliztion(), ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    mps_clear = (device.type == "mps") and hasattr(torch.mps, "empty_cache")
    info_df = todo_df

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            video_id = info_df.iloc[i]["video_id"]
            fps = float(info_df.iloc[i]["fps"])
            hr_gt = float(info_df.iloc[i]["hr_gt"])

            out_npy = out_dir / f"{video_id}.npy"
            inputs = batch["video_x"].to(device)
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
            hr_mean = float(np.mean([h for h in hr_per_clip if h > 0])) \
                if any(h > 0 for h in hr_per_clip) else 0.0

            np.save(out_npy, rppg_long)
            rows.append({"video_id": video_id, "fps": fps, "hr_gt": hr_gt,
                         "hr_pred_median": hr_pred, "hr_pred_mean": hr_mean,
                         "n_clips": len(hr_per_clip)})
            print(f"[ok]       {video_id}: HR_pred={hr_pred:.1f} (mean={hr_mean:.1f}) "
                  f"vs HR_gt={hr_gt:.1f}  |  MAE={abs(hr_pred - hr_gt):.1f}", flush=True)
            del inputs, rppg_segments
            if mps_clear:
                torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    csv_out = out_dir / "results_stream.csv"
    df.to_csv(csv_out, index=False)
    valid = df[df["hr_gt"] > 0]
    if len(valid):
        mae = (valid["hr_pred_median"] - valid["hr_gt"]).abs().mean()
        print(f"\n[summary]  MAE(median) = {mae:.2f} bpm over {len(valid)} videos")
    print(f"[summary]  wrote {csv_out}")


if __name__ == "__main__":
    main()
