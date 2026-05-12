"""Extract per-frame mean RGB traces from preprocessed face crops.

Reads data/VIPL_frames/<video_id>/image_NNNNN.png sequences and writes
data/rgb_traces/<video_id>.npz with:
  - rgb:  (T, 3) float32 array, channel order R, G, B
  - fps:  float, sampling rate
  - skin_frac: (T,) float32, fraction of crop classified as skin (0 if --no_skin)

Skin mask uses HSV thresholds typical for rPPG papers; falls back to whole-crop
mean when fewer than 5% of pixels qualify (avoids divide-by-near-zero on bad detections).

Designed to be the canonical input for classical methods (POS, CHROM, PBV, etc.).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def skin_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([0, 30, 60], dtype=np.uint8)
    hi = np.array([20, 150, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi) > 0


def rgb_trace_from_folder(folder: Path, use_skin: bool) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(folder.glob("image_*.png"))
    rgb = np.zeros((len(files), 3), dtype=np.float32)
    skin_frac = np.zeros(len(files), dtype=np.float32)
    for i, f in enumerate(files):
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        if use_skin:
            mask = skin_mask_hsv(bgr)
            frac = float(mask.sum()) / mask.size
            skin_frac[i] = frac
            if frac >= 0.05:
                px = bgr[mask]
                mean_bgr = px.mean(axis=0)
            else:
                mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
        else:
            mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
        rgb[i, 0] = mean_bgr[2]  # R
        rgb[i, 1] = mean_bgr[1]  # G
        rgb[i, 2] = mean_bgr[0]  # B
    return rgb, skin_frac


def process_one(args: tuple[Path, Path, bool]) -> tuple[str, int, str | None]:
    folder, out_dir, use_skin = args
    out_path = out_dir / f"{folder.name}.npz"
    if out_path.exists():
        return folder.name, 0, "skip"
    fps_file = folder / "_fps.txt"
    if not fps_file.exists():
        return folder.name, 0, "no_fps"
    fps = float(fps_file.read_text())
    try:
        rgb, skin_frac = rgb_trace_from_folder(folder, use_skin)
    except Exception as e:
        return folder.name, 0, f"err: {e}"
    np.savez(out_path, rgb=rgb, fps=np.float32(fps), skin_frac=skin_frac)
    return folder.name, len(rgb), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", type=Path, default=Path("data/VIPL_frames"))
    ap.add_argument("--out_root", type=Path, default=Path("data/rgb_traces"))
    ap.add_argument("--no_skin", action="store_true",
                    help="Skip HSV skin mask, use whole-crop mean")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames_root.iterdir() if p.is_dir())
    print(f"[scan]   {len(folders)} folders, workers={args.workers}, skin={not args.no_skin}")

    work = [(f, args.out_root, not args.no_skin) for f in folders]
    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, w): w[0].name for w in work}
        for fut in as_completed(futures):
            name, n, status = fut.result()
            if status == "skip":
                skipped += 1
            elif status is not None:
                errors += 1
                print(f"[err]    {name}: {status}")
            else:
                done += 1
                if done % 25 == 0:
                    print(f"[ok]     {done}/{len(folders)} done ({skipped} skipped, {errors} err)")
    print(f"[done]   {done} new, {skipped} skipped, {errors} errors -> {args.out_root}")


if __name__ == "__main__":
    main()
