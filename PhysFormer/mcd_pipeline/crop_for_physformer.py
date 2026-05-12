"""Stage 1a: produce 128x128 face crop PNG sequences for PhysFormer.

Reads raw videos + precomputed MediaPipe landmarks, extracts square face crop
(smoothed bbox from Stage 0), resizes to 128x128, writes PNG sequence per video.

Output:
  data/face_crops_mp/<video_id>/image_NNNNN.png  (1-indexed to match VIPL format)
  data/face_crops_mp/<video_id>/_fps.txt
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def process_one(args: tuple[Path, Path, Path]) -> tuple[str, int, str | None]:
    video_path, landmarks_path, out_root = args
    video_id = video_path.stem
    out_dir = out_root / video_id
    fps_file = out_dir / "_fps.txt"
    if fps_file.exists() and any(out_dir.glob("image_*.png")):
        return video_id, 0, "skip"
    if not landmarks_path.exists():
        return video_id, 0, "no_landmarks"

    out_dir.mkdir(parents=True, exist_ok=True)

    lm = np.load(landmarks_path)
    bboxes = lm["bboxes"].astype(np.int32)
    valid = lm["valid"]
    fps = float(lm["fps"])
    width = int(lm["width"])
    height = int(lm["height"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_id, 0, "open_fail"

    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= len(valid):
            break
        if not valid[frame_idx]:
            frame_idx += 1
            continue  # skip frames with no detected face (don't reuse stale)
        x1, y1, x2, y2 = bboxes[frame_idx]
        # Clamp into frame
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(width, int(x2)); y2 = min(height, int(y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            frame_idx += 1
            continue
        crop = frame[y1:y2, x1:x2]
        # Force square (smoothed bbox is already square in float space, but clamp-to-frame
        # at edges may have shrunk one side; re-pad with edge replication if needed).
        h, w = crop.shape[:2]
        if h != w:
            side = max(h, w)
            top = (side - h) // 2; bottom = side - h - top
            left = (side - w) // 2; right = side - w - left
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                      cv2.BORDER_REPLICATE)
        face_128 = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        written += 1
        cv2.imwrite(str(out_dir / f"image_{written:05d}.png"), face_128)
        frame_idx += 1
    cap.release()

    if written == 0:
        return video_id, 0, "no_valid_frames"

    fps_file.write_text(str(fps))
    return video_id, written, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=Path, default=Path("data/mcd_videos"))
    ap.add_argument("--landmarks_root", type=Path, default=Path("data/face_landmarks"))
    ap.add_argument("--out_root", type=Path, default=Path("data/face_crops_mp"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.videos_root.glob("*.avi"))
    work = [(v, args.landmarks_root / f"{v.stem}.npz", args.out_root) for v in videos]
    print(f"[scan]   {len(videos)} videos, workers={args.workers}")

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, w) for w in work]
        for fut in as_completed(futures):
            vid, n, status = fut.result()
            if status == "skip":
                skipped += 1
            elif status is not None:
                errors += 1
                print(f"[err]    {vid}: {status}")
            else:
                done += 1
                if done % 25 == 0:
                    print(f"[ok]     {done} new, {skipped} skipped, {errors} err")
    print(f"[done]   {done} new, {skipped} skipped, {errors} errors -> {args.out_root}")


if __name__ == "__main__":
    main()
