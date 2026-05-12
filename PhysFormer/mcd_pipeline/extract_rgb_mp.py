"""Stage 1b: extract per-frame mean RGB time series from raw videos using
precomputed MediaPipe landmarks (Stage 0) + Pedro's full-face skin masking logic.

For each frame: build face mask from landmarks (convex hull of all face points
minus convex hulls of eyes/brows/lips, dilated), apply YCrCb skin filter,
morphological cleanup, then mean RGB over the resulting mask.

Output per video at data/rgb_traces_mp/<video_id>.npz:
  rgb:        (T, 3) float32  channel order R, G, B (skipped frames omitted)
  frame_idx:  (T,)   int32    original frame indices kept (for alignment)
  fps:        float32
  skin_frac:  (T,)   float32  fraction of face pixels classified as skin
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# Landmark index sets — same as in IppgSignalObtainer.extractFullFaceSkinRGBFromVideo
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
              14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

YCRCB_LO = np.array([0, 133, 77], dtype=np.uint8)
YCRCB_HI = np.array([255, 173, 127], dtype=np.uint8)
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def build_skin_mask(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    face_hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, face_hull, 255)

    exclude = np.zeros((h, w), dtype=np.uint8)
    for idxs in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS_OUTER):
        poly = cv2.convexHull(landmarks[idxs])
        cv2.fillConvexPoly(exclude, poly, 255)
    exclude = cv2.dilate(exclude, DILATE_KERNEL, iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    color_mask = cv2.inRange(ycrcb, YCRCB_LO, YCRCB_HI)
    mask = cv2.bitwise_and(mask, color_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, DILATE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, DILATE_KERNEL)
    return mask


def process_one(args: tuple[Path, Path, Path]) -> tuple[str, int, str | None]:
    video_path, landmarks_path, out_root = args
    video_id = video_path.stem
    out_path = out_root / f"{video_id}.npz"
    if out_path.exists():
        return video_id, 0, "skip"
    if not landmarks_path.exists():
        return video_id, 0, "no_landmarks"

    lm = np.load(landmarks_path)
    landmarks = lm["landmarks"].astype(np.int32)
    valid = lm["valid"]
    fps = float(lm["fps"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_id, 0, "open_fail"

    rgb_list, idx_list, skin_frac_list = [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(valid):
            break
        if not valid[frame_idx]:
            frame_idx += 1
            continue
        mask = build_skin_mask(frame, landmarks[frame_idx])
        skin = mask > 0
        frac = float(skin.sum()) / mask.size
        if frac < 0.005:  # essentially no skin found, skip
            frame_idx += 1
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = float(rgb_frame[:, :, 0][skin].mean())
        g = float(rgb_frame[:, :, 1][skin].mean())
        b = float(rgb_frame[:, :, 2][skin].mean())
        rgb_list.append((r, g, b))
        idx_list.append(frame_idx)
        skin_frac_list.append(frac)
        frame_idx += 1
    cap.release()

    if not rgb_list:
        return video_id, 0, "no_valid_frames"

    np.savez_compressed(out_path,
                        rgb=np.array(rgb_list, dtype=np.float32),
                        frame_idx=np.array(idx_list, dtype=np.int32),
                        fps=np.float32(fps),
                        skin_frac=np.array(skin_frac_list, dtype=np.float32))
    return video_id, len(rgb_list), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=Path, default=Path("data/mcd_videos"))
    ap.add_argument("--landmarks_root", type=Path, default=Path("data/face_landmarks"))
    ap.add_argument("--out_root", type=Path, default=Path("data/rgb_traces_mp"))
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
