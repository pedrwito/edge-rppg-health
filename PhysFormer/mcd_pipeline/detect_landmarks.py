"""Stage 0: detect MediaPipe FaceMesh landmarks for every frame in every video.

Runs MediaPipe FaceMesh once per video, saves landmarks + smoothed bboxes to disk
so downstream pipelines (PhysFormer crops, iPPG RGB traces) reuse the same
face localization without redoing detection.

Output per video at data/face_landmarks/<video_id>.npz:
  landmarks: (T, 478, 2) int16   pixel (x, y) per landmark per frame
  bboxes:    (T, 4) int16        smoothed [x1, y1, x2, y2] face bbox per frame
  valid:     (T,) bool           True if MediaPipe found a face in that frame
  fps, width, height: scalar metadata
"""
import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Mediapipe is verbose at import-time; mute the worst of it.
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")


# Smoothing on bbox (cx, cy, side). Higher alpha = more smoothing, more lag.
SMOOTH_ALPHA = 0.7
# Pad detected face bbox by this factor to give margin for movement.
BBOX_PAD = 1.25


def smooth_bboxes(bboxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """EMA smoothing on bbox center + size; linear-interpolate gaps."""
    T = len(bboxes)
    if not valid.any():
        return bboxes.copy()
    # Convert to (cx, cy, side)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    side = np.maximum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]).astype(np.float32)

    # Linear interpolate over invalid frames using valid-frame indices
    valid_idx = np.where(valid)[0]
    all_idx = np.arange(T)
    cx = np.interp(all_idx, valid_idx, cx[valid_idx])
    cy = np.interp(all_idx, valid_idx, cy[valid_idx])
    side = np.interp(all_idx, valid_idx, side[valid_idx])

    # EMA
    out_cx, out_cy, out_side = np.empty(T), np.empty(T), np.empty(T)
    out_cx[0], out_cy[0], out_side[0] = cx[0], cy[0], side[0]
    for t in range(1, T):
        out_cx[t] = SMOOTH_ALPHA * out_cx[t-1] + (1 - SMOOTH_ALPHA) * cx[t]
        out_cy[t] = SMOOTH_ALPHA * out_cy[t-1] + (1 - SMOOTH_ALPHA) * cy[t]
        out_side[t] = SMOOTH_ALPHA * out_side[t-1] + (1 - SMOOTH_ALPHA) * side[t]

    # Convert back to bboxes
    half = out_side / 2.0
    smooth = np.empty_like(bboxes, dtype=np.int16)
    smooth[:, 0] = (out_cx - half).astype(np.int16)
    smooth[:, 1] = (out_cy - half).astype(np.int16)
    smooth[:, 2] = (out_cx + half).astype(np.int16)
    smooth[:, 3] = (out_cy + half).astype(np.int16)
    return smooth


def process_one(args: tuple[Path, Path]) -> tuple[str, int, str | None]:
    video_path, out_dir = args
    out_path = out_dir / f"{video_path.stem}.npz"
    if out_path.exists():
        return video_path.name, 0, "skip"

    # Import inside worker so mediapipe loads fresh per process
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_path.name, 0, "open_fail"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarks_per_frame: list[np.ndarray] = []
    bboxes_raw: list[np.ndarray] = []
    valid_list: list[bool] = []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0]
                pts = np.array([(int(p.x * width), int(p.y * height)) for p in fl.landmark],
                               dtype=np.int16)
                landmarks_per_frame.append(pts)
                # Tight bbox from convex hull, then pad
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                side = max(x_max - x_min, y_max - y_min) * BBOX_PAD
                half = side / 2.0
                bbox = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.int16)
                bboxes_raw.append(bbox)
                valid_list.append(True)
            else:
                landmarks_per_frame.append(np.zeros((478, 2), dtype=np.int16))
                bboxes_raw.append(np.zeros(4, dtype=np.int16))
                valid_list.append(False)
    cap.release()

    T = len(valid_list)
    if T == 0:
        return video_path.name, 0, "no_frames"
    valid = np.array(valid_list, dtype=bool)
    if not valid.any():
        return video_path.name, T, "no_face_detected_anywhere"

    landmarks = np.stack(landmarks_per_frame).astype(np.int16)
    bboxes_raw_arr = np.stack(bboxes_raw).astype(np.int16)
    bboxes_smooth = smooth_bboxes(bboxes_raw_arr, valid)

    np.savez_compressed(out_path,
                        landmarks=landmarks,
                        bboxes=bboxes_smooth,
                        valid=valid,
                        fps=np.float32(fps),
                        width=np.int32(width),
                        height=np.int32(height))
    return video_path.name, T, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=Path, default=Path("data/mcd_videos"))
    ap.add_argument("--out_root", type=Path, default=Path("data/face_landmarks"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only this many videos (for testing)")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.videos_root.glob("*.avi"))
    if args.limit:
        videos = videos[:args.limit]
    print(f"[scan]   {len(videos)} videos, workers={args.workers}")

    work = [(v, args.out_root) for v in videos]
    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, w) for w in work]
        for fut in as_completed(futures):
            name, T, status = fut.result()
            if status == "skip":
                skipped += 1
            elif status is not None:
                errors += 1
                print(f"[err]    {name}: {status} (T={T})")
            else:
                done += 1
                if done % 25 == 0:
                    print(f"[ok]     {done} new, {skipped} skipped, {errors} err "
                          f"({done+skipped+errors}/{len(work)})")
    print(f"[done]   {done} new, {skipped} skipped, {errors} errors -> {args.out_root}")


if __name__ == "__main__":
    main()
