"""Build UBFC_test_stream.txt index from UBFC videos + Stage-0 landmarks.

UBFC2 ground_truth.txt is 3 whitespace-separated lines:
    line 1: PPG signal samples (length == N_video_frames, sampled at video fps)
    line 2: per-sample HR trace in bpm
    line 3: timestamps in seconds

We take the mean of line 2 as hr_gt. Line 1 is also kept on disk so the
inference step / eval can use the full PPG waveform as ground truth.
"""
import argparse
from pathlib import Path

import numpy as np

CLIP_STRIDE = 160
CLIP_FRAMES = 220


def load_ubfc_ppg(gt_path: Path) -> tuple[np.ndarray, float]:
    """Return (ppg_signal, mean_hr_bpm). Raises if file malformed."""
    lines = gt_path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"ground_truth.txt has only {len(lines)} lines")
    ppg = np.fromstring(lines[0], sep=" ", dtype=np.float32)
    hr_trace = np.fromstring(lines[1], sep=" ", dtype=np.float32)
    if hr_trace.size == 0:
        return ppg, 0.0
    hr_trace = hr_trace[np.isfinite(hr_trace) & (hr_trace > 20) & (hr_trace < 240)]
    mean_hr = float(np.mean(hr_trace)) if hr_trace.size else 0.0
    return ppg, mean_hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=Path, required=True,
                    help="UBFC root containing subjectN/vid.avi + ground_truth.txt")
    ap.add_argument("--landmarks_root", type=Path,
                    default=Path("data/face_landmarks_ubfc"))
    ap.add_argument("--ppg_out", type=Path, default=Path("data/ubfc_ppg"),
                    help="Directory where ground-truth PPG arrays are cached as .npy")
    ap.add_argument("--out", type=Path,
                    default=Path("data/UBFC_test_stream.txt"))
    args = ap.parse_args()

    args.ppg_out.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.parent for p in args.videos_root.glob("*/vid.avi"))
    print(f"[scan]   {len(subjects)} UBFC subjects in {args.videos_root}")

    rows = []
    skipped = 0
    for subj_dir in subjects:
        subject = subj_dir.name
        lm_path = args.landmarks_root / f"{subject}.npz"
        gt_path = subj_dir / "ground_truth.txt"
        if not lm_path.exists():
            print(f"[skip]   {subject}: no landmarks")
            skipped += 1; continue
        if not gt_path.exists():
            print(f"[skip]   {subject}: no ground_truth.txt")
            skipped += 1; continue

        lm = np.load(lm_path)
        valid = lm["valid"]
        n_valid = int(valid.sum())
        if n_valid < CLIP_FRAMES + CLIP_STRIDE:
            print(f"[skip]   {subject}: only {n_valid} valid frames")
            skipped += 1; continue
        n_clips = max((n_valid - (CLIP_FRAMES - CLIP_STRIDE)) // CLIP_STRIDE, 0)
        fps = float(lm["fps"])

        try:
            ppg, hr_gt = load_ubfc_ppg(gt_path)
        except Exception as e:
            print(f"[skip]   {subject}: GT parse error: {e}")
            skipped += 1; continue
        np.save(args.ppg_out / f"{subject}.npy", ppg)

        rows.append(f"{subject} {n_clips} {fps:.4f} {hr_gt:.4f}")

    args.out.write_text("\n".join(rows) + "\n")
    print(f"[done]   wrote {len(rows)} entries to {args.out} (skipped {skipped})")
    print(f"[done]   GT PPG arrays cached to {args.ppg_out}")


if __name__ == "__main__":
    main()
